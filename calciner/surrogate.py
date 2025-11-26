"""
Neural Surrogate for Flash Calciner Dynamics

Learns discrete-time dynamics of the full 140-dimensional PDE/ODE system:
    x_{k+1} = f_θ(x_k, u_k)

This enables fast MPC by replacing expensive scipy.integrate.solve_ivp calls
with a single neural network forward pass.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Tuple, Optional, Dict, List

from .physics import SimplifiedFlashCalciner, N_SPECIES, L

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Physics Model Wrapper for Data Generation
# =============================================================================

class CalcinerSimulator:
    """
    Wrapper around SimplifiedFlashCalciner for generating training data.
    Simulates discrete-time transitions with control inputs.
    """
    
    def __init__(self, N_z: int = 20, dt: float = 0.1):
        self.N_z = N_z
        self.dt = dt
        self.model = SimplifiedFlashCalciner(N_z=N_z)
        
        self.state_dim = N_SPECIES * N_z + 2 * N_z  # 140 for N_z=20
        self.control_dim = 2  # T_g_in, T_s_in
        
        self.c_in_default = np.array([0.15, 0.79, 0.31, 5.81, 3.74])
        
    def state_to_vector(self, c: np.ndarray, T_s: np.ndarray, T_g: np.ndarray) -> np.ndarray:
        return self.model.pack(c, T_s, T_g)
    
    def vector_to_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.model.unpack(x)
    
    def step(self, x: np.ndarray, u: np.ndarray, c_in: Optional[np.ndarray] = None) -> np.ndarray:
        if c_in is None:
            c_in = self.c_in_default
            
        T_g_in, T_s_in = u[0], u[1]
        c, T_s, T_g = self.vector_to_state(x)
        y0 = self.model.pack(c, T_s, T_g)
        
        sol = solve_ivp(
            lambda t, y: self.model.rhs(t, y, c_in, T_s_in, T_g_in),
            (0.0, self.dt),
            y0,
            method='RK45',
            rtol=1e-4,
            atol=1e-6
        )
        return sol.y[:, -1]
    
    def generate_trajectory(self, x0: np.ndarray, u_seq: np.ndarray, 
                           c_in: Optional[np.ndarray] = None) -> np.ndarray:
        T = len(u_seq)
        x_traj = np.zeros((T + 1, self.state_dim))
        x_traj[0] = x0
        
        x = x0.copy()
        for t in range(T):
            x = self.step(x, u_seq[t], c_in)
            x_traj[t + 1] = x
        return x_traj
    
    def sample_random_state(self) -> np.ndarray:
        c = np.zeros((N_SPECIES, self.N_z))
        c[0, :] = np.random.uniform(0.01, 0.20, self.N_z) * np.linspace(1, 0.1, self.N_z)
        c[1, :] = np.random.uniform(0.5, 1.0, self.N_z)
        c[2, :] = np.random.uniform(0.1, 0.5, self.N_z) * np.linspace(0.1, 1, self.N_z)
        c[3, :] = np.random.uniform(4.0, 8.0, self.N_z)
        c[4, :] = np.random.uniform(2.0, 6.0, self.N_z) * np.linspace(1, 1.5, self.N_z)
        
        z = np.linspace(0, 1, self.N_z)
        T_s_in = np.random.uniform(600, 750)
        T_s_out = np.random.uniform(900, 1100)
        T_s = T_s_in + (T_s_out - T_s_in) * (1 - np.exp(-3*z))
        
        T_g_in = np.random.uniform(1100, 1350)
        T_g_out = np.random.uniform(900, 1100)
        T_g = T_g_in + (T_g_out - T_g_in) * z
        
        return self.state_to_vector(c, T_s, T_g)
    
    def sample_random_control(self) -> np.ndarray:
        T_g_in = np.random.uniform(1000, 1350)
        T_s_in = np.random.uniform(600, 750)
        return np.array([T_g_in, T_s_in])


# =============================================================================
# Dataset
# =============================================================================

class TransitionDataset(Dataset):
    """Dataset of (x, u, x_next) transitions."""
    
    def __init__(self, states: np.ndarray, controls: np.ndarray, 
                 next_states: np.ndarray, normalize: bool = True):
        self.normalize = normalize
        
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-6
        self.control_mean = controls.mean(axis=0)
        self.control_std = controls.std(axis=0) + 1e-6
        
        if normalize:
            self.states = (states - self.state_mean) / self.state_std
            self.next_states = (next_states - self.state_mean) / self.state_std
            self.controls = (controls - self.control_mean) / self.control_std
        else:
            self.states = states
            self.next_states = next_states
            self.controls = controls
        
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.controls = torch.tensor(self.controls, dtype=torch.float32)
        self.next_states = torch.tensor(self.next_states, dtype=torch.float32)
        self.residuals = self.next_states - self.states
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'control': self.controls[idx],
            'next_state': self.next_states[idx],
            'residual': self.residuals[idx]
        }
    
    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        return {
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'control_mean': self.control_mean,
            'control_std': self.control_std
        }


def generate_training_data(simulator: CalcinerSimulator, 
                          n_trajectories: int = 100,
                          trajectory_length: int = 50,
                          verbose: bool = True) -> TransitionDataset:
    """Generate training data by running physics simulation."""
    all_states = []
    all_controls = []
    all_next_states = []
    
    start_time = time.time()
    
    for traj_idx in range(n_trajectories):
        x0 = simulator.sample_random_state()
        u_base = simulator.sample_random_control()
        u_seq = np.zeros((trajectory_length, simulator.control_dim))
        
        for t in range(trajectory_length):
            if t == 0 or np.random.rand() < 0.1:
                u_base = simulator.sample_random_control()
            u_seq[t] = u_base + np.random.randn(simulator.control_dim) * np.array([20, 10])
            u_seq[t, 0] = np.clip(u_seq[t, 0], 900, 1400)
            u_seq[t, 1] = np.clip(u_seq[t, 1], 550, 800)
        
        x_traj = simulator.generate_trajectory(x0, u_seq)
        
        for t in range(trajectory_length):
            all_states.append(x_traj[t])
            all_controls.append(u_seq[t])
            all_next_states.append(x_traj[t + 1])
        
        if verbose and (traj_idx + 1) % 10 == 0:
            print(f"  Generated {traj_idx + 1}/{n_trajectories} trajectories", flush=True)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"  ✓ Generated {len(all_states)} transitions in {elapsed:.1f}s", flush=True)
    
    return TransitionDataset(np.array(all_states), np.array(all_controls), np.array(all_next_states))


# =============================================================================
# Neural Network Architectures
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        
    def forward(self, x):
        return x + self.net(x)


class MLPDynamics(nn.Module):
    """Simple MLP for learning dynamics residual."""
    
    def __init__(self, state_dim: int, control_dim: int, 
                 hidden_dims: List[int] = [512, 512, 256]):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        layers = []
        input_dim = state_dim + control_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, state_dim))
        
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([x, u], dim=-1)
        dx = self.net(xu)
        return x + dx


class SpatiallyAwareDynamics(nn.Module):
    """Architecture that respects spatial structure of the PDE."""
    
    def __init__(self, N_z: int = 20, hidden_dim: int = 256, n_species: int = 5):
        super().__init__()
        
        self.N_z = N_z
        self.n_species = n_species
        self.conc_dim = n_species * N_z
        self.temp_dim = 2 * N_z
        self.state_dim = self.conc_dim + self.temp_dim
        self.control_dim = 2
        
        self.conc_conv = nn.Sequential(
            nn.Conv1d(n_species + 2 + 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, n_species, kernel_size=3, padding=1),
        )
        
        self.temp_conv = nn.Sequential(
            nn.Conv1d(2 + n_species + 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, 2, kernel_size=3, padding=1),
        )
        
        for conv in [self.conc_conv[-1], self.temp_conv[-1]]:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
    
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        
        conc = x[:, :self.conc_dim].view(batch, self.n_species, self.N_z)
        temps = x[:, self.conc_dim:].view(batch, 2, self.N_z)
        u_spatial = u.unsqueeze(-1).expand(-1, -1, self.N_z)
        
        conc_input = torch.cat([conc, temps, u_spatial], dim=1)
        d_conc = self.conc_conv(conc_input)
        
        temp_input = torch.cat([temps, conc, u_spatial], dim=1)
        d_temps = self.temp_conv(temp_input)
        
        d_conc_flat = d_conc.view(batch, -1)
        d_temps_flat = d_temps.view(batch, -1)
        dx = torch.cat([d_conc_flat, d_temps_flat], dim=1)
        
        return x + dx


# =============================================================================
# Training
# =============================================================================

def train_surrogate(model: nn.Module, 
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None,
                   n_epochs: int = 100,
                   lr: float = 1e-3,
                   weight_decay: float = 1e-5,
                   device: torch.device = DEVICE) -> Dict:
    """Train surrogate model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            state = batch['state'].to(device)
            control = batch['control'].to(device)
            next_state = batch['next_state'].to(device)
            
            optimizer.zero_grad()
            pred = model(state, control)
            loss = torch.mean((pred - next_state) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        history['train_loss'].append(train_loss)
        
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    state = batch['state'].to(device)
                    control = batch['control'].to(device)
                    next_state = batch['next_state'].to(device)
                    pred = model(state, control)
                    val_loss += torch.mean((pred - next_state) ** 2).item()
                    n_val += 1
            val_loss /= n_val
            history['val_loss'].append(val_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            val_str = f", val={history['val_loss'][-1]:.2e}" if val_loader else ""
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={train_loss:.2e}{val_str}", flush=True)
    
    return history


# =============================================================================
# Surrogate Model Wrapper
# =============================================================================

class SurrogateModel:
    """Wrapper for neural surrogate with normalization handling."""
    
    def __init__(self, model: nn.Module, norm_params: Dict[str, np.ndarray],
                 device: torch.device = DEVICE):
        self.model = model.to(device)
        self.device = device
        
        self.state_mean = torch.tensor(norm_params['state_mean'], dtype=torch.float32, device=device)
        self.state_std = torch.tensor(norm_params['state_std'], dtype=torch.float32, device=device)
        self.control_mean = torch.tensor(norm_params['control_mean'], dtype=torch.float32, device=device)
        self.control_std = torch.tensor(norm_params['control_std'], dtype=torch.float32, device=device)
        
        self.state_dim = len(norm_params['state_mean'])
        self.control_dim = len(norm_params['control_mean'])
        
    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.state_mean) / self.state_std
    
    def denormalize_state(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.state_std + self.state_mean
    
    def normalize_control(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.control_mean) / self.control_std
    
    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_norm = self.normalize_state(x)
        u_norm = self.normalize_control(u)
        x_next_norm = self.model(x_norm, u_norm)
        return self.denormalize_state(x_next_norm)
    
    def rollout(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        batch, T, _ = u_seq.shape
        x_traj = torch.zeros(batch, T + 1, self.state_dim, device=self.device)
        x_traj[:, 0] = x0
        
        x = x0
        for t in range(T):
            x = self.step(x, u_seq[:, t])
            x_traj[:, t + 1] = x
        return x_traj
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self):
        self.model.train()
        return self

