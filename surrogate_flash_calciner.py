"""
Neural Surrogate for Flash Calciner Dynamics

Learns discrete-time dynamics of the full 140-dimensional PDE/ODE system:
    x_{k+1} = f_θ(x_k, u_k)

This enables fast MPC by replacing expensive scipy.integrate.solve_ivp calls
with a single neural network forward pass.

Key features:
1. Residual learning: predicts Δx = x_{k+1} - x_k for better conditioning
2. Physics-informed normalization based on known concentration/temperature scales
3. Multiple architecture options (MLP, ResNet, FNO-inspired)
4. Differentiable for gradient-based MPC optimization
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

# Import physics model
from flash_calciner import SimplifiedFlashCalciner, N_SPECIES, L

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# Physics Model Wrapper for Data Generation
# =============================================================================

class CalcinerSimulator:
    """
    Wrapper around SimplifiedFlashCalciner for generating training data.
    Simulates discrete-time transitions with control inputs.
    """
    
    def __init__(self, N_z: int = 20, dt: float = 0.1):
        """
        Parameters
        ----------
        N_z : int
            Number of spatial discretization cells
        dt : float
            Discrete time step for surrogate [s]
        """
        self.N_z = N_z
        self.dt = dt
        self.model = SimplifiedFlashCalciner(N_z=N_z)
        
        # State dimension: 5 species × N_z + 2 temperatures × N_z
        self.state_dim = N_SPECIES * N_z + 2 * N_z  # 140 for N_z=20
        self.control_dim = 2  # T_g_in, T_s_in (or just T_g_in)
        
        # Default inlet concentrations [mol/m³]
        self.c_in_default = np.array([0.15, 0.79, 0.31, 5.81, 3.74])
        
        # Normalization statistics (computed from data or set physically)
        self.state_mean = None
        self.state_std = None
        self.control_mean = None
        self.control_std = None
        
    def state_to_vector(self, c: np.ndarray, T_s: np.ndarray, T_g: np.ndarray) -> np.ndarray:
        """Convert (c, T_s, T_g) to flat state vector."""
        return self.model.pack(c, T_s, T_g)
    
    def vector_to_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert flat state vector to (c, T_s, T_g)."""
        return self.model.unpack(x)
    
    def step(self, x: np.ndarray, u: np.ndarray, c_in: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Single discrete-time step using physics simulator.
        
        Parameters
        ----------
        x : np.ndarray, shape (state_dim,)
            Current state
        u : np.ndarray, shape (2,)
            Control: [T_g_in, T_s_in] in Kelvin
        c_in : np.ndarray, optional
            Inlet concentrations. If None, use default.
            
        Returns
        -------
        x_next : np.ndarray, shape (state_dim,)
            Next state after dt seconds
        """
        if c_in is None:
            c_in = self.c_in_default
            
        T_g_in, T_s_in = u[0], u[1]
        
        c, T_s, T_g = self.vector_to_state(x)
        
        # Solve ODE for dt seconds
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
        """
        Generate trajectory from initial state with control sequence.
        
        Parameters
        ----------
        x0 : np.ndarray, shape (state_dim,)
            Initial state
        u_seq : np.ndarray, shape (T, control_dim)
            Control sequence
        c_in : np.ndarray, optional
            Inlet concentrations
            
        Returns
        -------
        x_traj : np.ndarray, shape (T+1, state_dim)
            State trajectory including initial state
        """
        T = len(u_seq)
        x_traj = np.zeros((T + 1, self.state_dim))
        x_traj[0] = x0
        
        x = x0.copy()
        for t in range(T):
            x = self.step(x, u_seq[t], c_in)
            x_traj[t + 1] = x
            
        return x_traj
    
    def sample_random_state(self) -> np.ndarray:
        """Sample a random but physically plausible state."""
        # Concentrations: perturb around typical values
        c = np.zeros((N_SPECIES, self.N_z))
        # Kaolinite: decreases along reactor
        c[0, :] = np.random.uniform(0.01, 0.20, self.N_z) * np.linspace(1, 0.1, self.N_z)
        # Quartz: roughly constant
        c[1, :] = np.random.uniform(0.5, 1.0, self.N_z)
        # Metakaolin: increases along reactor
        c[2, :] = np.random.uniform(0.1, 0.5, self.N_z) * np.linspace(0.1, 1, self.N_z)
        # N2: roughly constant
        c[3, :] = np.random.uniform(4.0, 8.0, self.N_z)
        # H2O: increases along reactor (from reaction)
        c[4, :] = np.random.uniform(2.0, 6.0, self.N_z) * np.linspace(1, 1.5, self.N_z)
        
        # Temperatures: smooth profiles
        z = np.linspace(0, 1, self.N_z)
        T_s_in = np.random.uniform(600, 750)
        T_s_out = np.random.uniform(900, 1100)
        T_s = T_s_in + (T_s_out - T_s_in) * (1 - np.exp(-3*z))
        
        T_g_in = np.random.uniform(1100, 1350)
        T_g_out = np.random.uniform(900, 1100)
        T_g = T_g_in + (T_g_out - T_g_in) * z
        
        return self.state_to_vector(c, T_s, T_g)
    
    def sample_random_control(self) -> np.ndarray:
        """Sample random control input."""
        T_g_in = np.random.uniform(1000, 1350)
        T_s_in = np.random.uniform(600, 750)
        return np.array([T_g_in, T_s_in])


# =============================================================================
# Dataset for Training
# =============================================================================

class TransitionDataset(Dataset):
    """Dataset of (x, u, x_next) transitions."""
    
    def __init__(self, states: np.ndarray, controls: np.ndarray, 
                 next_states: np.ndarray, normalize: bool = True):
        """
        Parameters
        ----------
        states : np.ndarray, shape (N, state_dim)
        controls : np.ndarray, shape (N, control_dim)
        next_states : np.ndarray, shape (N, state_dim)
        normalize : bool
            Whether to normalize data
        """
        self.normalize = normalize
        
        # Compute statistics
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
        
        # Convert to tensors
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.controls = torch.tensor(self.controls, dtype=torch.float32)
        self.next_states = torch.tensor(self.next_states, dtype=torch.float32)
        
        # Residuals for residual learning
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
        """Return normalization parameters for inference."""
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
    """
    Generate training data by running physics simulation.
    
    Parameters
    ----------
    simulator : CalcinerSimulator
        Physics-based simulator
    n_trajectories : int
        Number of trajectories to generate
    trajectory_length : int
        Number of steps per trajectory
    verbose : bool
        Print progress
        
    Returns
    -------
    dataset : TransitionDataset
        Dataset of transitions
    """
    all_states = []
    all_controls = []
    all_next_states = []
    
    start_time = time.time()
    
    for traj_idx in range(n_trajectories):
        if verbose and traj_idx % 10 == 0:
            print(f"  Generating trajectory {traj_idx + 1}/{n_trajectories}...")
            
        # Random initial state
        x0 = simulator.sample_random_state()
        
        # Random control sequence (with some smoothness)
        u_base = simulator.sample_random_control()
        u_seq = np.zeros((trajectory_length, simulator.control_dim))
        for t in range(trajectory_length):
            # Slowly varying control with occasional jumps
            if t == 0 or np.random.rand() < 0.1:
                u_base = simulator.sample_random_control()
            u_seq[t] = u_base + np.random.randn(simulator.control_dim) * np.array([20, 10])
            u_seq[t, 0] = np.clip(u_seq[t, 0], 900, 1400)  # T_g_in bounds
            u_seq[t, 1] = np.clip(u_seq[t, 1], 550, 800)   # T_s_in bounds
        
        # Generate trajectory
        x_traj = simulator.generate_trajectory(x0, u_seq)
        
        # Extract transitions
        for t in range(trajectory_length):
            all_states.append(x_traj[t])
            all_controls.append(u_seq[t])
            all_next_states.append(x_traj[t + 1])
    
    elapsed = time.time() - start_time
    
    states = np.array(all_states)
    controls = np.array(all_controls)
    next_states = np.array(all_next_states)
    
    if verbose:
        print(f"  Generated {len(states)} transitions in {elapsed:.1f}s")
        print(f"  ({elapsed/len(states)*1000:.2f} ms per transition)")
    
    return TransitionDataset(states, controls, next_states)


# =============================================================================
# Neural Network Architectures
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual MLP block with skip connection."""
    
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
    """
    Simple MLP for learning dynamics residual.
    
    Predicts: Δx = f_θ(x, u)
    Next state: x_{k+1} = x_k + Δx
    """
    
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
        
        # Initialize last layer to small values (helps with residual learning)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Predict next state.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, state_dim)
            Current normalized state
        u : torch.Tensor, shape (batch, control_dim)
            Normalized control
            
        Returns
        -------
        x_next : torch.Tensor, shape (batch, state_dim)
            Predicted normalized next state
        """
        xu = torch.cat([x, u], dim=-1)
        dx = self.net(xu)
        return x + dx  # Residual connection


class ResNetDynamics(nn.Module):
    """
    ResNet-style architecture for dynamics.
    Better for capturing complex nonlinear dynamics.
    """
    
    def __init__(self, state_dim: int, control_dim: int,
                 hidden_dim: int = 512, n_blocks: int = 4):
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        # Initial embedding
        self.embed = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2)
            for _ in range(n_blocks)
        ])
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Initialize output to zero
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([x, u], dim=-1)
        h = self.embed(xu)
        for block in self.blocks:
            h = block(h)
        dx = self.output(h)
        return x + dx


class SpatiallyAwareDynamics(nn.Module):
    """
    Architecture that respects spatial structure of the PDE.
    Uses separate networks for concentration and temperature,
    with coupling through attention or concatenation.
    
    State structure for N_z=20:
    - Concentrations: [5 × 20 = 100] (reshaped to 5 × 20)
    - Temperatures: [2 × 20 = 40] (T_s and T_g)
    """
    
    def __init__(self, N_z: int = 20, hidden_dim: int = 256, n_species: int = 5):
        super().__init__()
        
        self.N_z = N_z
        self.n_species = n_species
        self.conc_dim = n_species * N_z  # 100
        self.temp_dim = 2 * N_z  # 40
        self.state_dim = self.conc_dim + self.temp_dim  # 140
        self.control_dim = 2
        
        # 1D convolutions for spatial processing
        self.conc_conv = nn.Sequential(
            nn.Conv1d(n_species + 2 + 2, hidden_dim, kernel_size=3, padding=1),  # +2 for temps, +2 for control broadcast
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, n_species, kernel_size=3, padding=1),
        )
        
        self.temp_conv = nn.Sequential(
            nn.Conv1d(2 + n_species + 2, hidden_dim // 2, kernel_size=3, padding=1),  # temps + avg conc + control
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, 2, kernel_size=3, padding=1),
        )
        
        # Initialize to small values
        for conv in [self.conc_conv[-1], self.temp_conv[-1]]:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
    
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        
        # Unpack state
        conc = x[:, :self.conc_dim].view(batch, self.n_species, self.N_z)  # (B, 5, 20)
        temps = x[:, self.conc_dim:].view(batch, 2, self.N_z)  # (B, 2, 20)
        
        # Broadcast control to spatial dimension
        u_spatial = u.unsqueeze(-1).expand(-1, -1, self.N_z)  # (B, 2, 20)
        
        # Process concentrations (with temperature and control context)
        conc_input = torch.cat([conc, temps, u_spatial], dim=1)  # (B, 9, 20)
        d_conc = self.conc_conv(conc_input)  # (B, 5, 20)
        
        # Process temperatures (with concentration and control context)
        temp_input = torch.cat([temps, conc, u_spatial], dim=1)  # (B, 9, 20)
        d_temps = self.temp_conv(temp_input)  # (B, 2, 20)
        
        # Pack residuals
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
    """
    Train surrogate model.
    
    Returns
    -------
    history : dict
        Training history with losses
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Training
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        history['train_loss'].append(train_loss)
        
        # Validation
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
        
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        if (epoch + 1) % 10 == 0:
            val_str = f", val={val_loss:.2e}" if val_loader else ""
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: train={train_loss:.2e}{val_str}, "
                  f"time={epoch_time:.1f}s")
    
    return history


# =============================================================================
# Surrogate Model Wrapper for MPC
# =============================================================================

class SurrogateModel:
    """
    Wrapper for neural surrogate with normalization handling.
    Provides interface compatible with MPC.
    """
    
    def __init__(self, model: nn.Module, norm_params: Dict[str, np.ndarray],
                 device: torch.device = DEVICE):
        self.model = model.to(device)
        self.device = device
        
        # Store normalization parameters as tensors
        self.state_mean = torch.tensor(norm_params['state_mean'], 
                                       dtype=torch.float32, device=device)
        self.state_std = torch.tensor(norm_params['state_std'], 
                                      dtype=torch.float32, device=device)
        self.control_mean = torch.tensor(norm_params['control_mean'], 
                                         dtype=torch.float32, device=device)
        self.control_std = torch.tensor(norm_params['control_std'], 
                                        dtype=torch.float32, device=device)
        
        self.state_dim = len(norm_params['state_mean'])
        self.control_dim = len(norm_params['control_mean'])
        
    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.state_mean) / self.state_std
    
    def denormalize_state(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.state_std + self.state_mean
    
    def normalize_control(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.control_mean) / self.control_std
    
    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Single step prediction (handles normalization).
        
        Parameters
        ----------
        x : torch.Tensor, shape (..., state_dim)
            Unnormalized state
        u : torch.Tensor, shape (..., control_dim)
            Unnormalized control
            
        Returns
        -------
        x_next : torch.Tensor, shape (..., state_dim)
            Predicted unnormalized next state
        """
        x_norm = self.normalize_state(x)
        u_norm = self.normalize_control(u)
        
        x_next_norm = self.model(x_norm, u_norm)
        return self.denormalize_state(x_next_norm)
    
    def rollout(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        """
        Multi-step rollout.
        
        Parameters
        ----------
        x0 : torch.Tensor, shape (batch, state_dim)
            Initial states
        u_seq : torch.Tensor, shape (batch, T, control_dim)
            Control sequences
            
        Returns
        -------
        x_traj : torch.Tensor, shape (batch, T+1, state_dim)
            State trajectories
        """
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


# =============================================================================
# MPC with Surrogate
# =============================================================================

class SurrogateMPC:
    """
    Economic MPC using neural surrogate for fast predictions.
    Uses gradient-based optimization through the differentiable surrogate.
    """
    
    def __init__(self, surrogate: SurrogateModel, 
                 horizon: int = 10,
                 u_min: np.ndarray = np.array([900, 550]),
                 u_max: np.ndarray = np.array([1350, 800]),
                 n_iter: int = 50,
                 lr: float = 5.0):
        """
        Parameters
        ----------
        surrogate : SurrogateModel
            Trained neural surrogate
        horizon : int
            Prediction horizon
        u_min, u_max : np.ndarray
            Control bounds [T_g_in, T_s_in]
        n_iter : int
            Number of optimization iterations
        lr : float
            Learning rate for control optimization
        """
        self.surrogate = surrogate.eval()
        self.horizon = horizon
        self.u_min = torch.tensor(u_min, dtype=torch.float32, device=surrogate.device)
        self.u_max = torch.tensor(u_max, dtype=torch.float32, device=surrogate.device)
        self.n_iter = n_iter
        self.lr = lr
        
        # Cost weights
        self.N_z = 20
        self.n_species = 5
        
    def compute_cost(self, x_traj: torch.Tensor, u_seq: torch.Tensor,
                    alpha_min: float = 0.95) -> torch.Tensor:
        """
        Compute MPC cost over trajectory.
        
        Cost = energy + constraint_penalty
        
        Energy: proportional to inlet gas temperature
        Constraint: conversion α ≥ α_min
        """
        # Energy cost (normalized by reference)
        T_g_in = u_seq[:, :, 0]  # (batch, T)
        energy_cost = torch.mean((T_g_in - 900) / (1350 - 900), dim=1)
        
        # Conversion constraint
        # Conversion α = 1 - c_kaolinite_out / c_kaolinite_in
        conc_dim = self.n_species * self.N_z
        c_kaolinite = x_traj[:, 1:, :conc_dim].view(-1, self.horizon, self.n_species, self.N_z)
        c_kao_out = c_kaolinite[:, :, 0, -1]  # Outlet kaolinite concentration
        c_kao_in = 0.15  # Reference inlet
        alpha = 1.0 - c_kao_out / (c_kao_in + 1e-6)
        alpha = torch.clamp(alpha, 0, 1)
        
        # Soft constraint penalty
        violation = torch.relu(alpha_min - alpha)
        constraint_cost = 1000.0 * torch.mean(violation ** 2, dim=1)
        
        # Terminal temperature target (want solid temp around 1066 K)
        T_s = x_traj[:, -1, conc_dim:conc_dim + self.N_z]
        T_s_out = T_s[:, -1]
        temp_cost = 0.01 * ((T_s_out - 1066) / 100) ** 2
        
        return energy_cost + constraint_cost + temp_cost
    
    def solve(self, x0: np.ndarray, alpha_min: float = 0.95) -> Tuple[np.ndarray, float]:
        """
        Solve MPC problem.
        
        Parameters
        ----------
        x0 : np.ndarray, shape (state_dim,)
            Current state
        alpha_min : float
            Minimum conversion requirement
            
        Returns
        -------
        u_opt : np.ndarray, shape (control_dim,)
            Optimal control for current step
        cost : float
            Optimal cost value
        """
        device = self.surrogate.device
        
        # Convert to tensor
        x0_t = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Initialize control sequence (warm start with previous solution or midpoint)
        u_init = (self.u_min + self.u_max) / 2
        u_seq = u_init.unsqueeze(0).unsqueeze(0).expand(1, self.horizon, -1).clone()
        u_seq = u_seq.requires_grad_(True)
        
        # Optimization
        optimizer = torch.optim.Adam([u_seq], lr=self.lr)
        
        best_u = None
        best_cost = float('inf')
        
        for i in range(self.n_iter):
            optimizer.zero_grad()
            
            # Project to feasible region
            with torch.no_grad():
                u_seq.data = torch.clamp(u_seq.data, self.u_min, self.u_max)
            
            # Rollout
            x_traj = self.surrogate.rollout(x0_t, u_seq)
            
            # Cost
            cost = self.compute_cost(x_traj, u_seq, alpha_min).mean()
            
            # Track best
            if cost.item() < best_cost:
                best_cost = cost.item()
                best_u = u_seq.detach().clone()
            
            # Backward
            cost.backward()
            optimizer.step()
        
        # Return first control
        u_opt = best_u[0, 0].cpu().numpy()
        return u_opt, best_cost


# =============================================================================
# Evaluation and Comparison
# =============================================================================

def compare_surrogate_physics(surrogate: SurrogateModel, 
                             simulator: CalcinerSimulator,
                             n_test: int = 10,
                             horizon: int = 20) -> Dict:
    """
    Compare surrogate predictions with physics simulation.
    """
    surrogate.eval()
    
    errors = []
    physics_times = []
    surrogate_times = []
    
    for i in range(n_test):
        x0 = simulator.sample_random_state()
        u_seq = np.array([simulator.sample_random_control() for _ in range(horizon)])
        
        # Physics simulation
        t0 = time.time()
        x_physics = simulator.generate_trajectory(x0, u_seq)
        physics_times.append(time.time() - t0)
        
        # Surrogate prediction
        x0_t = torch.tensor(x0, dtype=torch.float32, device=surrogate.device).unsqueeze(0)
        u_seq_t = torch.tensor(u_seq, dtype=torch.float32, device=surrogate.device).unsqueeze(0)
        
        t0 = time.time()
        with torch.no_grad():
            x_surrogate = surrogate.rollout(x0_t, u_seq_t).cpu().numpy()[0]
        surrogate_times.append(time.time() - t0)
        
        # Relative error
        rel_error = np.mean(np.abs(x_surrogate - x_physics) / (np.abs(x_physics) + 1e-6))
        errors.append(rel_error)
    
    return {
        'mean_rel_error': np.mean(errors),
        'std_rel_error': np.std(errors),
        'physics_time_ms': np.mean(physics_times) * 1000,
        'surrogate_time_ms': np.mean(surrogate_times) * 1000,
        'speedup': np.mean(physics_times) / np.mean(surrogate_times)
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Neural Surrogate for Flash Calciner")
    print("=" * 70)
    
    # Create simulator
    N_z = 20
    dt = 0.1  # 100ms discrete time step
    simulator = CalcinerSimulator(N_z=N_z, dt=dt)
    print(f"\nState dimension: {simulator.state_dim}")
    print(f"Control dimension: {simulator.control_dim}")
    print(f"Discrete time step: {dt}s")
    
    # Generate training data
    print("\n" + "-" * 50)
    print("Generating training data...")
    print("-" * 50)
    
    train_data = generate_training_data(
        simulator, 
        n_trajectories=100, 
        trajectory_length=50
    )
    
    # Split into train/val
    n_total = len(train_data)
    n_val = n_total // 5
    indices = np.random.permutation(n_total)
    
    train_dataset = torch.utils.data.Subset(train_data, indices[n_val:])
    val_dataset = torch.utils.data.Subset(train_data, indices[:n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\n" + "-" * 50)
    print("Training neural surrogate...")
    print("-" * 50)
    
    # Try spatially-aware architecture
    model = SpatiallyAwareDynamics(N_z=N_z)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    history = train_surrogate(
        model, 
        train_loader, 
        val_loader,
        n_epochs=100,
        lr=1e-3
    )
    
    # Create surrogate wrapper
    norm_params = train_data.get_normalization_params()
    surrogate = SurrogateModel(model, norm_params)
    
    # Evaluate
    print("\n" + "-" * 50)
    print("Evaluating surrogate...")
    print("-" * 50)
    
    metrics = compare_surrogate_physics(surrogate, simulator, n_test=20, horizon=20)
    print(f"Mean relative error: {metrics['mean_rel_error']*100:.2f}%")
    print(f"Physics simulation: {metrics['physics_time_ms']:.1f} ms/rollout")
    print(f"Surrogate prediction: {metrics['surrogate_time_ms']:.2f} ms/rollout")
    print(f"Speedup: {metrics['speedup']:.0f}x")
    
    # Test MPC
    print("\n" + "-" * 50)
    print("Testing MPC with surrogate...")
    print("-" * 50)
    
    mpc = SurrogateMPC(surrogate, horizon=10)
    
    # Run MPC for a few steps
    x = simulator.sample_random_state()
    mpc_times = []
    
    for step in range(5):
        t0 = time.time()
        u_opt, cost = mpc.solve(x, alpha_min=0.95)
        mpc_time = time.time() - t0
        mpc_times.append(mpc_time)
        
        print(f"Step {step+1}: u=[{u_opt[0]:.0f}, {u_opt[1]:.0f}] K, "
              f"cost={cost:.3f}, time={mpc_time*1000:.0f}ms")
        
        # Apply control
        x = simulator.step(x, u_opt)
    
    print(f"\nAverage MPC solve time: {np.mean(mpc_times)*1000:.0f}ms")
    
    # Save model
    save_path = Path("surrogate_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
        'model_class': model.__class__.__name__,
        'N_z': N_z,
        'dt': dt
    }, save_path)
    print(f"\nSaved model to {save_path}")
    
    # Plot training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history['train_loss'], label='Train')
    ax.semilogy(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Surrogate Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('surrogate_training.png', dpi=150, bbox_inches='tight')
    print("Saved: surrogate_training.png")
    plt.close()
    
    return surrogate, simulator, mpc


if __name__ == "__main__":
    surrogate, simulator, mpc = main()

