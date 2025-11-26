"""
MPC Controllers for Flash Calciner

Includes:
- MPPI (Model Predictive Path Integral) - sampling-based
- Gradient-based MPC - optimization-based
- Classical MPC - non-surrogate baseline
"""

import numpy as np
import torch
from scipy.optimize import minimize, Bounds
from typing import Tuple, Optional

from .surrogate import SurrogateModel


# =============================================================================
# MPPI Controller (Sampling-based)
# =============================================================================

class SurrogateMPPI:
    """
    MPPI (Model Predictive Path Integral) controller using neural surrogate.
    
    Fast sampling-based MPC that works well with neural network dynamics.
    """
    
    def __init__(self, surrogate: SurrogateModel,
                 horizon: int = 10,
                 n_samples: int = 256,
                 temperature: float = 0.1,
                 u_min: np.ndarray = np.array([900, 550]),
                 u_max: np.ndarray = np.array([1350, 800]),
                 noise_sigma: np.ndarray = np.array([50, 25])):
        
        self.surrogate = surrogate.eval()
        self.horizon = horizon
        self.n_samples = n_samples
        self.temperature = temperature
        
        self.device = surrogate.device
        self.u_min = torch.tensor(u_min, dtype=torch.float32, device=self.device)
        self.u_max = torch.tensor(u_max, dtype=torch.float32, device=self.device)
        self.noise_sigma = torch.tensor(noise_sigma, dtype=torch.float32, device=self.device)
        
        self.u_mean = (self.u_min + self.u_max) / 2
        self.u_mean = self.u_mean.unsqueeze(0).expand(horizon, -1).clone()
        
        self.N_z = 20
        self.n_species = 5
    
    def compute_costs(self, x_traj: torch.Tensor, u_seq: torch.Tensor,
                     alpha_min: float = 0.95) -> torch.Tensor:
        n_samples = x_traj.shape[0]
        
        # Energy cost
        T_g_in = u_seq[:, :, 0]
        energy_cost = torch.mean((T_g_in - 900) / (1350 - 900), dim=1)
        
        # Conversion constraint
        conc_dim = self.n_species * self.N_z
        c_kaolinite = x_traj[:, 1:, :conc_dim].view(n_samples, self.horizon, self.n_species, self.N_z)
        c_kao_out = c_kaolinite[:, :, 0, -1]
        alpha = 1.0 - c_kao_out / (0.15 + 1e-6)
        alpha = torch.clamp(alpha, 0, 1)
        
        violation = torch.relu(alpha_min - alpha)
        constraint_cost = 5000.0 * torch.mean(violation ** 2, dim=1)
        
        # Terminal temperature
        T_s = x_traj[:, -1, conc_dim:conc_dim + self.N_z]
        T_s_out = T_s[:, -1]
        temp_cost = 0.01 * ((T_s_out - 1066) / 100) ** 2
        
        return energy_cost + constraint_cost + temp_cost
    
    @torch.no_grad()
    def solve(self, x0: np.ndarray, alpha_min: float = 0.95) -> Tuple[np.ndarray, float]:
        x0_t = torch.tensor(x0, dtype=torch.float32, device=self.device)
        x0_batch = x0_t.unsqueeze(0).expand(self.n_samples, -1)
        
        noise = torch.randn(self.n_samples, self.horizon, 2, device=self.device)
        noise = noise * self.noise_sigma
        
        u_seq = self.u_mean.unsqueeze(0) + noise
        u_seq = torch.clamp(u_seq, self.u_min, self.u_max)
        
        x_traj = self.surrogate.rollout(x0_batch, u_seq)
        costs = self.compute_costs(x_traj, u_seq, alpha_min)
        
        costs_shifted = costs - costs.min()
        weights = torch.exp(-costs_shifted / self.temperature)
        weights = weights / weights.sum()
        
        u_opt_seq = torch.sum(weights.view(-1, 1, 1) * u_seq, dim=0)
        
        self.u_mean = u_opt_seq.clone()
        self.u_mean[:-1] = self.u_mean[1:].clone()
        self.u_mean[-1] = (self.u_min + self.u_max) / 2
        
        u_opt = u_opt_seq[0].cpu().numpy()
        best_cost = (weights * costs).sum().item()
        
        return u_opt, best_cost


# =============================================================================
# Gradient-based MPC
# =============================================================================

class SurrogateMPC:
    """Gradient-based MPC using neural surrogate."""
    
    def __init__(self, surrogate: SurrogateModel, 
                 horizon: int = 10,
                 u_min: np.ndarray = np.array([900, 550]),
                 u_max: np.ndarray = np.array([1350, 800]),
                 n_iter: int = 50,
                 lr: float = 5.0):
        self.surrogate = surrogate.eval()
        self.horizon = horizon
        self.u_min = torch.tensor(u_min, dtype=torch.float32, device=surrogate.device)
        self.u_max = torch.tensor(u_max, dtype=torch.float32, device=surrogate.device)
        self.n_iter = n_iter
        self.lr = lr
        self.N_z = 20
        self.n_species = 5
        
    def compute_cost(self, x_traj: torch.Tensor, u_seq: torch.Tensor,
                    alpha_min: float = 0.95) -> torch.Tensor:
        T_g_in = u_seq[:, :, 0]
        energy_cost = torch.mean((T_g_in - 900) / (1350 - 900), dim=1)
        
        conc_dim = self.n_species * self.N_z
        c_kaolinite = x_traj[:, 1:, :conc_dim].view(-1, self.horizon, self.n_species, self.N_z)
        c_kao_out = c_kaolinite[:, :, 0, -1]
        alpha = 1.0 - c_kao_out / (0.15 + 1e-6)
        alpha = torch.clamp(alpha, 0, 1)
        
        violation = torch.relu(alpha_min - alpha)
        constraint_cost = 5000.0 * torch.mean(violation ** 2, dim=1)
        
        T_s = x_traj[:, -1, conc_dim:conc_dim + self.N_z]
        T_s_out = T_s[:, -1]
        temp_cost = 0.01 * ((T_s_out - 1066) / 100) ** 2
        
        return energy_cost + constraint_cost + temp_cost
    
    def solve(self, x0: np.ndarray, alpha_min: float = 0.95) -> Tuple[np.ndarray, float]:
        device = self.surrogate.device
        x0_t = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
        
        u_init = (self.u_min + self.u_max) / 2
        u_seq = u_init.unsqueeze(0).unsqueeze(0).expand(1, self.horizon, -1).clone()
        u_seq = u_seq.requires_grad_(True)
        
        optimizer = torch.optim.Adam([u_seq], lr=self.lr)
        best_u, best_cost = None, float('inf')
        
        for i in range(self.n_iter):
            optimizer.zero_grad()
            with torch.no_grad():
                u_seq.data = torch.clamp(u_seq.data, self.u_min, self.u_max)
            
            x_traj = self.surrogate.rollout(x0_t, u_seq)
            cost = self.compute_cost(x_traj, u_seq, alpha_min).mean()
            
            if cost.item() < best_cost:
                best_cost = cost.item()
                best_u = u_seq.detach().clone()
            
            cost.backward()
            optimizer.step()
        
        return best_u[0, 0].cpu().numpy(), best_cost


# =============================================================================
# Classical MPC (Simplified Dynamics - Non-surrogate baseline)
# =============================================================================

class CalcinerDynamics:
    """
    Simple first-order dynamics for outlet conversion.
    Used as a baseline comparison (non-surrogate).
    """
    
    def __init__(self, tau: float = 2.0, dt: float = 0.5):
        self.tau = tau
        self.dt = dt
        self.a = np.exp(-dt / tau)
        
    def steady_state_conversion(self, T_g_in: float) -> float:
        alpha_max = 0.999
        T_mid = 1000.0
        k = 0.025
        return alpha_max / (1.0 + np.exp(-k * (T_g_in - T_mid)))
    
    def heater_power(self, T_g_in: float, T_cold: float = 300.0) -> float:
        c = 0.46 / (1261 - 300)
        return c * (T_g_in - T_cold)
    
    def step(self, alpha: float, u: float, disturbance: float = 0.0) -> float:
        alpha_ss = self.steady_state_conversion(u) + disturbance
        alpha_ss = np.clip(alpha_ss, 0, 0.999)
        return self.a * alpha + (1 - self.a) * alpha_ss
    
    def simulate(self, alpha0: float, u_seq: np.ndarray, 
                 disturbances: Optional[np.ndarray] = None) -> np.ndarray:
        N = len(u_seq)
        if disturbances is None:
            disturbances = np.zeros(N)
        
        alphas = [alpha0]
        alpha = alpha0
        for k in range(N):
            alpha = self.step(alpha, u_seq[k], disturbances[k])
            alphas.append(alpha)
        return np.array(alphas)


class ClassicalMPC:
    """Classical MPC with simplified dynamics (non-surrogate baseline)."""
    
    def __init__(self, model: CalcinerDynamics, horizon: int = 20,
                 u_min: float = 900.0, u_max: float = 1300.0):
        self.model = model
        self.horizon = horizon
        self.u_min = u_min
        self.u_max = u_max
        
    def solve(self, alpha0: float, alpha_min_seq: np.ndarray,
              disturbances: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        N = self.horizon
        if disturbances is None:
            disturbances = np.zeros(N)
        
        def objective(u_flat):
            u = u_flat.reshape(-1)
            alphas = self.model.simulate(alpha0, u, disturbances[:N])
            
            # Energy cost
            power = sum(self.model.heater_power(u[k]) for k in range(N))
            
            # Constraint violation
            violation = 0.0
            for k in range(N):
                margin = alpha_min_seq[k] - alphas[k+1]
                if margin > 0:
                    violation += 1000 * margin**2
            
            return power + violation
        
        u0 = np.ones(N) * 1100.0
        bounds = Bounds(self.u_min, self.u_max)
        
        result = minimize(objective, u0, method='L-BFGS-B', bounds=bounds)
        
        return result.x, result.fun

