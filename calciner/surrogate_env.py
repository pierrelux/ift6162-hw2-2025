"""
Gym-like Environment for Part 2: Full 140D Flash Calciner

This environment wraps the neural surrogate for RL training.
Students are free to modify the reward function or state representation.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional

from .surrogate import SurrogateModel
from .physics import N_SPECIES


class SurrogateCalcinerEnv:
    """
    RL environment for the full 140-dimensional flash calciner.
    
    Uses neural surrogate for fast simulation during training.
    Validate learned policies on the true physics simulator afterward.
    
    State: 140D (5 species × 20 cells + 2 temps × 20 cells)
    Action: 2D continuous [T_g_in, T_s_in] or 1D [T_g_in] with fixed T_s_in
    """
    
    def __init__(self, surrogate: SurrogateModel, 
                 episode_length: int = 50,
                 alpha_min: float = 0.95,
                 control_T_s: bool = False):
        """
        Args:
            surrogate: Trained neural surrogate model
            episode_length: Episode horizon in steps
            alpha_min: Target minimum conversion
            control_T_s: If True, action is [T_g_in, T_s_in]; else just [T_g_in]
        """
        self.surrogate = surrogate.eval()
        self.episode_length = episode_length
        self.alpha_min = alpha_min
        self.control_T_s = control_T_s
        
        self.N_z = 20
        self.n_species = N_SPECIES
        self.state_dim = self.n_species * self.N_z + 2 * self.N_z  # 140
        
        self.action_dim = 2 if control_T_s else 1
        self.T_g_min, self.T_g_max = 900.0, 1350.0
        self.T_s_min, self.T_s_max = 550.0, 800.0
        self.T_s_default = 657.15  # Used when not controlling T_s
        
        self.c_in_nominal = 0.15  # Nominal kaolinite inlet concentration
        
        self.t = 0
        self.state = None
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset to initial condition.
        
        Uses a cold start: uniform concentrations, low temperatures.
        This is challenging but realistic for startup scenarios.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.t = 0
        
        # Cold start initial condition
        c = np.zeros((self.n_species, self.N_z))
        c[0, :] = 0.15  # Kaolinite (unreacted)
        c[1, :] = 0.1   # Quartz
        c[2, :] = 0.05  # Metakaolin (small amount)
        c[3, :] = 18.0  # N2 (high)
        c[4, :] = 0.1   # H2O (low)
        
        T_s = np.ones(self.N_z) * 600.0
        T_g = np.ones(self.N_z) * 600.0
        
        self.state = np.concatenate([
            c.flatten(),  # 100D
            T_s,          # 20D
            T_g,          # 20D
        ])
        
        return self.state.copy()
    
    def _compute_conversion(self, state: np.ndarray) -> float:
        """Compute conversion from kaolinite concentration at outlet."""
        c_kao_out = state[self.N_z - 1]  # Last cell of first species
        alpha = 1.0 - c_kao_out / self.c_in_nominal
        return np.clip(alpha, 0, 1)
    
    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Reward = -energy_cost - constraint_penalty
        
        Students can modify this if they want to experiment with
        different reward shaping strategies.
        """
        T_g_in = action[0]
        
        # Energy cost (linear in temperature)
        energy = (T_g_in - 900.0) / (1350.0 - 900.0)
        
        # Constraint penalty
        alpha = self._compute_conversion(state)
        violation = max(0, self.alpha_min - alpha)
        penalty = 100.0 * violation ** 2
        
        return -energy - penalty
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one step in the environment.
        
        Args:
            action: If control_T_s=False, scalar T_g_in
                    If control_T_s=True, [T_g_in, T_s_in]
        
        Returns:
            next_state, reward, done, info
        """
        # Parse action
        if self.action_dim == 1:
            action = np.atleast_1d(action)
            T_g_in = np.clip(action[0], self.T_g_min, self.T_g_max)
            T_s_in = self.T_s_default
        else:
            T_g_in = np.clip(action[0], self.T_g_min, self.T_g_max)
            T_s_in = np.clip(action[1], self.T_s_min, self.T_s_max)
        
        u = np.array([T_g_in, T_s_in])
        
        # Simulate with surrogate
        with torch.no_grad():
            state_t = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
            u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
            next_state_t = self.surrogate.step(state_t, u_t)
            next_state = next_state_t.cpu().numpy().squeeze()
        
        # Compute reward
        reward = self._compute_reward(next_state, u)
        
        # Update state and time
        self.state = next_state
        self.t += 1
        done = self.t >= self.episode_length
        
        # Info dict
        alpha = self._compute_conversion(next_state)
        info = {
            'alpha': alpha,
            'T_g_in': T_g_in,
            'T_s_in': T_s_in,
            'energy': (T_g_in - 900.0) / (1350.0 - 900.0),
            'violation': max(0, self.alpha_min - alpha),
        }
        
        return self.state.copy(), reward, done, info


