"""
Baselines for Flash Calciner Control

Includes:
- CalcinerEnv: Gym-like environment for RL training
- Constant temperature baseline
"""

import numpy as np
from typing import Tuple, Dict, Optional

from .mpc import CalcinerDynamics


# =============================================================================
# RL Environment
# =============================================================================

class CalcinerEnv:
    """
    Gym-like environment for RL training on flash calciner control.
    Uses simplified first-order dynamics.
    """
    
    def __init__(self, episode_length: int = 40, dt: float = 0.5):
        self.model = CalcinerDynamics(tau=2.0, dt=dt)
        self.episode_length = episode_length
        self.dt = dt
        self.u_min = 900.0
        self.u_max = 1300.0
        
        self.t = 0
        self.alpha = 0.0
        self.alpha_min = None
        self.disturbances = None
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        
        self.t = 0
        self.alpha = 0.90
        
        # Time-varying conversion requirements
        self.alpha_min = np.ones(self.episode_length + 10) * 0.95
        self.alpha_min[10:25] = 0.99
        self.alpha_min[30:] = 0.90
        
        # Disturbances
        self.disturbances = np.zeros(self.episode_length + 10)
        self.disturbances[15:20] = -0.03
        self.disturbances[35:] = 0.02
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.alpha,
            self.alpha_min[self.t],
            self.t / self.episode_length,
        ], dtype=np.float32)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        u = np.clip(action, self.u_min, self.u_max)
        
        self.alpha = self.model.step(self.alpha, u, self.disturbances[self.t])
        
        power = self.model.heater_power(u)
        target = self.alpha_min[self.t]
        
        # Reward: minimize energy, penalize constraint violations
        reward = -power * 2.0
        margin = self.alpha - target
        if margin < 0:
            reward -= 10.0 * margin**2
        elif margin < 0.02:
            reward += 0.5 * margin
        
        self.t += 1
        done = self.t >= self.episode_length
        
        return self._get_obs(), reward, done, {"power": power, "u": u, "alpha": self.alpha}


# =============================================================================
# Constant Temperature Baseline
# =============================================================================

class ConstantTemperatureController:
    """Baseline controller with constant gas inlet temperature."""
    
    def __init__(self, T_g_in: float = 1261.15):
        self.T_g_in = T_g_in
    
    def get_action(self, obs: np.ndarray) -> float:
        return self.T_g_in


def evaluate_baseline(env: CalcinerEnv, controller, n_episodes: int = 5) -> Dict:
    """Evaluate a baseline controller on the environment."""
    
    results = {
        'total_energy': [],
        'violations': [],
        'final_conversion': [],
    }
    
    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        total_energy = 0.0
        violations = 0
        
        done = False
        while not done:
            action = controller.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_energy += info['power']
            
            if info['alpha'] < env.alpha_min[env.t - 1]:
                violations += 1
        
        results['total_energy'].append(total_energy)
        results['violations'].append(violations)
        results['final_conversion'].append(info['alpha'])
    
    return {
        'mean_energy': np.mean(results['total_energy']),
        'mean_violations': np.mean(results['violations']),
        'mean_final_conversion': np.mean(results['final_conversion']),
    }

