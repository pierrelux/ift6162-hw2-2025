"""
Simplified Dynamics Model for Flash Calciner

This module provides a first-order approximation of the calciner dynamics
used for the Part 1 simplified control problem.
"""

import numpy as np
from typing import Optional


class CalcinerDynamics:
    """
    Simple first-order dynamics for outlet conversion.
    
    Models the relationship between gas inlet temperature and kaolinite conversion
    as a first-order lag system approaching a sigmoidal steady state.
    
    State: α ∈ [0, 1] (conversion fraction)
    Control: u = T_g,in ∈ [900, 1300] K (gas inlet temperature)
    
    Dynamics: α_{k+1} = a·α_k + (1-a)·α_ss(u_k)
    where a = exp(-dt/τ) and α_ss is the steady-state conversion at temperature u.
    """
    
    def __init__(self, tau: float = 2.0, dt: float = 0.5):
        """
        Args:
            tau: Time constant [s] for first-order response
            dt: Discretization time step [s]
        """
        self.tau = tau
        self.dt = dt
        self.a = np.exp(-dt / tau)
        
    def steady_state_conversion(self, T_g_in: float) -> float:
        """
        Compute steady-state conversion at given gas inlet temperature.
        
        Uses a sigmoidal relationship calibrated to match the physics model:
        - T = 900K: ~8% conversion
        - T = 1000K: ~50% conversion (inflection point)
        - T = 1100K: ~92% conversion
        - T = 1261K: ~99.8% conversion
        """
        alpha_max = 0.999
        T_mid = 1000.0
        k = 0.025
        return alpha_max / (1.0 + np.exp(-k * (T_g_in - T_mid)))
    
    def heater_power(self, T_g_in: float, T_cold: float = 300.0) -> float:
        """
        Compute normalized heater power (energy cost).
        
        Linear relationship: power ∝ (T_g_in - T_cold)
        Normalized so that T_g_in = 1261K gives power ≈ 0.46 MW.
        """
        c = 0.46 / (1261 - 300)
        return c * (T_g_in - T_cold)
    
    def step(self, alpha: float, u: float, disturbance: float = 0.0) -> float:
        """
        Simulate one time step.
        
        Args:
            alpha: Current conversion
            u: Gas inlet temperature [K]
            disturbance: Additive disturbance to steady-state conversion
            
        Returns:
            Next conversion value
        """
        alpha_ss = self.steady_state_conversion(u) + disturbance
        alpha_ss = np.clip(alpha_ss, 0, 0.999)
        return self.a * alpha + (1 - self.a) * alpha_ss
    
    def simulate(self, alpha0: float, u_seq: np.ndarray, 
                 disturbances: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate a trajectory given control sequence.
        
        Args:
            alpha0: Initial conversion
            u_seq: Control sequence of length N
            disturbances: Optional disturbance sequence of length N
            
        Returns:
            Conversion trajectory of length N+1 (including initial state)
        """
        N = len(u_seq)
        if disturbances is None:
            disturbances = np.zeros(N)
        
        alphas = [alpha0]
        alpha = alpha0
        for k in range(N):
            alpha = self.step(alpha, u_seq[k], disturbances[k])
            alphas.append(alpha)
        return np.array(alphas)
