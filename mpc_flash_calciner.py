"""
Economic MPC for Flash Calciner - With Dynamics and Disturbances

Demonstrates MPC responding to:
1. Time-varying conversion requirements
2. Inlet temperature disturbances
3. Transient dynamics
"""

import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Dynamic Model with State
# -----------------------------------------------------------------------------

class CalcinerDynamics:
    """
    Simple first-order dynamics for outlet conversion.
    
    Captures the thermal lag of the calciner:
    d(alpha)/dt = (alpha_ss(u) - alpha) / tau
    
    where alpha_ss is the steady-state conversion and tau is time constant.
    """
    
    def __init__(self, tau=2.0, dt=0.5):
        """
        Parameters
        ----------
        tau : float
            Time constant [s] - how fast system responds
        dt : float
            Discrete time step [s]
        """
        self.tau = tau
        self.dt = dt
        # Discretized: alpha_{k+1} = a*alpha_k + (1-a)*alpha_ss(u_k)
        self.a = np.exp(-dt / tau)
        
    def steady_state_conversion(self, T_g_in):
        """Steady-state conversion vs temperature."""
        alpha_max = 0.999
        T_mid = 1000.0
        k = 0.025
        return alpha_max / (1.0 + np.exp(-k * (T_g_in - T_mid)))
    
    def heater_power(self, T_g_in, T_cold=300.0):
        """Heater power [MW]."""
        c = 0.46 / (1261 - 300)
        return c * (T_g_in - T_cold)
    
    def step(self, alpha, u, disturbance=0.0):
        """
        One discrete time step.
        
        Parameters
        ----------
        alpha : float
            Current conversion
        u : float
            Control input (T_g,in)
        disturbance : float
            Additive disturbance on conversion
            
        Returns
        -------
        alpha_next : float
            Next conversion
        """
        alpha_ss = self.steady_state_conversion(u) + disturbance
        alpha_ss = np.clip(alpha_ss, 0, 0.999)
        alpha_next = self.a * alpha + (1 - self.a) * alpha_ss
        return alpha_next
    
    def simulate(self, alpha0, u_seq, disturbances=None):
        """Simulate trajectory."""
        N = len(u_seq)
        if disturbances is None:
            disturbances = np.zeros(N)
        
        alphas = [alpha0]
        alpha = alpha0
        for k in range(N):
            alpha = self.step(alpha, u_seq[k], disturbances[k])
            alphas.append(alpha)
        return np.array(alphas)


# -----------------------------------------------------------------------------
# MPC Controller
# -----------------------------------------------------------------------------

class EconomicMPC:
    """
    Economic MPC with prediction horizon.
    """
    
    def __init__(self, model, horizon=8, u_min=900, u_max=1300):
        self.model = model
        self.N = horizon
        self.u_min = u_min
        self.u_max = u_max
        self.P_ref = model.heater_power(1100)
        
    def solve(self, alpha0, alpha_min_trajectory, penalty=1e4):
        """
        Solve MPC problem.
        
        Parameters
        ----------
        alpha0 : float
            Current conversion
        alpha_min_trajectory : array of shape (N,)
            Minimum conversion requirement over horizon
        """
        def objective(u):
            # Simulate forward
            alphas = self.model.simulate(alpha0, u)[1:]  # exclude initial
            
            # Energy cost
            cost = np.sum(self.model.heater_power(u)) / self.P_ref
            
            # Constraint violations
            violations = np.maximum(0, alpha_min_trajectory[:len(alphas)] - alphas)
            cost += penalty * np.sum(violations**2)
            
            return cost
        
        u_init = np.ones(self.N) * 1150
        bounds = Bounds(self.u_min * np.ones(self.N), self.u_max * np.ones(self.N))
        
        result = minimize(objective, u_init, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 50})
        return result.x


# -----------------------------------------------------------------------------
# Simulation with Disturbances and Setpoint Changes
# -----------------------------------------------------------------------------

def run_mpc_scenario():
    """
    Run MPC with realistic scenario:
    - Setpoint changes (varying quality requirements)
    - Disturbances (varying feedstock quality)
    """
    n_steps = 40
    dt = 0.5
    horizon = 8
    
    model = CalcinerDynamics(tau=2.0, dt=dt)
    mpc = EconomicMPC(model, horizon=horizon)
    
    # Time-varying setpoint (conversion requirement)
    alpha_min = np.ones(n_steps + horizon) * 0.95
    alpha_min[10:25] = 0.99  # Higher quality demand mid-run
    alpha_min[30:] = 0.90    # Relaxed requirement at end
    
    # Disturbances (e.g., feedstock variability)
    disturbances = np.zeros(n_steps)
    disturbances[15:20] = -0.03  # Harder-to-process feedstock
    disturbances[35:] = 0.02     # Easier feedstock
    
    # Initial state
    alpha = 0.90
    
    # Storage
    alphas = [alpha]
    controls = []
    powers = []
    setpoints = []
    
    print("=" * 70)
    print("MPC Scenario: Time-varying setpoint + Disturbances")
    print("=" * 70)
    print(f"  t=0-5s:   α_min = 95% (normal)")
    print(f"  t=5-12s:  α_min = 99% (high quality demand)")
    print(f"  t=7.5-10s: disturbance (hard feedstock)")
    print(f"  t=15-20s: α_min = 90% (relaxed)")
    print("=" * 70)
    
    for k in range(n_steps):
        # Get setpoint trajectory for horizon
        sp_traj = alpha_min[k:k+horizon]
        
        # Solve MPC
        u_opt = mpc.solve(alpha, sp_traj)
        u_k = u_opt[0]  # Apply first control
        
        # Record
        controls.append(u_k)
        powers.append(model.heater_power(u_k))
        setpoints.append(alpha_min[k])
        
        # Step plant with disturbance
        alpha = model.step(alpha, u_k, disturbances[k])
        alphas.append(alpha)
        
        if k % 10 == 0:
            print(f"t={k*dt:5.1f}s: u={u_k:7.1f} K, α={alpha:.4f}, "
                  f"α_min={alpha_min[k]:.2f}, P={powers[-1]:.3f} MW")
    
    # Also run baseline (constant control)
    alpha_base = 0.90
    alphas_base = [alpha_base]
    for k in range(n_steps):
        alpha_base = model.step(alpha_base, 1261.15, disturbances[k])
        alphas_base.append(alpha_base)
    
    print("\n" + "-" * 50)
    print(f"MPC avg power:      {np.mean(powers):.3f} MW")
    print(f"Baseline avg power: {model.heater_power(1261.15):.3f} MW")
    print(f"Energy savings:     {(1 - np.mean(powers)/model.heater_power(1261.15))*100:.1f}%")
    print("-" * 50)
    
    return {
        't': np.arange(n_steps + 1) * dt,
        't_ctrl': np.arange(n_steps) * dt,
        'alphas': np.array(alphas),
        'alphas_base': np.array(alphas_base),
        'controls': np.array(controls),
        'powers': np.array(powers),
        'setpoints': np.array(setpoints),
        'disturbances': disturbances,
        'baseline_u': 1261.15,
    }


def plot_scenario(result):
    """Plot MPC scenario results."""
    t = result['t']
    t_ctrl = result['t_ctrl']
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    
    # 1) Control input
    ax = axes[0]
    ax.step(t_ctrl, result['controls'], where='post', color='b', linewidth=2, label='MPC')
    ax.axhline(result['baseline_u'], color='orange', linestyle='--', 
               linewidth=2, label='Baseline')
    ax.set_ylabel(r'$T_{g,in}$ [K]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Economic MPC: Responding to Setpoint Changes & Disturbances')
    
    # 2) Conversion vs setpoint
    ax = axes[1]
    ax.plot(t, result['alphas'], 'b-', linewidth=2, label='MPC')
    ax.plot(t, result['alphas_base'], color='orange', linestyle='--', linewidth=2, label='Baseline')
    ax.step(t_ctrl, result['setpoints'], where='post', color='r', linestyle=':', linewidth=2, label=r'$\alpha_{min}$')
    ax.fill_between(t_ctrl, 0, result['setpoints'], alpha=0.1, color='red', step='post')
    ax.set_ylabel(r'Conversion $\alpha$')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.02])
    
    # 3) Power
    ax = axes[2]
    ax.step(t_ctrl, result['powers'], where='post', color='b', linewidth=2, label='MPC')
    ax.axhline(result['powers'].mean(), color='b', linestyle=':', alpha=0.7)
    P_base = 0.46  # Baseline power
    ax.axhline(P_base, color='orange', linestyle='--', linewidth=2, label='Baseline')
    ax.set_ylabel('Power [MW]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 4) Disturbances
    ax = axes[3]
    ax.step(t_ctrl, result['disturbances'] * 100, where='post', color='g', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.fill_between(t_ctrl, 0, result['disturbances']*100, alpha=0.3, color='green', step='post')
    ax.set_ylabel('Disturbance [%]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-5, 5])
    
    plt.tight_layout()
    plt.savefig('mpc_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved: mpc_results.png")
    plt.close()


if __name__ == "__main__":
    result = run_mpc_scenario()
    plot_scenario(result)
