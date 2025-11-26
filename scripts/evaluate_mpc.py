#!/usr/bin/env python3
"""
Evaluate MPC Controllers on Flash Calciner

Compares surrogate-based MPPI against constant-temperature baseline.

Usage:
    python scripts/evaluate_mpc.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from calciner import (
    CalcinerSimulator,
    SurrogateModel,
    SurrogateMPPI,
    SpatiallyAwareDynamics,
    N_SPECIES,
)

ROOT = Path(__file__).parent.parent


def load_surrogate():
    """Load trained surrogate model."""
    model_path = ROOT / "models" / "surrogate_model.pt"
    ckpt = torch.load(model_path, weights_only=False)
    
    model = SpatiallyAwareDynamics(N_z=ckpt['N_z'])
    model.load_state_dict(ckpt['model_state_dict'])
    
    norm_params = {k: np.array(v) if isinstance(v, list) else v 
                   for k, v in ckpt['norm_params'].items()}
    
    return SurrogateModel(model, norm_params), ckpt['dt'], ckpt['N_z']


def compute_conversion(x, N_z=20):
    c_kao_out = x[N_z - 1]
    return np.clip(1.0 - c_kao_out / 0.15, 0, 1)


def run_mpc_control(surrogate, simulator, n_steps=80, alpha_target=0.95):
    """Run closed-loop MPC control."""
    mppi = SurrogateMPPI(
        surrogate,
        horizon=12,
        n_samples=96,
        temperature=0.05,
        noise_sigma=np.array([60, 25]),
    )
    
    # Initial state
    N_z = 20
    c_init = np.array([0.15, 0.79, 0.31, 5.81, 3.74])
    c = np.zeros((N_SPECIES, N_z))
    for i in range(N_SPECIES):
        c[i, :] = c_init[i]
    T_s = np.ones(N_z) * 700.0
    T_g = np.ones(N_z) * 700.0
    x = simulator.state_to_vector(c, T_s, T_g)
    
    history = {'time': [], 'conversion': [], 'T_g_in': [], 'energy': []}
    
    for step in range(n_steps):
        alpha = compute_conversion(x, N_z)
        
        u_opt, _ = mppi.solve(x, alpha_min=alpha_target)
        T_g_in = u_opt[0]
        energy = (T_g_in - 900) / (1350 - 900)
        
        history['time'].append(step * simulator.dt)
        history['conversion'].append(alpha)
        history['T_g_in'].append(T_g_in)
        history['energy'].append(energy)
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: α={alpha:.2%}, T_g_in={T_g_in:.0f}K")
        
        x = simulator.step(x, u_opt)
    
    history['conversion'].append(compute_conversion(x, N_z))
    history['time'].append(n_steps * simulator.dt)
    
    return history


def run_baseline(simulator, n_steps=80, T_g_in_fixed=1261.15):
    """Run constant temperature baseline."""
    N_z = 20
    c_init = np.array([0.15, 0.79, 0.31, 5.81, 3.74])
    c = np.zeros((N_SPECIES, N_z))
    for i in range(N_SPECIES):
        c[i, :] = c_init[i]
    T_s = np.ones(N_z) * 700.0
    T_g = np.ones(N_z) * 700.0
    x = simulator.state_to_vector(c, T_s, T_g)
    
    u = np.array([T_g_in_fixed, 657.15])
    history = {'time': [], 'conversion': [], 'T_g_in': [], 'energy': []}
    
    for step in range(n_steps):
        alpha = compute_conversion(x, N_z)
        energy = (T_g_in_fixed - 900) / (1350 - 900)
        
        history['time'].append(step * simulator.dt)
        history['conversion'].append(alpha)
        history['T_g_in'].append(T_g_in_fixed)
        history['energy'].append(energy)
        
        x = simulator.step(x, u)
    
    history['conversion'].append(compute_conversion(x, N_z))
    history['time'].append(n_steps * simulator.dt)
    
    return history


def plot_results(mpc_hist, baseline_hist):
    """Create comparison plot."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    t_mpc = mpc_hist['time']
    t_base = baseline_hist['time']
    
    # Conversion
    axes[0].plot(t_mpc, mpc_hist['conversion'], 'b-', lw=2, label='MPC')
    axes[0].plot(t_base, baseline_hist['conversion'], 'r--', lw=2, label='Baseline')
    axes[0].axhline(0.95, color='g', ls=':', lw=1.5, label='Target')
    axes[0].set_ylabel('Conversion α')
    axes[0].legend()
    axes[0].set_title('Surrogate MPC vs Baseline')
    
    # Control
    axes[1].step(t_mpc[:-1], mpc_hist['T_g_in'], 'b-', lw=2, where='post', label='MPC')
    axes[1].axhline(1261.15, color='r', ls='--', lw=2, label='Baseline')
    axes[1].set_ylabel('T_g,in [K]')
    axes[1].legend()
    
    # Energy
    mpc_cumul = np.cumsum([0] + mpc_hist['energy'])
    base_cumul = np.cumsum([0] + baseline_hist['energy'])
    axes[2].plot(t_mpc, mpc_cumul, 'b-', lw=2, label='MPC')
    axes[2].plot(t_base, base_cumul, 'r--', lw=2, label='Baseline')
    axes[2].set_ylabel('Cumulative Energy')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend()
    
    savings = (1 - mpc_cumul[-1] / base_cumul[-1]) * 100
    axes[2].annotate(f'Savings: {savings:.1f}%', xy=(t_mpc[-1]*0.6, base_cumul[-1]*0.5),
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    save_path = ROOT / "figures" / "mpc_results.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    return savings


def main():
    print("=" * 70)
    print("MPC Evaluation: Surrogate vs Baseline")
    print("=" * 70)
    
    surrogate, dt, N_z = load_surrogate()
    simulator = CalcinerSimulator(N_z=N_z, dt=dt)
    
    print("\nRunning MPC control...")
    mpc_hist = run_mpc_control(surrogate, simulator)
    
    print("\nRunning baseline...")
    baseline_hist = run_baseline(simulator)
    
    savings = plot_results(mpc_hist, baseline_hist)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  MPC final conversion:      {mpc_hist['conversion'][-1]:.1%}")
    print(f"  Baseline final conversion: {baseline_hist['conversion'][-1]:.1%}")
    print(f"  Energy savings:            {savings:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

