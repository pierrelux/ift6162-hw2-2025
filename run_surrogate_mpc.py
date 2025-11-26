"""
Closed-Loop MPC Control with Neural Surrogate

Runs a full simulation comparing:
1. Surrogate-based MPPI controller
2. Baseline constant-temperature control
3. Validates predictions against physics simulator
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from pathlib import Path

from flash_calciner import SimplifiedFlashCalciner, N_SPECIES, L
from surrogate_flash_calciner import (
    SurrogateModel, SurrogateMPPI, CalcinerSimulator,
    SpatiallyAwareDynamics
)

# Plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_surrogate():
    """Load trained surrogate model."""
    print("Loading surrogate model...", flush=True)
    ckpt = torch.load('surrogate_model.pt', weights_only=False)
    
    model = SpatiallyAwareDynamics(N_z=ckpt['N_z'])
    model.load_state_dict(ckpt['model_state_dict'])
    
    # Convert lists back to numpy arrays
    norm_params = {k: np.array(v) if isinstance(v, list) else v 
                   for k, v in ckpt['norm_params'].items()}
    
    surrogate = SurrogateModel(model, norm_params)
    print(f"  ✓ Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    
    return surrogate, ckpt['dt'], ckpt['N_z']


def compute_conversion(x, N_z=20, n_species=5):
    """Compute kaolinite conversion from state."""
    c_kao_in = 0.15  # Inlet kaolinite concentration
    c_kao_out = x[0 * N_z + N_z - 1]  # Outlet kaolinite (last cell of species 0)
    alpha = 1.0 - c_kao_out / (c_kao_in + 1e-8)
    return np.clip(alpha, 0, 1)


def compute_outlet_temp(x, N_z=20, n_species=5):
    """Get outlet solid temperature."""
    conc_dim = n_species * N_z
    T_s_out = x[conc_dim + N_z - 1]  # Last cell of T_s
    return T_s_out


def compute_energy(T_g_in, T_ref=900):
    """Compute normalized energy cost."""
    # Energy proportional to heating above reference
    return (T_g_in - T_ref) / (1350 - T_ref)


def run_mpc_control(surrogate, simulator, n_steps=50, alpha_target=0.95):
    """
    Run closed-loop MPC control simulation.
    
    Returns history of states, controls, and metrics.
    """
    print(f"\nRunning MPC control for {n_steps} steps...", flush=True)
    print(f"  Target conversion: α ≥ {alpha_target:.0%}", flush=True)
    
    # Create controller - balanced speed vs quality
    mppi = SurrogateMPPI(
        surrogate, 
        horizon=12,  # Good horizon for planning
        n_samples=96,  # Reasonable samples
        temperature=0.05,  # Greedy toward low cost
        noise_sigma=np.array([60, 25])  # Exploration
    )
    
    # Initial state (cold start)
    N_z = 20
    c_init = np.array([0.15, 0.79, 0.31, 5.81, 3.74])  # Inlet concentrations
    T_init = 700.0
    
    c = np.zeros((N_SPECIES, N_z))
    for i in range(N_SPECIES):
        c[i, :] = c_init[i]
    T_s = np.ones(N_z) * T_init
    T_g = np.ones(N_z) * T_init
    
    x = simulator.state_to_vector(c, T_s, T_g)
    
    # Storage
    history = {
        'time': [],
        'conversion': [],
        'T_s_out': [],
        'T_g_in': [],
        'T_s_in': [],
        'energy': [],
        'mpc_time': [],
        'states': [],
    }
    
    total_energy = 0
    start_time = time.time()
    
    for step in range(n_steps):
        # Current metrics
        alpha = compute_conversion(x, N_z)
        T_s_out = compute_outlet_temp(x, N_z)
        
        # Solve MPC
        t0 = time.time()
        u_opt, cost = mppi.solve(x, alpha_min=alpha_target)
        mpc_time = time.time() - t0
        
        T_g_in, T_s_in = u_opt
        energy = compute_energy(T_g_in)
        total_energy += energy
        
        # Record
        history['time'].append(step * simulator.dt)
        history['conversion'].append(alpha)
        history['T_s_out'].append(T_s_out)
        history['T_g_in'].append(T_g_in)
        history['T_s_in'].append(T_s_in)
        history['energy'].append(energy)
        history['mpc_time'].append(mpc_time)
        history['states'].append(x.copy())
        
        # Progress - more frequent updates
        if step % 5 == 0:
            print(f"  Step {step:3d}: α={alpha:.2%}, T_out={T_s_out:.0f}K, "
                  f"T_g_in={T_g_in:.0f}K, MPC={mpc_time*1000:.0f}ms", flush=True)
        
        # Apply control (using physics simulator for ground truth)
        x = simulator.step(x, u_opt)
    
    # Final state
    alpha = compute_conversion(x, N_z)
    history['conversion'].append(alpha)
    history['time'].append(n_steps * simulator.dt)
    history['T_s_out'].append(compute_outlet_temp(x, N_z))
    
    elapsed = time.time() - start_time
    
    print(f"\n  ✓ MPC simulation complete in {elapsed:.1f}s", flush=True)
    print(f"  ✓ Final conversion: {alpha:.2%}", flush=True)
    print(f"  ✓ Average MPC time: {np.mean(history['mpc_time'])*1000:.0f}ms", flush=True)
    print(f"  ✓ Total energy: {total_energy:.2f} (normalized)", flush=True)
    
    return history


def run_baseline_control(simulator, n_steps=50, T_g_in_fixed=1261.15):
    """
    Run baseline with constant high temperature.
    """
    print(f"\nRunning baseline control (T_g_in={T_g_in_fixed:.0f}K)...", flush=True)
    
    N_z = 20
    c_init = np.array([0.15, 0.79, 0.31, 5.81, 3.74])
    T_init = 700.0
    
    c = np.zeros((N_SPECIES, N_z))
    for i in range(N_SPECIES):
        c[i, :] = c_init[i]
    T_s = np.ones(N_z) * T_init
    T_g = np.ones(N_z) * T_init
    
    x = simulator.state_to_vector(c, T_s, T_g)
    
    history = {
        'time': [],
        'conversion': [],
        'T_s_out': [],
        'T_g_in': [],
        'energy': [],
    }
    
    total_energy = 0
    u = np.array([T_g_in_fixed, 657.15])  # Fixed control
    
    for step in range(n_steps):
        alpha = compute_conversion(x, N_z)
        T_s_out = compute_outlet_temp(x, N_z)
        energy = compute_energy(T_g_in_fixed)
        total_energy += energy
        
        history['time'].append(step * simulator.dt)
        history['conversion'].append(alpha)
        history['T_s_out'].append(T_s_out)
        history['T_g_in'].append(T_g_in_fixed)
        history['energy'].append(energy)
        
        x = simulator.step(x, u)
    
    alpha = compute_conversion(x, N_z)
    history['conversion'].append(alpha)
    history['time'].append(n_steps * simulator.dt)
    history['T_s_out'].append(compute_outlet_temp(x, N_z))
    
    print(f"  ✓ Final conversion: {alpha:.2%}", flush=True)
    print(f"  ✓ Total energy: {total_energy:.2f} (normalized)", flush=True)
    
    return history


def plot_results(mpc_hist, baseline_hist, alpha_target=0.95):
    """Create comparison plot."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    t_mpc = mpc_hist['time']
    t_base = baseline_hist['time']
    
    # 1. Conversion
    ax = axes[0]
    ax.plot(t_mpc, mpc_hist['conversion'], 'b-', linewidth=2, label='MPC')
    ax.plot(t_base, baseline_hist['conversion'], 'r--', linewidth=2, label='Baseline')
    ax.axhline(alpha_target, color='g', linestyle=':', linewidth=1.5, label=f'Target ({alpha_target:.0%})')
    ax.fill_between(t_mpc, 0, alpha_target, alpha=0.1, color='green')
    ax.set_ylabel('Conversion α')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right')
    ax.set_title('Surrogate MPC vs Baseline Control', fontsize=12, fontweight='bold')
    
    # 2. Outlet Temperature
    ax = axes[1]
    ax.plot(t_mpc, mpc_hist['T_s_out'], 'b-', linewidth=2, label='MPC')
    ax.plot(t_base, baseline_hist['T_s_out'], 'r--', linewidth=2, label='Baseline')
    ax.axhline(1066, color='g', linestyle=':', linewidth=1.5, label='Target (1066K)')
    ax.set_ylabel('Outlet Temp [K]')
    ax.legend(loc='lower right')
    
    # 3. Control Input (Gas Temperature)
    ax = axes[2]
    ax.step(t_mpc[:-1], mpc_hist['T_g_in'], 'b-', linewidth=2, where='post', label='MPC')
    ax.axhline(baseline_hist['T_g_in'][0], color='r', linestyle='--', linewidth=2, label='Baseline')
    ax.set_ylabel('Gas Inlet Temp [K]')
    ax.legend(loc='upper right')
    ax.set_ylim([850, 1350])
    
    # 4. Cumulative Energy
    ax = axes[3]
    mpc_cumul = np.cumsum([0] + mpc_hist['energy'])
    base_cumul = np.cumsum([0] + baseline_hist['energy'])
    ax.plot(t_mpc, mpc_cumul, 'b-', linewidth=2, label='MPC')
    ax.plot(t_base, base_cumul, 'r--', linewidth=2, label='Baseline')
    ax.set_ylabel('Cumulative Energy')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='upper left')
    
    # Add savings annotation
    final_mpc = mpc_cumul[-1]
    final_base = base_cumul[-1]
    savings = (1 - final_mpc / final_base) * 100
    ax.annotate(f'Energy savings: {savings:.1f}%', 
                xy=(t_mpc[-1]*0.6, final_base*0.5),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('mpc_control_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: mpc_control_results.png", flush=True)
    plt.close()
    
    return savings


def plot_state_profiles(mpc_hist, simulator, save_name='mpc_state_profiles.png'):
    """Plot spatial profiles at different times."""
    N_z = 20
    z = np.linspace(0, L, N_z)
    
    # Select time snapshots
    n_states = len(mpc_hist['states'])
    indices = [0, n_states//4, n_states//2, 3*n_states//4, n_states-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    for idx, color in zip(indices, colors):
        x = mpc_hist['states'][idx]
        t = mpc_hist['time'][idx]
        c, T_s, T_g = simulator.vector_to_state(x)
        
        # Kaolinite concentration
        axes[0, 0].plot(z, c[0, :], color=color, label=f't={t:.1f}s')
        
        # Metakaolin concentration
        axes[0, 1].plot(z, c[2, :], color=color, label=f't={t:.1f}s')
        
        # Solid temperature
        axes[1, 0].plot(z, T_s, color=color, label=f't={t:.1f}s')
        
        # Gas temperature
        axes[1, 1].plot(z, T_g, color=color, label=f't={t:.1f}s')
    
    axes[0, 0].set_ylabel('Kaolinite [mol/m³]')
    axes[0, 0].set_title('Kaolinite Concentration')
    axes[0, 0].legend(fontsize=8)
    
    axes[0, 1].set_ylabel('Metakaolin [mol/m³]')
    axes[0, 1].set_title('Metakaolin Concentration')
    axes[0, 1].legend(fontsize=8)
    
    axes[1, 0].set_xlabel('Position [m]')
    axes[1, 0].set_ylabel('Temperature [K]')
    axes[1, 0].set_title('Solid Temperature')
    axes[1, 0].legend(fontsize=8)
    
    axes[1, 1].set_xlabel('Position [m]')
    axes[1, 1].set_ylabel('Temperature [K]')
    axes[1, 1].set_title('Gas Temperature')
    axes[1, 1].legend(fontsize=8)
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Spatial Profiles During MPC Control', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_name}", flush=True)
    plt.close()


def main():
    print("=" * 70, flush=True)
    print("Closed-Loop MPC Control with Neural Surrogate", flush=True)
    print("=" * 70, flush=True)
    
    # Load surrogate
    surrogate, dt, N_z = load_surrogate()
    
    # Create physics simulator for ground-truth stepping
    simulator = CalcinerSimulator(N_z=N_z, dt=dt)
    
    # Run MPC control
    alpha_target = 0.95
    n_steps = 80  # Longer simulation to reach steady state
    
    mpc_hist = run_mpc_control(
        surrogate, simulator, 
        n_steps=n_steps, 
        alpha_target=alpha_target
    )
    
    # Run baseline for comparison
    baseline_hist = run_baseline_control(
        simulator, 
        n_steps=n_steps,
        T_g_in_fixed=1261.15
    )
    
    # Plot results
    print("\n" + "-" * 50, flush=True)
    print("Generating plots...", flush=True)
    print("-" * 50, flush=True)
    
    savings = plot_results(mpc_hist, baseline_hist, alpha_target)
    plot_state_profiles(mpc_hist, simulator)
    
    # Summary
    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    
    mpc_alpha_final = mpc_hist['conversion'][-1]
    base_alpha_final = baseline_hist['conversion'][-1]
    mpc_energy = sum(mpc_hist['energy'])
    base_energy = sum(baseline_hist['energy'])
    
    print(f"  MPC Controller:", flush=True)
    print(f"    Final conversion:    {mpc_alpha_final:.1%}", flush=True)
    print(f"    Total energy:        {mpc_energy:.2f}", flush=True)
    print(f"    Avg MPC solve time:  {np.mean(mpc_hist['mpc_time'])*1000:.0f}ms", flush=True)
    
    print(f"\n  Baseline Controller:", flush=True)
    print(f"    Final conversion:    {base_alpha_final:.1%}", flush=True)
    print(f"    Total energy:        {base_energy:.2f}", flush=True)
    
    print(f"\n  Improvement:", flush=True)
    print(f"    Energy savings:      {savings:.1f}%", flush=True)
    print(f"    Conversion met:      {'✓ Yes' if mpc_alpha_final >= alpha_target else '✗ No'}", flush=True)
    
    print("\n" + "=" * 70, flush=True)
    
    return mpc_hist, baseline_hist


if __name__ == "__main__":
    mpc_hist, baseline_hist = main()

