#!/usr/bin/env python3
"""
Evaluate RL Algorithms on Flash Calciner

Trains and compares TD3 and PPO against MPC baseline.

Usage:
    python scripts/evaluate_rl.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from calciner import CalcinerEnv, CalcinerDynamics, ClassicalMPC
from calciner.rl import TD3Agent, PPOAgent
from calciner.rl.td3 import train_td3, ReplayBuffer
from calciner.rl.ppo import train_ppo

ROOT = Path(__file__).parent.parent


def run_mpc_baseline(env, n_steps):
    """Run classical MPC for comparison."""
    model = CalcinerDynamics(tau=2.0, dt=env.dt)
    mpc = ClassicalMPC(model, horizon=8)
    
    env.reset(seed=42)
    alpha = env.alpha
    alpha_min_seq = env.alpha_min
    disturbances = env.disturbances
    
    actions, alphas, powers = [], [alpha], []
    
    for t in range(n_steps):
        sp_traj = alpha_min_seq[t:t+8]
        if len(sp_traj) < 8:
            sp_traj = np.pad(sp_traj, (0, 8-len(sp_traj)), constant_values=sp_traj[-1])
        
        u_opt, _ = mpc.solve(alpha, sp_traj, disturbances[t:t+8])
        u = u_opt[0]
        
        actions.append(u)
        powers.append(model.heater_power(u))
        alpha = model.step(alpha, u, disturbances[t])
        alphas.append(alpha)
    
    return {'actions': actions, 'alphas': alphas, 'powers': powers}


def evaluate_agent(agent, env, seed=42):
    """Evaluate trained agent."""
    env.reset(seed=seed)
    n_steps = env.episode_length
    
    actions, alphas, powers = [], [env.alpha], []
    
    for _ in range(n_steps):
        obs = env._get_obs()
        action = agent.select_action(obs, deterministic=True) if hasattr(agent, 'select_action') else agent.policy.get_action(obs, deterministic=True)
        
        _, _, _, info = env.step(action)
        
        actions.append(info['u'])
        alphas.append(info['alpha'])
        powers.append(info['power'])
    
    return {'actions': actions, 'alphas': alphas, 'powers': powers}


def plot_comparison(td3_results, ppo_results, mpc_results, env, save_path):
    """Create comparison plot."""
    n_steps = env.episode_length
    dt = env.dt
    t_ctrl = np.arange(n_steps) * dt
    t_state = np.arange(n_steps + 1) * dt
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Actions
    ax = axes[0]
    ax.step(t_ctrl, td3_results['actions'], 'g-', lw=2, where='post', label='TD3')
    ax.step(t_ctrl, ppo_results['actions'], 'b-', lw=2, where='post', label='PPO')
    ax.step(t_ctrl, mpc_results['actions'], 'orange', lw=2, ls='--', where='post', label='MPC')
    ax.set_ylabel('T_g,in [K]')
    ax.legend()
    ax.set_title('RL vs MPC: Flash Calciner Control')
    
    # Conversion
    ax = axes[1]
    ax.plot(t_state, td3_results['alphas'], 'g-', lw=2, label='TD3')
    ax.plot(t_state, ppo_results['alphas'], 'b-', lw=2, label='PPO')
    ax.plot(t_state, mpc_results['alphas'], color='orange', ls='--', lw=2, label='MPC')
    ax.step(t_ctrl, env.alpha_min[:n_steps], 'r:', lw=2, where='post', label='Target')
    ax.set_ylabel('Conversion α')
    ax.legend()
    
    # Power
    ax = axes[2]
    ax.step(t_ctrl, td3_results['powers'], 'g-', lw=2, where='post', label='TD3')
    ax.step(t_ctrl, ppo_results['powers'], 'b-', lw=2, where='post', label='PPO')
    ax.step(t_ctrl, mpc_results['powers'], color='orange', ls='--', lw=2, where='post', label='MPC')
    ax.set_ylabel('Power [MW]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_training(td3_history, ppo_history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    
    # Power
    ax = axes[0]
    window = 20
    if len(td3_history['power']) > window:
        td3_smooth = np.convolve(td3_history['power'], np.ones(window)/window, mode='valid')
        ax.plot(td3_history['episode'][:len(td3_smooth)], td3_smooth, 'g-', label='TD3')
    if len(ppo_history['power']) > window:
        ppo_smooth = np.convolve(ppo_history['power'], np.ones(window)/window, mode='valid')
        ax.plot(ppo_history['episode'][:len(ppo_smooth)], ppo_smooth, 'b-', label='PPO')
    ax.set_ylabel('Avg Power [MW]')
    ax.legend()
    ax.set_title('RL Training Progress')
    
    # Violations
    ax = axes[1]
    if len(td3_history['violations']) > window:
        td3_smooth = np.convolve(td3_history['violations'], np.ones(window)/window, mode='valid')
        ax.plot(td3_history['episode'][:len(td3_smooth)], td3_smooth, 'g-', label='TD3')
    if len(ppo_history['violations']) > window:
        ppo_smooth = np.convolve(ppo_history['violations'], np.ones(window)/window, mode='valid')
        ax.plot(ppo_history['episode'][:len(ppo_smooth)], ppo_smooth, 'b-', label='PPO')
    ax.set_ylabel('Violations')
    ax.set_xlabel('Episode')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("RL Evaluation: TD3 and PPO vs MPC")
    print("=" * 70)
    
    env = CalcinerEnv()
    
    # Train TD3
    print("\n--- Training TD3 ---")
    td3_agent, td3_history = train_td3(
        env, n_episodes=300, batch_size=128, 
        start_steps=500, exploration_noise=30.0
    )
    
    # Train PPO
    print("\n--- Training PPO ---")
    ppo_agent, ppo_history = train_ppo(
        env, n_episodes=200, trajectories_per_update=10
    )
    
    # Evaluate
    print("\n--- Evaluating ---")
    env.reset(seed=42)
    td3_results = evaluate_agent(td3_agent, env)
    
    env.reset(seed=42)
    ppo_results = evaluate_agent(ppo_agent, env)
    
    mpc_results = run_mpc_baseline(env, env.episode_length)
    
    # Plot
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plot_comparison(td3_results, ppo_results, mpc_results, env, 
                    figures_dir / "rl_vs_mpc.png")
    plot_training(td3_history, ppo_history, figures_dir / "rl_training.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Algorithm':<12} {'Avg Power':<12} {'Final α':<12}")
    print("-" * 36)
    print(f"{'TD3':<12} {np.mean(td3_results['powers']):.3f} MW     {td3_results['alphas'][-1]:.3f}")
    print(f"{'PPO':<12} {np.mean(ppo_results['powers']):.3f} MW     {ppo_results['alphas'][-1]:.3f}")
    print(f"{'MPC':<12} {np.mean(mpc_results['powers']):.3f} MW     {mpc_results['alphas'][-1]:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

