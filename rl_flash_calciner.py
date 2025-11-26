"""
RL Baseline for Flash Calciner Control

Simple policy gradient (REINFORCE) to learn energy-optimal control.
Compares against MPC performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpc_flash_calciner import CalcinerDynamics, EconomicMPC


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

class CalcinerEnv:
    """
    Gym-like environment for flash calciner control.
    
    State: [α, α_min]  (2D - simple)
    Action: T_g_in normalized to [-1, 1] → [900, 1300] K
    Reward: -power - penalty * constraint_violation
    """
    
    def __init__(self, episode_length=40, dt=0.5):
        self.model = CalcinerDynamics(tau=2.0, dt=dt)
        self.episode_length = episode_length
        self.dt = dt
        
        # Action bounds
        self.u_min = 900.0
        self.u_max = 1300.0
        
        # Reward scaling
        self.P_ref = self.model.heater_power(1100)  # ~0.38 MW
        self.penalty = 100.0
        
    def reset(self, seed=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
            
        self.t = 0
        self.alpha = 0.90
        
        # Fixed scenario for reproducibility
        self.alpha_min = np.ones(self.episode_length + 10) * 0.95
        self.alpha_min[10:25] = 0.99
        self.alpha_min[30:] = 0.90
        
        self.disturbances = np.zeros(self.episode_length + 10)
        self.disturbances[15:20] = -0.03
        
        return self._get_obs()
    
    def _get_obs(self):
        """Current observation: [α, α_min] normalized."""
        return np.array([
            (self.alpha - 0.9) / 0.1,  # normalize to ~[-1, 1]
            (self.alpha_min[self.t] - 0.9) / 0.1,
        ])
    
    def _action_to_temp(self, action):
        """Convert normalized action [-1, 1] to temperature [900, 1300]."""
        return 1100.0 + action * 200.0  # Maps [-1,1] to [900, 1300]
    
    def step(self, action):
        """Take action, return (obs, reward, done, info)."""
        # Convert action to temperature
        u = self._action_to_temp(np.clip(action, -1, 1))
        
        # Step dynamics
        self.alpha = self.model.step(self.alpha, u, self.disturbances[self.t])
        
        # Compute reward
        power = self.model.heater_power(u)
        violation = max(0, self.alpha_min[self.t] - self.alpha)
        reward = -power / self.P_ref - self.penalty * violation**2
        
        self.t += 1
        done = self.t >= self.episode_length
        
        return self._get_obs(), reward, done, {"power": power, "u": u, "alpha": self.alpha}


# -----------------------------------------------------------------------------
# Simple Policy (Linear + Softmax for simplicity)
# -----------------------------------------------------------------------------

class LinearPolicy:
    """
    Linear Gaussian policy: μ = w·s + b, fixed σ.
    Very simple - should learn quickly if problem is easy.
    """
    
    def __init__(self, obs_dim=2, lr=0.1):
        self.w = np.zeros(obs_dim)
        self.b = 0.0  # Initial mean action = 0 → T = 1100 K
        self.log_std = np.log(0.5)  # std = 0.5 → ~100 K
        self.lr = lr
        
    def forward(self, obs):
        """Compute mean action."""
        return np.dot(self.w, obs) + self.b
    
    def sample(self, obs):
        """Sample action from Gaussian policy."""
        mean = self.forward(obs)
        std = np.exp(self.log_std)
        action = mean + std * np.random.randn()
        return action, mean, std
    
    def update(self, trajectories):
        """REINFORCE update."""
        # Collect all data
        all_obs = []
        all_actions = []
        all_returns = []
        
        for traj in trajectories:
            obs_list, action_list, reward_list = zip(*traj)
            
            # Compute returns (cumulative reward, no discount for simplicity)
            returns = np.cumsum(reward_list[::-1])[::-1]
            
            all_obs.extend(obs_list)
            all_actions.extend(action_list)
            all_returns.extend(returns)
        
        obs = np.array(all_obs)
        actions = np.array(all_actions)
        returns = np.array(all_returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute gradients
        means = obs @ self.w + self.b
        std = np.exp(self.log_std)
        
        # d log π / d w = (a - μ) / σ² * s
        # d log π / d b = (a - μ) / σ²
        diff = (actions - means) / (std**2)
        
        grad_w = np.mean(returns[:, None] * diff[:, None] * obs, axis=0)
        grad_b = np.mean(returns * diff)
        
        # Update
        self.w += self.lr * grad_w
        self.b += self.lr * grad_b


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_rl(n_epochs=200, episodes_per_epoch=5):
    """Train RL agent."""
    env = CalcinerEnv()
    policy = LinearPolicy(obs_dim=2, lr=0.05)
    
    history = {'epoch': [], 'return': [], 'power': [], 'violations': []}
    
    print("=" * 60)
    print("Training RL Agent (REINFORCE with Linear Policy)")
    print("=" * 60)
    
    for epoch in range(n_epochs):
        trajectories = []
        epoch_returns = []
        epoch_powers = []
        epoch_violations = []
        
        for ep in range(episodes_per_epoch):
            obs = env.reset(seed=epoch * 100 + ep)
            traj = []
            total_return = 0
            total_power = 0
            violations = 0
            
            while True:
                action, _, _ = policy.sample(obs)
                next_obs, reward, done, info = env.step(action)
                
                traj.append((obs, action, reward))
                total_return += reward
                total_power += info["power"]
                if info["alpha"] < env.alpha_min[env.t - 1]:
                    violations += 1
                
                obs = next_obs
                if done:
                    break
            
            trajectories.append(traj)
            epoch_returns.append(total_return)
            epoch_powers.append(total_power / env.episode_length)
            epoch_violations.append(violations)
        
        # Update policy
        policy.update(trajectories)
        
        # Log
        history['epoch'].append(epoch)
        history['return'].append(np.mean(epoch_returns))
        history['power'].append(np.mean(epoch_powers))
        history['violations'].append(np.mean(epoch_violations))
        
        if epoch % 40 == 0:
            avg_pwr = np.mean(epoch_powers)  # Already in MW from heater_power
            print(f"Epoch {epoch:3d}: return={np.mean(epoch_returns):7.1f}, "
                  f"power={avg_pwr:.3f} MW, "
                  f"violations={np.mean(epoch_violations):.1f}")
    
    print(f"\nFinal policy: w={policy.w}, b={policy.b:.2f}")
    return policy, history


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate(policy, n_episodes=10):
    """Compare RL vs MPC."""
    env = CalcinerEnv()
    mpc = EconomicMPC(env.model, horizon=8)
    
    rl_powers = []
    rl_violations = []
    mpc_powers = []
    mpc_violations = []
    
    print("\n" + "=" * 60)
    print("Evaluation: RL vs MPC (same scenarios)")
    print("=" * 60)
    
    for ep in range(n_episodes):
        # --- RL rollout ---
        obs = env.reset(seed=1000 + ep)
        alpha_min_traj = env.alpha_min.copy()
        disturbances = env.disturbances.copy()
        alpha0 = env.alpha
        
        rl_total_power = 0
        rl_viol = 0
        rl_actions = []
        
        while env.t < env.episode_length:
            action = policy.forward(env._get_obs())  # Use mean (no exploration)
            _, _, done, info = env.step(action)
            rl_total_power += info["power"]
            rl_actions.append(info["u"])
            if info["alpha"] < alpha_min_traj[env.t - 1]:
                rl_viol += 1
        
        rl_powers.append(rl_total_power / env.episode_length)
        rl_violations.append(rl_viol)
        
        # --- MPC rollout ---
        alpha = alpha0
        mpc_total_power = 0
        mpc_viol = 0
        mpc_actions = []
        
        for t in range(env.episode_length):
            sp_traj = alpha_min_traj[t:t + 8]
            if len(sp_traj) < 8:
                sp_traj = np.pad(sp_traj, (0, 8 - len(sp_traj)), constant_values=sp_traj[-1])
            u_opt = mpc.solve(alpha, sp_traj)
            u = u_opt[0]
            
            mpc_total_power += env.model.heater_power(u)
            mpc_actions.append(u)
            alpha = env.model.step(alpha, u, disturbances[t])
            if alpha < alpha_min_traj[t]:
                mpc_viol += 1
        
        mpc_powers.append(mpc_total_power / env.episode_length)
        mpc_violations.append(mpc_viol)
        
        if ep == 0:
            # Plot first episode comparison
            plot_episode(rl_actions, mpc_actions, alpha_min_traj[:40], env)
    
    # Summary
    print("\n" + "-" * 55)
    print(f"{'Metric':<25} {'RL':>12} {'MPC':>12} {'Winner':>8}")
    print("-" * 55)
    
    rl_avg = np.mean(rl_powers)  # Already in MW
    mpc_avg = np.mean(mpc_powers)  # Already in MW
    print(f"{'Avg Power [MW]':<25} {rl_avg:>12.3f} {mpc_avg:>12.3f} "
          f"{'RL' if rl_avg < mpc_avg else 'MPC':>8}")
    
    print(f"{'Avg Violations':<25} {np.mean(rl_violations):>12.1f} {np.mean(mpc_violations):>12.1f} "
          f"{'RL' if np.mean(rl_violations) < np.mean(mpc_violations) else 'MPC':>8}")
    print("-" * 55)
    
    if rl_avg < mpc_avg and np.mean(rl_violations) <= np.mean(mpc_violations):
        print("\n🏆 RL BEATS MPC! 🏆")
    elif rl_avg < mpc_avg * 1.02 and np.mean(rl_violations) <= np.mean(mpc_violations):
        print("\n≈ RL matches MPC (within 2%)")
    else:
        print("\nMPC wins (RL needs more training or problem is too easy for MPC)")
    
    return {'rl_powers': rl_powers, 'mpc_powers': mpc_powers}


def plot_episode(rl_actions, mpc_actions, alpha_min, env):
    """Plot one episode comparison."""
    t = np.arange(len(rl_actions)) * env.dt
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(t, rl_actions, where='post', label='RL', linewidth=2)
    ax.step(t, mpc_actions, where='post', label='MPC', linewidth=2, linestyle='--')
    ax.axhline(1261, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'$T_{g,in}$ [K]')
    ax.set_title('RL vs MPC Control Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_vs_mpc.png', dpi=150, bbox_inches='tight')
    print("\nSaved: rl_vs_mpc.png")
    plt.close()


def plot_training(history):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    
    ax = axes[0]
    ax.plot(history['epoch'], history['power'])  # Already in MW
    ax.axhline(0.42, color='r', linestyle='--', alpha=0.7, label='MPC target')
    ax.set_ylabel('Avg Power [MW]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('RL Training Progress')
    
    ax = axes[1]
    ax.plot(history['epoch'], history['violations'])
    ax.set_ylabel('Violations / Episode')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_training.png', dpi=150, bbox_inches='tight')
    print("Saved: rl_training.png")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    policy, history = train_rl(n_epochs=200, episodes_per_epoch=5)
    plot_training(history)
    evaluate(policy, n_episodes=10)
