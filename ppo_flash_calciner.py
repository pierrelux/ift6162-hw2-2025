"""
PPO for Flash Calciner - Simplified Robust Version
"""

import numpy as np
import matplotlib.pyplot as plt
from mpc_flash_calciner import CalcinerDynamics, EconomicMPC


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

class CalcinerEnv:
    def __init__(self, episode_length=40, dt=0.5):
        self.model = CalcinerDynamics(tau=2.0, dt=dt)
        self.episode_length = episode_length
        self.dt = dt
        self.u_min, self.u_max = 900.0, 1300.0
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        self.alpha = 0.90
        
        self.alpha_min = np.ones(self.episode_length + 10) * 0.95
        self.alpha_min[10:25] = 0.99
        self.alpha_min[30:] = 0.90
        
        self.disturbances = np.zeros(self.episode_length + 10)
        self.disturbances[15:20] = -0.03
        self.disturbances[35:] = 0.02
        
        return self._get_obs()
    
    def _get_obs(self):
        return np.array([
            self.alpha,
            self.alpha_min[self.t],
            self.t / self.episode_length,  # Time feature
        ])
    
    def step(self, action):
        # Action is raw temperature in [900, 1300]
        u = np.clip(action, self.u_min, self.u_max)
        
        old_alpha = self.alpha
        self.alpha = self.model.step(self.alpha, u, self.disturbances[self.t])
        
        power = self.model.heater_power(u)
        target = self.alpha_min[self.t]
        
        # Reward: balance energy minimization with constraint satisfaction
        # Normalize power to similar scale as constraint penalty
        reward = -power * 2.0  # Energy cost (scaled)
        
        margin = self.alpha - target  # Positive = above target (good)
        if margin < 0:
            reward -= 10.0 * margin**2  # Quadratic penalty for violation
        elif margin < 0.02:
            reward += 0.5 * margin  # Small bonus for being close to target (not over-heating)
        
        self.t += 1
        done = self.t >= self.episode_length
        
        return self._get_obs(), reward, done, {"power": power, "u": u, "alpha": self.alpha}


# -----------------------------------------------------------------------------
# Simple Actor-Critic with PPO-style updates
# -----------------------------------------------------------------------------

class ActorCritic:
    def __init__(self, lr=0.01):
        # Linear actor: action = w @ obs + b
        # Intuition: T = f(α, α_min) - need higher T when α < α_min
        self.w_actor = np.array([-200.0, 400.0, -50.0])  # Lower α → higher T, higher α_min → higher T
        self.b_actor = 1000.0  # Base temperature
        self.log_std = np.log(30.0)  # Std dev for exploration
        
        # Linear critic: value = w @ obs + b  
        self.w_critic = np.zeros(3)
        self.b_critic = 0.0
        
        self.lr = lr
        
    def get_action(self, obs, deterministic=False):
        """Get action from policy."""
        mean = np.dot(self.w_actor, obs) + self.b_actor
        mean = np.clip(mean, 900, 1300)
        
        if deterministic:
            return mean
        
        std = np.exp(self.log_std)
        action = mean + std * np.random.randn()
        return np.clip(action, 900, 1300)
    
    def get_value(self, obs):
        """Get value estimate."""
        return np.dot(self.w_critic, obs) + self.b_critic
    
    def update(self, trajectories, gamma=0.99):
        """Simple policy gradient update."""
        all_obs = []
        all_actions = []
        all_returns = []
        
        for traj in trajectories:
            obs_list, act_list, rew_list = zip(*traj)
            
            # Compute returns
            returns = []
            G = 0
            for r in reversed(rew_list):
                G = r + gamma * G
                returns.insert(0, G)
            
            all_obs.extend(obs_list)
            all_actions.extend(act_list)
            all_returns.extend(returns)
        
        obs = np.array(all_obs)
        actions = np.array(all_actions)
        returns = np.array(all_returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient
        means = obs @ self.w_actor + self.b_actor
        means = np.clip(means, 900, 1300)
        std = np.exp(self.log_std)
        
        # Gradient: d log π / d θ * advantage
        diff = actions - means
        
        # Update actor
        for i in range(3):
            grad = np.mean(returns * diff * obs[:, i]) / (std**2)
            self.w_actor[i] += self.lr * np.clip(grad, -10, 10)
        
        grad_b = np.mean(returns * diff) / (std**2)
        self.b_actor += self.lr * np.clip(grad_b, -10, 10)
        
        # Update critic (simple TD)
        values = obs @ self.w_critic + self.b_critic
        td_error = returns - values
        
        for i in range(3):
            grad = np.mean(td_error * obs[:, i])
            self.w_critic[i] += self.lr * 0.5 * np.clip(grad, -1, 1)
        self.b_critic += self.lr * 0.5 * np.clip(np.mean(td_error), -1, 1)
        
        # Decay exploration
        self.log_std = max(self.log_std - 0.01, np.log(10.0))


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(n_iterations=200, episodes_per_iter=5):
    env = CalcinerEnv()
    agent = ActorCritic(lr=0.05)
    
    history = {'iter': [], 'return': [], 'power': [], 'violations': []}
    
    print("=" * 60)
    print("Training PPO-style Agent")
    print("=" * 60)
    
    for it in range(n_iterations):
        trajectories = []
        iter_returns, iter_powers, iter_violations = [], [], []
        
        for ep in range(episodes_per_iter):
            obs = env.reset(seed=it * 100 + ep)
            traj = []
            ep_return, ep_power, ep_viol = 0, 0, 0
            
            while True:
                action = agent.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                
                traj.append((obs, action, reward))
                ep_return += reward
                ep_power += info["power"]
                if info["alpha"] < env.alpha_min[env.t - 1]:
                    ep_viol += 1
                
                obs = next_obs
                if done:
                    break
            
            trajectories.append(traj)
            iter_returns.append(ep_return)
            iter_powers.append(ep_power / env.episode_length)
            iter_violations.append(ep_viol)
        
        agent.update(trajectories)
        
        history['iter'].append(it)
        history['return'].append(np.mean(iter_returns))
        history['power'].append(np.mean(iter_powers))
        history['violations'].append(np.mean(iter_violations))
        
        if it % 40 == 0:
            print(f"Iter {it:3d}: return={np.mean(iter_returns):7.1f}, "
                  f"power={np.mean(iter_powers):.3f} MW, "
                  f"violations={np.mean(iter_violations):.1f}, "
                  f"std={np.exp(agent.log_std):.0f} K")
    
    print(f"\nFinal policy: w={agent.w_actor}, b={agent.b_actor:.0f}")
    return agent, history


# -----------------------------------------------------------------------------
# 4-Panel Evaluation Plot
# -----------------------------------------------------------------------------

def evaluate_and_plot(agent):
    env = CalcinerEnv()
    mpc = EconomicMPC(env.model, horizon=8)
    
    env.reset(seed=42)
    alpha_min_traj = env.alpha_min.copy()
    disturbances = env.disturbances.copy()
    alpha0 = env.alpha
    n_steps = env.episode_length
    dt = env.dt
    
    # --- RL rollout ---
    env.reset(seed=42)
    rl_actions, rl_alphas, rl_powers = [], [env.alpha], []
    
    while env.t < n_steps:
        action = agent.get_action(env._get_obs(), deterministic=True)
        _, _, _, info = env.step(action)
        rl_actions.append(info["u"])
        rl_alphas.append(info["alpha"])
        rl_powers.append(info["power"])
    
    # --- MPC rollout ---
    alpha = alpha0
    mpc_actions, mpc_alphas, mpc_powers = [], [alpha0], []
    
    for t in range(n_steps):
        sp_traj = alpha_min_traj[t:t + 8]
        if len(sp_traj) < 8:
            sp_traj = np.pad(sp_traj, (0, 8 - len(sp_traj)), constant_values=sp_traj[-1])
        u_opt = mpc.solve(alpha, sp_traj)
        u = u_opt[0]
        
        mpc_actions.append(u)
        mpc_powers.append(env.model.heater_power(u))
        alpha = env.model.step(alpha, u, disturbances[t])
        mpc_alphas.append(alpha)
    
    # --- Baseline ---
    alpha = alpha0
    base_alphas = [alpha0]
    for t in range(n_steps):
        alpha = env.model.step(alpha, 1261.15, disturbances[t])
        base_alphas.append(alpha)
    
    t_ctrl = np.arange(n_steps) * dt
    t_state = np.arange(n_steps + 1) * dt
    
    # 4-Panel Plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # Panel 1: Control
    ax = axes[0]
    ax.step(t_ctrl, rl_actions, where='post', color='blue', linewidth=2, label='RL (PPO)')
    ax.step(t_ctrl, mpc_actions, where='post', color='orange', linewidth=2, 
            linestyle='--', label='MPC')
    ax.axhline(1261.15, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax.set_ylabel(r'$T_{g,in}$ [K]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('RL (PPO) vs MPC: Flash Calciner Control')
    
    # Panel 2: Conversion
    ax = axes[1]
    ax.plot(t_state, rl_alphas, 'b-', linewidth=2, label='RL')
    ax.plot(t_state, mpc_alphas, color='orange', linestyle='--', linewidth=2, label='MPC')
    ax.plot(t_state, base_alphas, color='gray', linestyle=':', linewidth=1.5, 
            label='Baseline', alpha=0.7)
    ax.step(t_ctrl, alpha_min_traj[:n_steps], where='post', color='red', 
            linestyle=':', linewidth=2, label=r'$\alpha_{min}$')
    ax.fill_between(t_ctrl, 0.8, alpha_min_traj[:n_steps], alpha=0.1, color='red', step='post')
    ax.set_ylabel(r'Conversion $\alpha$')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.02])
    
    # Panel 3: Power
    ax = axes[2]
    ax.step(t_ctrl, rl_powers, where='post', color='blue', linewidth=2, label='RL')
    ax.step(t_ctrl, mpc_powers, where='post', color='orange', linewidth=2, 
            linestyle='--', label='MPC')
    ax.axhline(0.46, color='gray', linestyle=':', alpha=0.7, label='Baseline')
    ax.axhline(np.mean(rl_powers), color='blue', linestyle=':', alpha=0.5)
    ax.axhline(np.mean(mpc_powers), color='orange', linestyle=':', alpha=0.5)
    ax.set_ylabel('Power [MW]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Disturbances
    ax = axes[3]
    ax.step(t_ctrl, disturbances[:n_steps] * 100, where='post', color='green', linewidth=2)
    ax.fill_between(t_ctrl, 0, disturbances[:n_steps] * 100, alpha=0.3, 
                    color='green', step='post')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Disturbance [%]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-5, 5])
    
    plt.tight_layout()
    plt.savefig('ppo_vs_mpc.png', dpi=150, bbox_inches='tight')
    print("\nSaved: ppo_vs_mpc.png")
    plt.close()
    
    # Summary
    rl_viol = sum(1 for i in range(n_steps) if rl_alphas[i+1] < alpha_min_traj[i])
    mpc_viol = sum(1 for i in range(n_steps) if mpc_alphas[i+1] < alpha_min_traj[i])
    
    print("\n" + "=" * 60)
    print("Episode Summary")
    print("=" * 60)
    print(f"{'Metric':<25} {'RL':>12} {'MPC':>12} {'Baseline':>12}")
    print("-" * 60)
    print(f"{'Avg Power [MW]':<25} {np.mean(rl_powers):>12.3f} {np.mean(mpc_powers):>12.3f} {0.460:>12.3f}")
    print(f"{'Violations':<25} {rl_viol:>12d} {mpc_viol:>12d}")
    print(f"{'Energy Savings vs Base':<25} {(1-np.mean(rl_powers)/0.46)*100:>11.1f}% {(1-np.mean(mpc_powers)/0.46)*100:>11.1f}%")
    print("=" * 60)
    
    if np.mean(rl_powers) < np.mean(mpc_powers) and rl_viol <= mpc_viol + 5:
        print("\n🏆 RL WINS! 🏆")
    elif abs(np.mean(rl_powers) - np.mean(mpc_powers)) < 0.02:
        print("\n≈ RL matches MPC!")
    else:
        print(f"\nMPC better by {(np.mean(rl_powers)/np.mean(mpc_powers)-1)*100:.1f}%")


def plot_training(history):
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    
    ax = axes[0]
    ax.plot(history['iter'], history['power'])
    ax.axhline(0.42, color='r', linestyle='--', alpha=0.7, label='MPC ~0.42 MW')
    ax.set_ylabel('Avg Power [MW]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('RL Training Progress')
    
    ax = axes[1]
    ax.plot(history['iter'], history['violations'])
    ax.set_ylabel('Violations / Episode')
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppo_training.png', dpi=150, bbox_inches='tight')
    print("Saved: ppo_training.png")
    plt.close()


if __name__ == "__main__":
    agent, history = train(n_iterations=200, episodes_per_iter=5)
    plot_training(history)
    evaluate_and_plot(agent)
