"""
TD3 (Twin Delayed DDPG) for Flash Calciner Control - PyTorch Version

Proper implementation with autodiff for fast training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpc_flash_calciner import CalcinerDynamics, EconomicMPC

device = torch.device("cpu")


# -----------------------------------------------------------------------------
# Environment (same as before)
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
        # Normalized observations for better learning
        return np.array([
            (self.alpha - 0.9) * 10,  # Deviation from 0.9, scaled
            (self.alpha_min[self.t] - 0.9) * 10,  # Setpoint deviation
            (self.alpha - self.alpha_min[self.t]) * 20,  # Constraint margin (key!)
            self.t / self.episode_length - 0.5,  # Centered time
        ], dtype=np.float32)
    
    def step(self, action):
        u = np.clip(action, self.u_min, self.u_max)
        self.alpha = self.model.step(self.alpha, u, self.disturbances[self.t])
        
        power = self.model.heater_power(u)
        target = self.alpha_min[self.t]
        
        # Simple reward: minimize power, penalize violations
        # Same structure as PPO (which works)
        power_normalized = (power - 0.35) / 0.15  # Normalize to ~[-1, 1]
        margin = self.alpha - target
        
        if margin >= 0:
            reward = -power_normalized + 0.1  # Energy cost + small bonus
        else:
            reward = -power_normalized - 5.0 * (-margin) - 0.5  # + violation penalty
        
        self.t += 1
        done = self.t >= self.episode_length
        
        return self._get_obs(), reward, done, {"power": power, "u": u, "alpha": self.alpha}


# -----------------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        )
    
    def __len__(self):
        return len(self.buffer)


# -----------------------------------------------------------------------------
# Neural Networks (PyTorch)
# -----------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim=4, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        
        # Initialize last layer to output ~0 (-> 1150 K)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        
        # Scale output to [900, 1300]
        self.action_scale = 150.0
        self.action_bias = 1150.0
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(self, obs_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden),  # obs + action
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.net(x)


# -----------------------------------------------------------------------------
# TD3 Agent
# -----------------------------------------------------------------------------

class TD3:
    def __init__(
        self,
        obs_dim=3,
        hidden=64,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=20.0,  # In action space [900, 1300]
        noise_clip=40.0,
        policy_delay=2,
    ):
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        # Actor
        self.actor = Actor(obs_dim, hidden).to(device)
        self.actor_target = Actor(obs_dim, hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Twin Critics
        self.critic1 = Critic(obs_dim, hidden).to(device)
        self.critic2 = Critic(obs_dim, hidden).to(device)
        self.critic1_target = Critic(obs_dim, hidden).to(device)
        self.critic2_target = Critic(obs_dim, hidden).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        self.total_it = 0
        
    def select_action(self, obs, noise=0.0):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = self.actor(obs_t).cpu().numpy()[0, 0]
        if noise > 0:
            action += np.random.randn() * noise
        return np.clip(action, 900, 1300)
    
    def train(self, replay_buffer, batch_size=128):
        self.total_it += 1
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(900, 1300)
            
            # Twin Q targets
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)
        
        # Update critics
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy update
        if self.total_it % self.policy_delay == 0:
            # Actor loss: maximize Q
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_td3(n_episodes=300, batch_size=128, start_steps=500, exploration_noise=30.0):
    env = CalcinerEnv()
    agent = TD3(obs_dim=4, hidden=128, lr=1e-3)  # Bigger network, faster learning
    buffer = ReplayBuffer(capacity=50000)
    
    history = {'episode': [], 'return': [], 'power': [], 'violations': []}
    
    print("=" * 60)
    print("Training TD3 Agent (PyTorch)")
    print("=" * 60)
    
    total_steps = 0
    
    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        ep_return, ep_power, ep_viol = 0, 0, 0
        
        done = False
        while not done:
            if total_steps < start_steps:
                # Warm start: sample around reasonable operating range
                action = np.random.uniform(1050, 1250)
            else:
                action = agent.select_action(obs, noise=exploration_noise)
            
            next_obs, reward, done, info = env.step(action)
            buffer.push(obs, action, reward, next_obs, float(done))
            
            ep_return += reward
            ep_power += info["power"]
            if info["alpha"] < env.alpha_min[env.t - 1]:
                ep_viol += 1
            
            obs = next_obs
            total_steps += 1
            
            if len(buffer) >= batch_size and total_steps >= start_steps:
                agent.train(buffer, batch_size)
        
        history['episode'].append(ep)
        history['return'].append(ep_return)
        history['power'].append(ep_power / env.episode_length)
        history['violations'].append(ep_viol)
        
        if ep % 50 == 0:
            recent = min(20, ep + 1)
            avg_power = np.mean(history['power'][-recent:])
            avg_viol = np.mean(history['violations'][-recent:])
            print(f"Episode {ep:3d}: return={ep_return:7.1f}, "
                  f"power={avg_power:.3f} MW, violations={avg_viol:.1f}")
    
    return agent, history


# -----------------------------------------------------------------------------
# Evaluation with 4-Panel Plot
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
    
    # --- TD3 rollout ---
    env.reset(seed=42)
    td3_actions, td3_alphas, td3_powers = [], [env.alpha], []
    
    while env.t < n_steps:
        action = agent.select_action(env._get_obs(), noise=0)
        _, _, _, info = env.step(action)
        td3_actions.append(info["u"])
        td3_alphas.append(info["alpha"])
        td3_powers.append(info["power"])
    
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
    
    ax = axes[0]
    ax.step(t_ctrl, td3_actions, where='post', color='green', linewidth=2, label='TD3')
    ax.step(t_ctrl, mpc_actions, where='post', color='orange', linewidth=2, 
            linestyle='--', label='MPC')
    ax.axhline(1261.15, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax.set_ylabel(r'$T_{g,in}$ [K]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('TD3 (PyTorch) vs MPC: Flash Calciner Control')
    
    ax = axes[1]
    ax.plot(t_state, td3_alphas, 'g-', linewidth=2, label='TD3')
    ax.plot(t_state, mpc_alphas, color='orange', linestyle='--', linewidth=2, label='MPC')
    ax.plot(t_state, base_alphas, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.step(t_ctrl, alpha_min_traj[:n_steps], where='post', color='red', 
            linestyle=':', linewidth=2, label=r'$\alpha_{min}$')
    ax.fill_between(t_ctrl, 0.8, alpha_min_traj[:n_steps], alpha=0.1, color='red', step='post')
    ax.set_ylabel(r'Conversion $\alpha$')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.02])
    
    ax = axes[2]
    ax.step(t_ctrl, td3_powers, where='post', color='green', linewidth=2, label='TD3')
    ax.step(t_ctrl, mpc_powers, where='post', color='orange', linewidth=2, linestyle='--', label='MPC')
    ax.axhline(0.46, color='gray', linestyle=':', alpha=0.7, label='Baseline')
    ax.axhline(np.mean(td3_powers), color='green', linestyle=':', alpha=0.5)
    ax.axhline(np.mean(mpc_powers), color='orange', linestyle=':', alpha=0.5)
    ax.set_ylabel('Power [MW]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[3]
    ax.step(t_ctrl, disturbances[:n_steps] * 100, where='post', color='purple', linewidth=2)
    ax.fill_between(t_ctrl, 0, disturbances[:n_steps] * 100, alpha=0.3, color='purple', step='post')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Disturbance [%]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-5, 5])
    
    plt.tight_layout()
    plt.savefig('td3_vs_mpc.png', dpi=150, bbox_inches='tight')
    print("\nSaved: td3_vs_mpc.png")
    plt.close()
    
    # Summary
    td3_viol = sum(1 for i in range(n_steps) if td3_alphas[i+1] < alpha_min_traj[i])
    mpc_viol = sum(1 for i in range(n_steps) if mpc_alphas[i+1] < alpha_min_traj[i])
    
    print("\n" + "=" * 60)
    print("Episode Summary")
    print("=" * 60)
    print(f"{'Metric':<25} {'TD3':>12} {'MPC':>12} {'Baseline':>12}")
    print("-" * 60)
    print(f"{'Avg Power [MW]':<25} {np.mean(td3_powers):>12.3f} {np.mean(mpc_powers):>12.3f} {0.460:>12.3f}")
    print(f"{'Violations':<25} {td3_viol:>12d} {mpc_viol:>12d}")
    print(f"{'Energy Savings':<25} {(1-np.mean(td3_powers)/0.46)*100:>11.1f}% {(1-np.mean(mpc_powers)/0.46)*100:>11.1f}%")
    print("=" * 60)
    
    if np.mean(td3_powers) < np.mean(mpc_powers) and td3_viol <= mpc_viol + 5:
        print("\n🏆 TD3 WINS! 🏆")
    elif abs(np.mean(td3_powers) - np.mean(mpc_powers)) < 0.015:
        print("\n≈ TD3 matches MPC!")
    else:
        diff = (np.mean(td3_powers)/np.mean(mpc_powers)-1)*100
        if diff > 0:
            print(f"\nMPC better by {diff:.1f}%")
        else:
            print(f"\nTD3 better by {-diff:.1f}%")


def plot_training(history):
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    
    window = 20
    power_smooth = np.convolve(history['power'], np.ones(window)/window, mode='valid')
    viol_smooth = np.convolve(history['violations'], np.ones(window)/window, mode='valid')
    
    ax = axes[0]
    ax.plot(history['episode'][:len(power_smooth)], power_smooth, 'g-')
    ax.axhline(0.419, color='orange', linestyle='--', alpha=0.7, label='MPC ~0.42 MW')
    ax.axhline(0.413, color='b', linestyle='--', alpha=0.7, label='PPO ~0.41 MW')
    ax.set_ylabel('Avg Power [MW]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('TD3 Training Progress (PyTorch)')
    
    ax = axes[1]
    ax.plot(history['episode'][:len(viol_smooth)], viol_smooth, 'g-')
    ax.axhline(23, color='orange', linestyle='--', alpha=0.5, label='MPC violations')
    ax.set_ylabel('Violations / Episode')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('td3_training.png', dpi=150, bbox_inches='tight')
    print("Saved: td3_training.png")
    plt.close()


if __name__ == "__main__":
    # Proper hyperparameters for TD3
    agent, history = train_td3(
        n_episodes=800,          # Even more training
        batch_size=256,
        start_steps=2000,
        exploration_noise=40.0,
    )
    plot_training(history)
    evaluate_and_plot(agent)
