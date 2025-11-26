"""
TD3 (Twin Delayed DDPG) for Flash Calciner Control

PyTorch implementation with autodiff for fast training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int):
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


# =============================================================================
# Neural Networks
# =============================================================================

class Actor(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 64,
                 action_low: float = 900.0, action_high: float = 1300.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.net(x)


# =============================================================================
# TD3 Agent
# =============================================================================

class TD3Agent:
    """TD3 (Twin Delayed DDPG) Agent for continuous control."""
    
    def __init__(
        self,
        obs_dim: int = 3,
        hidden: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 20.0,
        noise_clip: float = 40.0,
        policy_delay: int = 2,
        action_low: float = 900.0,
        action_high: float = 1300.0,
    ):
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.action_low = action_low
        self.action_high = action_high
        
        # Actor
        self.actor = Actor(obs_dim, hidden, action_low, action_high).to(device)
        self.actor_target = Actor(obs_dim, hidden, action_low, action_high).to(device)
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
        
    def select_action(self, obs: np.ndarray, noise: float = 0.0) -> float:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = self.actor(obs_t).cpu().numpy()[0, 0]
        if noise > 0:
            action += np.random.randn() * noise
        return np.clip(action, self.action_low, self.action_high)
    
    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int = 128):
        self.total_it += 1
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                self.action_low, self.action_high
            )
            
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy update
        if self.total_it % self.policy_delay == 0:
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


def train_td3(env, n_episodes: int = 300, batch_size: int = 128,
              start_steps: int = 500, exploration_noise: float = 30.0,
              verbose: bool = True) -> Tuple[TD3Agent, Dict]:
    """Train TD3 agent on the given environment."""
    
    agent = TD3Agent(obs_dim=3, hidden=128, lr=1e-3)
    buffer = ReplayBuffer(capacity=50000)
    
    history = {'episode': [], 'return': [], 'power': [], 'violations': []}
    total_steps = 0
    
    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        ep_return, ep_power, ep_viol = 0, 0, 0
        
        done = False
        while not done:
            if total_steps < start_steps:
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
                agent.train_step(buffer, batch_size)
        
        history['episode'].append(ep)
        history['return'].append(ep_return)
        history['power'].append(ep_power / env.episode_length)
        history['violations'].append(ep_viol)
        
        if verbose and ep % 50 == 0:
            recent = min(20, ep + 1)
            avg_power = np.mean(history['power'][-recent:])
            avg_viol = np.mean(history['violations'][-recent:])
            print(f"  TD3 Episode {ep:3d}: power={avg_power:.3f} MW, violations={avg_viol:.1f}")
    
    return agent, history

