"""
PPO (Proximal Policy Optimization) for Flash Calciner Control

Simple linear actor-critic implementation.
"""

import numpy as np
from typing import Tuple, Dict


# =============================================================================
# Linear Actor-Critic
# =============================================================================

class LinearActorCritic:
    """Simple linear actor-critic for interpretable policies."""
    
    def __init__(self, obs_dim: int = 3, lr: float = 0.01,
                 action_low: float = 900.0, action_high: float = 1300.0):
        self.obs_dim = obs_dim
        self.action_low = action_low
        self.action_high = action_high
        
        # Linear actor: action = w @ obs + b
        self.w_actor = np.array([-200.0, 400.0, -50.0])
        self.b_actor = 1000.0
        self.log_std = np.log(30.0)
        
        # Linear critic
        self.w_critic = np.zeros(obs_dim)
        self.b_critic = 0.0
        
        self.lr = lr
        
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> float:
        mean = np.dot(self.w_actor, obs) + self.b_actor
        mean = np.clip(mean, self.action_low, self.action_high)
        
        if deterministic:
            return mean
        
        std = np.exp(self.log_std)
        action = mean + std * np.random.randn()
        return np.clip(action, self.action_low, self.action_high)
    
    def get_value(self, obs: np.ndarray) -> float:
        return np.dot(self.w_critic, obs) + self.b_critic
    
    def log_prob(self, obs: np.ndarray, action: float) -> float:
        mean = np.dot(self.w_actor, obs) + self.b_actor
        mean = np.clip(mean, self.action_low, self.action_high)
        std = np.exp(self.log_std)
        return -0.5 * ((action - mean) / std) ** 2 - self.log_std


# =============================================================================
# PPO Agent
# =============================================================================

class PPOAgent:
    """PPO agent with simple linear policy."""
    
    def __init__(self, obs_dim: int = 3, lr: float = 0.01,
                 gamma: float = 0.99, clip_epsilon: float = 0.2):
        self.policy = LinearActorCritic(obs_dim=obs_dim, lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> float:
        return self.policy.get_action(obs, deterministic)
    
    def update(self, trajectories: list):
        """Update policy using collected trajectories."""
        all_obs = []
        all_actions = []
        all_returns = []
        all_advantages = []
        all_old_log_probs = []
        
        for traj in trajectories:
            obs_list, action_list, reward_list = traj['obs'], traj['actions'], traj['rewards']
            
            # Compute returns and advantages
            T = len(reward_list)
            returns = np.zeros(T)
            returns[-1] = reward_list[-1]
            for t in range(T - 2, -1, -1):
                returns[t] = reward_list[t] + self.gamma * returns[t + 1]
            
            values = np.array([self.policy.get_value(o) for o in obs_list])
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            old_log_probs = [self.policy.log_prob(o, a) for o, a in zip(obs_list, action_list)]
            
            all_obs.extend(obs_list)
            all_actions.extend(action_list)
            all_returns.extend(returns)
            all_advantages.extend(advantages)
            all_old_log_probs.extend(old_log_probs)
        
        # PPO update
        for _ in range(5):
            for i in range(len(all_obs)):
                obs = all_obs[i]
                action = all_actions[i]
                ret = all_returns[i]
                adv = all_advantages[i]
                old_log_prob = all_old_log_probs[i]
                
                new_log_prob = self.policy.log_prob(obs, action)
                ratio = np.exp(new_log_prob - old_log_prob)
                
                clip_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -min(ratio * adv, clip_ratio * adv)
                
                # Update actor
                mean = np.dot(self.policy.w_actor, obs) + self.policy.b_actor
                mean = np.clip(mean, self.policy.action_low, self.policy.action_high)
                std = np.exp(self.policy.log_std)
                
                d_mean = (action - mean) / (std ** 2) * adv * self.lr * 0.1
                self.policy.w_actor += d_mean * obs
                self.policy.b_actor += d_mean
                
                # Update critic
                value = self.policy.get_value(obs)
                value_error = ret - value
                self.policy.w_critic += self.lr * 0.5 * value_error * obs
                self.policy.b_critic += self.lr * 0.5 * value_error


def train_ppo(env, n_episodes: int = 200, trajectories_per_update: int = 10,
              verbose: bool = True) -> Tuple[PPOAgent, Dict]:
    """Train PPO agent on the given environment."""
    
    agent = PPOAgent(obs_dim=3, lr=0.01)
    history = {'episode': [], 'return': [], 'power': [], 'violations': []}
    
    ep = 0
    while ep < n_episodes:
        trajectories = []
        
        for _ in range(trajectories_per_update):
            obs = env.reset(seed=ep)
            traj = {'obs': [], 'actions': [], 'rewards': []}
            ep_return, ep_power, ep_viol = 0, 0, 0
            
            done = False
            while not done:
                action = agent.select_action(obs)
                traj['obs'].append(obs)
                traj['actions'].append(action)
                
                next_obs, reward, done, info = env.step(action)
                traj['rewards'].append(reward)
                
                ep_return += reward
                ep_power += info["power"]
                if info["alpha"] < env.alpha_min[env.t - 1]:
                    ep_viol += 1
                
                obs = next_obs
            
            trajectories.append(traj)
            
            history['episode'].append(ep)
            history['return'].append(ep_return)
            history['power'].append(ep_power / env.episode_length)
            history['violations'].append(ep_viol)
            
            ep += 1
            if ep >= n_episodes:
                break
        
        agent.update(trajectories)
        
        if verbose and ep % 50 == 0:
            recent = min(20, ep)
            avg_power = np.mean(history['power'][-recent:])
            avg_viol = np.mean(history['violations'][-recent:])
            print(f"  PPO Episode {ep:3d}: power={avg_power:.3f} MW, violations={avg_viol:.1f}")
    
    return agent, history

