"""
Hierarchical Storage Architecture

Efficient storage management for multi-agent RL training
"""

import numpy as np
import torch
from collections import deque
import random

# Utility Functions

def compute_gae_returns(rewards, values, dones, next_values, gamma=0.99, gae_lambda=0.95):
    """Compute GAE advantages and returns"""
    num_steps, num_agents = rewards.shape
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for step in reversed(range(num_steps)):
        if step == num_steps - 1:
            next_value = next_values
        else:
            next_value = values[step + 1]
        
        mask = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
    
    returns = advantages + values
    return returns, advantages

# User Agent Circular Buffer

class UserCircularBuffer:
    """
    Circular PPO rollout buffer with episodic storage
    - Fixed capacity with circular overwriting
    - Compatible interface with standard PPO buffer
    - Efficient memory reuse for training
    """
    
    def __init__(self, buffer_capacity, num_agents, obs_dim, action_dim, device, 
                 action_mask_dim=None, central_obs_dim=None):
        self.num_steps = buffer_capacity
        self.buffer_capacity = buffer_capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        print(f"📊 Circular PPO Rollout Buffer:")
        print(f"   Buffer Capacity: {buffer_capacity} timesteps")
        print(f"   Number of Agents: {num_agents}")
        print(f"   Total Capacity: {buffer_capacity * num_agents} transitions")
        self.observations = np.zeros((buffer_capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_capacity, num_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((buffer_capacity, num_agents), dtype=np.float32)
        self.rewards = np.zeros((buffer_capacity, num_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_capacity, num_agents), dtype=np.float32)
        self.values = np.zeros((buffer_capacity + 1, num_agents), dtype=np.float32)
        self.cost_values = np.zeros((buffer_capacity + 1, num_agents), dtype=np.float32)
        self.costs = np.zeros((buffer_capacity, num_agents), dtype=np.float32)
        
        if action_mask_dim is not None:
            self.action_masks = np.zeros((buffer_capacity, num_agents, action_mask_dim), dtype=np.float32)
        else:
            self.action_masks = None
            
        if central_obs_dim is not None:
            self.central_observations = np.zeros((buffer_capacity, num_agents, central_obs_dim), dtype=np.float32)
            self.central_obs_dim = central_obs_dim
        else:
            self.central_observations = None
            self.central_obs_dim = None
        
        self.step = 0
        self.write_ptr = 0
        self.is_full = False
        
        total_memory_MB = (buffer_capacity * num_agents * (obs_dim + action_dim + 6 + (action_mask_dim or 0) + (central_obs_dim or 0))) * 4 / (1024**2)
        print(f"   Estimated Memory: {total_memory_MB:.1f} MB")
    
    def insert(self, obs, actions, log_probs, rewards, dones, values, costs, action_masks=None, central_obs=None, cost_values=None):
        """Insert transition data with circular overwriting"""
        self.observations[self.write_ptr] = obs
        self.actions[self.write_ptr] = actions
        self.log_probs[self.write_ptr] = log_probs
        self.rewards[self.write_ptr] = rewards
        self.dones[self.write_ptr] = dones
        self.values[self.write_ptr] = values
        self.costs[self.write_ptr] = costs
        if cost_values is not None:
            self.cost_values[self.write_ptr] = cost_values
        
        if action_masks is not None and self.action_masks is not None:
            self.action_masks[self.write_ptr] = action_masks.astype(np.float32)
        if central_obs is not None and self.central_observations is not None:
            self.central_observations[self.write_ptr] = central_obs
    
    def advance_step(self):
        """Advance write pointer"""
        self.step += 1
        self.write_ptr = (self.write_ptr + 1) % self.buffer_capacity
        if self.write_ptr == 0 and self.step > 0:
            self.is_full = True
    
    def insert_final_values(self, final_values, final_cost_values=None):
        """Insert bootstrap values for GAE"""
        self.values[self.write_ptr] = final_values
        if final_cost_values is not None:
            self.cost_values[self.write_ptr] = final_cost_values
    
    def get_training_data(self, normalize_adv=True):
        """Get training batch with computed returns and advantages"""
        returns, advantages = self.compute_returns_and_advantages()
        cost_returns, cost_advantages = self.compute_cost_returns_and_advantages()
        
        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Determine valid data range and reorder if buffer is full
        if self.is_full:
            end_idx = self.buffer_capacity
            # Reorder all data to correct time sequence
            ordered_obs = np.zeros((end_idx, self.num_agents, self.obs_dim), dtype=np.float32)
            ordered_actions = np.zeros((end_idx, self.num_agents, self.action_dim), dtype=np.float32)
            ordered_log_probs = np.zeros((end_idx, self.num_agents), dtype=np.float32)
            ordered_values = np.zeros((end_idx, self.num_agents), dtype=np.float32)
            ordered_costs = np.zeros((end_idx, self.num_agents), dtype=np.float32)
            ordered_cost_values = np.zeros((end_idx, self.num_agents), dtype=np.float32)
            
            for i in range(end_idx):
                src_idx = (self.write_ptr + i) % self.buffer_capacity
                ordered_obs[i] = self.observations[src_idx]
                ordered_actions[i] = self.actions[src_idx]
                ordered_log_probs[i] = self.log_probs[src_idx]
                ordered_values[i] = self.values[src_idx]
                ordered_costs[i] = self.costs[src_idx]
                ordered_cost_values[i] = self.cost_values[src_idx]
                
            # Optional data reordering
            if self.action_masks is not None:
                ordered_action_masks = np.zeros((end_idx, self.num_agents, self.action_masks.shape[-1]), dtype=np.float32)
                for i in range(end_idx):
                    src_idx = (self.write_ptr + i) % self.buffer_capacity
                    ordered_action_masks[i] = self.action_masks[src_idx]
            else:
                ordered_action_masks = None
                
            if self.central_observations is not None:
                ordered_central_obs = np.zeros((end_idx, self.num_agents, self.central_observations.shape[-1]), dtype=np.float32)
                for i in range(end_idx):
                    src_idx = (self.write_ptr + i) % self.buffer_capacity
                    ordered_central_obs[i] = self.central_observations[src_idx]
            else:
                ordered_central_obs = None
        else:
            end_idx = self.step
            ordered_obs = self.observations[:end_idx]
            ordered_actions = self.actions[:end_idx]
            ordered_log_probs = self.log_probs[:end_idx]
            ordered_values = self.values[:end_idx]
            ordered_costs = self.costs[:end_idx]
            ordered_cost_values = self.cost_values[:end_idx]
            ordered_action_masks = self.action_masks[:end_idx] if self.action_masks is not None else None
            ordered_central_obs = self.central_observations[:end_idx] if self.central_observations is not None else None
            
        if end_idx == 0:
            return None
        
        # Flatten data following original PPORolloutBuffer approach
        data = {
            'observations': self.to_torch(ordered_obs.reshape(-1, self.obs_dim)),
            'actions': self.to_torch(ordered_actions.reshape(-1, self.action_dim)),
            'log_probs': self.to_torch(ordered_log_probs.reshape(-1)),
            'returns': self.to_torch(returns.reshape(-1)),
            'advantages': self.to_torch(advantages.reshape(-1)),
            'values': self.to_torch(ordered_values.reshape(-1)),
            'costs': self.to_torch(ordered_costs.reshape(-1)),  # PPO-Lagrangian
            'cost_returns': self.to_torch(cost_returns.reshape(-1)),
            'cost_advantages': self.to_torch(cost_advantages.reshape(-1)),
            'cost_values': self.to_torch(ordered_cost_values.reshape(-1))
        }
        
        # Add action_masks if available
        if ordered_action_masks is not None:
            mask_dim = ordered_action_masks.shape[-1]
            data['action_masks'] = self.to_torch(ordered_action_masks.reshape(-1, mask_dim))
            
        # Add central_obs for MAPPO
        if ordered_central_obs is not None:
            data['central_observations'] = self.to_torch(ordered_central_obs.reshape(-1, self.central_obs_dim))
            
        return data
    
    def get_buffer_stats(self):
        """获取buffer统计信息"""
        if self.is_full:
            utilization = 1.0
            total_samples = self.buffer_capacity * self.num_agents
        else:
            utilization = self.step / self.buffer_capacity
            total_samples = self.step * self.num_agents
            
        return {
            'total_samples': int(total_samples),
            'avg_utilization': float(utilization)
        }
    
    # Maintain identical interface
    def to_torch(self, arr, unsqueeze_dim=None):
        """转换为tensor - 与原storage相同"""
        tensor = torch.from_numpy(arr).to(self.device)
        if unsqueeze_dim is not None:
            tensor = tensor.unsqueeze(unsqueeze_dim)
        return tensor
    
    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """Compute returns and advantages with circular buffer data reordering"""
        if self.is_full:
            # Reorder data to correct time sequence
            end_idx = self.buffer_capacity
            ordered_rewards = np.zeros((end_idx, self.num_agents), dtype=np.float32)
            ordered_values = np.zeros((end_idx + 1, self.num_agents), dtype=np.float32)
            ordered_dones = np.zeros((end_idx, self.num_agents), dtype=np.float32)
            
            for i in range(end_idx):
                src_idx = (self.write_ptr + i) % self.buffer_capacity
                ordered_rewards[i] = self.rewards[src_idx]
                ordered_values[i] = self.values[src_idx]
                ordered_dones[i] = self.dones[src_idx]
            
            # Bootstrap value at current write position
            ordered_values[-1] = self.values[self.write_ptr]
            
            return compute_gae_returns(ordered_rewards, ordered_values[:-1], ordered_dones, ordered_values[-1], gamma, gae_lambda)
        else:
            # Not full, use original logic
            end_idx = self.step
            return compute_gae_returns(self.rewards[:end_idx], self.values[:end_idx], self.dones[:end_idx], self.values[end_idx], gamma, gae_lambda)

    def compute_cost_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """Compute cost returns and advantages with circular buffer data reordering"""
        if self.is_full:
            # Reorder data to correct time sequence
            end_idx = self.buffer_capacity
            ordered_costs = np.zeros((end_idx, self.num_agents), dtype=np.float32)
            ordered_cost_values = np.zeros((end_idx + 1, self.num_agents), dtype=np.float32)
            ordered_dones = np.zeros((end_idx, self.num_agents), dtype=np.float32)
            
            for i in range(end_idx):
                src_idx = (self.write_ptr + i) % self.buffer_capacity
                ordered_costs[i] = self.costs[src_idx]
                ordered_cost_values[i] = self.cost_values[src_idx]
                ordered_dones[i] = self.dones[src_idx]
            
            # Bootstrap value at current write position
            ordered_cost_values[-1] = self.cost_values[self.write_ptr]
            
            return compute_gae_returns(ordered_costs, ordered_cost_values[:-1], ordered_dones, ordered_cost_values[-1], gamma, gae_lambda)
        else:
            # Not full, use original logic
            end_idx = self.step
            return compute_gae_returns(self.costs[:end_idx], self.cost_values[:end_idx], self.dones[:end_idx], self.cost_values[end_idx], gamma, gae_lambda)
    
    def after_update(self):
        """Operation after training update - circular buffer does not reset"""
        pass  # Circular buffer retains data

# PPO Rollout Buffer (original, backup)

class PPORolloutBuffer:
    """PPO rollout buffer - reference original storage style"""
    
    def __init__(self, num_steps, num_agents, obs_dim, action_dim, device, action_mask_dim=None, central_obs_dim=None):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Pre-allocate arrays
        self.observations = np.zeros((num_steps, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.values = np.zeros((num_steps + 1, num_agents), dtype=np.float32)
        # Cost value function for PPO-Lagrangian GAE-C
        self.cost_values = np.zeros((num_steps + 1, num_agents), dtype=np.float32)
        self.costs = np.zeros((num_steps, num_agents), dtype=np.float32)  # PPO-Lagrangian
        
        # Pre-allocate action_masks if dimension provided
        if action_mask_dim is not None:
            self.action_masks = np.zeros((num_steps, num_agents, action_mask_dim), dtype=np.float32)
        else:
            self.action_masks = None
            
        # Pre-allocate central_obs for MAPPO
        if central_obs_dim is not None:
            self.central_observations = np.zeros((num_steps, num_agents, central_obs_dim), dtype=np.float32)
            self.central_obs_dim = central_obs_dim
        else:
            self.central_observations = None
            self.central_obs_dim = None
        
        self.step = 0
    
    def insert(self, obs, actions, log_probs, rewards, dones, values, costs, action_masks=None, central_obs=None, cost_values=None):
        """插入一步数据 - 要求提供costs与cost_values；支持central_obs与action_masks"""
        if self.step >= self.num_steps:
            return
            
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.costs[self.step] = costs
        self.cost_values[self.step] = cost_values
        
        # Handle action_masks
        if action_masks is not None and self.action_masks is not None:
            # Convert bool mask to float32
            self.action_masks[self.step] = action_masks.astype(np.float32)
            
        # Handle central_obs for MAPPO
        if central_obs is not None and self.central_observations is not None:
            self.central_observations[self.step] = central_obs
        
    def advance_step(self):
        """前进一步"""
        self.step += 1
    
    def insert_final_values(self, final_values, final_cost_values=None):
        """插入最终值与最终cost值"""
        self.values[self.step] = final_values
        if final_cost_values is not None:
            self.cost_values[self.step] = final_cost_values
    
    def to_torch(self, arr, unsqueeze_dim=None):
        """转换为tensor - 与原storage相同"""
        tensor = torch.from_numpy(arr).to(self.device)
        if unsqueeze_dim is not None:
            tensor = tensor.unsqueeze(unsqueeze_dim)
        return tensor
    
    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """计算回报和优势"""
        return compute_gae_returns(
            self.rewards, self.values[:-1], self.dones,
            self.values[-1], gamma, gae_lambda
        )

    def compute_cost_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """计算cost的回报与优势（用于Lagrangian约束）"""
        return compute_gae_returns(
            self.costs, self.cost_values[:-1], self.dones,
            self.cost_values[-1], gamma, gae_lambda
        )
    
    def get_training_data(self, normalize_adv=True):
        """获取训练数据"""
        returns, advantages = self.compute_returns_and_advantages()
        cost_returns, cost_advantages = self.compute_cost_returns_and_advantages()
        
        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # Note: Do not normalize cost_advantages to avoid biasing constraint signals
        
        # Flatten data
        data = {
            'observations': self.to_torch(self.observations.reshape(-1, self.obs_dim)),
            'actions': self.to_torch(self.actions.reshape(-1, self.action_dim)),
            'log_probs': self.to_torch(self.log_probs.reshape(-1)),
            'returns': self.to_torch(returns.reshape(-1)),
            'advantages': self.to_torch(advantages.reshape(-1)),
            'values': self.to_torch(self.values[:-1].reshape(-1)),
            'costs': self.to_torch(self.costs.reshape(-1)),  # PPO-Lagrangian
            'cost_returns': self.to_torch(cost_returns.reshape(-1)),
            'cost_advantages': self.to_torch(cost_advantages.reshape(-1)),
            'cost_values': self.to_torch(self.cost_values[:-1].reshape(-1))
        }
        
        # Add action_masks if available
        if self.action_masks is not None:
            mask_dim = self.action_masks.shape[-1]
            data['action_masks'] = self.to_torch(self.action_masks.reshape(-1, mask_dim))
            
        # Add central_obs for MAPPO
        if self.central_observations is not None:
            data['central_observations'] = self.to_torch(self.central_observations.reshape(-1, self.central_obs_dim))
        
        return data
    
    def after_update(self):
        """更新后重置"""
        self.step = 0

# SAC Replay Buffer (allocation layer)

class SACReplayBuffer:
    """SAC replay buffer - concise implementation"""
    
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Pre-allocate arrays
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.size = 0
        self.pointer = 0
    
    def push(self, obs, action, reward, next_obs, done):
        """添加经验"""
        self.observations[self.pointer] = obs
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_observations[self.pointer] = next_obs
        self.dones[self.pointer] = done
        
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """采样批次"""
        if self.size < batch_size:
            return None
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'observations': torch.from_numpy(self.observations[indices]).to(self.device),
            'actions': torch.from_numpy(self.actions[indices]).to(self.device),
            'rewards': torch.from_numpy(self.rewards[indices]).to(self.device),
            'next_observations': torch.from_numpy(self.next_observations[indices]).to(self.device),
            'dones': torch.from_numpy(self.dones[indices]).to(self.device)
        }
    
    def __len__(self):
        return self.size

# Deployment Agent Circular Buffer (maintain original storage structure)

class DeploymentCircularBuffer:
    """
    Circular version of AutoregressiveBuffer - maintain original storage structure, just become circular queue
    - Storage structure identical: (num_servers, buffer_capacity, ...)
    - Circular writing: new data overwrites oldest data
    - Interface compatible: identical to original AutoregressiveBuffer
    - Solve data waste: retain historical data for training
    """
    
    def __init__(self, num_servers, device, num_services, buffer_capacity, deployment_update_interval, obs_dim=90):
        """
        Args:
            buffer_capacity (int): 固定buffer大小（循环队列长度）
                                  替代原来的trajectory_length，但存储结构完全一样
                                  注意：这里的单位是"部署决策次数"，不是transition！
                                  例如：buffer_capacity=50 表示每个服务器保留最近50次部署决策
        """
        self.num_servers = num_servers
        self.device = device
        
        # Maintain same attribute names as original AutoregressiveBuffer
        self.trajectory_length = buffer_capacity
        self.buffer_capacity = buffer_capacity
        self.max_deployment_seq_len = num_services
        self.obs_dim = obs_dim
        
        print(f"📊 DeploymentCircularBuffer (circular buffer):")
        print(f"   Number of servers: {num_servers} (one trajectory per server)")
        print(f"   Buffer capacity: {buffer_capacity} deployment decisions")
        print(f"   Max sequence length per decision: {self.max_deployment_seq_len} (total services)")
        print(f"   Observation dimension: {self.obs_dim}")
        
        # === Identical storage structure (num_servers, buffer_capacity, ...) ===
        self.deploy_obs = np.zeros((self.num_servers, self.buffer_capacity, self.obs_dim), dtype=np.float32)
        self.deploy_actions = np.zeros((self.num_servers, self.buffer_capacity, self.max_deployment_seq_len), dtype=np.int32)
        self.deploy_rewards = np.zeros((self.num_servers, self.buffer_capacity), dtype=np.float32)
        self.deploy_values = np.zeros((self.num_servers, self.buffer_capacity, self.max_deployment_seq_len + 1), dtype=np.float32)
        self.deploy_log_probs = np.zeros((self.num_servers, self.buffer_capacity, self.max_deployment_seq_len), dtype=np.float32)
        self.deploy_dones = np.zeros((self.num_servers, self.buffer_capacity), dtype=np.bool_)
        self.deployment_seq_lengths = np.zeros((self.num_servers, self.buffer_capacity), dtype=np.int32)
        
        # === Circular queue logic ===
        self.current_step = np.zeros(self.num_servers, dtype=np.int32)  # Current write position (compatible with original interface externally)
        self.write_ptrs = np.zeros(self.num_servers, dtype=np.int32)    # actual write pointers
        self.is_full = np.zeros(self.num_servers, dtype=np.bool_)       # whether each server has written a full round
        
        total_memory_MB = (self.num_servers * buffer_capacity * (self.obs_dim + self.max_deployment_seq_len * 3 + 1)) * 4 / (1024**2)
        print(f"   Estimated memory usage: {total_memory_MB:.1f} MB")
    
    def insert_deployment_decision(self, server_id, obs, deployment_actions, seq_length, sequence_reward, step_values, step_log_probs):
        """
        Insert data - maintain original interface, just change to circular writing
        """
        # Write to current pointer position
        write_ptr = self.write_ptrs[server_id]
        
        # Handle observation
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy().flatten()
        else:
            obs_np = np.array(obs).flatten()
            
        # Store observation
        self.deploy_obs[server_id, write_ptr, :obs_np.shape[0]] = obs_np
        
        # Store action sequence and length
        self.deploy_actions[server_id, write_ptr] = deployment_actions
        self.deployment_seq_lengths[server_id, write_ptr] = seq_length
        
        # Store trajectory level reward (scalar) and sequence level data
        self.deploy_rewards[server_id, write_ptr] = sequence_reward
        self.deploy_values[server_id, write_ptr] = step_values  
        self.deploy_log_probs[server_id, write_ptr] = step_log_probs
        self.deploy_dones[server_id, write_ptr] = False  # circular buffer does not use done
        
        # Update step count and pointer (circular logic)
        self.current_step[server_id] += 1
        self.write_ptrs[server_id] = (write_ptr + 1) % self.buffer_capacity
        
        # Check if a full round has been written
        if self.write_ptrs[server_id] == 0 and self.current_step[server_id] > 0:
            self.is_full[server_id] = True
    
    def get_training_data(self, gamma=0.99, gae_lambda=0.95):
        """
        Get autoregressive training data - exactly as in original AutoregressiveBuffer
        """
        # === Use maximum dimension directly to avoid dynamic calculation ===
        max_batch_size = self.num_servers * self.buffer_capacity
        
        # Pre-allocate full batch arrays
        batch_obs = np.zeros((max_batch_size, self.obs_dim), dtype=np.float32)
        batch_actions = np.zeros((max_batch_size, self.max_deployment_seq_len), dtype=np.int32) 
        batch_seq_lengths = np.zeros(max_batch_size, dtype=np.int32)
        batch_step_log_probs = np.zeros((max_batch_size, self.max_deployment_seq_len), dtype=np.float32)
        batch_step_returns = np.zeros((max_batch_size, self.max_deployment_seq_len), dtype=np.float32)
        batch_step_advantages = np.zeros((max_batch_size, self.max_deployment_seq_len), dtype=np.float32)
        
        valid_samples = 0
        
        # ✅ Correct: Compute GAE on the coarse-grained trajectory of each server
        for server_id in range(self.num_servers):
            if self.is_full[server_id]:
                num_decisions = self.buffer_capacity
                # Reorder data to correct time order
                ordered_rewards = np.zeros(num_decisions, dtype=np.float32)
                ordered_dones = np.zeros(num_decisions, dtype=np.float32)
                ordered_obs = np.zeros((num_decisions, self.obs_dim), dtype=np.float32)
                ordered_actions = np.zeros((num_decisions, self.max_deployment_seq_len), dtype=np.int32)
                ordered_seq_lengths = np.zeros(num_decisions, dtype=np.int32)
                ordered_log_probs = np.zeros((num_decisions, self.max_deployment_seq_len), dtype=np.float32)
                ordered_values = np.zeros((num_decisions, self.max_deployment_seq_len + 1), dtype=np.float32)
                
                for i in range(num_decisions):
                    src_idx = (self.write_ptrs[server_id] + i) % self.buffer_capacity
                    ordered_rewards[i] = self.deploy_rewards[server_id, src_idx]
                    ordered_dones[i] = self.deploy_dones[server_id, src_idx]
                    ordered_obs[i] = self.deploy_obs[server_id, src_idx]
                    ordered_actions[i] = self.deploy_actions[server_id, src_idx]
                    ordered_seq_lengths[i] = self.deployment_seq_lengths[server_id, src_idx]
                    ordered_log_probs[i] = self.deploy_log_probs[server_id, src_idx]
                    ordered_values[i] = self.deploy_values[server_id, src_idx]
            else:
                num_decisions = self.current_step[server_id]
                ordered_rewards = self.deploy_rewards[server_id, :num_decisions]
                ordered_dones = self.deploy_dones[server_id, :num_decisions]
                ordered_obs = self.deploy_obs[server_id, :num_decisions]
                ordered_actions = self.deploy_actions[server_id, :num_decisions]
                ordered_seq_lengths = self.deployment_seq_lengths[server_id, :num_decisions]
                ordered_log_probs = self.deploy_log_probs[server_id, :num_decisions]
                ordered_values = self.deploy_values[server_id, :num_decisions]
                
            if num_decisions == 0:
                continue

            # Compute sequence average value for each point in the trajectory
            trajectory_values = np.zeros(num_decisions, dtype=np.float32)
            for i in range(num_decisions):
                seq_len = ordered_seq_lengths[i]
                if seq_len > 0:
                    trajectory_values[i] = np.mean(ordered_values[i, :seq_len])
            
            # Value of the last point used for bootstrap
            last_seq_len = ordered_seq_lengths[num_decisions - 1]
            bootstrap_value = np.mean(ordered_values[num_decisions - 1, :last_seq_len]) if last_seq_len > 0 else 0.0

            # ✅ Compute GAE on coarse-grained trajectory
            trajectory_advantages, trajectory_returns = self._compute_trajectory_gae(
                ordered_rewards,
                trajectory_values,
                ordered_dones,
                bootstrap_value,
                gamma,
                gae_lambda
            )

            # Fill the calculated trajectory level advantages and returns back into batch data
            for i in range(num_decisions):
                seq_len = ordered_seq_lengths[i]
                if seq_len > 0:
                    start_index = valid_samples + i
                    batch_obs[start_index] = ordered_obs[i]
                    batch_actions[start_index] = ordered_actions[i]
                    batch_seq_lengths[start_index] = seq_len
                    batch_step_log_probs[start_index, :seq_len] = ordered_log_probs[i, :seq_len]
                    
                    # Broadcast trajectory level advantage and return to all steps in the corresponding sequence
                    batch_step_advantages[start_index, :seq_len] = trajectory_advantages[i]
                    batch_step_returns[start_index, :seq_len] = trajectory_returns[i]

            valid_samples += num_decisions
        
        if valid_samples == 0:
            return {}
        
        # Truncate to valid samples
        return {
            'observations': torch.from_numpy(batch_obs[:valid_samples]).to(self.device, dtype=torch.float32),
            'action_sequences': torch.from_numpy(batch_actions[:valid_samples]).to(self.device, dtype=torch.long),
            'sequence_lengths': torch.from_numpy(batch_seq_lengths[:valid_samples]).to(self.device, dtype=torch.long),
            'step_log_probs': torch.from_numpy(batch_step_log_probs[:valid_samples]).to(self.device, dtype=torch.float32),
            'step_returns': torch.from_numpy(batch_step_returns[:valid_samples]).to(self.device, dtype=torch.float32),
            'step_advantages': torch.from_numpy(batch_step_advantages[:valid_samples]).to(self.device, dtype=torch.float32)
        }
    
    def _compute_trajectory_gae(self, rewards, values, dones, bootstrap_value, gamma, gae_lambda):
        """Correctly compute GAE on coarse-grained trajectory - identical to original AutoregressiveBuffer"""
        advantages = np.zeros_like(rewards)
        gae = 0.0
        T = len(rewards)
        for t in reversed(range(T)):
            # done signal now comes from trajectory, not from inside sequence
            is_last_step = (t == T - 1)
            next_nonterminal = 1.0 - dones[t]
            
            if is_last_step:
                 next_value = bootstrap_value
            else:
                next_value = values[t + 1]
                
            # GAE formula
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns
    
    def get_buffer_stats(self):
        """Get buffer statistics"""
        total_samples = 0
        utilizations = []
        
        for server_id in range(self.num_servers):
            if self.is_full[server_id]:
                utilization = 1.0
                samples = self.buffer_capacity
            else:
                utilization = self.current_step[server_id] / self.buffer_capacity
                samples = self.current_step[server_id]
            
            utilizations.append(utilization)
            total_samples += samples
            
        return {
            'total_samples': int(total_samples),
            'avg_utilization': float(np.mean(utilizations))
        }
    
    # === Maintain the same interface ===
    def reset_trajectories(self):
        """Reset all trajectories - circular buffer does not reset"""
        pass  # circular buffer retains data, no reset
    
    def after_update(self):
        """Operation after training update - circular buffer does not reset"""
        pass  # circular buffer retains data, no reset

# Original AutoregressiveBuffer (backup)

class AutoregressiveBuffer:
    """Efficient autoregressive sequence buffer - use numpy arrays, consistent style with other buffers"""
    
    def __init__(self, num_servers, device, num_services, num_steps, deployment_update_interval, obs_dim=90):
        self.num_servers = num_servers
        self.device = device
        
        # === Correct trajectory understanding: each server has one long trajectory ===
        # Number of trajectories = num_servers (5 servers = 5 trajectories)
        # Trajectory length = number of deployment decisions = num_steps // deployment_update_interval
        self.trajectory_length = num_steps // deployment_update_interval
        # Max deployment sequence length = total number of services (max number of services selected per decision)
        self.max_deployment_seq_len = num_services
        self.obs_dim = obs_dim
        
        print(f"📊 AutoregressiveBuffer buffer size (corrected version):")
        print(f"   Number of servers: {num_servers} (one trajectory per server)")
        print(f"   Trajectory length per server: {self.trajectory_length} (deployment decisions: {num_steps} / {deployment_update_interval})")
        print(f"   Max sequence length per decision: {self.max_deployment_seq_len} (total services)")
        print(f"   Observation dimension: {self.obs_dim}")
        
        # === Efficient numpy array pre-allocation - correct dimensions ===
        # Shape: (num_servers, trajectory_length, obs_dim) - observation sequence per server
        self.deploy_obs = np.zeros((self.num_servers, self.trajectory_length, self.obs_dim), dtype=np.float32)
        # Shape: (num_servers, trajectory_length, max_deployment_seq_len) - deployment plan per decision
        self.deploy_actions = np.zeros((self.num_servers, self.trajectory_length, self.max_deployment_seq_len), dtype=np.int32)
        # Shape: (num_servers, trajectory_length) - trajectory level reward storage (scalar)
        self.deploy_rewards = np.zeros((self.num_servers, self.trajectory_length), dtype=np.float32)
        self.deploy_values = np.zeros((self.num_servers, self.trajectory_length, self.max_deployment_seq_len + 1), dtype=np.float32)  # +1 for bootstrap
        self.deploy_log_probs = np.zeros((self.num_servers, self.trajectory_length, self.max_deployment_seq_len), dtype=np.float32)  # sequence level log_prob
        self.deploy_dones = np.zeros((self.num_servers, self.trajectory_length), dtype=np.bool_)  # maintain trajectory level
        # Shape: (num_servers, trajectory_length) - actual deployment sequence length per decision
        self.deployment_seq_lengths = np.zeros((self.num_servers, self.trajectory_length), dtype=np.int32)
        
        # === Index tracking ===
        # Shape: (num_servers,) - current time step per server
        self.current_step = np.zeros(self.num_servers, dtype=np.int32)
        
        print(f"   Pre-allocated memory: {num_servers} × {self.trajectory_length} × {obs_dim} = {num_servers * self.trajectory_length * obs_dim / 1e6:.1f}M elements (observations)")
        print(f"               + {num_servers} × {self.trajectory_length} × {self.max_deployment_seq_len} = {num_servers * self.trajectory_length * self.max_deployment_seq_len / 1e6:.1f}M elements (actions)")
    
    def insert_deployment_decision(self, server_id, obs, deployment_actions, seq_length, sequence_reward, step_values, step_log_probs):
        """
        Insert one complete deployment decision - trajectory level storage
        
        Args:
            server_id: Server ID
            obs: Observation (shape: obs_dim)
            deployment_actions: numpy array (max_seq_len,) padded, -1 for invalid
            seq_length: int, actual sequence length
            sequence_reward: float, trajectory level reward (scalar, not split)
            step_values: numpy array (max_seq_len + 1,) value + bootstrap for each step in autoregressive sequence
            step_log_probs: numpy array (max_seq_len,) log_prob for each step in autoregressive sequence
        """
        step_idx = self.current_step[server_id]
        
        # Check boundary
        if step_idx >= self.trajectory_length:
            print(f"⚠️ Server {server_id} trajectory is full ({step_idx}/{self.trajectory_length})")
            return
            
        # Handle observation
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy().flatten()
        else:
            obs_np = np.array(obs).flatten()
            
        # Store observation
        self.deploy_obs[server_id, step_idx, :obs_np.shape[0]] = obs_np
        
        # Store action sequence and length
        self.deploy_actions[server_id, step_idx] = deployment_actions
        self.deployment_seq_lengths[server_id, step_idx] = seq_length
        
        # Store trajectory level reward (scalar) and sequence level data
        self.deploy_rewards[server_id, step_idx] = sequence_reward  # scalar reward
        self.deploy_values[server_id, step_idx] = step_values  
        self.deploy_log_probs[server_id, step_idx] = step_log_probs
        self.deploy_dones[server_id, step_idx] = (step_idx == self.trajectory_length - 1)
        
        # Update step count
        self.current_step[server_id] += 1
    

    
    def reset_trajectories(self):
        """Reset all trajectories (start new training epoch)"""
        self.current_step.fill(0)
    
    def get_training_data(self, gamma=0.99, gae_lambda=0.95):
        """
        Get autoregressive training data - directly use maximum dimension, sequence level pre-allocation storage
        
        Output format:
        - observations: (batch_size, obs_dim)
        - action_sequences: (batch_size, max_seq_len) - padded sequences
        - sequence_lengths: (batch_size,) - actual sequence lengths
        - step_log_probs: (batch_size, max_seq_len) - sequence level log_prob
        - step_returns: (batch_size, max_seq_len) - sequence level return
        - step_advantages: (batch_size, max_seq_len) - sequence level advantage
        """
        # === Use maximum dimension directly to avoid dynamic calculation ===
        max_batch_size = self.num_servers * self.trajectory_length
        
        # Pre-allocate full batch arrays
        batch_obs = np.zeros((max_batch_size, self.obs_dim), dtype=np.float32)
        batch_actions = np.zeros((max_batch_size, self.max_deployment_seq_len), dtype=np.int32) 
        batch_seq_lengths = np.zeros(max_batch_size, dtype=np.int32)
        batch_step_log_probs = np.zeros((max_batch_size, self.max_deployment_seq_len), dtype=np.float32)
        batch_step_returns = np.zeros((max_batch_size, self.max_deployment_seq_len), dtype=np.float32)
        batch_step_advantages = np.zeros((max_batch_size, self.max_deployment_seq_len), dtype=np.float32)
        
        valid_samples = 0
        
        # ✅ Correct: Compute GAE on coarse-grained trajectory per server
        for server_id in range(self.num_servers):
            num_decisions = self.current_step[server_id]
            if num_decisions == 0:
                continue

            # Extract data for the entire trajectory
            trajectory_rewards = self.deploy_rewards[server_id, :num_decisions]
            trajectory_dones = self.deploy_dones[server_id, :num_decisions]
            
            # Compute sequence average value for each point in the trajectory
            trajectory_values = np.zeros(num_decisions, dtype=np.float32)
            for i in range(num_decisions):
                seq_len = self.deployment_seq_lengths[server_id, i]
                if seq_len > 0:
                    trajectory_values[i] = np.mean(self.deploy_values[server_id, i, :seq_len])
            
            # Value of the last point used for bootstrap
            last_seq_len = self.deployment_seq_lengths[server_id, num_decisions - 1]
            bootstrap_value = np.mean(self.deploy_values[server_id, num_decisions - 1, :last_seq_len]) if last_seq_len > 0 else 0.0

            # ✅ Compute GAE on coarse-grained trajectory
            trajectory_advantages, trajectory_returns = self._compute_trajectory_gae(
                trajectory_rewards,
                trajectory_values,
                trajectory_dones,
                bootstrap_value,
                gamma,
                gae_lambda
            )

            # Fill calculated trajectory level advantages and returns back into batch data
            for i in range(num_decisions):
                seq_len = self.deployment_seq_lengths[server_id, i]
                if seq_len > 0:
                    start_index = valid_samples + i
                    batch_obs[start_index] = self.deploy_obs[server_id, i]
                    batch_actions[start_index] = self.deploy_actions[server_id, i]
                    batch_seq_lengths[start_index] = seq_len
                    batch_step_log_probs[start_index, :seq_len] = self.deploy_log_probs[server_id, i, :seq_len]
                    
                    # Broadcast trajectory level advantage and return to all steps in the corresponding sequence
                    batch_step_advantages[start_index, :seq_len] = trajectory_advantages[i]
                    batch_step_returns[start_index, :seq_len] = trajectory_returns[i]

            valid_samples += num_decisions
        
        if valid_samples == 0:
            return {}
        
        # Truncate to valid samples
        return {
            'observations': torch.from_numpy(batch_obs[:valid_samples]).to(self.device, dtype=torch.float32),
            'action_sequences': torch.from_numpy(batch_actions[:valid_samples]).to(self.device, dtype=torch.long),
            'sequence_lengths': torch.from_numpy(batch_seq_lengths[:valid_samples]).to(self.device, dtype=torch.long),
            'step_log_probs': torch.from_numpy(batch_step_log_probs[:valid_samples]).to(self.device, dtype=torch.float32),
            'step_returns': torch.from_numpy(batch_step_returns[:valid_samples]).to(self.device, dtype=torch.float32),
            'step_advantages': torch.from_numpy(batch_step_advantages[:valid_samples]).to(self.device, dtype=torch.float32)
        }

    def _compute_trajectory_gae(self, rewards, values, dones, bootstrap_value, gamma, gae_lambda):
        """Correctly compute GAE"""
        advantages = np.zeros_like(rewards)
        gae = 0.0
        T = len(rewards)
        for t in reversed(range(T)):
            # done signal now comes from trajectory, not from inside sequence
            is_last_step = (t == T - 1)
            next_nonterminal = 1.0 - dones[t]
            
            if is_last_step:
                 next_value = bootstrap_value
            else:
                 next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def after_update(self):
        """Clear after update - efficient numpy version"""
        # Reset all counters (much faster than recreating arrays)
        self.current_step.fill(0)
        # Optional: zero array contents (usually not necessary as they will be overwritten)
        # self.deploy_obs.fill(0)
        # self.deploy_actions.fill(0)
        # self.deploy_rewards.fill(0)
        # self.deploy_values.fill(0)
        # self.deploy_dones.fill(False)
        # self.deployment_seq_lengths.fill(0)

# Main storage class

class HierarchicalStorage:
    """Hierarchical storage - maintain original storage's concise style"""
    
    def __init__(self, config, device):
        self.device = device
        self.config = config
        
        # === User Layer Buffer (supports replay or rollout mode) ===
        use_replay_buffers = config.get('use_replay_buffers', False)
        
        if use_replay_buffers:
            self.user_buffer = UserCircularBuffer(
                buffer_capacity=config.get('user_buffer_capacity', 1000),
                num_agents=config['num_users'],
                obs_dim=config['user_obs_dim'],
                action_dim=config['user_action_dim'],
                device=device,
                action_mask_dim=config.get('user_action_mask_dim', None),
                central_obs_dim=config.get('central_obs_dim', None)  # MAPPO needs
            )
        else:
            self.user_buffer = PPORolloutBuffer(
                num_steps=config['num_steps'],
                num_agents=config['num_users'],
                obs_dim=config['user_obs_dim'],
                action_dim=config['user_action_dim'],
                device=device,
                action_mask_dim=config.get('user_action_mask_dim', None),
                central_obs_dim=config.get('central_obs_dim', None)  # MAPPO needs
            )
        
        self.allocation_buffer = SACReplayBuffer(
            capacity=config.get('sac_buffer_size', 10000),
            obs_dim=config['alloc_obs_dim'],
            action_dim=config['alloc_action_dim'],
            device=device
        )
        
        # === Use circular buffer to replace the original one-time buffer ===
        self.deployment_buffer = DeploymentCircularBuffer(
            num_servers=config['num_servers'],
            device=device,
            num_services=config['num_services'],
            buffer_capacity=config.get('deployment_buffer_capacity', 50),  # Read from config file
            deployment_update_interval=config.get('deployment_update_interval', 5),
            obs_dim=config.get('deploy_obs_dim', 90)  # Keep the same default value as original AutoregressiveBuffer
        )
    
    # === User Layer Interface ===
    def insert_user_step(self, obs, actions, log_probs, rewards, dones, values, costs, action_masks=None, central_obs=None, cost_values=None):
        """Insert user data, including costs, cost_values, action_masks, and central_obs for PPO-Lagrangian and MAPPO"""
        self.user_buffer.insert(obs, actions, log_probs, rewards, dones, values, costs, action_masks, central_obs, cost_values)
    
    def advance_step(self):
        """Advance step"""
        self.user_buffer.advance_step()
    
    def insert_user_final_values(self, final_values, final_cost_values=None):
        """Insert final values and final cost values for user"""
        self.user_buffer.insert_final_values(final_values, final_cost_values)
    
    def get_user_training_data(self):
        """Get user training data"""
        return self.user_buffer.get_training_data()
    
    # === Allocation Layer Interface ===
    def insert_allocation_experience(self, obs, action, reward, next_obs, done):
        """Insert allocation experience"""
        # Store separately for each server
        for server_id in range(self.config['num_servers']):
            self.allocation_buffer.push(
                obs[server_id], action[server_id], reward[server_id],
                next_obs[server_id], done[server_id]
            )
    
    def sample_allocation_batch(self, batch_size):
        """Sample allocation batch"""
        return self.allocation_buffer.sample(batch_size)
    
    # === Deployment Layer Interface ===
    def insert_deployment_decision(self, server_id, obs, deployment_actions, seq_length, sequence_reward, step_values, step_log_probs):
        """
        Insert a complete deployment decision - sequence level storage
        
        Args:
            server_id: Server ID
            obs: Observation
            deployment_actions: numpy array (max_seq_len,) padded, -1 for invalid
            seq_length: int, actual sequence length
            sequence_reward: float, trajectory level reward (scalar, not split)
            step_values: numpy array (max_seq_len + 1,) value + bootstrap for each step in autoregressive sequence
            step_log_probs: numpy array (max_seq_len,) log_prob for each step in autoregressive sequence
        """
        self.deployment_buffer.insert_deployment_decision(server_id, obs, deployment_actions, seq_length, sequence_reward, step_values, step_log_probs)
    
    def get_deployment_training_data(self):
        """Get deployment training data"""
        return self.deployment_buffer.get_training_data()
    
    def get_deployment_buffer_stats(self):
        """Get deployment buffer statistics"""
        return self.deployment_buffer.get_buffer_stats()
    
    def get_user_buffer_stats(self):
        """Get user buffer statistics"""
        if hasattr(self.user_buffer, 'get_buffer_stats'):
            return self.user_buffer.get_buffer_stats()
        else:
            # Original PPORolloutBuffer has no statistics
            return {'total_samples': getattr(self.user_buffer, 'step', 0) * self.config['num_users']}
    
    # === Global Interface ===
    def after_update(self):
        """Reset after update (circular buffer does not reset, only user buffer resets)"""
        self.user_buffer.after_update()
        # Note: Circular buffer does not reset, data accumulates
    
    def ready_for_user_training(self):
        """Check if user layer is ready for training"""
        if hasattr(self.user_buffer, 'buffer_capacity'):
            # UserCircularBuffer mode: can train if there is data
            return self.user_buffer.step > 0
        else:
            # Original PPORolloutBuffer mode
            return self.user_buffer.step >= self.config['num_steps']
    
    def ready_for_allocation_training(self, min_batch_size=256):
        """Check if allocation layer is ready for training"""
        return len(self.allocation_buffer) >= min_batch_size
    
    def ready_for_deployment_training(self):
        """Check if deployment layer is ready for training (circular buffer is always ready for training)"""
        stats = self.deployment_buffer.get_buffer_stats()
        return stats['total_samples'] > 0