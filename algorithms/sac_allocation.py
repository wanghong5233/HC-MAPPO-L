# algorithms/sac_allocation.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    """SAC Q network"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.q = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q(x)

class SACAllocation:
    """SAC algorithm - dedicated to resource allocation layer"""
    
    def __init__(self, actor_policy, obs_dim, action_dim, config, device):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Configuration parameters
        self.lr = float(config.get('agent.sac_lr'))
        # Temperature parameter (automatic adjustment)
        alpha_init = float(config.get('agent.sac_alpha', 0.2))
        self.log_alpha = torch.tensor(np.log(max(alpha_init, 1e-6)), requires_grad=True, device=device)
        self.alpha = np.exp(self.log_alpha.item())
        # Target entropy (for two resource heads, total entropy ≈2*log(K); take ratio for easy tuning)
        target_entropy_ratio = float(config.get('agent.sac_target_entropy_ratio', 0.7))
        self.num_users = action_dim // 2  # action_dim = num_users * 2 (comp + band)
        self.target_entropy = target_entropy_ratio * (2.0 * np.log(max(self.num_users, 1)))
        self.tau = float(config.get('agent.sac_tau'))
        self.gamma = float(config.get('agent.gamma'))
        self.num_users = action_dim // 2  # action_dim = num_users * 2 (comp + band)
        
        # Networks
        self.actor = actor_policy
        self.q1 = QNetwork(obs_dim, action_dim).to(device)
        self.q2 = QNetwork(obs_dim, action_dim).to(device)
        self.q1_target = QNetwork(obs_dim, action_dim).to(device)
        self.q2_target = QNetwork(obs_dim, action_dim).to(device)
        
        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.lr)
    
    def get_action(self, obs, deterministic=False):
        """Get action, support deterministic and stochastic modes"""
        with torch.no_grad():
            action_mean = self.actor.get_action(obs)

            if deterministic:
                return action_mean
            else:
                # Exploration: Do masked softmax normalization within comp/band separately according to association mask
                noise = torch.randn_like(action_mean) * 0.1
                action_noisy = action_mean + noise
                # Extract association mask
                assoc_mask = obs[:, :self.num_users].bool()
                comp_logits = action_noisy[:, :self.num_users]
                band_logits = action_noisy[:, self.num_users:]
                # Set non-associated users to -Inf, softmax only normalizes on associated subset
                comp_logits = comp_logits.masked_fill(~assoc_mask, float('-inf'))
                band_logits = band_logits.masked_fill(~assoc_mask, float('-inf'))
                comp_probs = torch.softmax(comp_logits, dim=-1)
                band_probs = torch.softmax(band_logits, dim=-1)
                return torch.cat([comp_probs, band_probs], dim=-1)
    
    def update(self, batch):
        """SAC update"""
        obs = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_observations']
        dones = batch['dones']
        
        # === Update Q networks ===
        with torch.no_grad():
            next_actions = self.actor.get_action(next_obs)
            
            # Compute entropy of next_actions (for SAC entropy regularization)
            # For softmax output, entropy = -sum(p * log(p))
            next_comp_probs = next_actions[:, :self.num_users]
            next_band_probs = next_actions[:, self.num_users:]
            
            # Robust entropy: clamp probabilities to avoid log(0)
            next_comp_probs = next_comp_probs.clamp(1e-8, 1.0)
            next_band_probs = next_band_probs.clamp(1e-8, 1.0)
            comp_entropy = -(next_comp_probs * torch.log(next_comp_probs)).sum(dim=-1)
            band_entropy = -(next_band_probs * torch.log(next_band_probs)).sum(dim=-1)
            next_entropy = comp_entropy + band_entropy
            
            q1_next = self.q1_target(next_obs, next_actions)
            q2_next = self.q2_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)
            
            # SAC target value includes entropy regularization
            alpha_t = self.log_alpha.exp().detach()
            target_q = rewards + self.gamma * (1 - dones) * (q_next + alpha_t * next_entropy)
            target_q = target_q.unsqueeze(-1)  # Match Q network output shape [batch, 1]
        
        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # === Update Actor ===
        new_actions = self.actor.get_action(obs)
        
        # Compute entropy of current policy
        comp_probs = new_actions[:, :self.num_users].clamp(1e-8, 1.0)
        band_probs = new_actions[:, self.num_users:].clamp(1e-8, 1.0)
        
        comp_entropy = -(comp_probs * torch.log(comp_probs)).sum(dim=-1)
        band_entropy = -(band_probs * torch.log(band_probs)).sum(dim=-1)
        current_entropy = comp_entropy + band_entropy
        
        q1_new = self.q1(obs, new_actions)
        q2_new = self.q2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new).squeeze(-1)
        
        # SAC Actor loss: Maximize Q value and entropy (using automatic temperature alpha)
        alpha_t = self.log_alpha.exp()
        actor_loss = -(q_new + alpha_t * current_entropy).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # === Automatic temperature adjustment ===
        # Optimization goal: Make current policy entropy close to target entropy
        alpha_loss = -(self.log_alpha * (current_entropy.detach().mean() - self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = float(self.log_alpha.exp().item())
        
        # === Soft update target networks ===
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy': current_entropy.mean().item(),
            'alpha': self.alpha,
            'alpha_loss': float(alpha_loss.item()),
            'target_entropy': float(self.target_entropy)
        }
    
    def _soft_update(self, local_model, target_model):
        """Soft update target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def get_action_cpu(self, obs_np, deterministic=False):
        """
        CPU inference: get actions without switching devices.
        
        Args:
            obs_np: NumPy observations (num_servers, obs_dim)
            deterministic: whether to use a deterministic policy
        
        Returns:
            actions_np: NumPy actions (num_servers, action_dim)
        """
        # Temporarily switch to CPU mode for inference
        actor_device_backup = next(self.actor.parameters()).device
        
        # If network is on GPU, temporarily move to CPU
        if actor_device_backup.type == 'cuda':
            self.actor.cpu()
        
        try:
            # Convert input to CPU tensor
            obs_tensor = torch.from_numpy(obs_np).float()
            
            # Inference on CPU
            with torch.no_grad():
                action_mean = self.actor.get_action(obs_tensor)

                if deterministic:
                    actions = action_mean
                else:
                    noise = torch.randn_like(action_mean) * 0.1
                    action_noisy = action_mean + noise
                    assoc_mask = obs_tensor[:, :self.num_users].bool()
                    comp_logits = action_noisy[:, :self.num_users]
                    band_logits = action_noisy[:, self.num_users:]
                    comp_logits = comp_logits.masked_fill(~assoc_mask, float('-inf'))
                    band_logits = band_logits.masked_fill(~assoc_mask, float('-inf'))
                    comp_probs = torch.softmax(comp_logits, dim=-1)
                    band_probs = torch.softmax(band_logits, dim=-1)
                    actions = torch.cat([comp_probs, band_probs], dim=-1)
            
            # Convert output to numpy
            actions_np = actions.cpu().numpy().astype(np.float32)
            
            return actions_np
            
        finally:
            # Restore original device position
            if actor_device_backup.type == 'cuda':
                self.actor.to(actor_device_backup)