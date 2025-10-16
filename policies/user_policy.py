# policies/user_policy.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a layer with orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class UserPolicy(nn.Module):
    """
    Policy network for the User Agent with learnable embeddings.
    Processes embedded_service_id + embedded_input_size + embedded_user_id + deployment_matrix.
    """
    def __init__(self, obs_spec, action_dims, hidden_dim=256):
        """
        Args:
            obs_spec (dict): Contains dimensions and embedding configurations:
                           'service_id', 'input_size', 'user_id', 'deployment',
                           'service_id_embedding_dim', 'input_size_embedding_dim', 'user_id_embedding_dim', etc.
            action_dims (List[int]): [num_edges, max_split + 1]
            hidden_dim (int): Size of hidden layers.
        """
        super(UserPolicy, self).__init__()
        
        self.action_dims = action_dims
        self.obs_spec = obs_spec

        # --- Extract dimensions from obs_spec for clarity ---
        num_services = obs_spec['num_services']
        service_id_emb_dim = obs_spec['service_id_embedding_dim']
        max_input_size = obs_spec['max_input_size']
        input_size_emb_dim = obs_spec['input_size_embedding_dim']
        num_users = obs_spec['num_users']
        user_id_emb_dim = obs_spec['user_id_embedding_dim']
        deployment_dim = obs_spec['deployment']

        # Cache dims for sanitization/clamp
        self._num_services = num_services
        self._max_input_size = max_input_size
        self._num_users = num_users

        # --- Learnable Embedding Layers ---
        self.service_id_embedding = nn.Embedding(num_services, service_id_emb_dim)
        self.input_size_embedding = nn.Embedding(max_input_size + 1, input_size_emb_dim)
        self.user_id_embedding = nn.Embedding(num_users, user_id_emb_dim)

        # --- Feature Extractors for different parts ---
        # Request features: service_id_embedding + input_size_embedding + user_id_embedding
        request_fused_dim = service_id_emb_dim + input_size_emb_dim + user_id_emb_dim
        
        self.request_extractor = nn.Sequential(
            layer_init(nn.Linear(request_fused_dim, hidden_dim // 2)),
            nn.Tanh()
        )
        
        self.deployment_extractor = nn.Sequential(
            layer_init(nn.Linear(deployment_dim, hidden_dim // 2)),
            nn.Tanh()
        )

        fused_dim = (hidden_dim // 2) * 2

        # --- Deepened Actor layers (4 layers to handle complex deployment matrix) ---
        self.actor_base = nn.Sequential(
            layer_init(nn.Linear(fused_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh()
        )
        
        self.association_head = layer_init(nn.Linear(hidden_dim, action_dims[0]), std=0.01)
        self.split_point_head = layer_init(nn.Linear(hidden_dim, action_dims[1]), std=0.01)
        
        # --- Deepened Critic layers (4 layers) ---
        self.critic_base = nn.Sequential(
             layer_init(nn.Linear(fused_dim, hidden_dim)),
             nn.Tanh(),
             layer_init(nn.Linear(hidden_dim, hidden_dim)),
             nn.Tanh(),
             layer_init(nn.Linear(hidden_dim, hidden_dim)),
             nn.Tanh(),
             layer_init(nn.Linear(hidden_dim, hidden_dim)),
             nn.Tanh()
        )
        self.reward_critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        self.cost_critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _sanitize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Sanitize observation tensor to avoid NaNs/Infs and clamp indices domain.
        Expects obs layout: [service_id, input_size, user_id, deployment...]
        """
        # Remove NaN/Inf in raw obs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp index-like fields to valid ranges before casting to long
        if obs.dim() == 2 and obs.size(1) >= 3:
            # service_id in [0, num_services-1]
            obs[:, 0] = obs[:, 0].clamp(min=0, max=max(0, self._num_services - 1))
            # input_size in [0, max_input_size]
            obs[:, 1] = obs[:, 1].clamp(min=0, max=max(0, self._max_input_size))
            # user_id in [0, num_users-1]
            obs[:, 2] = obs[:, 2].clamp(min=0, max=max(0, self._num_users - 1))

        return obs

    def _fuse_obs(self, obs):
        """Splits observation, applies embeddings, and fuses features."""
        # Split observations: [service_id, input_size, user_id, deployment_matrix]
        obs = self._sanitize_obs(obs)
        obs_idx = 0
        
        # Extract and embed service ID
        service_id_raw = obs[:, obs_idx].long()  # Convert to long for embedding
        service_id_emb = self.service_id_embedding(service_id_raw)
        obs_idx += 1
        
        # Extract and embed input size
        input_size_raw = obs[:, obs_idx].long()  # Convert to long for embedding
        input_size_emb = self.input_size_embedding(input_size_raw)
        obs_idx += 1
        
        # Extract and embed user ID
        user_id_raw = obs[:, obs_idx].long()  # Convert to long for embedding  
        user_id_emb = self.user_id_embedding(user_id_raw)
        obs_idx += 1
        
        # Extract deployment matrix
        deployment_obs = obs[:, obs_idx:]
        
        # Fuse request features: service_id_embedding + input_size_embedding + user_id_embedding
        request_fused = torch.cat([service_id_emb, input_size_emb, user_id_emb], dim=-1)
        
        # Process each part through extractors
        request_features = self.request_extractor(request_fused)
        deployment_features = self.deployment_extractor(deployment_obs)
        
        # Final fusion
        fused_features = torch.cat([request_features, deployment_features], dim=-1)
        return fused_features

    def forward(self, obs):
        """
        Forward pass through the policy network.
        
        Args:
            obs (torch.Tensor): The observation tensor of shape (batch_size, obs_dim).
            
        Returns:
            A tuple containing:
            - A list of action distributions for each action dimension.
            - The estimated state value from the reward critic.
            - The estimated state-cost value from the cost critic.
        """
        fused_features = self._fuse_obs(obs)
        
        # Actor forward pass
        actor_features = self.actor_base(fused_features)
        association_logits = self.association_head(actor_features)
        split_point_logits = self.split_point_head(actor_features)

        # Clean logits to avoid NaN/Inf propagation
        association_logits = torch.nan_to_num(association_logits, nan=0.0, posinf=0.0, neginf=0.0)
        split_point_logits = torch.nan_to_num(split_point_logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create categorical distributions
        association_dist = Categorical(logits=association_logits)
        split_point_dist = Categorical(logits=split_point_logits)
        
        # Critic forward pass
        critic_features = self.critic_base(fused_features)
        reward_value = self.reward_critic_head(critic_features)
        cost_value = self.cost_critic_head(critic_features)
        
        return [association_dist, split_point_dist], reward_value, cost_value

    def get_value(self, obs, central_obs=None):
        """
        Returns both reward and cost values for a given observation.
        IPPO ignores the central_obs.
        """
        fused_features = self._fuse_obs(obs)
        critic_features = self.critic_base(fused_features)
        return self.reward_critic_head(critic_features), self.cost_critic_head(critic_features)

    def get_action_and_value(self, obs, central_obs=None, action=None, action_masks=None):
        """
        Main interface function. IPPO ignores the central_obs.
        """
        dists, reward_value, cost_value = self.forward(obs)

        # Apply action masks by rebuilding masked distributions to avoid in-place -inf
        if action_masks is not None:
            masked_dists = []
            for i, dist in enumerate(dists):
                start_idx = sum(self.action_dims[:i])
                end_idx = start_idx + self.action_dims[i]
                mask = action_masks[:, start_idx:end_idx]
                # ensure boolean mask
                if mask.dtype is not torch.bool:
                    mask = mask.to(dtype=torch.bool)
                logits = torch.nan_to_num(dist.logits, nan=0.0, posinf=0.0, neginf=0.0)
                masked_logits = logits.masked_fill(~mask, -1e9)
                masked_dists.append(Categorical(logits=masked_logits))
            dists = masked_dists

        if action is None:
            # Sample new actions with masking support
            actions_sampled = []
            
            for _, dist in enumerate(dists):
                actions_sampled.append(dist.sample())
            
            action = torch.stack(actions_sampled, dim=-1)

        log_probs = torch.stack([dist.log_prob(act) for dist, act in zip(dists, torch.unbind(action, dim=-1))], dim=-1).sum(dim=-1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=-1).sum(dim=-1)
        
        return action, log_probs, entropy, reward_value, cost_value

    def get_action(self, obs, action_masks=None):
        """
        Samples an action from the policy distributions.
        
        Args:
            obs (torch.Tensor): The observation tensor of shape (batch_size, obs_dim).
            action_masks (torch.Tensor, optional): A boolean tensor for masking invalid actions.
                                                    Shape: (batch_size, sum(action_dims)).
        
        Returns:
            A tuple containing:
            - actions (torch.Tensor): The sampled actions, shape (batch_size, num_action_dims).
            - log_probs (torch.Tensor): The log probability of the sampled actions, shape (batch_size,).
        """
        dists, _, _ = self.forward(obs)
        
        actions = []
        log_probs = []
        
        # Rebuild masked distributions if masks provided
        if action_masks is not None:
            masked_dists = []
            for i, dist in enumerate(dists):
                start_idx = sum(self.action_dims[:i])
                end_idx = start_idx + self.action_dims[i]
                mask = action_masks[:, start_idx:end_idx]
                if mask.dtype is not torch.bool:
                    mask = mask.to(dtype=torch.bool)
                logits = torch.nan_to_num(dist.logits, nan=0.0, posinf=0.0, neginf=0.0)
                masked_logits = logits.masked_fill(~mask, -1e9)
                masked_dists.append(Categorical(logits=masked_logits))
            dists = masked_dists

        for dist in dists:
            action = dist.sample()
            actions.append(action.unsqueeze(1))
            log_probs.append(dist.log_prob(action))
            
        return torch.cat(actions, dim=1), torch.sum(torch.stack(log_probs), dim=0)

    def evaluate_actions(self, obs, actions, action_masks=None):
        """
        Evaluates the given actions under the current policy.

        Args:
            obs (torch.Tensor): The observation tensor of shape (batch_size, obs_dim).
            actions (torch.Tensor): The actions to evaluate, shape (batch_size, num_action_dims).
            action_masks (torch.Tensor, optional): Action masks.

        Returns:
            A tuple containing:
            - log_probs (torch.Tensor): The log probability of the given actions.
            - entropy (torch.Tensor): The entropy of the action distributions.
            - reward_value (torch.Tensor): The estimated state value from the reward critic.
            - cost_value (torch.Tensor): The estimated state-cost value from the cost critic.
        """
        dists, reward_value, cost_value = self.forward(obs)
        
        log_probs = []
        entropies = []
        
        # Rebuild masked distributions if masks provided
        if action_masks is not None:
            masked_dists = []
            for i, dist in enumerate(dists):
                start_idx = sum(self.action_dims[:i])
                end_idx = start_idx + self.action_dims[i]
                mask = action_masks[:, start_idx:end_idx]
                if mask.dtype is not torch.bool:
                    mask = mask.to(dtype=torch.bool)
                logits = torch.nan_to_num(dist.logits, nan=0.0, posinf=0.0, neginf=0.0)
                masked_logits = logits.masked_fill(~mask, -1e9)
                masked_dists.append(Categorical(logits=masked_logits))
            dists = masked_dists

        for i, dist in enumerate(dists):
            action_log_prob = dist.log_prob(actions[:, i])
            log_probs.append(action_log_prob)
            entropies.append(dist.entropy())
            
        return torch.sum(torch.stack(log_probs), dim=0), torch.sum(torch.stack(entropies), dim=0), reward_value, cost_value
    
    def get_action_and_value_cpu(self, obs_np, central_obs_np=None, action_masks=None):
        """
        🚀 CPU inference version: accepts numpy input, returns numpy output, avoids GPU switching.
        Use this method during data collection for significant performance improvement.
        
        Args:
            obs_np: numpy array of observations (num_users, obs_dim)
            action_masks: numpy array of action masks (num_users, action_mask_dim)
        
        Returns:
            tuple: (actions_np, log_probs_np, values_np, cost_values_np) - all in numpy format.
        """
        # Temporarily switch to CPU mode for inference
        device_backup = next(self.parameters()).device
        
        # If the network is on GPU, temporarily move to CPU
        if device_backup.type == 'cuda':
            self.cpu()
        
        try:
            # Convert input to CPU tensor
            obs_tensor = torch.from_numpy(obs_np).float()
            action_masks_tensor = torch.from_numpy(action_masks).bool() if action_masks is not None else None
            
            # Inference on CPU
            with torch.no_grad():
                actions, log_probs, _, values, cost_values = self.get_action_and_value(
                    obs_tensor, central_obs=None, action_masks=action_masks_tensor)
            
            # Convert output to numpy
            actions_np = actions.cpu().numpy().astype(np.int32)
            log_probs_np = log_probs.cpu().numpy().astype(np.float32)
            values_np = values.cpu().numpy().flatten().astype(np.float32)
            cost_values_np = cost_values.cpu().numpy().flatten().astype(np.float32)
            
            return actions_np, log_probs_np, values_np, cost_values_np
            
        finally:
            # Restore original device location
            if device_backup.type == 'cuda':
                self.to(device_backup) 