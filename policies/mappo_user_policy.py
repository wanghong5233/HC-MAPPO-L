# policies/mappo_user_policy.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a layer with orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MAPPOUserPolicy(nn.Module):
    """
    Multi-Agent PPO (MAPPO) Policy for the User Agent.
    - Actor is decentralized, based on local observations.
    - Critic is centralized, based on a dedicated global state (central_obs).
    """
    def __init__(self, obs_spec, action_dims, hidden_dim=256):
        super(MAPPOUserPolicy, self).__init__()
        
        self.action_dims = action_dims
        self.obs_spec = obs_spec
        self.num_users = obs_spec['num_users']
        self.num_services = obs_spec['num_services']

        # --- Actor components (decentralized, same as UserPolicy) ---
        # This part processes local observations for action selection.
        # Fix: Hardcode smaller Embedding dimensions within the policy to balance features
        service_id_emb_dim = 16
        input_size_emb_dim = 16
        user_id_emb_dim = 16
        max_input_size = obs_spec['max_input_size']
        deployment_dim = obs_spec['num_edges'] 

        self.actor_service_id_embedding = nn.Embedding(self.num_services, service_id_emb_dim)
        self.actor_input_size_embedding = nn.Embedding(max_input_size + 1, input_size_emb_dim)
        self.actor_user_id_embedding = nn.Embedding(self.num_users, user_id_emb_dim)

        request_fused_dim = service_id_emb_dim + input_size_emb_dim + user_id_emb_dim
        
        # Ultimate fix: Set the dimensions of request features and deployment features to be equally important
        request_feature_dim = hidden_dim // 2 
        deployment_feature_dim = hidden_dim // 2

        self.actor_request_extractor = nn.Sequential(
            layer_init(nn.Linear(request_fused_dim, request_feature_dim)), nn.Tanh()
        )
        self.actor_deployment_extractor = nn.Sequential(
            # Input dimension is num_edges, output dimension is equal to request feature
            layer_init(nn.Linear(deployment_dim, deployment_feature_dim)), nn.Tanh()
        )
        
        actor_input_dim = request_feature_dim + deployment_feature_dim
        # --- Deepened Actor layers (4 layers for complex deployment matrix) ---
        self.actor_base = nn.Sequential(
            layer_init(nn.Linear(actor_input_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh()
        )
        self.association_head = layer_init(nn.Linear(hidden_dim, action_dims[0]), std=0.01)
        self.split_point_head = layer_init(nn.Linear(hidden_dim, action_dims[1]), std=0.01)
        
        # --- Critic components (centralized) ---
        # This part processes the dedicated global state (central_obs) for value estimation.
        central_obs_spec = obs_spec['central_obs_spec']
        
        # Use normalized features; no embedding
        # Extractor for the aggregated request features from all users
        # Input: num_users * 2 (service_id_normalized + input_size_normalized)
        critic_request_dim = self.num_users * 2  # Each user has 2 normalized features
        self.critic_request_extractor = nn.Sequential(
            layer_init(nn.Linear(critic_request_dim, hidden_dim // 2)), nn.Tanh()
        )

        # Extractor for the global deployment matrix
        critic_deployment_dim = central_obs_spec['deployment_matrix']
        self.critic_deployment_extractor = nn.Sequential(
            layer_init(nn.Linear(critic_deployment_dim, hidden_dim // 2)), nn.Tanh()
        )
        
        critic_input_dim = (hidden_dim // 2) * 2
        # --- Deepened Critic layers (4 layers) ---
        self.critic_base = nn.Sequential(
            layer_init(nn.Linear(critic_input_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh()
        )
        # Standard MAPPO: centralized critic outputs global scalar value/cost (shape: (batch, 1))
        self.reward_critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        self.cost_critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _fuse_local_obs(self, local_obs):
        """Processes a single agent's local observation for the actor."""
        obs_idx = 0
        service_id_raw = local_obs[:, obs_idx].long()
        service_id_emb = self.actor_service_id_embedding(service_id_raw)
        obs_idx += 1
        
        input_size_raw = local_obs[:, obs_idx].long()
        input_size_emb = self.actor_input_size_embedding(input_size_raw)
        obs_idx += 1
        
        user_id_raw = local_obs[:, obs_idx].long()
        user_id_emb = self.actor_user_id_embedding(user_id_raw)
        obs_idx += 1
        
        # Fix: Now deployment observation dimension is num_edges
        deployment_obs = local_obs[:, obs_idx:]
        
        request_fused = torch.cat([service_id_emb, input_size_emb, user_id_emb], dim=-1)
        request_features = self.actor_request_extractor(request_fused)
        deployment_features = self.actor_deployment_extractor(deployment_obs)
        
        return torch.cat([request_features, deployment_features], dim=-1)

    def _get_central_value(self, central_obs):
        """Processes the dedicated central_obs to get a global value."""
        # Deconstruct the flattened central_obs tensor
        idx = 0
        
        # All users' normalized request IDs (already normalized in [0,1])
        normalized_req_ids = central_obs[:, idx : idx + self.num_users]
        idx += self.num_users
        
        # All users' normalized input sizes (already normalized in [0,1])
        normalized_input_sizes = central_obs[:, idx : idx + self.num_users]
        idx += self.num_users

        # Global deployment matrix
        deployment_matrix = central_obs[:, idx:]
        
        # Use normalized features; no embedding
        # Shape: (batch, num_users, 2) -> (batch, num_users * 2)
        all_req_features = torch.cat([normalized_req_ids.unsqueeze(-1), normalized_input_sizes.unsqueeze(-1)], dim=-1).view(central_obs.shape[0], -1)
        
        # Extract features
        request_features = self.critic_request_extractor(all_req_features)
        deployment_features = self.critic_deployment_extractor(deployment_matrix)
        
        # Fuse and get value
        fused_features = torch.cat([request_features, deployment_features], dim=-1)
        critic_features = self.critic_base(fused_features)
        
        # Return shape: (batch, 1)
        return self.reward_critic_head(critic_features), self.cost_critic_head(critic_features)

    def get_value(self, obs, central_obs=None):
        # MAPPO must be called with central_obs
        if central_obs is None:
            raise ValueError("MAPPOUserPolicy's get_value must be called with central_obs.")
        
        reward_value, cost_value = self._get_central_value(central_obs)
        # Returns both reward and cost values for a given observation.
        return reward_value, cost_value

    def get_action_and_value(self, obs, central_obs=None, action=None, action_masks=None):
        # --- Actor (Decentralized) ---
        local_features = self._fuse_local_obs(obs)
        actor_features = self.actor_base(local_features)
        association_logits = self.association_head(actor_features)
        split_point_logits = self.split_point_head(actor_features)
        # Numerical robustness: Replace potential NaN/Inf to avoid distribution construction failure; and clip to reasonable range to prevent overflow
        association_logits = torch.nan_to_num(association_logits, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        split_point_logits = torch.nan_to_num(split_point_logits, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        
        # --- Apply action masks to logits BEFORE constructing distributions ---
        LARGE_NEG = -1e9
        if action_masks is not None:
            # Slice masks for each action head and move to correct device
            assoc_start = 0
            assoc_end = self.action_dims[0]
            split_start = assoc_end
            split_end = assoc_end + self.action_dims[1]

            assoc_mask = action_masks[:, assoc_start:assoc_end].to(association_logits.device).bool()
            split_mask = action_masks[:, split_start:split_end].to(split_point_logits.device).bool()

            association_logits_masked = association_logits.masked_fill(~assoc_mask, LARGE_NEG)
            split_point_logits_masked = split_point_logits.masked_fill(~split_mask, LARGE_NEG)
            association_logits_masked = torch.nan_to_num(association_logits_masked, nan=0.0, posinf=1e6, neginf=LARGE_NEG).clamp(LARGE_NEG, 1e6)
            split_point_logits_masked = torch.nan_to_num(split_point_logits_masked, nan=0.0, posinf=1e6, neginf=LARGE_NEG).clamp(LARGE_NEG, 1e6)
        else:
            association_logits_masked = association_logits
            split_point_logits_masked = split_point_logits

        # Construct masked distributions (always use masked logits for consistency)
        dists = [
            Categorical(logits=association_logits_masked),
            Categorical(logits=split_point_logits_masked)
        ]
        
        if action is None:
            actions_sampled = [dist.sample() for dist in dists]
            action = torch.stack(actions_sampled, dim=-1)

        log_probs = torch.stack([dist.log_prob(act) for dist, act in zip(dists, torch.unbind(action, dim=-1))], dim=-1).sum(dim=-1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=-1).sum(dim=-1)
        
        # --- Critic (Centralized) ---
        if central_obs is None:
            raise ValueError("MAPPOUserPolicy must be called with central_obs.")

        reward_value, cost_value = self._get_central_value(central_obs)
        
        # Returns both reward and cost values for a given observation.
        return action, log_probs, entropy, reward_value, cost_value

    def get_action_and_value_cpu(self, obs_np, central_obs_np=None, action_masks=None):
        device_backup = next(self.parameters()).device
        if device_backup.type == 'cuda':
            self.cpu()
        
        try:
            obs_tensor = torch.from_numpy(obs_np).float()
            central_obs_tensor = torch.from_numpy(central_obs_np).float() if central_obs_np is not None else None
            action_masks_tensor = torch.from_numpy(action_masks).bool() if action_masks is not None else None
            
            with torch.no_grad():
                actions, log_probs, _, values, cost_values = self.get_action_and_value(
                    obs_tensor, central_obs=central_obs_tensor, action_masks=action_masks_tensor)
            
            actions_np = actions.cpu().numpy().astype(np.int32)
            log_probs_np = log_probs.cpu().numpy().astype(np.float32)
            values_np = values.cpu().numpy().flatten().astype(np.float32)
            cost_values_np = cost_values.cpu().numpy().flatten().astype(np.float32)
            
            return actions_np, log_probs_np, values_np, cost_values_np
            
        finally:
            if device_backup.type == 'cuda':
                self.to(device_backup)
