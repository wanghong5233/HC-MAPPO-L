import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DeploymentPolicy(nn.Module):
    def __init__(self, obs_spec, num_models, hidden_dim=256, embed_dim=64,
                 sampling_temperature: float = 1.0,
                 size_bias_beta: float = 0.0):
        super(DeploymentPolicy, self).__init__()
        # No explicit stop action - rely on storage constraints for natural stopping
        self.num_actions = num_models
        self.num_models = num_models
        self.hidden_dim = hidden_dim
        self.obs_spec = obs_spec
        # Temperature and size bias used only during sampling (does not change distribution definition during training)
        self.sampling_temperature = max(float(sampling_temperature), 1e-6)
        self.size_bias_beta = max(float(size_bias_beta), 0.0)
        
        # Note: Environment now directly provides normalized frequency data, no additional normalization needed

        # --- Structured Observation Encoders ---
        service_requests_dim = obs_spec['request_history']  # Frequency of all services requested by users in the previous window
        server_requests_dim = obs_spec['hit_history']       # Frequency of all services requested by visiting users for each server
        global_deployment_dim = obs_spec['global_deployment'] # Global deployment matrix vec(X)
        deployment_state_dim = obs_spec['deployment_state'] # Current deployment state (temporary variable)
        
        # --- Deepened four encoders ---
        branch_dim = hidden_dim // 4
        self.service_requests_encoder = nn.Sequential(
            layer_init(nn.Linear(service_requests_dim, branch_dim)), nn.Tanh(),
            layer_init(nn.Linear(branch_dim, branch_dim)), nn.Tanh(),
            layer_init(nn.Linear(branch_dim, branch_dim)), nn.Tanh()
        )
        self.server_requests_encoder = nn.Sequential(
            layer_init(nn.Linear(server_requests_dim, branch_dim)), nn.Tanh(),
            layer_init(nn.Linear(branch_dim, branch_dim)), nn.Tanh(),
            layer_init(nn.Linear(branch_dim, branch_dim)), nn.Tanh()
        )
        self.global_deployment_encoder = nn.Sequential(
            layer_init(nn.Linear(global_deployment_dim, branch_dim)), nn.Tanh(),
            layer_init(nn.Linear(branch_dim, branch_dim)), nn.Tanh(),
            layer_init(nn.Linear(branch_dim, branch_dim)), nn.Tanh()
        )
        self.deployment_state_encoder = nn.Sequential(
            layer_init(nn.Linear(deployment_state_dim, branch_dim)), nn.Tanh(),
            layer_init(nn.Linear(branch_dim, branch_dim)), nn.Tanh(),
            layer_init(nn.Linear(branch_dim, branch_dim)), nn.Tanh()
        )
        
        # --- Core Auto-regressive Components ---
        # Embedding includes model actions + start token
        self.action_embedding = nn.Embedding(self.num_actions + 1, embed_dim) # +1 for initial start token
        self.gru_cell = nn.GRUCell(embed_dim, hidden_dim)
        
        # --- Actor and Critic Heads ---
        self.actor_head = layer_init(nn.Linear(hidden_dim, self.num_actions), std=0.01)
        self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        
    def _get_initial_h_state(self, obs):
        # Handle possible 3D tensor (batch, seq_len, features) -> (batch, features)
        if obs.dim() == 3:
            # During training may be (batch, seq_len, features), take the last time step
            obs = obs[:, -1, :]  # Take the last time step
        
        # Three-part observation: system requests + server hits + deployment state
        req_dim = self.obs_spec['request_history']   # 45
        hit_dim = self.obs_spec['hit_history']       # 45  
        glob_dim = self.obs_spec['global_deployment'] # I*J
        dep_dim = self.obs_spec['deployment_state']  # 45
        
        req_obs = obs[:, :req_dim]
        hit_obs = obs[:, req_dim:req_dim + hit_dim]
        glob_obs = obs[:, req_dim + hit_dim : req_dim + hit_dim + glob_dim]
        dep_obs = obs[:, req_dim + hit_dim + glob_dim : req_dim + hit_dim + glob_dim + dep_dim]

        req_obs_norm = req_obs
        hit_obs_norm = hit_obs
        glob_obs_norm = glob_obs
        dep_obs_norm = dep_obs

        req_feat = self.service_requests_encoder(req_obs_norm)
        hit_feat = self.server_requests_encoder(hit_obs_norm)
        glob_feat = self.global_deployment_encoder(glob_obs_norm)
        dep_feat = self.deployment_state_encoder(dep_obs_norm)

        initial_h = torch.cat([req_feat, hit_feat, glob_feat, dep_feat], dim=-1)  # Concatenate the four features
        
        # Each of the four encoders produces hidden_dim//4 features, totaling hidden_dim, so padding is usually not needed.
        if initial_h.shape[1] < self.hidden_dim:
            padding = torch.zeros(initial_h.shape[0], self.hidden_dim - initial_h.shape[1], device=initial_h.device)
            initial_h = torch.cat([initial_h, padding], dim=-1)

        return initial_h

    def forward(self, h_state, last_action_id):
        action_emb = self.action_embedding(last_action_id)
        next_h_state = self.gru_cell(action_emb, h_state)
        logits = self.actor_head(next_h_state)
        value = self.critic_head(next_h_state)
        return logits, value, next_h_state

    def get_value(self, obs):
        initial_h = self._get_initial_h_state(obs)
        return self.critic_head(initial_h)

    def get_action_and_value(self, obs, action_sequences=None, sequence_lengths=None):
        """
        Training method supporting variable-length sequences - using Padding+Masking.
        
        Args:
            obs: observation (batch_size, obs_dim)
            action_sequences: padded action sequence (batch_size, max_seq_len), with -1 for padding
            sequence_lengths: actual sequence lengths (batch_size,)
        """
        initial_h = self._get_initial_h_state(obs)
        h_state = initial_h
        
        if action_sequences is None: # Inference mode
            return self.get_value(obs)

        batch_size = obs.size(0)
        max_seq_len = action_sequences.size(1)
        device = obs.device
        
        log_probs = []
        entropies = []
        values = []

        # Start token is num_models
        last_action = torch.full((batch_size,), self.num_actions, dtype=torch.long, device=device)

        for t in range(max_seq_len):
            # First update the state based on the last action, then use the same state to compute
            # both actor and value to avoid baseline/policy misalignment.
            action_emb = self.action_embedding(last_action)
            h_state = self.gru_cell(action_emb, h_state)
            
            logits = self.actor_head(h_state)
            values.append(self.critic_head(h_state))
            dist = Categorical(logits=logits)
            
            action_step = action_sequences[:, t]
            
            # === Key: Masking logic ===
            # Create mask: mask out if the current step exceeds sequence length or the action is padding (-1).
            valid_mask = (t < sequence_lengths)  # Whether the current step t is within the valid range.
            valid_mask = valid_mask & (action_step != -1)  # padding token is -1
            
            # Calculate log_prob (will be masked out for invalid positions later)
            step_log_probs = dist.log_prob(action_step.clamp(0, self.num_actions - 1))  # clamp to prevent errors from -1
            step_entropies = dist.entropy()
            
            # Apply mask (invalid positions are set to 0)
            step_log_probs = step_log_probs * valid_mask.float()
            step_entropies = step_entropies * valid_mask.float()
            
            log_probs.append(step_log_probs)
            entropies.append(step_entropies)
            
            # Update last_action (invalid positions keep the start token)
            last_action = torch.where(valid_mask, action_step, last_action)
            
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, max_seq_len)
        entropies = torch.stack(entropies, dim=1)
        values = torch.stack(values, dim=1)

        # Sum for each sequence (only counting valid parts)
        sequence_log_probs = log_probs.sum(dim=1)  # (batch_size,)
        sequence_entropies = entropies.sum(dim=1)
        
        # Improvement: use sequence-average value (more stable value estimate) - vectorized version
        # Create a mask to identify valid positions, avoiding loops.
        batch_size = values.size(0)
        device = values.device
        
        # Create sequence mask: (batch_size, max_seq_len)
        seq_mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        
        # Apply mask and calculate the mean (vectorized)
        # Fix: values is 3D [batch_size, seq_len, 1], need to squeeze the last dimension or expand the mask dimension.
        values_2d = values.squeeze(-1)  # [batch_size, seq_len, 1] -> [batch_size, seq_len]
        masked_values = values_2d * seq_mask.float()  # Invalid positions become 0
        valid_counts = seq_mask.sum(dim=1).float().clamp(min=1)  # Avoid division by zero
        sequence_values = masked_values.sum(dim=1) / valid_counts  # Effective average value

        return None, sequence_log_probs, sequence_entropies, sequence_values

    def get_full_action_trajectory(self, obs, server_storage, service_sizes_tensor):
        """
        Generates a full deployment plan with proper stop conditions and feasibility masking.
        ✅ Uses both GRU implicit history and explicit deployment state information
        Args:
            obs: initial observation (batch_size, obs_dim) including initial deployment state (all zeros)
            service_sizes_tensor: pre-calculated service size tensor (GB) - performance optimization
        """
        device = obs.device
        batch_size = obs.size(0)
        
        # ✅ Pre-allocate numpy arrays instead of lists (for max possible deployment of all services)
        max_possible_services = self.num_actions  # At most, deploy all services
        deployment_actions = np.full(max_possible_services, -1, dtype=np.int32)  # -1 for padding
        log_probs_array = np.zeros(max_possible_services, dtype=np.float32)
        values_array = np.zeros(max_possible_services, dtype=np.float32)
        
        # ✅ Split obs into four parts: [req_history, hit_history, global_deployment, deployment_state]
        req_dim = self.obs_spec['request_history']    # 45
        hit_dim = self.obs_spec['hit_history']        # 45  
        glob_dim = self.obs_spec['global_deployment'] # I*J
        dep_dim = self.obs_spec['deployment_state']   # 45
        
        base_obs = obs[:, :req_dim + hit_dim + glob_dim]  # Fixed part: req + hit + global_deployment
        deployment_state = obs[:, req_dim + hit_dim + glob_dim : req_dim + hit_dim + glob_dim + dep_dim].clone()  # Dynamic part: deployment_state
        
        # Start token is num_actions
        last_action = torch.tensor([self.num_actions], dtype=torch.long, device=device)
        
        # Mask for already selected services
        selected_mask = torch.ones(self.num_actions, dtype=torch.bool, device=device)
        remaining_storage = float(server_storage)  # Remaining storage capacity (MB), ensure float type
        
        # ✅ Use a counter instead of list.append
        action_count = 0
        
        # Continue deploying until storage is insufficient
        while action_count < max_possible_services:
            # ✅ Construct the full current observation at each step (base_obs + current deployment_state)
            current_obs = torch.cat([base_obs, deployment_state], dim=-1)
            
            # ✅ Re-calculate the initial h_state based on the current observation (fusing explicit state)
            if action_count == 0:
                h_state = self._get_initial_h_state(current_obs)
            else:
                # 🔧 Simplification: only use GRU's auto-regressive update
                action_emb = self.action_embedding(last_action)
                h_state = self.gru_cell(action_emb, h_state)
            
            # Generate action and value
            value = self.critic_head(h_state)
            logits = self.actor_head(h_state)
            
            # --- Feasibility Masking ---
            current_mask = selected_mask.clone()  # Mask for already selected services
            service_sizes_MB = service_sizes_tensor * 1024.0  # GB -> MB
            capacity_mask = service_sizes_MB <= remaining_storage  # Capacity check
            current_mask &= capacity_mask  # Apply capacity limit to all services

            # If no models can be deployed, stop automatically
            if not current_mask.any():
                break

            # Sampling temperature and size bias (only effective during sampling, does not affect training definition)
            logits_for_sampling = logits.clone()
            if self.size_bias_beta > 0:
                size_norm = service_sizes_tensor / (service_sizes_tensor.max().clamp_min(1e-6))
                logits_for_sampling[0, :] = logits_for_sampling[0, :] - self.size_bias_beta * size_norm
            logits_for_sampling = logits_for_sampling / self.sampling_temperature

            logits_for_sampling[0, ~current_mask] = -float('inf')
            
            dist = Categorical(logits=logits_for_sampling)
            action = dist.sample()
            action_id = action.item()
            log_prob = dist.log_prob(action)
            
            # ✅ Write directly to numpy array (avoids list operations)
            deployment_actions[action_count] = action_id
            log_probs_array[action_count] = log_prob.item()
            values_array[action_count] = value.item()
            
            # ✅ Update state: explicit deployment state + other masks
            deployment_state[0, action_id] = 1.0  # Mark as deployed
            remaining_storage -= float(service_sizes_tensor[action_id].item()) * 1024.0
            selected_mask[action_id] = False
            last_action = action
            action_count += 1
            
        # ✅ Return numpy arrays and actual length (conforms to padding+masking pattern)
        actual_length = action_count
        return deployment_actions, log_probs_array, values_array, actual_length
    
    def get_full_action_trajectory_cpu(self, obs_np, server_storage, service_sizes_array):
        """
        🚀 CPU inference version: accepts numpy input, returns numpy output, avoids GPU switching.
        Use this method during data collection for significant performance improvement.
        
        Args:
            obs_np: numpy array observation (obs_dim,) containing the full observation
            server_storage: server storage capacity (MB)
            service_sizes_array: numpy array of service sizes (GB)
        
        Returns:
            tuple: (deployment_actions, log_probs_array, values_array, actual_length) - all in numpy format
        """
        # Temporarily switch to CPU mode for inference
        device_backup = next(self.parameters()).device
        
        # If the network is on GPU, temporarily move to CPU
        if device_backup.type == 'cuda':
            self.cpu()
        
        try:
            # Convert input to CPU tensor
            obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0)  # Add batch dimension
            service_sizes_tensor = torch.from_numpy(service_sizes_array).float()
            
            # Inference on CPU
            with torch.no_grad():
                return self.get_full_action_trajectory(obs_tensor, server_storage, service_sizes_tensor)
            
        finally:
            # Restore original device location
            if device_backup.type == 'cuda':
                self.to(device_backup)


