# policies/allocation_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a layer with orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AllocationPolicy(nn.Module):
    """
    Efficient Attention + SAC resource allocation policy
    
    Input: Fixed-length observation [association mask, raw req_id, raw input_size] - 3*num_users dimensions
    Processing:
    - Global query: Normalize data within policy (numerical stability, no precision loss)
    - Key: Raw integer ID + embedding (rich semantic expression)
    Output: Normalized allocation weights, directly used by environment
    
    Design advantages: Precision guarantee + numerical stability + rich semantics + computational efficiency
    """
    def __init__(self, obs_spec, action_shape, hidden_dim=128, max_input_size=32, num_services=40, 
                 max_split_point=20, use_pure_attention=False):
        super().__init__()
        self.obs_spec = obs_spec
        self.num_users = action_shape[0] // 2  # comp + band
        self.max_input_size = max_input_size
        self.num_services = num_services
        self.max_split_point = max(max_split_point, 1)
        self.hidden_dim = hidden_dim
        self.use_pure_attention = use_pure_attention
        
        # Get embedding dimension config from obs_spec
        service_emb_dim = obs_spec['service_id_embedding_dim']
        input_size_emb_dim = obs_spec['input_size_embedding_dim']
        split_point_emb_dim = obs_spec['split_point_embedding_dim']
        
        # Embedding layers consistent with user agent
        self.service_id_embedding = nn.Embedding(num_services, service_emb_dim)
        self.input_size_embedding = nn.Embedding(max_input_size + 1, input_size_emb_dim)
        self.split_point_embedding = nn.Embedding(max_split_point + 1, split_point_emb_dim)

        # Global context encoder
        # Efficient design: Global query uses raw data, key uses embedding
        # Global features = Raw observation (y, req, D_in, z, W_E, D_up)
        global_feature_dim = 6 * self.num_users  # 💡 Dimension: num_users * 6
        
        # 💡 user_features dimension: service_emb + input_size_emb + split_point_emb
        user_feature_dim = service_emb_dim + input_size_emb_dim + split_point_emb_dim
        self.global_encoder = nn.Sequential(
            layer_init(nn.Linear(global_feature_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh()
        )

        # --- Attention mechanism to handle dynamic associated users ---
        self.attention_dim = hidden_dim
        
        # Attention for compute resource allocation (simplified version: Query-Key mechanism)
        self.comp_query = layer_init(nn.Linear(hidden_dim, self.attention_dim))
        self.comp_key = layer_init(nn.Linear(user_feature_dim, self.attention_dim))  # [service_emb, input_size_emb]
        
        # Attention for bandwidth resource allocation  
        self.band_query = layer_init(nn.Linear(hidden_dim, self.attention_dim))
        self.band_key = layer_init(nn.Linear(user_feature_dim, self.attention_dim))

        # --- Value Network (Critic) ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def _process_obs(self, obs):
        """
        Process observation and extract global context and associated user features
        
        Args:
            obs: [batch, 3*num_users] Contains [association_vector(num_users), masked_req_ids(num_users), masked_input_sizes(num_users)]
                环境提供的格式：关联掩码 + 已掩码的请求ID + 已掩码的输入大小
            
        Returns:
            global_context: [batch, hidden_dim] Global context vector
            associated_users_info: dict Contains features and index information of associated users
        """
        
        # Extract components - based on actual observation format from environment
        assoc_mask = obs[..., :self.num_users]  # [batch, num_users] Association mask (0/1)
        masked_req_ids = obs[..., self.num_users:2*self.num_users]  # [batch, num_users]
        masked_input_sizes = obs[..., 2*self.num_users:3*self.num_users]  # [batch, num_users]
        masked_split_points = obs[..., 3*self.num_users:4*self.num_users] # [batch, num_users]
        masked_server_workload = obs[..., 4*self.num_users:5*self.num_users] # [batch, num_users]
        masked_upload_data = obs[..., 5*self.num_users:6*self.num_users] # [batch, num_users]
        
        # 🎯 Strategy-level normalization: Environment provides raw data, avoid precision loss
        # Use raw integer IDs directly for embedding
        req_ids_obs = masked_req_ids.long()
        input_sizes_obs = masked_input_sizes.long()
        split_points_obs = masked_split_points.long()
        
        # Process discrete features using embedding
        req_ids_emb = self.service_id_embedding(req_ids_obs)  # [batch, num_users, service_emb_dim]
        input_sizes_emb = self.input_size_embedding(input_sizes_obs)  # [batch, num_users, input_size_emb_dim]
        split_points_emb = self.split_point_embedding(split_points_obs) # [batch, num_users, split_point_emb_dim]
        
        # 🎯 Strategy-level normalization: Provide normalized data for global network (numerical stability)
        # Normalize observation data for global network
        obs_normalized = obs.clone()
        
        # Normalize service ID: [0, num_services-1] → [0, 1]
        obs_normalized[:, self.num_users:2*self.num_users] = obs[:, self.num_users:2*self.num_users] / (self.num_services - 1)
        
        # Normalize input size: [1, 32] → [0, 1] 
        obs_normalized[:, 2*self.num_users:3*self.num_users] = (obs[:, 2*self.num_users:3*self.num_users] - 1) / 31

        # Normalize split point: [0, max_split_point] -> [0, 1]
        obs_normalized[:, 3*self.num_users:4*self.num_users] = obs[:, 3*self.num_users:4*self.num_users] / self.max_split_point
        
        # 💡 Proportional normalization of server workload and upload data
        # W_E
        total_workload = torch.sum(masked_server_workload, dim=1, keepdim=True) + 1e-8
        obs_normalized[:, 4*self.num_users:5*self.num_users] = masked_server_workload / total_workload
        # D_up
        total_upload = torch.sum(masked_upload_data, dim=1, keepdim=True) + 1e-8
        obs_normalized[:, 5*self.num_users:6*self.num_users] = masked_upload_data / total_upload

        # Global network uses normalized data
        global_context = self.global_encoder(obs_normalized)  # [batch, hidden_dim]
        
        # Extract associated user information
        associated_users_info = {
            'association_mask': assoc_mask,  # [batch, num_users]
            'req_ids_emb': req_ids_emb,     # [batch, num_users, service_emb_dim] Note: Non-associated users embedding corresponds to ID=0
            'input_sizes_emb': input_sizes_emb,  # [batch, num_users, input_size_emb_dim] Note: Non-associated users embedding corresponds to size=0
            'split_points_emb': split_points_emb # [batch, num_users, split_point_emb_dim]
        }
        
        return global_context, associated_users_info

    def _compute_attention_weights_pure(self, query, users_info, resource_type='comp'):
        """
        True variable length attention calculation: Only process associated users, avoid invalid calculations and gradient interference
        
        Args:
            query: [batch, attention_dim] Global query vector
            users_info: dict Associated user information
            resource_type: 'comp' or 'band'
        
        Returns:
            weights: [batch, num_users] Full user weights (0 for non-associated users)
        """
        batch_size = query.shape[0]
        association_mask = users_info['association_mask']  # [batch, num_users]
        req_ids_emb = users_info['req_ids_emb']           # [batch, num_users, service_emb_dim]
        input_sizes_emb = users_info['input_sizes_emb']   # [batch, num_users, input_size_emb_dim]
        split_points_emb = users_info['split_points_emb'] # [batch, num_users, split_point_emb_dim]
        
        # Select corresponding key network
        key_network = self.comp_key if resource_type == 'comp' else self.band_key
        
        # 💡 Build user features: [service_emb, input_size_emb, split_point_emb]
        user_features = torch.cat([req_ids_emb, input_sizes_emb, split_points_emb], dim=-1)  # [batch, num_users, feature_dim]
        
        weights = torch.zeros(batch_size, self.num_users, device=query.device)
        
        for batch_idx in range(batch_size):
            # Get associated user indices
            assoc_mask_b = association_mask[batch_idx]  # [num_users]
            associated_indices = torch.nonzero(assoc_mask_b, as_tuple=True)[0]  # associated user indices
            
            if len(associated_indices) == 0:
                continue  # no associated users
                
            # Only extract associated user features - true variable length processing
            associated_features = user_features[batch_idx, associated_indices]  # [num_associated, feature_dim]
            
            # Compute keys only for associated users
            associated_keys = key_network(associated_features)  # [num_associated, attention_dim]
            
            # Compute attention scores
            current_query = query[batch_idx:batch_idx+1]  # [1, attention_dim]
            scores = torch.mm(current_query, associated_keys.t())  # [1, num_associated]
            scores = scores.squeeze(0) / (self.attention_dim ** 0.5)  # [num_associated]
            
            # Direct softmax, no mask needed
            attention_weights = F.softmax(scores, dim=-1)  # [num_associated]
            
            # Assign weights back to full user space
            weights[batch_idx, associated_indices] = attention_weights
            
        return weights

    def _compute_attention_weights(self, query, users_info, resource_type='comp'):
        """
        Vectorized attention calculation: Compute for all users then mask (consistent with Transformer standard practice)
        """
        association_mask = users_info['association_mask']  # [batch, num_users]
        req_ids_emb = users_info['req_ids_emb']           # [batch, num_users, service_emb_dim]
        input_sizes_emb = users_info['input_sizes_emb']   # [batch, num_users, input_size_emb_dim]
        split_points_emb = users_info['split_points_emb'] # [batch, num_users, split_point_emb_dim]
        
        # Select corresponding key network
        key_network = self.comp_key if resource_type == 'comp' else self.band_key
        
        # 💡 Build user features: [service_emb, input_size_emb, split_point_emb]
        user_features = torch.cat([req_ids_emb, input_sizes_emb, split_points_emb], dim=-1)  # [batch, num_users, feature_dim]
        
        # Compute keys for all users (including 0-value embedding for non-associated users)
        all_keys = key_network(user_features)  # [batch, num_users, attention_dim]
        
        # Compute attention scores
        scores = torch.bmm(query.unsqueeze(1), all_keys.transpose(1, 2))  # [batch, 1, num_users]
        scores = scores.squeeze(1) / (self.attention_dim ** 0.5)  # [batch, num_users]
        
        # Transformer standard practice: Mask before softmax
        scores = scores.masked_fill(~association_mask.bool(), float('-inf'))
        
        # Compute weights (softmax automatically handles -inf), output all 0 when no associated users, avoid NaN
        weights = F.softmax(scores, dim=-1)  # [batch, num_users]
        
        return weights

    def forward(self, obs):
        """SAC Actor: Output deterministic allocation weights"""
        global_context, users_info = self._process_obs(obs)
        
        # Select attention calculation method based on configuration
        attention_fn = self._compute_attention_weights_pure if self.use_pure_attention else self._compute_attention_weights
        
        # === Compute resource Attention ===
        comp_query = self.comp_query(global_context)  # [batch, attention_dim]
        comp_weights = attention_fn(comp_query, users_info, 'comp')
        
        # === Bandwidth resource Attention ===
        band_query = self.band_query(global_context)  # [batch, attention_dim]
        band_weights = attention_fn(band_query, users_info, 'band')
        
        # Return normalized allocation weights
        action = torch.cat([comp_weights, band_weights], dim=-1)
        return action

    def get_action(self, obs):
        """SAC interface: Get deterministic action"""
        return self.forward(obs)
    
    def get_value(self, obs):
        """SAC interface: Get state value"""
        global_context, _ = self._process_obs(obs)
        return self.critic(global_context)
    
    def test_attention_consistency(self, obs, tolerance=1e-5):
        """
        Test consistency of two attention methods
        
        Args:
            obs: Input observation
            tolerance: Numerical tolerance
            
        Returns:
            dict: Dictionary containing test results
        """
        global_context, users_info = self._process_obs(obs)
        
        # Compute resource attention test
        comp_query = self.comp_query(global_context)
        comp_weights_pure = self._compute_attention_weights_pure(comp_query, users_info, 'comp')
        comp_weights_vec = self._compute_attention_weights(comp_query, users_info, 'comp')
        
        # Bandwidth resource attention test  
        band_query = self.band_query(global_context)
        band_weights_pure = self._compute_attention_weights_pure(band_query, users_info, 'band')
        band_weights_vec = self._compute_attention_weights(band_query, users_info, 'band')
        
        # Compute differences
        comp_diff = torch.abs(comp_weights_pure - comp_weights_vec).max().item()
        band_diff = torch.abs(band_weights_pure - band_weights_vec).max().item()
        
        # Check consistency
        comp_consistent = comp_diff < tolerance
        band_consistent = band_diff < tolerance
        
        return {
            'comp_max_diff': comp_diff,
            'band_max_diff': band_diff,
            'comp_consistent': comp_consistent,
            'band_consistent': band_consistent,
            'overall_consistent': comp_consistent and band_consistent,
            'comp_weights_pure_shape': comp_weights_pure.shape,
            'comp_weights_vec_shape': comp_weights_vec.shape,
            'band_weights_pure_shape': band_weights_pure.shape,
            'band_weights_vec_shape': band_weights_vec.shape
        }
    
    def get_value_cpu(self, obs_np):
        """
        🚀 CPU inference version: Get state value, avoid GPU switching
        
        Args:
            obs_np: numpy array of observations (num_servers, obs_dim)
        
        Returns:
            values_np: numpy array of values (num_servers,)
        """
        # Temporarily switch to CPU mode for inference
        device_backup = next(self.parameters()).device
        
        # If network is on GPU, temporarily move to CPU
        if device_backup.type == 'cuda':
            self.cpu()
        
        try:
            # Convert input to CPU tensor
            obs_tensor = torch.from_numpy(obs_np).float()
            
            # Inference on CPU
            with torch.no_grad():
                values = self.get_value(obs_tensor)
            
            # Convert output to numpy
            values_np = values.cpu().numpy().flatten().astype(np.float32)
            
            return values_np
            
        finally:
            # Restore original device position
            if device_backup.type == 'cuda':
                self.to(device_backup)