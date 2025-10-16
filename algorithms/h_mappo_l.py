# algorithms/h_mappo_l.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from policies.user_policy import UserPolicy
from policies.mappo_user_policy import MAPPOUserPolicy
from policies.deployment_policy import DeploymentPolicy
from policies.allocation_policy import AllocationPolicy
from algorithms.sac_allocation import SACAllocation
from storage import HierarchicalStorage
from torch.utils.data import SubsetRandomSampler, BatchSampler

class HMAPPO_Lagrangian:
    """
    The Hierarchical MAPPO-Lagrangian (H-MAPPO-L) trainer.
    This class orchestrates the training of the three-tiered agent hierarchy.
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device("cpu")

        # Hyperparameters
        self.lr = float(self.config.get('agent.learning_rate'))
        self.gamma = float(self.config.get('agent.gamma'))
        self.gae_lambda = float(self.config.get('agent.gae_lambda'))
        self.clip_coef = float(self.config.get('agent.clip_coef'))
        self.ent_coef = float(self.config.get('agent.ent_coef'))
        self.vf_coef = float(self.config.get('agent.vf_coef'))
        self.cost_vf_coef = float(self.config.get('agent.cost_vf_coef'))
        self.max_grad_norm = float(self.config.get('agent.max_grad_norm'))
        self.lagrangian_lr = float(self.config.get('agent.lagrangian_lr'))
        self.num_epochs = int(self.config.get('agent.num_epochs'))
        self.num_minibatches = int(self.config.get('agent.num_minibatches'))
        self.deploy_ent_coef = float(self.config.get('agent.deploy_ent_coef', self.ent_coef))
        self.cost_limit = self.env.latency_constraint

        self._setup_agents()

        lagrangian_init = float(self.config.get('agent.lagrangian_init', 0.1))
        self.lagrangian_multiplier = torch.tensor(lagrangian_init, dtype=torch.float32, requires_grad=True, device=self.device)
        self.lagrangian_optimizer = optim.Adam([self.lagrangian_multiplier], lr=self.lagrangian_lr)

    def _setup_agents(self):
        """Initializes policies, value functions, and optimizers for all agents."""
        
        # --- User Agent ---
        user_action_dims = [self.env.num_edges, max(s.num_split_points for s in self.env.services) + 1]
        
        user_obs_spec = {
            'deployment': self.env.num_edges,
            'num_edges': self.env.num_edges,
            'service_id_embedding_dim': int(self.config.get('agent.embedding.service_id_dim', 64)),
            'input_size_embedding_dim': int(self.config.get('agent.embedding.input_size_dim', 64)),
            'user_id_embedding_dim': int(self.config.get('agent.embedding.user_id_dim', 64)),
            'num_services': self.env.num_services,
            'max_input_size': self.config.get('user.input_size_range', [1, 32])[1],
            'num_users': self.env.num_users,
            'central_obs_spec': {
                'deployment_matrix': self.env.num_services * self.env.num_edges
            }
        }

        use_mappo = self.config.get('agent.use_mappo', False)
        if use_mappo:
            self.user_policy = MAPPOUserPolicy(
                obs_spec=user_obs_spec,
                action_dims=user_action_dims,
                hidden_dim=int(self.config.get('agent.user_hidden_dim', 256))
            ).to(self.device)
        else:
            self.user_policy = UserPolicy(
                obs_spec=user_obs_spec,
                action_dims=user_action_dims,
                hidden_dim=int(self.config.get('agent.user_hidden_dim', 256))
            ).to(self.device)
        self.user_optimizer = optim.Adam(self.user_policy.parameters(), lr=self.lr, eps=1e-5)

        # --- Allocation Agent ---
        alloc_action_shape = self.env.allocation_action_space.shape
        
        # Define the structure of the allocation observation space
        alloc_obs_spec = {
            'association': self.env.num_users,
            'request_ids': self.env.num_users,
            'input_sizes': self.env.num_users,
            'service_id_embedding_dim': int(self.config.get('agent.embedding.service_id_dim', 16)),
            'input_size_embedding_dim': int(self.config.get('agent.embedding.input_size_dim', 8)),
            'split_point_embedding_dim': int(self.config.get('agent.embedding.split_point_dim', 8))
        }

        # Allocation Agent (SAC)
        max_split_point = max(s.num_split_points for s in self.env.services)
        self.allocation_policy = AllocationPolicy(
            obs_spec=alloc_obs_spec,
            action_shape=alloc_action_shape,
            hidden_dim=int(self.config.get('agent.alloc_hidden_dim', 128)),
            max_input_size=self.config.get('user.input_size_range', [1, 32])[1],
            num_services=self.env.num_services,
            max_split_point=max_split_point,
            use_pure_attention=bool(self.config.get('agent.use_pure_attention', False))
        ).to(self.device)
        
        self.sac_allocation = SACAllocation(
            actor_policy=self.allocation_policy,
            obs_dim=self.env.allocation_observation_space.shape[0],
            action_dim=self.env.num_users * 2,
            config=self.config,
            device=self.device
        )
        
        # Deployment Agent
        deploy_obs_spec = {
            'request_history': self.env.num_services,
            'hit_history': self.env.num_services,
            'global_deployment': self.env.num_services * self.env.num_edges,
            'deployment_state': self.env.num_services
        }

        self.deployment_policy = DeploymentPolicy(
            obs_spec=deploy_obs_spec,
            num_models=self.env.num_services,
            hidden_dim=int(self.config.get('agent.deploy_hidden_dim', 256)),
            embed_dim=int(self.config.get('agent.deploy_embed_dim', 64)),
            sampling_temperature=float(self.config.get('agent.sampling_temperature', 1.0)),
            size_bias_beta=float(self.config.get('agent.size_bias_beta', 0.0))
        ).to(self.device)
        self.deployment_optimizer = optim.Adam(self.deployment_policy.parameters(), lr=self.lr, eps=1e-5)


    def compute_advantages(self, rewards, dones, values, gamma, gae_lambda):
        """Computes GAE for a single agent type."""
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(rewards.size(0))):
            next_nonterminal = 1.0 - dones[t+1]
            next_values = values[t+1]
            delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
        return advantages

    def train(self, storage: HierarchicalStorage):
        """Hierarchical training: PPO (User) + SAC (Allocation) + PPO (Deployment)"""

        # User-level training (PPO)
        user_data = storage.get_user_training_data()
        b_user_obs = user_data['observations'] 
        b_user_actions = user_data['actions']
        b_user_log_probs = user_data['log_probs']
        b_user_advantages = user_data['advantages']
        b_user_returns = user_data['returns']
        b_user_action_masks = user_data.get('action_masks', None)
        b_central_obs = user_data.get('central_observations', None)
        batch_size = b_user_obs.size(0)
        minibatch_size = batch_size // self.num_minibatches
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=True)

        for _ in range(self.num_epochs):
            for indices in sampler:
                mb_user_obs = b_user_obs[indices]
                mb_user_actions = b_user_actions[indices]
                mb_user_log_probs = b_user_log_probs[indices]
                mb_user_advantages = b_user_advantages[indices]
                mb_user_returns = b_user_returns[indices]

                mb_action_masks = b_user_action_masks[indices] if b_user_action_masks is not None else None
                mb_central_obs = b_central_obs[indices] if b_central_obs is not None else None
                
                # Numerical stability: clip advantages
                mb_user_advantages = mb_user_advantages.clamp(-50.0, 50.0)
                mb_cost_advantages = user_data['cost_advantages'][indices].clamp(-50.0, 50.0)
                mb_cost_returns = user_data['cost_returns'][indices]
                _, new_log_prob, entropy, new_value, new_cost_value = self.user_policy.get_action_and_value(
                    mb_user_obs, central_obs=mb_central_obs, action=mb_user_actions, action_masks=mb_action_masks)
                
                ratio = torch.exp(new_log_prob - mb_user_log_probs)

                # Lagrangian advantage reshaping for constraint handling
                with torch.no_grad():
                    adv_std = mb_user_advantages.std()
                    cost_adv_std = mb_cost_advantages.std()
                    scale = torch.clamp(adv_std / (cost_adv_std + 1e-8), 0.1, 10.0)
                lambda_scaled = self.lagrangian_multiplier.detach() * scale
                lambda_scaled = torch.clamp(lambda_scaled, -50.0, 50.0)
                combined_advantages = (mb_user_advantages - lambda_scaled * mb_cost_advantages)
                combined_advantages = torch.clamp(combined_advantages, -100.0, 100.0)

                pg_loss1 = -combined_advantages * ratio
                pg_loss2 = -combined_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value.view(-1) - mb_user_returns) ** 2).mean()
                cost_v_loss = 0.5 * ((new_cost_value.view(-1) - mb_cost_returns) ** 2).mean()
                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss + self.cost_vf_coef * cost_v_loss

                self.user_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.user_policy.parameters(), self.max_grad_norm)
                self.user_optimizer.step()

        # Allocation Agent Training (SAC)
        if storage.ready_for_allocation_training(int(self.config.get('agent.sac_batch_size'))):
            sac_updates = int(self.config.get('agent.sac_updates_per_epoch', 3))
            for _ in range(sac_updates):
                batch = storage.sample_allocation_batch(int(self.config.get('agent.sac_batch_size')))
                if batch is not None:
                    sac_losses = self.sac_allocation.update(batch)
        
        # Deployment-level training (PPO)
        deploy_data = storage.get_deployment_training_data()
        if not deploy_data:
            return
        b_deploy_obs = deploy_data['observations']
        b_deploy_action_sequences = deploy_data['action_sequences']
        b_deploy_seq_lengths = deploy_data['sequence_lengths']
        b_deploy_step_log_probs = deploy_data['step_log_probs']
        b_deploy_step_returns = deploy_data['step_returns']
        b_deploy_step_advantages = deploy_data['step_advantages']
        
        batch_size = b_deploy_obs.size(0)
        device = b_deploy_obs.device
        
        # Create sequence masks for training
        max_seq_len = b_deploy_step_log_probs.size(1)
        seq_mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < b_deploy_seq_lengths.unsqueeze(1)
        
        # Normalize advantages for valid positions only
        valid_advantages = b_deploy_step_advantages[seq_mask]
        normalized_advantages = (valid_advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        b_deploy_step_advantages = b_deploy_step_advantages.clone()
        b_deploy_step_advantages[seq_mask] = normalized_advantages
        
        minibatch_size = max(1, batch_size // self.num_minibatches)
            
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=False)
        
        for _ in range(self.num_epochs):
            for indices in sampler:
                mb_obs = b_deploy_obs[indices]
                mb_action_sequences = b_deploy_action_sequences[indices]
                mb_seq_lengths = b_deploy_seq_lengths[indices]
                mb_step_log_probs = b_deploy_step_log_probs[indices]
                mb_step_advantages = b_deploy_step_advantages[indices]
                mb_step_returns = b_deploy_step_returns[indices]
                
                mb_seq_mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < mb_seq_lengths.unsqueeze(1)
                
                _, new_seq_log_prob, entropy, new_seq_value = self.deployment_policy.get_action_and_value(
                    mb_obs, mb_action_sequences, mb_seq_lengths
                )
                
                old_seq_log_probs = (mb_step_log_probs * mb_seq_mask.float()).sum(dim=1)
                seq_advantages = (mb_step_advantages * mb_seq_mask.float()).sum(dim=1) / mb_seq_lengths.float()
                seq_returns = (mb_step_returns * mb_seq_mask.float()).sum(dim=1) / mb_seq_lengths.float()
                ratio = torch.exp(new_seq_log_prob - old_seq_log_probs)
                pg_loss1 = -seq_advantages * ratio
                pg_loss2 = -seq_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                vf_loss = 0.5 * ((new_seq_value.view(-1) - seq_returns) ** 2).mean()
                loss = pg_loss - self.deploy_ent_coef * entropy.mean() + self.vf_coef * vf_loss

                self.deployment_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.deployment_policy.parameters(), self.max_grad_norm)
                self.deployment_optimizer.step()
        
        # Lagrangian Multiplier Update
        with torch.no_grad():
            avg_cost_per_step = user_data['costs'].mean()
            target_limit = torch.tensor(self.cost_limit, dtype=torch.float32, device=self.device)

        lagrangian_loss = -(self.lagrangian_multiplier * (avg_cost_per_step - target_limit).detach())

        self.lagrangian_optimizer.zero_grad()
        lagrangian_loss.backward()
        self.lagrangian_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0, 100.0)

    def get_actions_and_values(self, obs_dict, env):
        """Gets actions and values for all agents from their respective policies."""
        user_obs = torch.Tensor(obs_dict['user']).to(self.device)
        alloc_obs = torch.Tensor(obs_dict['allocation']).to(self.device)

        actions = {}
        log_probs = {}
        values = {}
        
        # User actions
        user_action_masks = env.get_user_action_masks()
        user_action_masks_tensor = torch.from_numpy(user_action_masks).to(self.device)
        user_action, user_log_prob, _, user_value, user_cost_value = self.user_policy.get_action_and_value(
            user_obs, action_masks=user_action_masks_tensor)
        actions['user'] = user_action
        log_probs['user'] = user_log_prob
        values['user'] = user_value
        values['user_cost'] = user_cost_value

        # Allocation actions
        alloc_action = self.sac_allocation.get_action(alloc_obs)
        alloc_value = self.allocation_policy.get_value(alloc_obs)
        actions['allocation'] = alloc_action
        values['allocation'] = alloc_value
        
        # Deployment actions (only when needed: t mod ΔT = 0)
        if env.needs_deployment_decision():
            deploy_actions = []
            deploy_log_probs = []
            deploy_values = []
            
            req_freq, hit_freq_matrix = env.get_deployment_observation_data()
            device = next(self.deployment_policy.parameters()).device
            service_sizes_tensor = torch.tensor([service.model_size_GB for service in env.services], 
                                               dtype=torch.float32, device=device)
            
            for j in range(self.env.num_edges):
                hit_freq_vector = hit_freq_matrix[j, :]
                obs_vector = np.concatenate([req_freq, hit_freq_vector])
                obs = torch.from_numpy(obs_vector).float().unsqueeze(0).to(device)
                
                server = env.servers[j]
                full_storage_capacity_MB = server.storage_capacity_MB
                actions_array, log_probs_array, values_array, actual_length = self.deployment_policy.get_full_action_trajectory(
                    obs, full_storage_capacity_MB, service_sizes_tensor)
                
                if actual_length > 0:
                    action_ids = actions_array[:actual_length].tolist()
                    log_probs_list = log_probs_array[:actual_length].tolist()
                    values_list = values_array[:actual_length].tolist()
                else:
                    action_ids = []
                    log_probs_list = []
                    values_list = []
                
                deploy_actions.append(action_ids)
                deploy_log_probs.append(log_probs_list)
                deploy_values.append(values_list)
            
            actions['deployment'] = np.array(deploy_actions, dtype=object) 
            log_probs['deployment'] = np.array(deploy_log_probs, dtype=object)
            values['deployment'] = np.array(deploy_values, dtype=object)
        else:
            # No deployment actions needed
            actions['deployment'] = None
            log_probs['deployment'] = None
            values['deployment'] = None

        return actions, log_probs, values

    def _create_critic(self, obs_dim, hidden_dim=256):
        """Helper to create a standard critic network."""
        return nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

 