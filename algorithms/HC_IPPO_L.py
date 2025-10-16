import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from torch.utils.data import SubsetRandomSampler, BatchSampler

from policies.user_policy import UserPolicy
from policies.deployment_policy import DeploymentPolicy
from policies.allocation_policy import AllocationPolicy
from algorithms.sac_allocation import SACAllocation
from storage import HierarchicalStorage


class IPPOAgent:
    """
    HC-IPPO-L: Hierarchical Constrained IPPO with Lagrangian constraints
    - User layer: IPPO with local observations and Lagrangian constraints
    - Allocation layer: SAC
    - Deployment layer: IPPO with shared parameters
    """

    def __init__(self, env, config, max_updates=None):
        self.env = env
        self.config = config
        self.device = torch.device('cpu')

        # Hyperparameters
        self.lr = float(self.config.get('agent.learning_rate'))
        self.lr_scheduler_type = str(self.config.get('agent.lr_scheduler', 'none'))
        self.T_max = max_updates if max_updates is not None else 1000
        try:
            _lr_T_max = self.config.get('agent.lr_T_max')
            self.lr_T_max = int(_lr_T_max) if _lr_T_max is not None else self.T_max
        except Exception:
            self.lr_T_max = self.T_max

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
        # Whether to decouple policy and value optimization to avoid mutual interference
        self.decouple_critics = bool(self.config.get('agent.decouple_critics', False))
        # Whether to normalize cost advantages to enhance stability
        self.normalize_cost_advantages = bool(self.config.get('agent.normalize_cost_advantages', False))
        
        # Learning rates: separate for each layer
        self.learning_rate = self.config.get('agent.learning_rate', 3e-4) # Global learning rate, mainly used by SAC (allocation layer)
        self.user_learning_rate = self.config.get('agent.user_learning_rate', self.learning_rate) # User layer specific learning rate
        self.deploy_learning_rate = self.config.get('agent.deploy_learning_rate', 3e-4)

        self.max_grad_norm = self.config.get('agent.max_grad_norm', 0.5)

        # Learning rate scheduler parameters (only affect user layer)
        self.lr_scheduler_type = str(self.config.get('agent.lr_scheduler', 'none'))
        self.lr_end_factor = float(self.config.get('agent.lr_end_factor', 0.1))
        self.T_max = max_updates if max_updates is not None else 1000
        # Synchronize once (prevent overwriting)
        try:
            _lr_T_max = self.config.get('agent.lr_T_max')
            self.lr_T_max = int(_lr_T_max) if _lr_T_max is not None else self.T_max
        except Exception:
            self.lr_T_max = self.T_max

        # Lagrangian-related hyperparameters
        lagrangian_init = self.config.get('agent.lagrangian_init', 0.1)
        self.lagrangian_multiplier = torch.tensor(lagrangian_init, dtype=torch.float32, requires_grad=True, device=self.device)
        # Ensure initial value does not exceed clamp upper limit
        self.lagrangian_multiplier.data.clamp_(0, 50.0)
        self.lagrangian_optimizer = optim.Adam([self.lagrangian_multiplier], lr=self.lagrangian_lr)

        # Constraint threshold (average latency upper limit)
        self.cost_limit = self.env.latency_constraint

        # Preset scheduler placeholders
        self.scheduler = None
        self.policy_scheduler = None
        self.value_scheduler = None

        self._setup_agents()

    def _setup_agents(self):
        # --- User: IPPO (UserPolicy) ---
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
        }
        self.user_policy = UserPolicy(
            obs_spec=user_obs_spec,
            action_dims=user_action_dims,
            hidden_dim=int(self.config.get('agent.user_hidden_dim', 256))
        ).to(self.device)
        # Build optimizers as needed
        if self.decouple_critics:
            # Policy parameters: actor_base + two action heads
            policy_params = list(self.user_policy.actor_base.parameters()) + \
                            list(self.user_policy.association_head.parameters()) + \
                            list(self.user_policy.split_point_head.parameters())
            # Value parameters (shared): critic_base + two heads, optimized together to avoid multi-optimizer conflict on same parameters
            value_params = list(self.user_policy.critic_base.parameters()) + \
                           list(self.user_policy.reward_critic_head.parameters()) + \
                           list(self.user_policy.cost_critic_head.parameters())
            self.policy_optimizer = optim.Adam(policy_params, lr=self.user_learning_rate)
            self.value_optimizer = optim.Adam(value_params, lr=self.user_learning_rate)

            # Schedulers
            if self.lr_scheduler_type == 'cosine':
                # Half-cycle cosine decay (half-cycle length = lr_T_max / 2)
                T_max_cosine = self.lr_T_max / 2
                eta_min_ratio = self.lr_end_factor
                lr_lambda = lambda update: (
                    eta_min_ratio + 0.5 * (1 - eta_min_ratio) * (1 + math.cos(math.pi * update / T_max_cosine))
                    if update < T_max_cosine
                    else eta_min_ratio
                )
                self.policy_scheduler = optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda)
                self.value_scheduler = optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda)
            elif self.lr_scheduler_type == 'linear':
                # Full-cycle linear decay (reach end_factor at lr_T_max)
                lr_lambda = lambda update: 1.0 - (1.0 - self.lr_end_factor) * (min(update, self.lr_T_max) / self.lr_T_max)
                self.policy_scheduler = optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda)
                self.value_scheduler = optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda)

        else: # Not decoupled
            self.user_optimizer = optim.Adam(self.user_policy.parameters(), lr=self.user_learning_rate)
            self.scheduler = None
            if self.lr_scheduler_type == 'cosine':
                # Half-cycle cosine decay (half-cycle length = lr_T_max / 2)
                T_max_cosine = self.lr_T_max / 2
                eta_min_ratio = self.lr_end_factor
                lr_lambda = lambda update: (
                    eta_min_ratio + 0.5 * (1 - eta_min_ratio) * (1 + math.cos(math.pi * update / T_max_cosine))
                    if update < T_max_cosine
                    else eta_min_ratio
                )
                self.scheduler = optim.lr_scheduler.LambdaLR(self.user_optimizer, lr_lambda)
            elif self.lr_scheduler_type == 'linear':
                # Full-cycle linear decay (reach end_factor at lr_T_max)
                lr_lambda = lambda update: 1.0 - (1.0 - self.lr_end_factor) * (min(update, self.lr_T_max) / self.lr_T_max)
                self.scheduler = optim.lr_scheduler.LambdaLR(self.user_optimizer, lr_lambda)

        # --- Allocation (SAC) ---
        alloc_action_shape = self.env.allocation_action_space.shape
        alloc_obs_spec = {
            'association': self.env.num_users,
            'request_ids': self.env.num_users,
            'input_sizes': self.env.num_users,
            'service_id_embedding_dim': int(self.config.get('agent.embedding.service_id_dim', 16)),
            'input_size_embedding_dim': int(self.config.get('agent.embedding.input_size_dim', 8)),
            'split_point_embedding_dim': int(self.config.get('agent.embedding.split_point_dim', 8))
        }
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

        # --- Deployment: IPPO (shared parameters, independent trajectories) ---
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
        self.deployment_optimizer = optim.Adam(self.deployment_policy.parameters(), lr=self.deploy_learning_rate)

    def train(self, storage: HierarchicalStorage):
        # --- User layer: PPO (IPPO) ---
        user_data = storage.get_user_training_data()
        if user_data is None:
            return
        b_obs = user_data['observations']
        b_actions = user_data['actions']
        b_logp = user_data['log_probs']
        b_adv = user_data['advantages']
        b_ret = user_data['returns']
        # Constraint (cost) related
        b_cost_adv = user_data['cost_advantages']
        b_cost_ret = user_data['cost_returns']
        b_masks = user_data.get('action_masks', None)

        batch_size = b_obs.size(0)
        minibatch_size = max(1, batch_size // self.num_minibatches)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=True)

        for _ in range(self.num_epochs):
            for indices in sampler:
                mb_obs = b_obs[indices]
                mb_actions = b_actions[indices]
                mb_old_logp = b_logp[indices]
                mb_adv = b_adv[indices].clamp(-50.0, 50.0)
                mb_ret = b_ret[indices]
                mb_cost_adv = b_cost_adv[indices].clamp(-50.0, 50.0)
                mb_cost_ret = b_cost_ret[indices]
                mb_masks = b_masks[indices] if b_masks is not None else None

                # Consistent with h_mappo_l: directly use policy interface to evaluate old actions, passing boolean mask
                # Additional robustness: ensure at least one valid action per dimension to avoid all-False leading to -inf logits → NaN
                mb_masks_bool = None
                if mb_masks is not None:
                    mb_masks_bool = mb_masks.to(torch.bool)
                    start_idx = 0
                    for dim in self.user_policy.action_dims:
                        seg = mb_masks_bool[:, start_idx:start_idx+dim]
                        invalid_rows = ~seg.any(dim=1)
                        if invalid_rows.any():
                            seg[invalid_rows, 0] = True
                        mb_masks_bool[:, start_idx:start_idx+dim] = seg
                        start_idx += dim
                _, new_logp, entropy, new_v, new_cost_v = self.user_policy.get_action_and_value(
                    mb_obs, action=mb_actions, action_masks=mb_masks_bool
                )
                # Consistent with h_mappo_l: directly calculate ratio (with log ratio clamping to prevent overflow)
                log_ratio = (new_logp - mb_old_logp).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                # Lagrangian: advantage reshaping A_total = A_r - lambda_scaled * A_c
                # More conservative cost advantage handling: light normalization only when necessary
                with torch.no_grad():
                    adv_std = mb_adv.std()
                    cost_adv_std = mb_cost_adv.std()
                    if self.normalize_cost_advantages:
                        # Light normalization: retain some original scale to avoid losing scale information
                        norm_cost_adv = (mb_cost_adv - mb_cost_adv.mean()) / (mb_cost_adv.std() + 1e-8)
                        cost_term = norm_cost_adv
                        # Still use std ratio, but do extra scaling on the standardized advantage
                        scale = torch.clamp(adv_std / (cost_adv_std + 1e-8), 0.1, 10.0) * 0.5
                    else:
                        cost_term = mb_cost_adv
                        # Add numerical stability: limit max scale to avoid gradient explosion
                        scale = torch.clamp(adv_std / (cost_adv_std + 1e-8), 0.1, 10.0)
                lambda_scaled = self.lagrangian_multiplier.detach() * scale
                # Expand lambda_scaled range to align with H-MAPPO-L
                lambda_scaled = torch.clamp(lambda_scaled, -50.0, 50.0)
                combined_adv = (mb_adv - lambda_scaled * cost_term)

                # Expand combined_adv range to align with H-MAPPO-L
                combined_adv = torch.clamp(combined_adv, -100.0, 100.0)

                pg1 = -combined_adv * ratio
                pg2 = -combined_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((new_v.view(-1) - mb_ret) ** 2).mean()
                cost_v_loss = 0.5 * ((new_cost_v.view(-1) - mb_cost_ret) ** 2).mean()
                # Cost value loss weight aligned with h_mappo_l, fixed as cost_vf_coef (do not scale with λ)
                if not self.decouple_critics:
                    loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss + self.cost_vf_coef * cost_v_loss

                    self.user_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.user_policy.parameters(), self.max_grad_norm)
                    self.user_optimizer.step()
                else:
                    # 1) Policy update: only update policy parameters (independent step)
                    policy_loss = pg_loss - self.ent_coef * entropy.mean()
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.user_policy.actor_base.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.user_policy.association_head.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.user_policy.split_point_head.parameters(), self.max_grad_norm)
                    self.policy_optimizer.step()

                    # 2) Value update: single forward pass, optimize both reward/cost value heads + shared critic_base simultaneously
                    new_r_v, new_c_v = self.user_policy.get_value(mb_obs)
                    v_loss_now = 0.5 * ((new_r_v.view(-1) - mb_ret) ** 2).mean()
                    cost_v_loss_now = 0.5 * ((new_c_v.view(-1) - mb_cost_ret) ** 2).mean()
                    value_loss = self.vf_coef * v_loss_now + self.cost_vf_coef * cost_v_loss_now
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.user_policy.critic_base.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.user_policy.reward_critic_head.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.user_policy.cost_critic_head.parameters(), self.max_grad_norm)
                    self.value_optimizer.step()

        # --- Allocation layer: SAC (same as original) ---
        if storage.ready_for_allocation_training(int(self.config.get('agent.sac_batch_size'))):
            sac_updates = int(self.config.get('agent.sac_updates_per_epoch', 3))
            for _ in range(sac_updates):
                batch = storage.sample_allocation_batch(int(self.config.get('agent.sac_batch_size')))
                if batch is not None:
                    _ = self.sac_allocation.update(batch)

        # --- Deployment layer: IPPO (sequence training like MAPPO, without centralized critic) ---
        deploy_data = storage.get_deployment_training_data()
        if not deploy_data:
            # Advance schedulers after each full training
            try:
                if self.decouple_critics:
                    if self.policy_scheduler is not None:
                        self.policy_scheduler.step()
                    if self.value_scheduler is not None:
                        self.value_scheduler.step()
                else:
                    if self.scheduler is not None:
                        self.scheduler.step()
            except Exception:
                pass
            return

        b_obs = deploy_data['observations']
        b_action_seq = deploy_data['action_sequences']
        b_seq_len = deploy_data['sequence_lengths']
        b_step_logp = deploy_data['step_log_probs']
        b_step_ret = deploy_data['step_returns']
        b_step_adv = deploy_data['step_advantages']

        batch_size = b_obs.size(0)
        device = b_obs.device
        max_seq = b_step_logp.size(1)
        seq_mask = torch.arange(max_seq, device=device).unsqueeze(0) < b_seq_len.unsqueeze(1)

        valid_adv = b_step_adv[seq_mask]
        norm_adv = (valid_adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)
        b_step_adv = b_step_adv.clone()
        b_step_adv[seq_mask] = norm_adv

        minibatch_size = max(1, batch_size // self.num_minibatches)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=False)

        for _ in range(self.num_epochs):
            for indices in sampler:
                mb_obs = b_obs[indices]
                mb_action_seq = b_action_seq[indices]
                mb_seq_len = b_seq_len[indices]
                mb_step_logp = b_step_logp[indices]
                mb_step_adv = b_step_adv[indices]
                mb_step_ret = b_step_ret[indices]

                mb_seq_mask = torch.arange(max_seq, device=device).unsqueeze(0) < mb_seq_len.unsqueeze(1)
                _, new_seq_logp, entropy, new_seq_value = self.deployment_policy.get_action_and_value(
                    mb_obs, mb_action_seq, mb_seq_len
                )
                old_seq_logp = (mb_step_logp * mb_seq_mask.float()).sum(dim=1)
                seq_adv = (mb_step_adv * mb_seq_mask.float()).sum(dim=1) / mb_seq_len.float()
                seq_ret = (mb_step_ret * mb_seq_mask.float()).sum(dim=1) / mb_seq_len.float()

                # Sequence layer also has value clipping
                log_ratio = (new_seq_logp - old_seq_logp).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                pg1 = -seq_adv * ratio
                pg2 = -seq_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                vf_loss = 0.5 * ((new_seq_value.view(-1) - seq_ret) ** 2).mean()
                loss = pg_loss - self.deploy_ent_coef * entropy.mean() + self.vf_coef * vf_loss

                self.deployment_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.deployment_policy.parameters(), self.max_grad_norm)
                self.deployment_optimizer.step()

        # Advance schedulers after each full training
        try:
            if self.decouple_critics:
                if self.policy_scheduler is not None:
                    self.policy_scheduler.step()
                if self.value_scheduler is not None:
                    self.value_scheduler.step()
            else:
                if self.scheduler is not None:
                    self.scheduler.step()
        except Exception:
            pass
        # --- Lagrangian multiplier update: based on gap between avg latency and limit ---
        with torch.no_grad():
            avg_cost_per_step = user_data['costs'].mean()
            target_limit = torch.tensor(self.cost_limit, dtype=torch.float32, device=self.device)

        # Natural equivalence: when lr=0 or λ=0, this term's gradient/update is 0 (no branching needed)
        if self.lagrangian_lr > 0.0:
            # Relax cost gap limit to make Lagrangian mechanism more effective
            cost_gap = (avg_cost_per_step - target_limit).detach()
            cost_gap = torch.clamp(cost_gap, -10.0, 10.0)  # Expand range
            lagrangian_loss = -(self.lagrangian_multiplier * cost_gap)
            self.lagrangian_optimizer.zero_grad()
            lagrangian_loss.backward()
            self.lagrangian_optimizer.step()
            # Relax λ upper limit to make constraint mechanism more effective
            self.lagrangian_multiplier.data.clamp_(0, 50.0)  # Expand range to match configuration

    def get_user_learning_rates(self):
        """Return current learning rates for user layer (policy_lr, value_lr).
        - When decouple_critics=False, both are the same from self.user_optimizer.
        - When decouple_critics=True, from self.policy_optimizer and self.value_optimizer respectively.
        """
        try:
            if self.decouple_critics:
                policy_lr = float(self.policy_optimizer.param_groups[0]['lr']) if hasattr(self, 'policy_optimizer') else None
                value_lr = float(self.value_optimizer.param_groups[0]['lr']) if hasattr(self, 'value_optimizer') else None
                return policy_lr, value_lr
            else:
                if hasattr(self, 'user_optimizer'):
                    lr = float(self.user_optimizer.param_groups[0]['lr'])
                    return lr, lr
                # Fallback: return None in extreme cases
                return None, None
        except Exception:
            return None, None
