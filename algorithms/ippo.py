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
    IPPO baseline agent with hierarchical structure:
    - User layer: IPPO with local observations
    - Allocation layer: SAC (same as main implementation)
    - Deployment layer: IPPO with shared parameters across servers
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device('cpu')

        # Hyperparameters
        self.lr = float(self.config.get('agent.learning_rate'))
        self.lr_scheduler_type = str(self.config.get('agent.lr_scheduler', 'none'))
        self.lr_end_factor = float(self.config.get('agent.lr_end_factor', 0.1))
        try:
            _lr_T_max = self.config.get('agent.lr_T_max')
            self.lr_T_max = int(_lr_T_max) if _lr_T_max is not None else 1000
        except Exception:
            self.lr_T_max = 1000
        self.scheduler = None
        self.gamma = float(self.config.get('agent.gamma'))
        self.gae_lambda = float(self.config.get('agent.gae_lambda'))
        self.clip_coef = float(self.config.get('agent.clip_coef'))
        self.ent_coef = float(self.config.get('agent.ent_coef'))
        self.vf_coef = float(self.config.get('agent.vf_coef'))
        self.max_grad_norm = float(self.config.get('agent.max_grad_norm'))
        self.num_epochs = int(self.config.get('agent.num_epochs'))
        self.num_minibatches = int(self.config.get('agent.num_minibatches'))
        self.deploy_ent_coef = float(self.config.get('agent.deploy_ent_coef', self.ent_coef))

        self._setup_agents()

        # For log compatibility
        self.lagrangian_multiplier = torch.tensor(0.0, dtype=torch.float32, device=self.device)

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
        self.user_optimizer = optim.Adam(self.user_policy.parameters(), lr=self.lr, eps=1e-5)
        # Scheduler: cosine / linear strategy consistent with HC-IPPO-L
        if self.lr_scheduler_type == 'cosine':
            # Half-cycle cosine decay (half-cycle length = lr_T_max / 2), then maintain minimum value
            T_max_cosine = self.lr_T_max / 2
            eta_min_ratio = self.lr_end_factor
            lr_lambda = lambda update: (
                eta_min_ratio + 0.5 * (1 - eta_min_ratio) * (1 + math.cos(math.pi * update / T_max_cosine))
                if update < T_max_cosine
                else eta_min_ratio
            )
            self.scheduler = optim.lr_scheduler.LambdaLR(self.user_optimizer, lr_lambda)
        elif self.lr_scheduler_type == 'linear':
            # Full-cycle linear decay (reaches end_factor at lr_T_max)
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
        self.deployment_optimizer = optim.Adam(self.deployment_policy.parameters(), lr=self.lr, eps=1e-5)

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
                mb_masks = b_masks[indices] if b_masks is not None else None

                # Compatible with MAPPO interface: if storage provides central_obs, pass it along; UserPolicy will ignore
                central_obs = user_data.get('central_observations', None)
                _, new_logp, entropy, new_v, _ = self.user_policy.get_action_and_value(
                    mb_obs, central_obs=central_obs, action=mb_actions, action_masks=mb_masks
                )
                ratio = torch.exp(new_logp - mb_old_logp)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((new_v - mb_ret) ** 2).mean()
                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss

                self.user_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.user_policy.parameters(), self.max_grad_norm)
                self.user_optimizer.step()

        # --- Allocation layer: SAC (same as original implementation) ---
        if storage.ready_for_allocation_training(int(self.config.get('agent.sac_batch_size'))):
            sac_updates = int(self.config.get('agent.sac_updates_per_epoch', 3))
            for _ in range(sac_updates):
                batch = storage.sample_allocation_batch(int(self.config.get('agent.sac_batch_size')))
                if batch is not None:
                    _ = self.sac_allocation.update(batch)

        # --- Deployment layer: IPPO (same sequence training as MAPPO version, but without centralized critic concept) ---
        deploy_data = storage.get_deployment_training_data()
        if not deploy_data:
            # Advance scheduler one step after each full training (if enabled)
            try:
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

                ratio = torch.exp(new_seq_logp - old_seq_logp)
                pg1 = -seq_adv * ratio
                pg2 = -seq_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                vf_loss = 0.5 * ((new_seq_value.view(-1) - seq_ret) ** 2).mean()
                loss = pg_loss - self.deploy_ent_coef * entropy.mean() + self.vf_coef * vf_loss

                self.deployment_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.deployment_policy.parameters(), self.max_grad_norm)
                self.deployment_optimizer.step()

        # Advance scheduler one step after each full training (if enabled)
        try:
            if self.scheduler is not None:
                self.scheduler.step()
        except Exception:
            pass


    def get_user_learning_rates(self):
        """Returns user learning rates (policy_lr, value_lr).
        IPPO does not decouple critic and actor, they share the same optimizer and learning rate.
        """
        try:
            if hasattr(self, 'user_optimizer'):
                lr = float(self.user_optimizer.param_groups[0]['lr'])
                return lr, lr
            return None, None
        except Exception:
            return None, None
