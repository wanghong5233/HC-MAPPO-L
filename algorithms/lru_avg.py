import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import SubsetRandomSampler, BatchSampler

from policies.mappo_user_policy import MAPPOUserPolicy
from algorithms.local_only import _RuleBasedAllocation


class _LRUDeploymentPolicy:
    """Based on LRU approximation rule-based deployment strategy.
    
    - Use the hit_history of each server from the previous window as "recently used" metric, prioritize retaining/deploying recently used services;
    - When capacity has remaining, fill with system-level request frequency req_history (global preference as secondary basis);
    - Interface consistent with training deployment strategy: provide obs_spec and get_full_action_trajectory_cpu.
    """

    def __init__(self, num_services: int, num_edges: int):
        self.num_services = num_services
        self.num_edges = num_edges
        self.obs_spec = {
            'request_history': num_services,
            'hit_history': num_services,
            'global_deployment': num_services * num_edges,
            'deployment_state': num_services,
        }
        # Maintain consistent placeholder parameters with training interface (for device inference)
        self._dummy_param = nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._dummy_param

    def get_full_action_trajectory_cpu(
        self,
        obs_np: np.ndarray,
        full_storage_capacity_MB: float,
        service_sizes_array: np.ndarray,
    ):
        """Generates a complete deployment sequence (autoregressive approximation, but no learning).

        Strategy:
        1) First select services that were hit on this server in the previous window (hit_history>0), sorted by hit frequency from high to low;
        2) If there is still space, fill with system-level request frequency (req_history) from high to low;
        3) All services are selected only once, truncated by capacity (MB).
        Returns: actions(int32, length I, padding=-1), logps(float32, I), values(float32, I), actual_length(int)
        """
        I = self.obs_spec['request_history']
        req_freq = obs_np[:I]
        hit_freq = obs_np[I:2 * I]

        # First sort by "recently used" (hit in the previous window), then use "system preference" as a fallback
        recent_ids = np.where(hit_freq > 0)[0]
        recent_order = recent_ids[np.argsort(-hit_freq[recent_ids])]

        all_ids = np.arange(I)
        remaining_ids = np.setdiff1d(all_ids, recent_order, assume_unique=False)
        global_order = remaining_ids[np.argsort(-req_freq[remaining_ids])]

        deploy_order = np.concatenate([recent_order, global_order])

        actions = np.full(I, -1, dtype=np.int32)
        logps = np.zeros(I, dtype=np.float32)
        values = np.zeros(I, dtype=np.float32)

        remaining = float(full_storage_capacity_MB)
        count = 0
        for i in deploy_order:
            size_mb = float(service_sizes_array[i]) * 1024.0  # GB -> MB
            if size_mb <= remaining:
                actions[count] = int(i)
                remaining -= size_mb
                count += 1
            if remaining <= 1e-6:
                break

        return actions, logps, values, count


class LRUAvgAgent:
    """HC-MAPPO-L user layer + average allocation + LRU (approximation) deployment pluggable baseline.
    
    - user_policy: MAPPOUserPolicy (centralized critic), following HC-MAPPO-L user layer structure and training (including Lagrangian term).
    - sac_allocation: Rule-based average (_RuleBasedAllocation).
    - deployment_policy: _LRUDeploymentPolicy (based on hit frequency in the previous window).
    - train(): Only train the user layer (PPO-Lagrangian), do not train the allocation and deployment layers.
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device('cpu')

        # --- Hyperparameters (aligned with H-MAPPO-L) ---
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

        self.cost_limit = self.env.latency_constraint

        # --- User Policy (MAPPO structure, aligned with H-MAPPO-L) ---
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
        self.user_policy = MAPPOUserPolicy(
            obs_spec=user_obs_spec,
            action_dims=user_action_dims,
            hidden_dim=int(self.config.get('agent.user_hidden_dim', 256))
        ).to(self.device)
        self.user_optimizer = optim.Adam(self.user_policy.parameters(), lr=self.lr, eps=1e-5)

        # --- Allocation: Rule-based average ---
        self.sac_allocation = _RuleBasedAllocation(self.env.num_edges, self.env.num_users)

        # --- Deployment: Approximate LRU rule ---
        self.deployment_policy = _LRUDeploymentPolicy(self.env.num_services, self.env.num_edges)

        # --- Lagrangian multipliers (only for user layer training) ---
        self.lagrangian_multiplier = torch.tensor(0.1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.lagrangian_optimizer = optim.Adam([self.lagrangian_multiplier], lr=self.lagrangian_lr)

    def compute_advantages(self, rewards, dones, values, gamma, gae_lambda):
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(rewards.size(0))):
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
            delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
        return advantages

    def train(self, storage):
        """Only train the user layer (PPO-Lagrangian), keep the rule policy for allocation and deployment."""
        user_data = storage.get_user_training_data()
        if user_data is None:
            return

        b_user_obs = user_data['observations']
        b_user_actions = user_data['actions']
        b_user_log_probs = user_data['log_probs']
        b_user_advantages = user_data['advantages']
        b_user_returns = user_data['returns']
        b_user_action_masks = user_data.get('action_masks', None)
        b_central_obs = user_data.get('central_observations', None)

        batch_size = b_user_obs.size(0)
        minibatch_size = max(1, batch_size // self.num_minibatches)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=True)

        for _ in range(self.num_epochs):
            for indices in sampler:
                mb_user_obs = b_user_obs[indices]
                mb_user_actions = b_user_actions[indices]
                mb_user_log_probs = b_user_log_probs[indices]
                mb_user_advantages = b_user_advantages[indices].clamp(-50.0, 50.0)
                mb_user_returns = b_user_returns[indices]
                mb_action_masks = b_user_action_masks[indices] if b_user_action_masks is not None else None
                mb_central_obs = b_central_obs[indices] if b_central_obs is not None else None

                # Cost advantages and returns (for Lagrangian term)
                mb_cost_advantages = user_data['cost_advantages'][indices].clamp(-50.0, 50.0)
                mb_cost_returns = user_data['cost_returns'][indices]

                _, new_log_prob, entropy, new_value, new_cost_value = self.user_policy.get_action_and_value(
                    mb_user_obs, central_obs=mb_central_obs, action=mb_user_actions, action_masks=mb_action_masks
                )

                ratio = torch.exp(new_log_prob - mb_user_log_probs)

                # Advantage reshaping: A_total = A_r - λ̃ · A_c
                with torch.no_grad():
                    adv_std = mb_user_advantages.std()
                    cost_adv_std = mb_cost_advantages.std()
                    scale = adv_std / (cost_adv_std + 1e-8)
                lambda_scaled = self.lagrangian_multiplier.detach() * scale
                combined_advantages = (mb_user_advantages - lambda_scaled * mb_cost_advantages)

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

        # --- Lagrangian multiplier update (based on average latency and threshold difference) ---
        with torch.no_grad():
            avg_cost_per_step = user_data['costs'].mean()
            target_limit = torch.tensor(self.cost_limit, dtype=torch.float32, device=self.device)

        lagrangian_loss = -(self.lagrangian_multiplier * (avg_cost_per_step - target_limit).detach())
        self.lagrangian_optimizer.zero_grad()
        lagrangian_loss.backward()
        self.lagrangian_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0)


