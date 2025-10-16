import numpy as np
import torch
import torch.nn as nn

from .local_only import _RuleBasedAllocation, _PopularModelsDeploymentPolicy


class _EdgeOnlyUserPolicy(nn.Module):
    """Edge computing baseline:
    - Association: Choose the server with the strongest channel that deployed the target service; if none deployed, choose the global strongest.
    - Split point: Fixed choice z=0 (full edge).
    - Interface consistent with main process, return numpy.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        self._dummy = nn.Parameter(torch.zeros(1))

    def get_action_and_value_cpu(self, obs_np: np.ndarray, central_obs_np=None, action_masks=None):
        num_users = self.env.num_users
        num_edges = self.env.num_edges

        associations = np.zeros(num_users, dtype=np.int32)
        split_points = np.zeros(num_users, dtype=np.int32)  # Full edge offloading: z=0

        deployed_sets = [set(srv.deployed_services.keys()) for srv in self.env.servers]
        gains = self.env.channel_gains  # Shape: (num_edges, num_users)

        for k in range(num_users):
            service_id = int(self.env.all_users_requested_service_ids[k])
            candidates = [j for j in range(num_edges) if service_id in deployed_sets[j]]
            if candidates:
                best_j = max(candidates, key=lambda j: gains[j, k])
            else:
                best_j = int(np.argmax(gains[:, k]))
            associations[k] = best_j

        actions = np.stack([associations, split_points], axis=1).astype(np.int32)
        logp = np.zeros(num_users, dtype=np.float32)
        v = np.zeros(num_users, dtype=np.float32)
        cv = np.zeros(num_users, dtype=np.float32)
        return actions, logp, v, cv

    def get_value(self, obs, central_obs=None):
        batch = obs.shape[0]
        device = self._dummy.device
        return (
            torch.zeros(batch, 1, device=device),
            torch.zeros(batch, 1, device=device),
        )


class EdgeOnlyAgent:
    """Edge-Only baseline. Interface consistent with HMAPPO_Lagrangian."""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        # The rule-based baseline runs on the CPU.
        self.device = torch.device('cpu')

        self.user_policy = _EdgeOnlyUserPolicy(env).to(self.device)
        self.sac_allocation = _RuleBasedAllocation(env.num_edges, env.num_users)
        self.deployment_policy = _PopularModelsDeploymentPolicy(env.num_services, env.num_edges)
        self.lagrangian_multiplier = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def train(self, storage):
        return


