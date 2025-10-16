import numpy as np
import torch
import torch.nn as nn

from .local_only import _RuleBasedAllocation, _PopularModelsDeploymentPolicy


class _GreedyUserPolicy(nn.Module):
    """Greedy heuristic:
    - Association: Choose the server with the strongest channel that deployed the target service; if none deployed, choose the global strongest.
    - Split: Search upwards from the deepest split point, choose the first z that satisfies the delay constraint; otherwise degenerate to z=0.
    - Resource Estimation: Uses an average approximation: f_jk = F_j / |U_j| and B_jk = B_j / |U_j|.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        self._dummy = nn.Parameter(torch.zeros(1))

    def _choose_associations(self) -> np.ndarray:
        num_users = self.env.num_users
        num_edges = self.env.num_edges
        gains = self.env.channel_gains
        deployed_sets = [set(srv.deployed_services.keys()) for srv in self.env.servers]
        assoc = np.zeros(num_users, dtype=np.int32)
        for k in range(num_users):
            service_id = int(self.env.all_users_requested_service_ids[k])
            candidates = [j for j in range(num_edges) if service_id in deployed_sets[j]]
            if candidates:
                best_j = max(candidates, key=lambda j: gains[j, k])
            else:
                best_j = int(np.argmax(gains[:, k]))
            assoc[k] = best_j
        return assoc

    def _estimate_split(self, k: int, assoc_j: int, user_count_on_j: int) -> int:
        service_id, input_size = self.env.users[k].current_request
        service = self.env.services[service_id]
        server = self.env.servers[assoc_j]
        # Equal sharing approximation
        denom = max(user_count_on_j, 1)
        approx_f = server.compute_cap_GFLOPS / denom
        approx_B = (server.bandwidth_MHz * 1e6) / denom

        # Search downwards from the deepest split point
        for z in range(service.num_split_points, -1, -1):
            delay, _, _, _ = self.env._calculate_performance_metrics(
                k, assoc_j, service_id, z, input_size, approx_f, approx_B
            )
            if delay <= self.env.latency_constraint:
                return z
        return 0

    def get_action_and_value_cpu(self, obs_np: np.ndarray, central_obs_np=None, action_masks=None):
        num_users = self.env.num_users
        associations = self._choose_associations()
        # Count the number of users associated with each server
        counts = np.bincount(associations, minlength=self.env.num_edges)
        splits = np.zeros(num_users, dtype=np.int32)
        for k in range(num_users):
            j = int(associations[k])
            splits[k] = self._estimate_split(k, j, int(counts[j]))
        actions = np.stack([associations.astype(np.int32), splits.astype(np.int32)], axis=1)
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


class GreedyPolicyAgent:
    """Greedy-Policy heuristic baseline."""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        # The greedy baseline runs on the CPU.
        self.device = torch.device('cpu')
        self.user_policy = _GreedyUserPolicy(env).to(self.device)
        self.sac_allocation = _RuleBasedAllocation(env.num_edges, env.num_users)
        self.deployment_policy = _PopularModelsDeploymentPolicy(env.num_services, env.num_edges)
        self.lagrangian_multiplier = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def train(self, storage):
        return


