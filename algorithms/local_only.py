import numpy as np
import torch
import torch.nn as nn


class _RuleBasedAllocation:
    """Simple rule allocator: evenly distribute computation and bandwidth across associated users for each server.

    Output action shape matches environment expectation: (num_edges, 2 * num_users)
    The first K dimensions are comp weights, and the last K are band weights.
    """

    def __init__(self, num_edges: int, num_users: int):
        self.num_edges = num_edges
        self.num_users = num_users

    def get_action_cpu(self, obs_np: np.ndarray, deterministic: bool = True) -> np.ndarray:
        actions = np.zeros((self.num_edges, 2 * self.num_users), dtype=np.float32)
        for j in range(self.num_edges):
            assoc = obs_np[j, : self.num_users] > 0.5
            idx = np.where(assoc)[0]
            if idx.size == 0:
                continue
            w = np.full(idx.size, 1.0 / idx.size, dtype=np.float32)
            actions[j, idx] = w  # comp
            actions[j, self.num_users + idx] = w  # band
        return actions


class _PopularModelsDeploymentPolicy:
    """Popular model deployment policy based on global request frequency:
    - For each window boundary, greedily load from high to low req_freq until storage capacity is exhausted.
    - Compatible with existing interfaces: provide obs_spec and get_full_action_trajectory_cpu.
    """

    def __init__(self, num_services: int, num_edges: int):
        self.obs_spec = {
            'request_history': num_services,
            'hit_history': num_services,
            'global_deployment': num_services * num_edges,
            'deployment_state': num_services,
        }
        # For compatibility with training/logging interfaces, use a dummy parameter
        self._dummy_param = nn.Parameter(torch.zeros(1))

    def parameters(self):  # For external security to get device
        yield self._dummy_param

    def get_full_action_trajectory_cpu(
        self,
        obs_np: np.ndarray,
        full_storage_capacity_MB: float,
        service_sizes_array: np.ndarray,
    ):
        # obs_np: (obs_dim,) = [req_freq(I), hit_freq(I), global_deploy(I*J), deployment_state(I)]
        I = self.obs_spec['request_history']
        req_freq = obs_np[:I]

        # Greedy: select from high to low request frequency
        order = np.argsort(-req_freq)
        actions = np.full(I, -1, dtype=np.int32)
        logps = np.zeros(I, dtype=np.float32)
        values = np.zeros(I, dtype=np.float32)

        remaining = float(full_storage_capacity_MB)
        count = 0
        for i in order:
            size_mb = float(service_sizes_array[i]) * 1024.0  # GB -> MB
            if size_mb <= remaining:
                actions[count] = int(i)
                remaining -= size_mb
                count += 1
            # If capacity is not met, skip and try the next one

        return actions, logps, values, count


class _LocalOnlyUserPolicy(nn.Module):
    """Local computing baseline:
    - Association: Prefer the server with the strongest channel that deployed the target service; if none deployed, choose the strongest channel server.
    - Split point: Choose the deepest split point available for the service (full local inference).
    - Return log_prob/value/cost_value set to zero to ensure compatibility with main loop interface.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        # Place a dummy parameter to avoid external next(parameters()) failure
        self._dummy = nn.Parameter(torch.zeros(1))

    def get_action_and_value_cpu(self, obs_np: np.ndarray, central_obs_np=None, action_masks=None):
        num_users = self.env.num_users
        num_edges = self.env.num_edges

        associations = np.zeros(num_users, dtype=np.int32)
        split_points = np.zeros(num_users, dtype=np.int32)

        # Pre-fetch deployed sets and channel gains
        deployed_sets = [set(srv.deployed_services.keys()) for srv in self.env.servers]
        gains = self.env.channel_gains  # shape (J, K)

        for k in range(num_users):
            service_id = int(self.env.all_users_requested_service_ids[k])
            # Find the server with the strongest channel that deployed this service
            candidates = [j for j in range(num_edges) if service_id in deployed_sets[j]]
            if candidates:
                best_j = max(candidates, key=lambda j: gains[j, k])
            else:
                best_j = int(np.argmax(gains[:, k]))
            associations[k] = best_j

            # Full local: choose the maximum split point for this service
            max_z = int(self.env.services[service_id].num_split_points)
            split_points[k] = max_z

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


class LocalOnlyAgent:
    """Local-Only baseline agent.

    Provide the same key interface fields as HMAPPO_Lagrangian:
    - user_policy: Provide get_action_and_value_cpu / get_value
    - sac_allocation: Provide get_action_cpu
    - deployment_policy: Provide obs_spec / get_full_action_trajectory_cpu
    - train(storage): Empty implementation
    - lagrangian_multiplier: Constant zero tensor for log reuse
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        # Rule-based baseline forces CPU to avoid occupying GPU
        self.device = torch.device('cpu')

        # User layer (rule-based)
        self.user_policy = _LocalOnlyUserPolicy(env).to(self.device)

        # Allocation layer (rule-based)
        self.sac_allocation = _RuleBasedAllocation(env.num_edges, env.num_users)

        # Deployment layer (popular models)
        self.deployment_policy = _PopularModelsDeploymentPolicy(env.num_services, env.num_edges)

        # Compatible with main loop logging
        self.lagrangian_multiplier = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def train(self, storage):
        # Rule-based baseline does not train
        return


