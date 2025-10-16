import numpy as np
import torch
import torch.nn as nn


class _RandomAllocation:
    """Random allocation: Give random weights to associated users (separately for comp/band)."""

    def __init__(self, num_edges: int, num_users: int, rng: np.random.RandomState):
        self.num_edges = num_edges
        self.num_users = num_users
        self.rng = rng

    def get_action_cpu(self, obs_np: np.ndarray, deterministic: bool = False) -> np.ndarray:
        actions = np.zeros((self.num_edges, 2 * self.num_users), dtype=np.float32)
        for j in range(self.num_edges):
            assoc = obs_np[j, : self.num_users] > 0.5
            idx = np.where(assoc)[0]
            if idx.size == 0:
                continue
            # Sample random positive weights and normalize them.
            w1 = self.rng.rand(idx.size).astype(np.float32)
            w2 = self.rng.rand(idx.size).astype(np.float32)
            s1 = float(w1.sum()) or 1.0
            s2 = float(w2.sum()) or 1.0
            actions[j, idx] = w1 / s1
            actions[j, self.num_users + idx] = w2 / s2
        return actions


class _RandomDeploymentPolicy:
    """Random deployment: Load in random order until capacity is exhausted."""

    def __init__(self, num_services: int, num_edges: int, rng: np.random.RandomState):
        self.num_services = num_services
        self.rng = rng
        self.obs_spec = {
            'request_history': num_services,
            'hit_history': num_services,
            'global_deployment': num_services * num_edges,
            'deployment_state': num_services,
        }
        self._dummy = nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._dummy

    def get_full_action_trajectory_cpu(self, obs_np: np.ndarray, full_storage_capacity_MB: float, service_sizes_array: np.ndarray):
        order = np.arange(self.num_services)
        self.rng.shuffle(order)
        actions = np.full(self.num_services, -1, dtype=np.int32)
        logps = np.zeros(self.num_services, dtype=np.float32)
        values = np.zeros(self.num_services, dtype=np.float32)

        remaining = float(full_storage_capacity_MB)
        count = 0
        for i in order:
            size_mb = float(service_sizes_array[i]) * 1024.0
            if size_mb <= remaining:
                actions[count] = int(i)
                remaining -= size_mb
                count += 1
        return actions, logps, values, count


class _RandomUserPolicy(nn.Module):
    """Random user policy: Random association and random split (according to the valid split range of each service)."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.rng = env.rng
        self._dummy = nn.Parameter(torch.zeros(1))

    def get_action_and_value_cpu(self, obs_np: np.ndarray, central_obs_np=None, action_masks=None):
        num_users = self.env.num_users
        num_edges = self.env.num_edges
        associations = np.zeros(num_users, dtype=np.int32)
        split_points = np.zeros(num_users, dtype=np.int32)
        for k in range(num_users):
            associations[k] = int(self.rng.randint(0, num_edges))
            service_id = int(self.env.all_users_requested_service_ids[k])
            max_z = int(self.env.services[service_id].num_split_points)
            split_points[k] = int(self.rng.randint(0, max_z + 1))
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


class RandomPolicyAgent:
    """Random policy baseline."""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        # The random baseline runs on the CPU.
        self.device = torch.device('cpu')
        self.user_policy = _RandomUserPolicy(env).to(self.device)
        self.sac_allocation = _RandomAllocation(env.num_edges, env.num_users, env.rng)
        self.deployment_policy = _RandomDeploymentPolicy(env.num_services, env.num_edges, env.rng)
        self.lagrangian_multiplier = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def train(self, storage):
        return


