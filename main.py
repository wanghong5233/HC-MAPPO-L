import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import os
import sys
from tqdm import tqdm
import csv
import json
import argparse
import shutil

from env.env import MARLEdgeInferenceEnv
from algorithms.h_mappo_l import HMAPPO_Lagrangian
from algorithms.local_only import LocalOnlyAgent
from algorithms.edge_only import EdgeOnlyAgent
from algorithms.random_policy import RandomPolicyAgent
from algorithms.greedy_policy import GreedyPolicyAgent
from algorithms.mappo_no_constraint import MAPPO_NoConstraint
from algorithms.ippo import IPPOAgent
from algorithms.HC_IPPO_L import IPPOAgent as HCIPPOAgent
from algorithms.lru_avg import LRUAvgAgent
from storage import HierarchicalStorage
from env.entities import SystemConfig
from utils.running_stat import RunningMeanStd

# Algorithm registry
AGENT_REGISTRY = {
    'h_mappo_l': HMAPPO_Lagrangian,
    'local_only': LocalOnlyAgent,
    'edge_only': EdgeOnlyAgent,
    'random_policy': RandomPolicyAgent,
    'greedy_policy': GreedyPolicyAgent,
    'mappo_no_constraint': MAPPO_NoConstraint,
    'ippo': IPPOAgent,
    'hc_ippo_l': HCIPPOAgent,
    'lru_avg': LRUAvgAgent,
}

# Patch agent constructors to accept max_updates parameter
def _patch_agent_inits():
    for name, agent_class in AGENT_REGISTRY.items():
        if name != 'hc_ippo_l':
            original_init = agent_class.__init__
            # Fix closure: capture original_init with default parameter
            def new_init(self, env, config, max_updates=None, _original_init=original_init):
                _original_init(self, env, config)
            agent_class.__init__ = new_init
_patch_agent_inits()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Hierarchical MARL Training")
    parser.add_argument(
        '--agent', 
        type=str, 
        default='h_mappo_l', 
        choices=list(AGENT_REGISTRY.keys()),
        help='Algorithm name to run'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Random seed (overrides config file)'
    )
    parser.add_argument(
        '--config-overrides', 
        type=str, 
        default="{}",
        help='JSON string to override config parameters, e.g., \'{"env.num_users": 100}\''
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name for organizing results'
    )
    parser.add_argument(
        '--max-updates',
        type=int,
        default=1000,
        help='Maximum training updates'
    )
    parser.add_argument(
        '--results-base-dir',
        type=str,
        default='experiment_result',
        help='Base directory for results (default: experiment_result)'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = SystemConfig('configs/system_config.yaml', 'configs/agent_config.yaml')
    
    # Apply command-line configuration overrides
    try:
        overrides = json.loads(args.config_overrides)
        for key, value in overrides.items():
            keys = key.split('.')
            d = config.config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        if overrides:
            try:
                print(f"Applied config overrides: {overrides}")
            except Exception:
                print("[INFO] Applied config overrides:", overrides)
    except json.JSONDecodeError:
        print(f"❌  Invalid JSON override string: {args.config_overrides}")
        return

    # Prepare config snapshot for logging
    config_snapshot = {
        "config": config.config,  # merged config parameters (including all overrides)
        "command_args": {
            "agent": args.agent,
            "seed": args.seed,
            "experiment_name": args.experiment_name,
            "max_updates": args.max_updates,
            "results_base_dir": args.results_base_dir,
            "config_overrides": overrides if 'overrides' in locals() else {}
        },
        "runtime_info": {
            "timestamp": int(time.time()),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform
        }
    }

    # Set random seed (command-line argument takes priority)
    seed = args.seed if args.seed is not None else config.get('env.seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    config.config['env']['seed'] = seed

    # Construct run name for experiment tracking
    num_users_str = f"{config.get('env.num_users')}users"
    num_servers_str = f"{config.get('env.num_edges')}servers"

    exp_type_for_name = None
    if args.experiment_name:
        _parts = args.experiment_name.split('_', 2)
        if len(_parts) >= 1:
            exp_type_for_name = _parts[0]

    if exp_type_for_name in ("sensitivity", "scalability"):
        run_name = None
    else:
        if args.experiment_name:
            run_name = f"{args.experiment_name}__{args.agent}__{num_users_str}_{num_servers_str}__{seed}_{int(time.time())}"
        else:
            run_name = f"{args.agent}__{num_users_str}_{num_servers_str}__{seed}_{int(time.time())}"

    tb_run_name = run_name if run_name is not None else (
        f"{args.experiment_name}__{args.agent}__{num_users_str}_{num_servers_str}__{seed}_{int(time.time())}"
        if args.experiment_name else f"{args.agent}__{num_users_str}_{num_servers_str}__{seed}_{int(time.time())}"
    )

    # Initialize device (CPU for compatibility)
    device = torch.device("cpu")
    config_snapshot["runtime_info"]["device"] = str(device)

    def _sanitize_for_path(s):
        return str(s).replace(' ', '').replace('[', '').replace(']', '').replace(',', '_').replace(':', '-').replace('/', '-')

    results_dir = None
    exp_type = None
    exp_name = None
    param_value_str = None
    if args.experiment_name:
        # Parse experiment name: <type>_<param>_<value>
        if '_' in args.experiment_name:
            exp_type, rest = args.experiment_name.split('_', 1)
            if exp_type in ("sensitivity", "scalability"):
                last_us = rest.rfind('_')
                if last_us != -1:
                    exp_name = rest[:last_us]
                    param_value_str = rest[last_us+1:]

    if exp_type in ("sensitivity", "scalability") and exp_name is not None and param_value_str is not None:
        param_value_dir = _sanitize_for_path(param_value_str)
        results_dir = os.path.join(
            args.results_base_dir,
            exp_name,
            param_value_dir,
            args.agent,
            f"seed_{seed}",
        )
    else:
        results_dir = os.path.join(args.results_base_dir, f"seed_{seed}", run_name)

    # Create results directory and save config snapshot
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    # Initialize metrics CSV
    csv_filepath = os.path.join(results_dir, "metrics.csv")
    csv_header = [
        "update", "success_rate", "avg_delay_s", "avg_constraint_violation",
        "avg_energy_j", "avg_privacy_cost", "avg_user_reward", "avg_alloc_reward",
        "avg_deploy_reward", "lagrangian_multiplier", "deployment_coverage",
        "user_policy_lr", "user_value_lr"
    ]
    with open(csv_filepath, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)
    
    # Per-client metrics CSV
    per_client_csv_filepath = os.path.join(results_dir, "per_client_metrics.csv")
    per_client_csv_header = [
        "update", "step_in_update", "client_id", "delay_s", "energy_j",
        "privacy_cost", "service_hit"
    ]
    with open(per_client_csv_filepath, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(per_client_csv_header)
    
    env = MARLEdgeInferenceEnv(config=config)
    
    # Instantiate algorithm from registry
    agent_class = AGENT_REGISTRY.get(args.agent)
    if not agent_class:
        raise ValueError(f"Unknown algorithm: {args.agent}. Please register in AGENT_REGISTRY.")
    
    agent = agent_class(env, config, max_updates=args.max_updates)

    # Calculate user action mask dimension for efficient storage
    max_split_points = max(s.num_split_points for s in env.services) + 1
    user_action_mask_dim = env.num_edges + max_split_points
    
    # Initialize hierarchical storage with buffer configuration
    storage_config = {
        'num_users': env.num_users,
        'num_servers': env.num_edges,
        'num_services': env.num_services,
        'num_steps': config.get('agent.num_steps'),
        'deployment_update_interval': config.get('framework.deployment_update_interval'),
        'user_obs_dim': env.user_observation_space.shape[0],
        'user_action_dim': 2,  # association + split_point
        'user_action_mask_dim': user_action_mask_dim,  # User action mask dimension
        'central_obs_dim': env.central_observation_space.shape[0],  # Centralized observation dimension required by MAPPO
        'alloc_obs_dim': env.allocation_observation_space.shape[0],
        'alloc_action_dim': env.num_users * 2,  # comp + band
        'deploy_obs_dim': env.deployment_observation_space.shape[0],  # 135 dimensions (request + hit + deployment_state)
        'sac_buffer_size': config.get('agent.sac_buffer_size'),
        'deployment_buffer_capacity': config.get('agent.deployment_buffer_capacity'),
        'use_replay_buffers': config.get('agent.use_replay_buffers'),
        'user_buffer_capacity': config.get('agent.user_buffer_capacity')
    }
    storage = HierarchicalStorage(storage_config, device)

    # Calculate maximum training updates
    num_updates = config.get('agent.total_timesteps') // config.get('agent.num_steps')
    num_updates = min(num_updates, args.max_updates)

    print("=" * 80)
    print(f"🚀 Starting {args.agent.upper()} Training on {device}")
    print(f"   Run Name: {run_name}")
    print(f"   Random Seed: {seed}")
    print(f"   Max Updates: {num_updates}")
    print("-" * 80)
    print(f"⚙️ Fail delay (seconds): {env.fail_delay:.4f}")

    global_step = 0
    
    user_obs, central_obs = env.reset()

    # Reward normalizers for stable training
    user_reward_rms = RunningMeanStd()
    alloc_reward_rms = RunningMeanStd()
    deploy_reward_rms = RunningMeanStd()

    csv_buffer = []

    for update in range(1, num_updates + 1):
        collect_start_time = time.time()
        
        per_client_records = []

        # Performance trackers with pre-allocated arrays
        num_collect_steps = config.get('agent.num_steps')
        deployment_update_interval = config.get('framework.deployment_update_interval')
        num_deployment_decisions = (num_collect_steps // deployment_update_interval) * env.num_edges
        
        ep_rewards = {
            'user': np.zeros(num_collect_steps * env.num_users, dtype=np.float32),
            'alloc': np.zeros(num_collect_steps * env.num_edges, dtype=np.float32),
            'deploy': np.zeros(num_deployment_decisions, dtype=np.float32)
        }
        ep_metrics = {
            'delay': np.zeros(num_collect_steps * env.num_users, dtype=np.float32),
            'energy': np.zeros(num_collect_steps * env.num_users, dtype=np.float32),
            'privacy_cost': np.zeros(num_collect_steps * env.num_users, dtype=np.float32),
            'hit': np.zeros(num_collect_steps * env.num_users, dtype=np.float32)
        }
        reward_idx = {'user': 0, 'alloc': 0, 'deploy': 0}
        metrics_idx = 0
        
        # Data collection with progress bar
        print(f"\n=== Update {update}/{num_updates} - Data Collection Phase ===")
        progress_bar = tqdm(range(config.get('agent.num_steps')), 
                           desc="Collecting", 
                           unit="step", 
                           ncols=80,
                           leave=False)

        # Pre-allocate numpy arrays for efficient data collection
        user_obs_cache = np.zeros((env.num_users, env.user_observation_space.shape[0]), dtype=np.float32)
        central_obs_cache = np.zeros((env.num_users, env.central_observation_space.shape[0]), dtype=np.float32)
        alloc_obs_cache = np.zeros((env.num_edges, env.allocation_observation_space.shape[0]), dtype=np.float32)
        
        prev_alloc_obs = None
        prev_alloc_actions = None
        prev_deployment_data = None
        
        for step in progress_bar:
            global_step += 1
            boundary_step = env.needs_deployment_decision()

            with torch.no_grad():
                # Get user actions
                user_obs_cache[:] = user_obs
                central_obs_cache[:] = central_obs
                
                user_action_masks = env.get_user_action_masks()
                user_actions_np, user_log_probs_np, user_values_np, user_cost_values_np = agent.user_policy.get_action_and_value_cpu(
                    user_obs_cache, central_obs_np=central_obs_cache, action_masks=user_action_masks)
                
                alloc_obs = env.get_allocation_observation(user_actions_np)
                alloc_obs_cache[:] = alloc_obs
                alloc_actions_np = agent.sac_allocation.get_action_cpu(alloc_obs_cache, deterministic=True)
                
                env_actions = {
                    'user': user_actions_np,
                    'allocation': alloc_actions_np,
                    'deployment': None # Handled below
                }

                # Handle deployment agent
                deployment_obs = None
                curr_deployment_data = None
                
                if boundary_step:
                    req_freq, hit_freq_matrix = env.get_deployment_observation_data()
                    obs_dim = (agent.deployment_policy.obs_spec['request_history'] + 
                                agent.deployment_policy.obs_spec['hit_history'] + 
                                agent.deployment_policy.obs_spec['global_deployment'] +
                                agent.deployment_policy.obs_spec['deployment_state'])
                    curr_deployment_data = {
                        'obs_array': np.zeros((env.num_edges, obs_dim), dtype=np.float32),
                        'actions_padded': np.full((env.num_edges, env.num_services), -1, dtype=np.int32),
                        'seq_lengths': np.zeros(env.num_edges, dtype=np.int32),
                        'step_log_probs': np.zeros((env.num_edges, env.num_services), dtype=np.float32),
                        'step_values': np.zeros((env.num_edges, env.num_services + 1), dtype=np.float32),
                        'sequence_rewards': np.zeros(env.num_edges, dtype=np.float32)
                    }
                    base_obs_vectors = np.concatenate([
                        np.tile(req_freq, (env.num_edges, 1)),
                        hit_freq_matrix,
                        np.tile(env._get_deployment_matrix().flatten(), (env.num_edges, 1))
                    ], axis=1)
                    service_sizes_array = np.array([service.model_size_GB for service in env.services], dtype=np.float32)
                    for j in range(env.num_edges):
                        base_obs_vector = base_obs_vectors[j]
                        server = env.servers[j]
                        full_storage_capacity_MB = server.storage_capacity_MB
                        
                        deployment_state = np.zeros(env.num_services, dtype=np.float32)
                        full_obs_vector = np.concatenate([base_obs_vector, deployment_state])
                        actions_array, log_probs_array, values_array, actual_length = agent.deployment_policy.get_full_action_trajectory_cpu(
                            full_obs_vector, full_storage_capacity_MB, service_sizes_array)
                        
                        curr_deployment_data['actions_padded'][j] = actions_array
                        curr_deployment_data['seq_lengths'][j] = actual_length
                        curr_deployment_data['step_log_probs'][j, :actual_length] = log_probs_array[:actual_length]
                        curr_deployment_data['step_values'][j, :actual_length] = values_array[:actual_length]
                        curr_deployment_data['step_values'][j, actual_length] = values_array[0]
                        curr_deployment_data['obs_array'][j] = full_obs_vector
                    env_actions['deployment'] = {
                        'actions_padded': curr_deployment_data['actions_padded'],
                        'seq_lengths': curr_deployment_data['seq_lengths']
                    }

            # Step environment
            (next_user_obs, next_central_obs), rewards, done, info = env.step(env_actions)

            # Collect per-client metrics for final update
            if update == num_updates:
                num_clients = len(info['delays'])
                for i in range(num_clients):
                    per_client_records.append([
                        update,
                        step,
                        i,
                        info['delays'][i],
                        info['energies'][i],
                        info['privacy_costs'][i],
                        int(info['service_hits'][i])
                    ])

            # Store deployment rewards at boundary steps
            if boundary_step and prev_deployment_data is not None and rewards.get('deployment') is not None:
                dep_rewards_raw = np.asarray(rewards['deployment'], dtype=np.float32)
                dep_rewards_norm = deploy_reward_rms.normalize(dep_rewards_raw)
                deploy_reward_rms.update(dep_rewards_raw)

                for j in range(env.num_edges):
                    seq_len = prev_deployment_data['seq_lengths'][j]
                    if seq_len > 0:
                        prev_deployment_data['sequence_rewards'][j] = dep_rewards_norm[j]

                for j in range(env.num_edges):
                    seq_len = prev_deployment_data['seq_lengths'][j]
                    if seq_len == 0:
                        continue
                    storage.insert_deployment_decision(
                        server_id=j,
                        obs=prev_deployment_data['obs_array'][j],
                        deployment_actions=prev_deployment_data['actions_padded'][j],
                        seq_length=seq_len,
                        sequence_reward=prev_deployment_data['sequence_rewards'][j],
                        step_values=prev_deployment_data['step_values'][j],
                        step_log_probs=prev_deployment_data['step_log_probs'][j]
                    )

            if boundary_step:
                prev_deployment_data = curr_deployment_data

            # Store transitions
            user_dones = np.full(env.num_users, done, dtype=np.float32)
            alloc_dones = np.full(env.num_edges, done, dtype=np.float32)
            user_rewards_raw = np.asarray(rewards['user'], dtype=np.float32)
            user_rewards_norm = user_reward_rms.normalize(user_rewards_raw)
            user_reward_rms.update(user_rewards_raw)

            alloc_rewards_raw = np.asarray(rewards['allocation'], dtype=np.float32)
            alloc_rewards_norm = alloc_reward_rms.normalize(alloc_rewards_raw)
            alloc_reward_rms.update(alloc_rewards_raw)

            storage.insert_user_step(
                user_obs,
                user_actions_np,
                user_log_probs_np,
                user_rewards_norm,
                user_dones,
                user_values_np,
                info['costs'],
                action_masks=user_action_masks,
                central_obs=central_obs,
                cost_values=user_cost_values_np
            )
            if step > 0:
                storage.insert_allocation_experience(
                    prev_alloc_obs,
                    prev_alloc_actions,
                    alloc_rewards_norm,
                    alloc_obs,  # next_obs
                    alloc_dones
                )
            storage.advance_step()

            user_obs = next_user_obs
            central_obs = next_central_obs
            prev_alloc_obs = alloc_obs
            prev_alloc_actions = alloc_actions_np
            user_count = len(rewards['user'])
            alloc_count = len(rewards['allocation'])
            
            ep_rewards['user'][reward_idx['user']:reward_idx['user']+user_count] = rewards['user']
            reward_idx['user'] += user_count
            
            ep_rewards['alloc'][reward_idx['alloc']:reward_idx['alloc']+alloc_count] = rewards['allocation']
            reward_idx['alloc'] += alloc_count
            
            if rewards.get('deployment') is not None:
                deploy_count = len(rewards['deployment'])
                ep_rewards['deploy'][reward_idx['deploy']:reward_idx['deploy']+deploy_count] = rewards['deployment']
                reward_idx['deploy'] += deploy_count
                
            metrics_count = len(info['delays'])
            ep_metrics['delay'][metrics_idx:metrics_idx+metrics_count] = info['delays']
            ep_metrics['energy'][metrics_idx:metrics_idx+metrics_count] = info['energies']
            ep_metrics['privacy_cost'][metrics_idx:metrics_idx+metrics_count] = info['privacy_costs']
            ep_metrics['hit'][metrics_idx:metrics_idx+metrics_count] = info['service_hits'].astype(np.float32)
            metrics_idx += metrics_count
            
            # Update progress bar with current step info
            if (step + 1) % 20 == 0:  # Update every 20 steps to avoid too frequent updates
                progress_bar.set_postfix({
                    'step': f"{step+1}/{config.get('agent.num_steps')}",
                    'global': global_step
                })
        
        progress_bar.close()
        
        # Compute bootstrap values for GAE
        with torch.no_grad():
            policy_device = next(agent.user_policy.parameters()).device
            final_user_obs_tensor = torch.from_numpy(user_obs).float().to(policy_device)
            final_central_obs_tensor = torch.from_numpy(central_obs).float().to(policy_device)
            final_reward_values, final_cost_values = agent.user_policy.get_value(
                final_user_obs_tensor, central_obs=final_central_obs_tensor)
            storage.insert_user_final_values(
                final_reward_values.cpu().numpy().flatten().astype(np.float32),
                final_cost_values.cpu().numpy().flatten().astype(np.float32)
            )
        
        collect_time = time.time() - collect_start_time
        print(f"✅ Data Collection completed in {collect_time:.2f}s | update {update}/{num_updates}")
        
        print("🧠 Training Phase...")
        train_start_time = time.time()
        agent.train(storage)
        storage.after_update()
        train_time = time.time() - train_start_time
        print(f"✅ Training completed in {train_time:.2f}s | update {update}/{num_updates}")
        
        # Compute performance statistics
        total_time = collect_time + train_time
        sps = int(config.get('agent.num_steps') / total_time)
        
        if metrics_idx > 0:
            total_timesteps = metrics_idx // env.num_users  
            delays_matrix = ep_metrics['delay'][:metrics_idx].reshape(total_timesteps, env.num_users)
            energies_matrix = ep_metrics['energy'][:metrics_idx].reshape(total_timesteps, env.num_users)
            privacy_cost_matrix = ep_metrics['privacy_cost'][:metrics_idx].reshape(total_timesteps, env.num_users)
            hit_matrix = ep_metrics['hit'][:metrics_idx].reshape(total_timesteps, env.num_users)
            success_rate = float(hit_matrix.mean())
            hit_sum = float(hit_matrix.sum())
            if hit_sum > 1e-8:
                avg_delay_per_timestep_s = float((delays_matrix * hit_matrix).sum() / hit_sum)
                avg_energy_per_timestep = float((energies_matrix * hit_matrix).sum() / hit_sum)
                avg_privacy_cost_per_timestep = float((privacy_cost_matrix * hit_matrix).sum() / hit_sum)
                avg_constraint_violation = avg_delay_per_timestep_s - env.latency_constraint
            else:
                avg_delay_per_timestep_s = avg_energy_per_timestep = avg_privacy_cost_per_timestep = 0.0
                avg_constraint_violation = 0.0
        else:
            avg_delay_per_timestep_s = avg_energy_per_timestep = avg_privacy_cost_per_timestep = 0
            avg_constraint_violation = 0.0
        
        avg_user_reward_per_timestep = np.mean(ep_rewards['user'][:reward_idx['user']]) if reward_idx['user'] > 0 else 0
        avg_alloc_reward_per_timestep = np.mean(ep_rewards['alloc'][:reward_idx['alloc']]) if reward_idx['alloc'] > 0 else 0
        avg_deploy_reward_per_timestep = np.mean(ep_rewards['deploy'][:reward_idx['deploy']]) if reward_idx['deploy'] > 0 else 0
        
        print(f"\n=== Update {update}/{num_updates} Summary ===")
        print(f"📊 Performance Metrics:")
        print(f"   • Success Rate:    {success_rate:8.3f}")
        print(f"   • Avg Delay (success only):   {avg_delay_per_timestep_s:8.4f} s")
        print(f"   • Avg Constraint Violation: {avg_constraint_violation:8.4f} s")
        print(f"   • Avg Energy (success only):  {avg_energy_per_timestep:8.3f} J")
        print(f"   • Avg Privacy Cost (success only): {avg_privacy_cost_per_timestep:8.6f}")
        
        print(f"🎯 Agent Rewards (Per Agent Per Timestep):")
        print(f"   • User Agent:       {avg_user_reward_per_timestep:8.3f}")
        print(f"   • Allocation Agent: {avg_alloc_reward_per_timestep:8.3f}")
        print(f"   • Deployment Agent: {avg_deploy_reward_per_timestep:8.3f}")
        
        print(f"⏱️ Timing: Collect {collect_time:.1f}s | Train {train_time:.1f}s | Total {total_time:.1f}s | SPS {sps}")
        
        print(f"🧠 Learning Progress:")
        print(f"   • Global Step:      {global_step:8d}")
        print(f"   • Lagrangian λ:     {agent.lagrangian_multiplier.item():8.4f}")
        deployment_matrix_now = env._get_deployment_matrix()
        coverage_rate = float((deployment_matrix_now.sum(axis=1) > 0).mean())
        print(f"   • Deployment coverage: {coverage_rate:6.3f}")
        
        deploy_stats = storage.get_deployment_buffer_stats()
        user_stats = storage.get_user_buffer_stats()
        print(f"📊 Buffer Statistics:")
        print(f"   • User Buffer Samples:     {user_stats['total_samples']:8d}")
        if 'avg_utilization' in user_stats:
            print(f"   • User Buffer Utilization: {user_stats['avg_utilization']:8.3f}")
        print(f"   • Deploy Buffer Samples:   {deploy_stats['total_samples']:8d}")
        print(f"   • Deploy Buffer Utilization: {deploy_stats['avg_utilization']:8.3f}")
        
        print("=" * 70)
        
        # Log per-client metrics
        if per_client_records:
            with open(per_client_csv_filepath, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(per_client_records)

        # Collect learning rates if available
        user_policy_lr = None
        user_value_lr = None
        if hasattr(agent, 'get_user_learning_rates'):
            try:
                user_policy_lr, user_value_lr = agent.get_user_learning_rates()
            except Exception:
                pass
        
        csv_buffer.append([
            update, success_rate, avg_delay_per_timestep_s, avg_constraint_violation,
            avg_energy_per_timestep, avg_privacy_cost_per_timestep,
            avg_user_reward_per_timestep, avg_alloc_reward_per_timestep,
            avg_deploy_reward_per_timestep, agent.lagrangian_multiplier.item(),
            coverage_rate, user_policy_lr, user_value_lr
        ])

        # Flush CSV buffer periodically
        if update % 100 == 0 or update == num_updates:
            try:
                num_records_to_flush = len(csv_buffer)
                with open(csv_filepath, 'a', newline='', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(csv_buffer)
                csv_buffer.clear()
                if update != num_updates and num_records_to_flush > 0:
                    print(f"📝 Flushed {num_records_to_flush} metrics to CSV.")
            except IOError as e:
                print(f"❌ Error writing to CSV file: {e}")

        # TensorBoard logging with update index as X-axis (1,2,3,...) instead of global_step
        # writer.add_scalar("charts/SPS", sps, update)
        # writer.add_scalar("performance/avg_delay_s", avg_delay_per_timestep_s, update)
        # writer.add_scalar("performance/avg_constraint_violation", avg_constraint_violation, update)
        # writer.add_scalar("performance/success_rate", success_rate, update)
        # writer.add_scalar("performance/avg_energy", avg_energy_per_timestep, update)
        # writer.add_scalar("performance/avg_privacy_cost", avg_privacy_cost_per_timestep, update)
        #
        # writer.add_scalar("rewards/user_agent", avg_user_reward_per_timestep, update)
        # writer.add_scalar("rewards/allocation_agent", avg_alloc_reward_per_timestep, update)
        # writer.add_scalar("rewards/deployment_agent", avg_deploy_reward_per_timestep, update)
        #
        # writer.add_scalar("charts/lagrangian_multiplier", agent.lagrangian_multiplier.item(), update)
        # writer.add_scalar("deployment/coverage_rate", coverage_rate, update)
        # writer.add_scalar("timing/collection_time", collect_time, update)
        # writer.add_scalar("timing/training_time", train_time, update)
        # writer.add_scalar("timing/total_time", total_time, update)
        #
        # # Buffer statistics log
        # writer.add_scalar("buffer/user_total_samples", user_stats['total_samples'], update)
        # if 'avg_utilization' in user_stats:
        #     writer.add_scalar("buffer/user_utilization", user_stats['avg_utilization'], update)
        # writer.add_scalar("buffer/deployment_total_samples", deploy_stats['total_samples'], update)
        # writer.add_scalar("buffer/deployment_utilization", deploy_stats['avg_utilization'], update)

    # Save final models
    print("\n💾 Saving final models...")
    models_saved = []
    save_path = os.path.join(results_dir, "models")
    os.makedirs(save_path, exist_ok=True)

    config_snapshot_path = os.path.join(results_dir, "config_snapshot.json")
    if os.path.exists(config_snapshot_path):
        shutil.copy(config_snapshot_path, os.path.join(save_path, "config_snapshot.json"))
        print(f"   - Copied config_snapshot.json to models directory.")

    if hasattr(agent, 'user_policy') and hasattr(agent.user_policy, 'state_dict'):
        torch.save(agent.user_policy.state_dict(), os.path.join(save_path, "user_policy.pth"))
        models_saved.append("user_policy")
    if hasattr(agent, 'sac_allocation'):
        if hasattr(agent.sac_allocation, 'actor') and hasattr(agent.sac_allocation.actor, 'state_dict'):
            torch.save(agent.sac_allocation.actor.state_dict(), os.path.join(save_path, "sac_allocation_actor.pth"))
            models_saved.append("sac_allocation_actor")
        if hasattr(agent.sac_allocation, 'q1') and hasattr(agent.sac_allocation.q1, 'state_dict'):
            torch.save(agent.sac_allocation.q1.state_dict(), os.path.join(save_path, "sac_allocation_q1.pth"))
            models_saved.append("sac_allocation_q1")
        if hasattr(agent.sac_allocation, 'q2') and hasattr(agent.sac_allocation.q2, 'state_dict'):
            torch.save(agent.sac_allocation.q2.state_dict(), os.path.join(save_path, "sac_allocation_q2.pth"))
            models_saved.append("sac_allocation_q2")

    if hasattr(agent, 'deployment_policy') and hasattr(agent.deployment_policy, 'state_dict'):
        torch.save(agent.deployment_policy.state_dict(), os.path.join(save_path, "deployment_policy.pth"))
        models_saved.append("deployment_policy")

    if hasattr(agent, 'lagrangian_multiplier'):
        torch.save(agent.lagrangian_multiplier, os.path.join(save_path, "lagrangian_multiplier.pth"))
        models_saved.append("lagrangian_multiplier")

    if 'user_reward_rms' in locals():
        torch.save(user_reward_rms, os.path.join(save_path, "user_reward_rms.pth"))
        models_saved.append("user_reward_rms")
    if 'alloc_reward_rms' in locals():
        torch.save(alloc_reward_rms, os.path.join(save_path, "alloc_reward_rms.pth"))
        models_saved.append("alloc_reward_rms")
    if 'deploy_reward_rms' in locals():
        torch.save(deploy_reward_rms, os.path.join(save_path, "deploy_reward_rms.pth"))
        models_saved.append("deploy_reward_rms")

    if models_saved:
        print(f"✅ Models saved in {save_path}: {', '.join(models_saved)}")
    else:
        print("   (No models to save for this agent type)")

    print("=" * 80)
    print("🎉 Training Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
