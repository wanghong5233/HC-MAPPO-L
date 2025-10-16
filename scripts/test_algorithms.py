#!/usr/bin/env python3
# Single seed learning algorithm quick comparison test script
# - Only run learning algorithms (default: h_mappo_l, mappo_no_constraint, ippo, lru_avg)
# - Each algorithm in a separate process

import os
import sys
import json
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path to ensure imports from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_LEARNING_ALGOS = [
    "h_mappo_l",
    "mappo_no_constraint",
    "ippo",
    "hc_ippo_l",
    "lru_avg",
]


def build_per_algo_overrides(favor_ours: bool, base_updates: int):
    """Construct per-algorithm overrides (JSON-serializable dict).

    Unified stable baseline: fewer deployment perturbations, more stable sampling (applies to all learning algorithms).
    """
    # Unified base overrides (applies to all algorithms, does not change relative strength)
    base = {
        "agent.num_steps": 200,  # Improve sampling stability per round
    }

    if not favor_ours:
        # Fair setting: strictly do not override any hyperparameters, use original values from config file
        return {
            "h_mappo_l": {},
            "mappo_no_constraint": {},
            "ippo": {},
            "hc_ippo_l": {},  # HC-IPPO-L: use original values from config file
            "lru_avg": {},
        }

    hp_hmappo = {
        "agent.learning_rate": 2e-4,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.02,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
        # Constrain not too strong to avoid suppressing policies
        "agent.lagrangian_lr": 0.01,
        # Allocation layer: moderate reinforcement
        "agent.sac_updates_per_epoch": 16,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.55,
        # Deployment: stable but not overly conservative
        "agent.deploy_ent_coef": 0.12,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.0,
        "agent.size_bias_beta": 0.0,
    }
    # Independent parameters for each comparison algorithm (do not share the same set)
    hp_ippo = {
        "agent.learning_rate": 2e-4,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.08,
        "agent.num_epochs": 4,
        "agent.num_minibatches": 8,
        "agent.sac_updates_per_epoch": 10,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.75,
        "agent.deploy_ent_coef": 0.20,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.2,
        "agent.size_bias_beta": 0.1,
    }
    hc_ippo_l = {
        # HC-IPPO-L: copy IPPO configuration, can be independently modified
        "agent.learning_rate": 2e-4,
        "agent.clip_coef": 0.12,
        # "agent.ent_coef": 0.08,
        "agent.ent_coef": 0.10,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
        # -----------------------------
        "agent.lagrangian_lr": 0.0,
        "agent.lagrangian_init": 0.0,
        "agent.cost_vf_coef": 0.0,
        # -------------------------------
        "agent.sac_updates_per_epoch": 16,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.55,
        "agent.deploy_ent_coef": 0.12,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.0,
        "agent.size_bias_beta": 0.0,

        #-----------------All parameters same as ippo--------------------------------
        # "agent.learning_rate": 2e-4,
        # "agent.clip_coef": 0.12,
        # # "agent.ent_coef": 0.15,
        # # "agent.ent_coef": 0.19,
        # # "agent.ent_coef": 0.10,
        # "agent.ent_coef": 0.2,
        # "agent.num_epochs": 4,
        # "agent.num_minibatches": 8,
        # "agent.sac_updates_per_epoch": 10,
        # "agent.sac_batch_size": 512,
        # "agent.sac_target_entropy_ratio": 0.75,
        # "agent.deploy_ent_coef": 0.20,
        # "weights.deployment_migration_weight": 0.02,
        # "agent.sampling_temperature": 1.2,
        # "agent.size_bias_beta": 0.1,
    }
    hp_mappo_no = {
        # H-MAPPO (no constraint) tuning: increase final return and suppress late decline
        # PPO is more stable
        "agent.learning_rate": 1e-4,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.01,
        "agent.num_epochs": 4,
        "agent.num_minibatches": 8,
        # SAC is stronger, reduce policy noise
        "agent.sac_updates_per_epoch": 20,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.50,
        # Deployment is more stable (reduce late perturbations)
        "agent.deploy_ent_coef": 0.08,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.0,
        "agent.size_bias_beta": 0.0,
    }
    hp_lru_avg = {
        # "agent.learning_rate": 5e-4,
        # "agent.clip_coef": 0.4,
        # "agent.learning_rate": 3e-4,
        "agent.clip_coef": 0.30,
        "agent.ent_coef": 0.05,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
    }

    return {
        "h_mappo_l": {**base, **hp_hmappo},
        "ippo": {**base, **hp_ippo},
        "hc_ippo_l": {**base, **hc_ippo_l},  # HC-IPPO-L: independent configuration, can be modified separately
        "mappo_no_constraint": {**base, **hp_mappo_no},
        "lru_avg": {**base, **hp_lru_avg},
    }


def run_one(algorithm: str, seed: int, max_updates: int, overrides: dict, threads_per_proc: int, results_base_dir: str, run_tag: str):
    """Run one algorithm in a separate subprocess."""
    cmd = [
        sys.executable, "main.py",
        "--agent", algorithm,
        "--seed", str(seed),
        "--experiment-name", "convergence",
        "--max-updates", str(max_updates),
        "--results-base-dir", results_base_dir,
        "--config-overrides", json.dumps(overrides, ensure_ascii=False)
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    # Limit numerical library/BLAS/Torch threads to avoid excessive parallelism
    env["OMP_NUM_THREADS"] = str(threads_per_proc)
    env["MKL_NUM_THREADS"] = str(threads_per_proc)
    env["OPENBLAS_NUM_THREADS"] = str(threads_per_proc)
    env["NUMEXPR_NUM_THREADS"] = str(threads_per_proc)
    env["TORCH_NUM_THREADS"] = str(threads_per_proc)

    print(f"🚀 [{algorithm}] seed={seed} updates={max_updates} tag={run_tag}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env
    )
    ok = (result.returncode == 0)
    if ok:
        print(f"✅ [{algorithm}] completed")
    else:
        print(f"❌ [{algorithm}] failed, returncode={result.returncode}\n{result.stderr}")
    return {
        "algorithm": algorithm,
        "status": "success" if ok else "failed",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description="Single seed learning algorithm quick comparison test")
    parser.add_argument("--seed", type=int, default=None, help="Single random seed (mutually exclusive with --seeds)")
    parser.add_argument("--seeds", type=str, default=None, help="Multiple seeds, comma-separated, e.g., 0,1,2 (mutually exclusive with --seed)")
    parser.add_argument("--max-updates", type=int, default=500, help="Maximum training updates (default 500)")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(DEFAULT_LEARNING_ALGOS),
        help=f"Comma-separated list of algorithms to run, optional: {', '.join(DEFAULT_LEARNING_ALGOS)} (default: {', '.join(DEFAULT_LEARNING_ALGOS)})"
    )
    # Default favor mode: inject independent tuned hyperparameters for each algorithm
    parser.add_argument("--favor-ours", action="store_true", default=True,
                        help="Default enabled: inject independent tuned hyperparameters for each algorithm; if you want to use original config parameters, add --common")
    parser.add_argument("--common", action="store_true", help="Use default config parameters, do not inject any overrides")
    parser.add_argument("--threads-per-proc", type=int, default=2, help="Number of threads per process (default 1)")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands, do not execute")
    parser.add_argument("--results-dir", type=str, default="experiment_result_test", help="Root directory for test results (default experiment_result_test)")
    parser.add_argument("--run-tag", type=str, default="test", help="Run tag, written to experiment-name for identification (default test)")
    # Allow external JSON file or string to provide overrides for each algorithm (highest priority)
    parser.add_argument("--extra-overrides", type=str, default="",
                        help="JSON string or file path: {alg:{key:value}} format, append/overwrite per-algo overrides")
    # Directly pass HC-IPPO-L Lagrangian parameters (for quick experimentation, no JSON escaping needed)
    parser.add_argument("--hc-lagrangian-init", type=float, default=None,
                        help="HC-IPPO-L Lagrangian multiplier initial value (overrides agent.lagrangian_init)")
    parser.add_argument("--hc-lagrangian-lr", type=float, default=None,
                        help="HC-IPPO-L Lagrangian learning rate (overrides agent.lagrangian_lr)")
    # Directly pass HC-IPPO-L cost critic loss coefficient
    parser.add_argument("--hc-cost-vf-coef", type=float, default=None,
                        help="HC-IPPO-L cost value loss coefficient (overrides agent.cost_vf_coef)")
    # Direct switch: policy-value decoupling optimization (only for HC-IPPO-L)
    parser.add_argument("--hc-decouple-critics", action="store_true", help="Enable separate policy and value optimizers for HC-IPPO-L")
    # Direct switch: standardized cost advantage (only for HC-IPPO-L)
    parser.add_argument("--hc-normalize-cost-adv", action="store_true", help="Enable cost advantage normalization for HC-IPPO-L")
    # Directly pass HC-IPPO-L entropy coefficient ent_coef
    parser.add_argument("--hc-ent-coef", type=float, default=None, help="HC-IPPO-L user layer entropy coefficient (overrides agent.ent_coef)")
    # Directly pass HC-IPPO-L learning rate
    parser.add_argument("--hc-user-lr", type=float, default=None, help="HC-IPPO-L user layer exclusive learning rate (overrides agent.user_learning_rate)")
    # Directly pass HC-IPPO-L learning rate scheduler
    parser.add_argument("--hc-lr-scheduler", type=str, default=None, choices=["none", "cosine", "linear"],
                        help="HC-IPPO-L user layer learning rate scheduler (overrides agent.lr_scheduler)")
    parser.add_argument("--hc-lr-end-factor", type=float, default=None, help="HC-IPPO-L user layer learning rate decay end factor (overrides agent.lr_end_factor)")
    # Directly pass HC-IPPO-L scheduling steps Tmax (decoupled from --max-updates)
    parser.add_argument("--hc-lr-tmax", type=int, default=None, help="HC-IPPO-L learning rate scheduling target steps Tmax (overrides agent.lr_T_max)")
    # Compatible aliases (case-insensitive)
    parser.add_argument("--hc-lr-Tmax", dest="hc_lr_tmax", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    algos = [a.strip() for a in args.algorithms.split(',') if a.strip()]
    # Parse seeds
    if args.seeds is not None and args.seed is not None:
        print("❌ --seed and --seeds are mutually exclusive, please provide only one")
        sys.exit(1)
    if args.seeds is not None:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except ValueError:
            print(f"❌ Could not parse --seeds: {args.seeds}")
            sys.exit(1)
    else:
        seeds = [int(args.seed) if args.seed is not None else 0]
    # When --common is specified, force using original config parameters; otherwise default to favor
    favor_mode = not args.common
    per_algo = build_per_algo_overrides(favor_mode, args.max_updates)

    # Apply command line direct HC-IPPO-L Lagrangian parameters
    if args.hc_lagrangian_init is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lagrangian_init"] = args.hc_lagrangian_init
    if args.hc_lagrangian_lr is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lagrangian_lr"] = args.hc_lagrangian_lr
    if args.hc_cost_vf_coef is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.cost_vf_coef"] = args.hc_cost_vf_coef
    if args.hc_decouple_critics:
        per_algo.setdefault("hc_ippo_l", {})["agent.decouple_critics"] = True
    if args.hc_normalize_cost_adv:
        per_algo.setdefault("hc_ippo_l", {})["agent.normalize_cost_advantages"] = True
    if args.hc_ent_coef is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.ent_coef"] = args.hc_ent_coef
    if args.hc_user_lr is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.user_learning_rate"] = args.hc_user_lr
    if args.hc_lr_scheduler is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lr_scheduler"] = args.hc_lr_scheduler
    if args.hc_lr_end_factor is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lr_end_factor"] = args.hc_lr_end_factor
    if args.hc_lr_tmax is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lr_T_max"] = args.hc_lr_tmax

    # Parse extra overrides
    if args.extra_overrides:
        extra = None
        if os.path.isfile(args.extra_overrides):
            with open(args.extra_overrides, 'r', encoding='utf-8') as f:
                extra = json.load(f)
        else:
            try:
                extra = json.loads(args.extra_overrides)
            except json.JSONDecodeError:
                print(f"❌ Could not parse --extra-overrides: {args.extra_overrides}")
                sys.exit(1)
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in per_algo:
                    per_algo[k] = {}
                per_algo[k].update(v or {})

    tasks = []
    for alg in algos:
        overrides = per_algo.get(alg, {})
        overrides = {**overrides}
        for sd in seeds:
            tasks.append((alg, sd, args.max_updates, overrides, args.threads_per_proc))

    print("📋 Planned runs:")
    for alg, seed, upd, ov, th in tasks:
        print(f"  - {alg} | seed={seed} updates={upd} | overrides={ov}")

    if args.dry_run:
        print("🔍 Dry run mode, no execution")
        return

    results = []
    # One process per algorithm
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futs = [ex.submit(run_one, *t, args.results_dir, args.run_tag) for t in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())

    ok = sum(1 for r in results if r["status"] == "success")
    print(f"\n📈 Completed: {ok}/{len(results)} successful")


if __name__ == "__main__":
    main()


