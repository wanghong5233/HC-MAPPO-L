# Safe Multi-Agent Deep Reinforcement Learning for Privacy-Aware Edge-Device Collaborative DNN Inference

Official code repository for the paper: **"Safe Multi-Agent Deep Reinforcement Learning for Privacy-Aware Edge-Device Collaborative DNN Inference"**

## Overview

This repository provides the official implementation and simulation framework for our paper on privacy-aware collaborative inference in mobile edge computing (MEC) systems. The framework implements the Hierarchical Constrained Multi-Agent Proximal Policy Optimization with Lagrangian relaxation (HC-MAPPO-L), a novel safe reinforcement learning algorithm.

Our approach co-optimizes model deployment, user-server association, privacy-aware DNN partitioning, and multi-dimensional resource allocation to balance system cost, energy efficiency, and privacy leakage under a long-term average delay constraint. The emphasis on privacy is motivated by threats like model inversion attacks. For a practical demonstration of this threat, see our accompanying repository: [Deep-Inversion-Attack](https://github.com/wanghong5233/Deep-Inversion-Attack).

## Key Features

- **Three-Layer Hierarchical Architecture**: Decomposes the complex problem into three complementary decision layers: (i) a deployment layer, (ii) an association–partitioning layer, and (iii) a resource allocation layer.
- **Safe MARL Framework**: Integrates adaptive Lagrangian dual updates into the MAPPO algorithm (HC-MAPPO-L) to rigorously enforce long-term Quality of Service (QoS) guarantees.
- **Advanced Policy Networks**: Employs an auto-regressive policy to handle the combinatorial action space of model deployment and an attention-based policy for dynamic, query-based resource allocation.
- **Comprehensive Joint Optimization**: Systematically co-optimizes the interdependent decisions of model caching, user association, model partitioning, and resource management under stringent constraints.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- See `requirements.txt` for the full list of dependencies.

## Installation

```bash
# Clone the repository
git clone https://github.com/wanghong5233/HC-MAPPO-L.git
cd HC-MAPPO-L

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running a Single Experiment

To run an experiment with the default configuration, use the following command:

```bash
# Run our proposed method (HC-MAPPO-L) with a specific random seed
python main.py --agent h_mappo_l --seed 42
```

You can override parameters directly from the command line for quick tests:

```bash
# Example: Change the number of users
python main.py --agent h_mappo_l --seed 42 --config-overrides '{"env.num_users": 50}'
```

### Available Algorithms

The following algorithms from the paper can be run by specifying the `--agent` argument:

- `h_mappo_l`: **HC-MAPPO-L** (our proposed method).
- `hc_ippo_l`: **HC-IPPO-L** (Ablation: Constrained IPPO without centralized training).
- `h_mappo`: **H-MAPPO** (Ablation: Unconstrained MAPPO without the Lagrangian mechanism).
- `h_ippo`: **H-IPPO** (Ablation: Unconstrained IPPO).
- `heuristic_mappo_l`: **Heuristic-MAPPO-L** (Ablation: Learned user policy with heuristic deployment and allocation).
- `greedy`: **Greedy** (Heuristic baseline).
- `local_only`: **Local-Only** (Heuristic baseline).
- `edge_only`: **Edge-Only** (Heuristic baseline).
- `random`: **Random** policy baseline.

## Project Structure

```
.
├── algorithms/          # Algorithm implementations (e.g., HC-MAPPO-L)
├── env/                 # Simulation environment and system models
├── policies/            # Policy network architectures (e.g., auto-regressive, attention)
├── configs/             # Configuration files
│   ├── agent_config.yaml
│   ├── system_config.yaml
│   └── model_configs/   # DNN model profiles (FLOPs, parameters, SSIM scores)
├── utils/               # Utility functions, loggers, etc.
├── main.py              # Main entry point for running experiments
└── requirements.txt     # Python dependencies
```

## Configuration

All system, agent, and model parameters are defined in the YAML and JSON files within the `configs/` directory:

- `system_config.yaml`: Defines system-level parameters, including the number of users/servers, channel conditions, and resource capacities.
- `agent_config.yaml`: Contains all agent hyperparameters, such as learning rates, network dimensions, and PPO-specific settings.
- `model_configs/*.json`: Provides detailed layer-wise profiles for each DNN model, including computational load, parameter sizes, and privacy scores (SSIM).

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
