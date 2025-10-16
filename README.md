# Hierarchical Multi-Agent Reinforcement Learning for Edge Inference

Official code repository for the paper: **"Hierarchical Multi-Agent Reinforcement Learning for Collaborative Edge Inference with Privacy and Energy Constraints"**

## Overview

This repository provides a simulation framework for collaborative edge inference in mobile edge computing (MEC) systems. The framework implements a hierarchical multi-agent reinforcement learning approach that jointly optimizes service deployment, model partitioning, and resource allocation under privacy and energy constraints.

## Key Features

- **Hierarchical Architecture**: Two-level decision framework with server-level deployment and user-level execution policies
- **Multi-Objective Optimization**: Balances delay, energy consumption, and privacy protection
- **Constraint Handling**: Lagrangian-based approach for hard constraint satisfaction
- **Baseline Comparisons**: Includes multiple baseline algorithms for benchmarking

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- See `requirements.txt` for complete dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/HC-MAPPO-L.git
cd HC-MAPPO-L

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running a Single Experiment

```bash
# Run with default configuration
python main.py --agent h_mappo_l --seed 42

# Run with custom parameters
python main.py --agent h_mappo_l --seed 42 --config-overrides '{"env.num_users": 50}'
```

### Available Algorithms

- `h_mappo_l`: Hierarchical MAPPO with Lagrangian constraints (proposed method)
- `hc_ippo_l`: Hierarchical IPPO with Lagrangian constraints
- `mappo_no_constraint`: MAPPO without constraints
- `ippo`: Independent PPO baseline
- `greedy_policy`: Greedy heuristic baseline
- `local_only`: Local-only execution baseline
- `edge_only`: Edge-only execution baseline
- `lru_avg`: LRU caching with average allocation baseline
- `random_policy`: Random policy baseline

## Project Structure

```
.
├── algorithms/          # Algorithm implementations
├── env/                 # Environment and system models
├── policies/            # Policy network architectures
├── configs/             # Configuration files
│   ├── agent_config.yaml
│   ├── system_config.yaml
│   └── model_configs/   # DNN model specifications
├── utils/               # Utility functions
├── main.py              # Main entry point
└── requirements.txt     # Dependencies
```

## Configuration

The system behavior can be configured through YAML files in the `configs/` directory:

- `system_config.yaml`: System-level parameters (number of users, servers, etc.)
- `agent_config.yaml`: Agent hyperparameters (learning rate, network architecture, etc.)
- `model_configs/*.json`: DNN model specifications (layers, FLOPs, parameters, etc.)


## License

This project is licensed under the MIT License - see the LICENSE file for details.





