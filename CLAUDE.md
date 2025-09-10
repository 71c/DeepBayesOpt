# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements research on using deep reinforcement learning techniques to learn acquisition functions for Bayesian optimization. Instead of relying on surrogate models like Gaussian Processes, this approach trains neural networks end-to-end on synthetic datasets of objective functions to approximate acquisition functions such as Expected Improvement and the Gittins index.

## Core Architecture

### Key Components

- **Neural Network Acquisition Functions** (`nn_af/`): PyTorch modules for acquisition function networks
  - `acquisition_function_net.py`: Core NN architectures (PointNet, Transformer-based)
  - `train_acquisition_function_net.py`: Training logic and model persistence
  - `acquisition_function_net_save_utils.py`: Model saving/loading and configuration parsing

- **Bayesian Optimization Loop** (`bayesopt/`): Core BO implementation
  - `bayesopt.py`: Optimizer classes (RandomSearch, GPAcquisitionOptimizer, NNAcquisitionOptimizer)
  - `stable_gittins.py`: Stable Gittins index computations

- **Dataset Generation**: Synthetic objective function datasets
  - `gp_acquisition_dataset.py`: Generate GP-based training datasets
  - Supporting classes in `datasets/` for dataset hierarchies

- **Utilities** (`utils/`):
  - `utils.py`: Outcome transformations, kernel setup, JSON serialization
  - `nn_utils.py`: Custom PyTorch modules (PointNet layers, pooling strategies)
  - `plot_utils.py`: Plotting utilities

### Training Methods

The codebase supports three main training approaches:
1. **Gittins Index** (`method: gittins`): Pandora's Box Gittins Index (PBGI) acquisition function
2. **Expected Improvement** (`method: mse_ei`): Expected Improvement via MSE loss minimization
3. **Policy Gradient** (`method: policy_gradient`): Direct policy optimization to maximize myopic improvement reward

## Common Development Commands

### Environment Setup
```bash
conda create --name nn_bo python=3.12.4
conda activate nn_bo
pip install -r requirements.txt
```

### Core Scripts

#### Single Model Training
```bash
python run_train.py --dimension 1 --lengthscale 0.05 --kernel Matern52 --min_history 1 --max_history 20 --replacement --train_n_candidates 1 --test_n_candidates 1 --train_acquisition_size 8192 --train_samples_size 10000 --test_expansion_factor 1 --test_samples_size 5000 --batch_size 512 --early_stopping --min_delta 0.0 --patience 30 --layer_width 200 --learning_rate 3e-4 --method gittins --lamda 1e-2 --architecture pointnet --epochs 3
```

#### Single BO Loop
```bash
python run_bo.py --n_initial_samples 1 --n_iter 20 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --nn_model_name v2/model_[hash] --num_restarts 160 --raw_samples 3200 --gen_candidates L-BFGS-B --bo_seed [seed] --objective_gp_seed [seed]
```

#### Batch Experiments
```bash
python bo_experiments_gp.py --nn_base_config config/train_acqf.yml --nn_experiment_config config/train_acqf_experiment_test_simple.yml --bo_base_config config/bo_config.yml --n_gp_draws 8 --seed 8 --sweep_name preliminary-test-small --mail user@domain.edu --gres gpu:1
```

#### Generate Plots
```bash
python bo_experiments_gp_plot.py --nn_base_config config/train_acqf.yml --nn_experiment_config config/train_acqf_experiment_1dim_example.yml --bo_base_config config/bo_config.yml --n_gp_draws 2 --seed 8 --use_rows --use_cols --center_stat mean --plots_group_name test_1dim --plots_name results
```

#### Check Status
```bash
python bo_experiments_gp_status.py --nn_base_config config/train_acqf.yml --nn_experiment_config config/train_acqf_experiment_test_simple.yml --bo_base_config config/bo_config.yml --n_gp_draws 8 --seed 8
```

### Configuration System

The project uses YAML-based hierarchical configuration:

- **Base configs**: `config/train_acqf.yml`, `config/bo_config.yml`
- **Experiment configs**: Override specific parameters for experiments
- **Configuration structure**: Nested parameters with `values` arrays for hyperparameter sweeps

Key configuration sections:
- `function_samples_dataset`: GP kernel parameters, dataset sizes
- `acquisition_dataset`: History lengths, candidate counts
- `architecture`: NN architecture (PointNet/Transformer), layer widths, features
- `training`: Loss methods, learning rates, regularization
- `optimizer`: BO optimization settings (restarts, candidates generation)

### Model Persistence

Trained models are saved with content-based hashing:
- Location: `v2/model_[hash]/`
- Contains: model weights, training config, dataset config
- Hash based on: all training hyperparameters and dataset settings

## Development Notes

### Adding New Parameters

To add new NN training parameters:
1. Add to `get_cmd_options_train_acqf()` in `train_acqf.py`
2. Update `_parse_af_train_cmd_args()` and `_get_run_train_parser()` in `nn_af/acquisition_function_net_save_utils.py`
3. For architecture params: modify `_get_model()`
4. For training params: modify `_get_training_config()` and `run_train()` in `run_train.py`

### SLURM Integration

The codebase includes automated SLURM job submission through `bo_experiments_gp.py`. Job dependencies are handled automatically - NN training jobs are submitted first, followed by BO loop jobs that depend on them.

### Dataset Caching

Datasets are cached in `datasets/` directory with content-based naming to avoid regeneration of identical datasets across experiments.

## Testing

No specific test framework is mentioned in the codebase. Verify changes by running small-scale experiments with `config/train_acqf_experiment_test_simple.yml`.