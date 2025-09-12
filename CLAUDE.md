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
  - `gp_acquisition_dataset_manager.py`: GP-based training datasets
  - `lr_acquisition_dataset_manager.py`: Logistic regression hyperparameter optimization datasets  
  - `dataset_factory.py`: Unified interface for multiple dataset types
  - Supporting classes in `datasets/` for dataset hierarchies including `LogisticRegressionRandomDataset`

- **Utilities** (`utils/`):
  - `utils.py`: Outcome transformations, kernel setup, JSON serialization
  - `nn_utils.py`: Custom PyTorch modules (PointNet layers, pooling strategies)
  - `plot_utils.py`: Plotting utilities

### Training Methods

The codebase supports three main training approaches:
1. **Gittins Index** (`method: gittins`): Pandora's Box Gittins Index (PBGI) acquisition function
2. **Expected Improvement** (`method: mse_ei`): Expected Improvement via MSE loss minimization
3. **Policy Gradient** (`method: policy_gradient`): Direct policy optimization to maximize myopic improvement reward

### Dataset Types

The codebase supports multiple dataset types for training acquisition functions:
1. **Gaussian Process** (`dataset_type: gp`): Traditional GP-based synthetic functions (default)
2. **Logistic Regression** (`dataset_type: logistic_regression`): Hyperparameter optimization for regularized logistic regression

## Common Development Commands

### Environment Setup
```bash
conda create --name nn_bo python=3.12.4
conda activate nn_bo
pip install -r requirements.txt
```

### Core Scripts

#### Single Model Training

**GP Dataset Example:**
```bash
python run_train.py --dimension 1 --lengthscale 0.05 --kernel Matern52 --min_history 1 --max_history 20 --replacement --train_n_candidates 1 --test_n_candidates 1 --train_acquisition_size 8192 --train_samples_size 10000 --test_expansion_factor 1 --test_samples_size 5000 --batch_size 512 --early_stopping --min_delta 0.0 --patience 30 --layer_width 200 --learning_rate 3e-4 --method gittins --lamda 1e-2 --architecture pointnet --epochs 3
```

**Logistic Regression Dataset Example:**
```bash  
python run_train.py --dataset_type logistic_regression --train_samples_size 5000 --test_samples_size 2000 --train_acquisition_size 8000 --batch_size 128 --epochs 200 --layer_width 300 --learning_rate 3e-4 --method gittins --lamda 1e-2 --architecture pointnet --train_n_candidates 5 --test_n_candidates 10 --min_history 1 --max_history 50 --lr_n_samples_range 100 1000 --lr_n_features_range 10 100 --lr_log_lambda_range -6 2 --early_stopping --patience 30
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
- `dataset_type`: Choose between 'gp' (default) or 'logistic_regression'  
- `function_samples_dataset`: Dataset parameters (GP kernel parameters OR logistic regression parameters), dataset sizes
- `acquisition_dataset`: History lengths, candidate counts
- `architecture`: NN architecture (PointNet/Transformer), layer widths, features
- `training`: Loss methods, learning rates, regularization
- `optimizer`: BO optimization settings (restarts, candidates generation)

**Logistic Regression Dataset Parameters:**
- `lr_n_samples_range`: [min, max] number of samples per synthetic dataset (e.g., [50, 2000])
- `lr_n_features_range`: [min, max] number of features per synthetic dataset (e.g., [5, 100])  
- `lr_bias_range`: [min, max] range for bias term b (e.g., [-2.0, 2.0])
- `lr_coefficient_std`: Standard deviation for coefficient vector c (e.g., 1.0)
- `lr_noise_range`: [min, max] range for noise standard deviation σ_y (e.g., [0.01, 1.0])
- `lr_log_lambda_range`: [min, max] log-space range for regularization mapping (e.g., [-6, 2])
- `lr_log_uniform_sampling`: Whether to use log-uniform sampling for parameter ranges (e.g., true)

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