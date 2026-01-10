# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements research on using deep reinforcement learning techniques to learn acquisition functions for Bayesian optimization. Instead of relying on surrogate models like Gaussian Processes, this approach trains neural networks end-to-end on synthetic datasets of objective functions to approximate acquisition functions such as Expected Improvement and the Gittins index.

## Core Architecture

### Key Components

- **Bayesian Optimization Loop** (`bayesopt/`): Core BO implementation
  - `bayesopt.py`: Optimizer classes (RandomSearch, GPAcquisitionOptimizer, NNAcquisitionOptimizer)
  - `stable_gittins.py`: Stable Gittins index computations

- **Dataset Generation**: Multiple dataset types for training acquisition functions
  - `dataset_factory.py`: Unified factory interface for creating different dataset types
  - `datasets/acquisition_dataset_manager.py`: Base classes and manager architecture for acquisition datasets
  - `datasets/gp_acquisition_dataset_manager.py`: GP-based synthetic training datasets
  - `datasets/lr_acquisition_dataset_manager.py`: Logistic regression hyperparameter optimization datasets
  - `datasets/hpob_acquisition_dataset_manager.py`: HPO-B benchmark datasets
  - `datasets/cancer_dosage_acquisition_dataset_manager.py`: Cancer dosage optimization datasets
  - Supporting classes in `datasets/` including `acquisition_dataset.py`, `function_samples_dataset.py`, `dataset_with_models.py`, `gp_acquisition_dataset_manager.py`, `logistic_regression_dataset.py`, `hpob_dataset.py`, and `cancer_dosage_dataset.py`

- **Experiment Management**: Centralized experiment registry and orchestration
  - `experiments/`: Core experiment registry system
    - `registry.py`: Central registry for managing experiment configurations
    - `runner.py`: Experiment execution and orchestration
    - `registry.yml`: YAML file containing all experiment configurations
  - `bin/`: Command-line tools (recommended entry points)
    - `experiment_manager.py`: High-level CLI for running registered experiments
    - `activate_env.sh`: Convenience script to activate conda environment

- **Utilities**: Split between two directories:
  - `utils/`: Domain-specific utilities for BO and acquisition functions
    - `utils.py`: Outcome transformations, kernel setup, JSON serialization
    - `nn_utils.py`: Custom PyTorch modules (PointNet layers, pooling strategies)
    - `plot_utils.py`: Plotting utilities for BO experiments
    - `plot_sorting.py`: Utilities for sorting and organizing plot results
    - `exact_gp_computations.py`: Exact GP posterior calculations
    - `constants.py`: Global constants (e.g., HPOB_DATA_DIR)
    - `basic_model_save_utils.py`: Basic model saving utilities
  - `utils_general/`: General-purpose utilities (recently refactored from `utils/`)
    - `utils.py`: General utility functions
    - `nn_utils.py`: General neural network utilities
    - `plot_utils.py`: General plotting utilities
    - `io_utils.py`: I/O helper functions
    - `math_utils.py`: General mathematical utilities
    - `tictoc.py`: Timing utilities
    - `saveable_object.py`: Object serialization/persistence
    - `basic_model_save_utils.py`: Basic model saving/loading utilities
    - `torch_module_save_utils.py`: PyTorch module saving/loading utilities
    - `training/`: Training utilities
      - `single_trainer.py`: Single model trainer class
      - `train_or_test_loop.py`: Training and testing loop implementations
      - `train_utils.py`: General training utilities
    - `experiments/`: SLURM job submission utilities
      - `experiment_config_utils.py`: Configuration management
      - `experiment_manager.py`: Experiment manager utilities
      - `registry.py`: General-purpose experiment registry
      - `runner.py`: General-purpose experiment runner
      - `submit_dependent_jobs.py`: Job dependency handling
      - `job_array.sub`: SLURM job array template
  - `utils_train/`: Training utilities specific to acquisition function networks
    - `acquisition_function_net.py`: Core NN architectures (PointNet, Transformer-based)
    - `acquisition_function_net_constants.py`: Constants for acquisition function networks
    - `train_acquisition_function_net.py`: Training logic and model persistence
    - `model_save_utils.py`: Model saving/loading and configuration parsing
    - `train_or_test_loop.py`: Training/testing loops for acquisition functions
    - `train_utils.py`: Training utilities for acquisition functions

- **Top-level Scripts**: Main entry points for various tasks
  - `single_train.py`: Train a single acquisition function network
  - `single_run.py`: Run a single Bayesian optimization loop
  - `submit_train.py`: Submit multiple NN training jobs to SLURM
  - `submit.py`: Submit complete experiments (training + BO loops) to SLURM
  - `status.py`: Check status of experiments
  - `plot_run.py`: Generate plots for BO experiment results
  - `plot_train.py`: Generate plots for NN training results
  - `plot_combined.py`: Generate combined plots across experiments
  - `plot_1d_objectives.py`: Plot 1D objective function visualizations
  - `get_training_stats.py`: Extract and display training statistics
  - `dataset_factory.py`: Factory for creating different dataset types
  - `single_train_baseline.py`: Baseline training script
  - `g2top.py`: GPU monitoring utility

### Training Methods

The codebase supports three main training approaches:
1. **Gittins Index** (`method: gittins`): Pandora's Box Gittins Index (PBGI) acquisition function
2. **Expected Improvement** (`method: mse_ei`): Expected Improvement via MSE loss minimization
3. **Policy Gradient** (`method: policy_gradient`): Direct policy optimization to maximize myopic improvement reward

### Dataset Types

The codebase supports multiple dataset types for training acquisition functions:
1. **Gaussian Process** (`dataset_type: gp`): Traditional GP-based synthetic functions (default)
2. **Logistic Regression** (`dataset_type: logistic_regression`): Hyperparameter optimization for regularized logistic regression
3. **HPO-B** (`dataset_type: hpob`): Real-world hyperparameter optimization benchmarks from the HPO-B dataset
4. **Cancer Dosage** (`dataset_type: cancer_dosage`): Cancer dosage optimization tasks

## Common Development Commands

### Environment Setup

Initial setup:
```bash
conda create --name nn_bo python=3.12.4
conda activate nn_bo
pip install -r requirements.txt
```

Activating the environment (recommended):
```bash
source ./bin/activate_env.sh
```
Note: Use `source` (or `.`) to ensure activation persists in your shell. The script automatically detects conda installations and activates the `nn_bo` environment.

### Experiment Manager CLI (Recommended)

For most workflows, use the centralized experiment manager instead of calling scripts directly:

```bash
# List available experiments
python bin/experiment_manager.py list

# Show experiment details and commands
python bin/experiment_manager.py show <experiment_name> --commands

# Run an experiment
python bin/experiment_manager.py run <experiment_name>

# Run with recompute options
python bin/experiment_manager.py run <experiment_name> --always-train  # Recompute NN training
python bin/experiment_manager.py run <experiment_name> --recompute-run  # Recompute all BO results

# Check status
python bin/experiment_manager.py status <experiment_name>

# Generate plots
python bin/experiment_manager.py plot <experiment_name>
python bin/experiment_manager.py plot <experiment_name> --type combined_plot --max-iterations-to-plot 20
```

See README.md for complete documentation of the Experiment Manager CLI.

### Core Scripts (Low-Level)

These scripts can be called directly, but `bin/experiment_manager.py` is recommended for most use cases.

#### Single Model Training

**GP Dataset Example:**
```bash
python single_train.py --dimension 1 --lengthscale 0.05 --kernel Matern52 --min_history 1 --max_history 20 --replacement --train_n_candidates 1 --test_n_candidates 1 --train_acquisition_size 8192 --train_samples_size 10000 --test_expansion_factor 1 --test_samples_size 5000 --batch_size 512 --early_stopping --min_delta 0.0 --patience 30 --layer_width 200 --learning_rate 3e-4 --method gittins --lamda 1e-2 --architecture pointnet --epochs 3
```

**Logistic Regression Dataset Example:**
```bash
python single_train.py --dataset_type logistic_regression --train_samples_size 5000 --test_samples_size 2000 --train_acquisition_size 8000 --batch_size 128 --epochs 200 --layer_width 300 --learning_rate 3e-4 --method gittins --lamda 1e-2 --architecture pointnet --train_n_candidates 5 --test_n_candidates 10 --min_history 1 --max_history 50 --lr_n_samples_range 100 1000 --lr_n_features_range 10 100 --lr_log_lambda_range -6 2 --early_stopping --patience 30
```

**HPO-B Dataset Example:**
```bash
python single_train.py --dataset_type hpob --hpob_search_space_id 5970 --min_history 1 --max_history 20 --train_acquisition_size 8000 --batch_size 128 --epochs 4000 --layer_width 16 --learning_rate 3e-4 --method gittins --lamda 1e-2 --architecture pointnet
```

#### Single BO Loop
```bash
python single_run.py --n_initial_samples 1 --n_iter 20 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --nn_model_name v2/model_[hash] --num_restarts 160 --raw_samples 3200 --gen_candidates L-BFGS-B --bo_seed [seed] --objective_gp_seed [seed]
```

#### Batch Experiments
```bash
python submit.py --train_base_config config/train_acqf.yml --train_experiment_config config/train_acqf_experiment_test_simple.yml --run_base_config config/bo_config.yml --n_gp_draws 8 --seed 8 --sweep_name preliminary-test-small --mail user@domain.edu --gres gpu:1
```

#### Generate Plots
```bash
python plot_run.py --train_base_config config/train_acqf.yml --train_experiment_config config/train_acqf_experiment_1dim_example.yml --run_base_config config/bo_config.yml --n_gp_draws 2 --seed 8 --use_rows --use_cols --center_stat mean --plots_group_name test_1dim --plots_name results
```

**Plot Formatting Options:**
- `--add_grid`: Add grid to plots
- `--add_markers`: Add markers at each iteration point
- `--min_regret_for_plot <value>`: Minimum regret value for log-scale plots (default: 1e-6)
- `--max_iterations_to_plot <N>`: Manually specify maximum iterations to display
- Auto-detection: When plotting regret without `--max_iterations_to_plot`, the script automatically determines optimal iteration count based on convergence

#### Check Status
```bash
python status.py --train_base_config config/train_acqf.yml --train_experiment_config config/train_acqf_experiment_test_simple.yml --run_base_config config/bo_config.yml --n_gp_draws 8 --seed 8
```

### Configuration System

The project uses YAML-based hierarchical configuration:

- **Base configs**: `config/train_acqf.yml`, `config/bo_config.yml`
- **Experiment configs**: Override specific parameters for experiments
- **Configuration structure**: Nested parameters with `values` arrays for hyperparameter sweeps

Key configuration sections:
- `dataset_type`: Choose between 'gp' (default), 'logistic_regression', 'hpob', or 'cancer_dosage'
- `function_samples_dataset`: Dataset parameters (GP kernel parameters, logistic regression parameters, HPO-B search space, OR cancer dosage parameters), dataset sizes
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

**HPO-B Dataset Parameters:**
- `hpob_search_space_id`: HPO-B search space identifier (e.g., '5970', '5860', '6766', '4796')
- Note: HPO-B datasets have fixed sizes and do not use `train_samples_size`/`test_samples_size` parameters

### Model Persistence

Trained models are saved with content-based hashing:
- Location: `data/saved_models/v2/model_[hash]/` (or `v2/model_[hash]/` for backward compatibility)
- Contains: model weights, training config, dataset config
- Hash based on: all training hyperparameters and dataset settings

### Data Directory Structure

The `data/` directory contains all runtime-generated files and cached data:
- `data/saved_models/`: Trained acquisition function network models
- `data/bayesopt_results/`: Results from Bayesian optimization runs
- `data/datasets/`: Cached datasets for training acquisition functions
- `data/plots/`: Generated plots and visualizations
- `data/sweeps/`: SLURM job submission files and logs
- `data/hpob-data/`: HPO-B benchmark dataset files (if using HPO-B)
- `data/saved-surrogates/`: Cached surrogate models (if applicable)

## Development Notes

### Recent Codebase Refactoring

The codebase recently underwent a refactoring where general-purpose utilities were moved from `utils/` to `utils_general/`:
- `utils/` now contains domain-specific utilities for BO and acquisition functions
- `utils_general/` contains reusable utilities that could be used across different projects
- When working with utility functions, check both directories to find the appropriate location

### Adding New Parameters

To add new NN training parameters:
1. Add to `get_cmd_options_train_acqf()` in `submit_train.py`
2. Update `validate_single_train_args` and `add_single_train_args_and_return_info` in `utils_train/model_save_utils.py`
3. For architecture params: modify `initialize_module_from_args` in `utils_train/model_save_utils.py`
4. For training params: modify `_get_training_config()` in `utils_train/model_save_utils.py` and `run_train()` in `single_train.py`

### SLURM Integration

The codebase includes automated SLURM job submission through `submit.py`. Job dependencies are handled automatically - NN training jobs are submitted first, followed by BO loop jobs that depend on them.

### Experiment Registry

The centralized experiment registry (`experiments/registry.py`) provides structured management of complex experiments:
- **Configuration Templates**: Reusable experiment configurations
- **Parameter Validation**: Automatic validation of experiment parameters
- **Command Generation**: Automatic generation of command-line arguments
- **Plotting Integration**: Structured plotting configurations with multiple variants

### Dataset Caching

Datasets are cached in `data/datasets/` directory with content-based naming to avoid regeneration of identical datasets across experiments.

### Notes Directory

The `notes/` directory is used for temporary planning documents, analysis files, and other working notes that should not be committed to the repository. This directory is ignored by git.

**When to use the notes directory:**
- Planning documents and implementation proposals (e.g., when using `/plan` or creating design documents)
- Temporary analysis files and visualizations (PDFs, plots, etc.)
- Scratch notes and working documents
- Any other files that are for reference or planning purposes but not part of the project codebase

**Default behavior for Claude Code:**
When asked to create planning documents, proposals, or similar reference materials without a specific path, save them to the `notes/` directory by default.

## Testing

No specific test framework is mentioned in the codebase. Verify changes by running small-scale experiments with `config/train_acqf_experiment_test_simple.yml`.

## Dataset-Specific Notes

### HPO-B Integration
- Requires HPO-B dataset files in the directory specified by `HPOB_DATA_DIR` in `utils/constants.py`
- Supports multiple search spaces: '5970', '5860', '6766', '4796'
- Fixed dataset sizes (train/validation/test splits are predetermined)
- No outcome transforms applied
