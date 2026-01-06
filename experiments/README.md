# Experiment Management System

This directory contains the centralized experiment management system for Bayesian optimization experiments. This system replaces the manual copy-paste workflow from `commands.txt` with an automated, organized approach.

## Overview

The experiment management system consists of:

- **Registry** (`registry.yml`): Central YAML file containing all experiment configurations
- **CLI Tool** (`../bin/experiment_manager.py`): Command-line interface for managing experiments
- **Auto-plotting** (`plot_helper.py`): Automatic plot configuration detection for plotting scripts

## Quick Start

### List available experiments
```bash
python bin/experiment_manager.py list
```

### Run an experiment
```bash
# Check status first
python bin/experiment_manager.py status pointnet_max_history_pbgi_1d

# Run with dry-run to see what would be executed
python bin/experiment_manager.py run pointnet_max_history_pbgi_1d --dry-run

# Actually run the experiment
python bin/experiment_manager.py run pointnet_max_history_pbgi_1d --no-submit
```

### Generate plots
```bash
# Generate BO plots
python bin/experiment_manager.py plot pointnet_max_history_pbgi_1d

# Generate training plots
python bin/experiment_manager.py plot pointnet_max_history_pbgi_1d --type train_acqf
```

### Show experiment details
```bash
# Show configuration
python bin/experiment_manager.py show pointnet_max_history_pbgi_1d

# Show equivalent commands.txt format
python bin/experiment_manager.py commands pointnet_max_history_pbgi_1d
```

## CLI Commands

### `list`
List all available experiments with descriptions.

### `show <experiment_name>`
Display detailed configuration for an experiment.
- `--commands`: Show equivalent commands.txt format instead of configuration

### `status [experiment_name]`
Check experiment status using existing status scripts.
- If no name provided, checks all experiments

### `run <experiment_name>`
Execute an experiment.
- `--dry-run`: Show commands without executing
- `--no-submit`: Prepare jobs but don't submit to SLURM
- `--training-only`: Run only the training part

### `plot <experiment_name>`
Generate plots for an experiment.
- `--type {bo_experiments,train_acqf}`: Type of plots (default: bo_experiments)
- `--n-iterations N`: Number of iterations for BO plots (default: 30)
- `--center-stat {mean,median}`: Center statistic (default: mean)
- `--dry-run`: Show commands without executing

### `commands <experiment_name>`
Show the equivalent commands.txt format for an experiment.

## Auto-Plotting

The plotting scripts (`train_acqf_plot.py` and `bo_experiments_gp_plot.py`) now automatically detect experiment configurations and apply the correct plotting parameters.

### How it works
1. When a plotting script runs, it checks the `train_experiment_config` parameter
2. It matches this against experiments in the registry
3. If found, it automatically sets `PRE`, `ATTR_A`, `ATTR_B`, and `POST` variables
4. If not found, it falls back to default or manual configuration

### Benefits
- No more hardcoded plot configurations in plotting scripts
- No more commenting/uncommenting plot settings
- Automatic detection based on experiment being run
- Maintains backward compatibility

## Registry Structure

The `registry.yml` file contains:

```yaml
experiments:
  experiment_name:
    description: "Human-readable description"
    created_date: "YYYY-MM-DD"
    parameters:
      seed: 8
      train_experiment_config: "path/to/config.yml"
      run_experiment_config: "path/to/config.yml"
      sweep_name: "unique_sweep_name"
      n_seeds: 64
      plots_group_name: "plot_group_name"
      run_plots_name: "plot_name"
    plotting:
      train_acqf:
        pre: [["param1", "param2"]]
        attr_a: ["param3"]
        attr_b: ["param4"]
        post: [["param5"]]
      bo_experiments:
        pre: [["nn.param1", "nn.param2"]]
        attr_a: ["nn.param3"]
        attr_b: ["nn.param4"]
        post: [["nn.param5"], ["other_param"]]

templates:
  template_name:
    description: "Template description"
    parameters: {...}
    plotting: {...}
```

## Migration from commands.txt

The registry currently contains 10 key experiments migrated from `commands.txt`. The original `commands.txt` file is preserved for reference and backward compatibility.

To migrate additional experiments:
1. Add them to `registry.yml` following the existing pattern
2. Include appropriate plotting configurations
3. Test with `python bin/experiment_manager.py commands <name>` to verify

## Backward Compatibility

- All existing scripts continue to work unchanged
- `commands.txt` is preserved and functional
- Plotting scripts fall back to manual configuration if auto-detection fails
- No breaking changes to existing workflow

## Adding New Experiments

1. Add to `registry.yml` with complete configuration
2. Include plotting parameters for both `train_acqf` and `bo_experiments`
3. Test with dry-run before actual execution

Example:
```bash
# Test configuration
python bin/experiment_manager.py show my_new_experiment
python bin/experiment_manager.py commands my_new_experiment

# Test execution
python bin/experiment_manager.py run my_new_experiment --dry-run
```