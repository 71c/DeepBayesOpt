"""
Experiment Runner Module

Handles execution of experiments defined in the registry.
"""
import argparse
import sys
from typing import Dict, List, Tuple

from utils_general.cmd_utils import dict_to_cmd_args
from utils_general.utils import get_arg_names
from utils_general.plot_utils import add_plot_args
from .registry import ExperimentRegistryBase


def _get_valid_plot_args_for_type(plot_type: str, add_extra_plot_args_func) -> set:
    """
    Dynamically determine valid plot arguments by introspecting the argument parser.

    Args:
        plot_type: Type of plot ('run_plot', 'train_plot', etc.)
        add_extra_plot_args_func: Function that adds extra plot-specific arguments

    Returns:
        Set of valid argument names (with underscores, as they appear in kwargs)
    """
    parser = argparse.ArgumentParser()

    # All plot types support common plot args
    add_plot_args(parser, add_plot_name_args=True)

    # Add extra plot-specific args for run_plot
    if plot_type == 'run_plot' and add_extra_plot_args_func is not None:
        add_extra_plot_args_func(parser)

    # Extract argument names using the utility function
    return set(get_arg_names(parser))


class ExperimentRunnerBase:
    """Handles execution of experiments from the registry."""

    def __init__(self, registry: ExperimentRegistryBase, add_extra_plot_args_func=None):
        """Initialize the runner."""
        self.registry = registry
        self.add_extra_plot_args_func = add_extra_plot_args_func

    def _build_command(self, script: str, args: Dict[str, str],
                      include_nn_config: bool = True,
                      include_bo_config: bool = False,
                      include_seed: bool = False) -> List[str]:
        """
        Build a command list for executing a script with experiment configs.

        Args:
            script: Script name (e.g., "submit.py")
            args: Experiment command arguments from registry
            include_nn_config: Whether to include NN config arguments
            include_bo_config: Whether to include BO config arguments
            include_seed: Whether to include seed argument

        Returns:
            Command list ready for subprocess execution
        """
        cmd = [sys.executable, script]

        if include_nn_config:
            cmd.extend([
                "--train_base_config", "config/train_base_config.yml",
                "--train_experiment_config", args['TRAIN_EXPERIMENT_CFG'].strip('"')
            ])

        if include_bo_config:
            cmd.extend([
                "--run_base_config", "config/run_base_config.yml",
                "--run_experiment_config", args['RUN_EXPERIMENT_CFG'].strip('"')
            ])

        if include_seed:
            cmd.extend(["--seed", args['SEED']])

        return cmd
    
    def run_status_check(self, name: str):
        """Run status check for an experiment using existing status script."""
        args = self.registry.get_experiment_command_args(name)

        # Build the status command
        cmd = self._build_command(
            "status.py", args,
            include_nn_config=True,
            include_bo_config=True,
            include_seed=True
        )

        # Add seeds configuration
        cmd.extend(args['SEEDS_CFG'].strip('"').split())

        return cmd

    def run_experiment(self, name: str, no_submit: bool = False,
                      always_train: bool = False, recompute_run: bool = False,
                      recompute_non_train_only: bool = False, no_train: bool = False):
        """Run an experiment."""
        args = self.registry.get_experiment_command_args(name)

        # Build the experiment command
        cmd = self._build_command(
            "submit.py", args,
            include_nn_config=True,
            include_bo_config=True,
            include_seed=True
        )

        # Add seeds configuration
        cmd.extend(args['SEEDS_CFG'].strip('"').split())

        # Add SLURM configuration
        slurm_cfg = args['SLURM_CFG'].strip('"').split()
        cmd.extend(slurm_cfg)

        # Add optional flags
        if no_submit:
            cmd.append("--no_submit")

        if always_train:
            cmd.append("--always_train")

        if recompute_run:
            cmd.append("--recompute-run")

        if recompute_non_train_only:
            cmd.append("--recompute-non-train-only")

        if no_train:
            cmd.append("--no_train")

        return cmd

    def run_training_only(self, name: str,
                          no_submit: bool = False,
                          always_train: bool = False,
                          recompute_run: bool = False,
                          recompute_non_train_only: bool = False):
        """Run only the training part of an experiment."""
        args = self.registry.get_experiment_command_args(name)

        # Build the training command
        cmd = self._build_command(
            "submit_train.py", args,
            include_nn_config=True,
            include_bo_config=False,
            include_seed=False
        )

        # Add SLURM configuration
        slurm_cfg = args['SLURM_CFG'].strip('"').split()
        cmd.extend(slurm_cfg)

        # Add optional flags
        if no_submit:
            cmd.append("--no_submit")

        if always_train:
            cmd.append("--always_train")

        # Note: recompute_run and recompute_non_train_only don't apply to training-only mode

        return cmd

    def generate_plots(self, name: str, plot_type: str = "run_plot", **plot_kwargs):
        """
        Generate plots for an experiment.

        Args:
            name: Experiment name
            plot_type: Type of plot to generate
            **plot_kwargs: Additional plot arguments (passed directly to plotting script)
                Common arguments: n_iterations, center_stat, variant, max_iterations_to_plot,
                add_grid, add_markers, min_regret_for_plot, plot_mode, alpha,
                interval_of_center, assume_normal, etc.
        """
        args = self.registry.get_experiment_command_args(name)

        # Build base command based on plot type
        if plot_type == "run_plot":
            cmd = self._build_command(
                "plot_run.py", args,
                include_nn_config=True,
                include_bo_config=True,
                include_seed=True
            )
            # Add seeds configuration
            cmd.extend(args['SEEDS_CFG'].strip('"').split())

            # Set plots_name from experiment config
            if 'plots_name' not in plot_kwargs:
                plot_kwargs['plots_name'] = args['RUN_PLOTS_NAME'].strip('"')
        elif plot_type == "train_plot":
            cmd = self._build_command(
                "plot_train.py", args,
                include_nn_config=True,
                include_bo_config=False,
                include_seed=False
            )
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        # Add plots configuration
        cmd.extend(args['PLOTS_CFG'].strip('"').split())

        # Filter plot_kwargs to only include arguments valid for this plot type
        valid_args = _get_valid_plot_args_for_type(plot_type, self.add_extra_plot_args_func)
        filtered_plot_kwargs = {
            k: v for k, v in plot_kwargs.items() if k in valid_args
        }

        # Add filtered plot-specific arguments dynamically
        cmd.extend(dict_to_cmd_args(filtered_plot_kwargs))

        return cmd

    def print_experiment_commands(self, name: str):
        """Print the commands that would be executed for an experiment (like commands.txt)."""
        args = self.registry.get_experiment_command_args(name)
        exp_config = self.registry.get_experiment(name)
        
        print(f"# {exp_config.get('description', name)}")
        print(f"SEED={args['SEED']};")
        print(f"TRAIN_EXPERIMENT_CFG={args['TRAIN_EXPERIMENT_CFG']};")
        print(f"RUN_EXPERIMENT_CFG={args['RUN_EXPERIMENT_CFG']};")
        print(f"SWEEP_NAME={args['SWEEP_NAME']};")
        print(f"SEEDS_CFG={args['SEEDS_CFG']};")
        print(f"PLOTS_GROUP_NAME={args['PLOTS_GROUP_NAME']};")
        print(f"RUN_PLOTS_NAME={args['RUN_PLOTS_NAME']};")
        print()
        print("# Commands:")
        print(f"python status.py {args['TRAIN_CFG']} {args['RUN_CFG']}")
        print(f"python submit.py {args['TRAIN_CFG']} {args['RUN_CFG']} {args['SLURM_CFG']} --no_submit")
        print(f"python submit.py {args['TRAIN_CFG']} {args['RUN_CFG']} {args['SLURM_CFG']}")
        print(f"python submit_train.py {args['TRAIN_CFG']} {args['SLURM_CFG']}")
        print(f"python plot_run.py {args['TRAIN_CFG']} {args['RUN_CFG']} {args['PLOTS_CFG']} --center_stat mean --interval_of_center --plots_name {args['RUN_PLOTS_NAME']} --n_iterations 30")
        print(f"python plot_train.py {args['TRAIN_CFG']} {args['PLOTS_CFG']}")
