"""
Experiment Runner Module

Handles execution of experiments defined in the registry.
"""

import argparse
from typing import Tuple
from utils_general.experiments.runner import ExperimentRunnerBase
from utils_general.cmd_utils import dict_to_cmd_args
from utils_general.utils import get_arg_names
from utils_general.plot_utils import add_plot_args
from plot_run import add_plot_interval_args, add_plot_formatting_args, add_plot_iterations_args


def _get_valid_plot_args_for_type_extended(plot_type: str) -> set:
    """
    Dynamically determine valid plot arguments by introspecting the argument parser.

    Args:
        plot_type: Type of plot ('run_plot', 'train_plot', or 'combined_plot')

    Returns:
        Set of valid argument names (with underscores, as they appear in kwargs)
    """
    parser = argparse.ArgumentParser()

    # All plot types support common plot args
    add_plot_args(parser, add_plot_name_args=True)

    # run_plot and combined_plot support additional args
    if plot_type in ('run_plot', 'combined_plot'):
        add_plot_interval_args(parser)
        add_plot_formatting_args(parser)
        add_plot_iterations_args(parser)

    # Extract argument names using the utility function
    return set(get_arg_names(parser))


class ExperimentRunner(ExperimentRunnerBase):
    def generate_plots(self, name: str, plot_type: str = "run_plot", **plot_kwargs):
        ### Identical code except adds one more plot type: combined_plot
        ### (we don't use combined_plot, but just keep it in this refactor
        ### so that we don't lose any code)
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
        elif plot_type == "combined_plot":
            cmd = self._build_command(
                "plot_combined.py", args,
                include_nn_config=True,
                include_bo_config=True,
                include_seed=True
            )
            # Add seeds and plots configuration
            cmd.extend(args['SEEDS_CFG'].strip('"').split())

            # Set plots_name from experiment config (with _combined suffix)
            if 'plots_name' not in plot_kwargs:
                run_plots_name = args['RUN_PLOTS_NAME'].strip('"')
                plot_kwargs['plots_name'] = f"{run_plots_name}_combined"
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        # Add plots configuration
        cmd.extend(args['PLOTS_CFG'].strip('"').split())

        # Filter plot_kwargs to only include arguments valid for this plot type
        valid_args = _get_valid_plot_args_for_type_extended(plot_type)
        filtered_plot_kwargs = {
            k: v for k, v in plot_kwargs.items() if k in valid_args
        }

        # Add filtered plot-specific arguments dynamically
        cmd.extend(dict_to_cmd_args(filtered_plot_kwargs))

        return cmd
