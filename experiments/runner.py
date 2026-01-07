"""
Experiment Runner Module

Handles execution of experiments defined in the registry.
"""

from typing import Tuple
from utils_general.experiments.runner import ExperimentRunnerBase
from utils_general.utils import dict_to_cmd_args


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

        # Add all plot-specific arguments dynamically
        cmd.extend(dict_to_cmd_args(plot_kwargs))

        return cmd
