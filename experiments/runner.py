"""
Experiment Runner Module

Handles execution of experiments defined in the registry.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils_general.utils import dict_to_cmd_args
from .registry import ExperimentRegistry


class ExperimentRunner:
    """Handles execution of experiments from the registry."""

    def __init__(self, registry: Optional[ExperimentRegistry] = None):
        """Initialize the runner."""
        self.registry = registry or ExperimentRegistry()

    def _build_command(self, script: str, args: Dict[str, str],
                      include_nn_config: bool = True,
                      include_bo_config: bool = False,
                      include_seed: bool = False) -> List[str]:
        """
        Build a command list for executing a script with experiment configs.

        Args:
            script: Script name (e.g., "bo_experiments_gp.py")
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
                "--train_base_config", "config/train_acqf.yml",
                "--train_experiment_config", args['TRAIN_EXPERIMENT_CFG'].strip('"')
            ])

        if include_bo_config:
            cmd.extend([
                "--run_base_config", "config/bo_config.yml",
                "--run_experiment_config", args['RUN_EXPERIMENT_CFG'].strip('"')
            ])

        if include_seed:
            cmd.extend(["--seed", args['SEED']])

        return cmd

    def _run_command_with_streaming(self, cmd: List[str]) -> Tuple[int, str, str]:
        """Run command with real-time output streaming using threads to avoid blocking."""
        stdout_lines = []
        stderr_lines = []

        def stream_output(pipe, output_list, output_file):
            """Read from pipe and write to output file in real-time."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        # Print immediately for real-time streaming
                        print(line.rstrip(), file=output_file, flush=True)
                        output_list.append(line)
            except Exception:
                pass
            finally:
                pipe.close()

        try:
            # Add -u flag to Python commands to force unbuffered output
            modified_cmd = cmd.copy()
            if modified_cmd[0] == sys.executable:
                modified_cmd.insert(1, '-u')

            process = subprocess.Popen(
                modified_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Create threads to read stdout and stderr concurrently
            stdout_thread = threading.Thread(
                target=stream_output,
                args=(process.stdout, stdout_lines, sys.stdout)
            )
            stderr_thread = threading.Thread(
                target=stream_output,
                args=(process.stderr, stderr_lines, sys.stderr)
            )

            # Start threads
            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to complete
            returncode = process.wait()

            # Wait for threads to finish reading all output
            stdout_thread.join()
            stderr_thread.join()

            return returncode, ''.join(stdout_lines), ''.join(stderr_lines)

        except Exception as e:
            return 1, "", f"Error running command: {str(e)}"
    
    def run_status_check(self, name: str, dry_run: bool = False) -> Tuple[int, str, str]:
        """Run status check for an experiment using existing status script."""
        try:
            args = self.registry.get_experiment_command_args(name)

            # Build the status command
            cmd = self._build_command(
                "bo_experiments_gp_status.py", args,
                include_nn_config=True,
                include_bo_config=True,
                include_seed=True
            )

            # Add seeds configuration
            seeds_cfg = args['SEEDS_CFG'].strip('"').split()
            cmd.extend(seeds_cfg)

            if dry_run:
                # In dry run mode, just print the command that would be executed
                cmd_str = ' '.join(cmd)
                return 0, f"Would execute: {cmd_str}", ""
            else:
                return self._run_command_with_streaming(cmd)

        except Exception as e:
            return 1, "", f"Error running status check: {str(e)}"
    
    def run_experiment(self, name: str, dry_run: bool = False, no_submit: bool = False,
                      always_train: bool = False, recompute_bo: bool = False,
                      recompute_non_nn_only: bool = False, no_train: bool = False) -> Tuple[int, str, str]:
        """Run an experiment."""
        try:
            args = self.registry.get_experiment_command_args(name)

            # Build the experiment command
            cmd = self._build_command(
                "bo_experiments_gp.py", args,
                include_nn_config=True,
                include_bo_config=True,
                include_seed=True
            )

            # Add seeds configuration
            seeds_cfg = args['SEEDS_CFG'].strip('"').split()
            cmd.extend(seeds_cfg)

            # Add SLURM configuration
            slurm_cfg = args['SLURM_CFG'].strip('"').split()
            cmd.extend(slurm_cfg)

            # Add optional flags
            if no_submit:
                cmd.append("--no_submit")

            if always_train:
                cmd.append("--always_train")

            if recompute_bo:
                cmd.append("--recompute-bo")

            if recompute_non_nn_only:
                cmd.append("--recompute-non-nn-only")

            if no_train:
                cmd.append("--no_train")

            if dry_run:
                print("Dry run - would execute command:")
                print(" ".join(cmd))
                return 0, " ".join(cmd), ""
            
            return self._run_command_with_streaming(cmd)
            
        except Exception as e:
            return 1, "", f"Error running experiment: {str(e)}"
    
    def run_training_only(self, name: str,
                          dry_run: bool = False,
                          no_submit: bool = False,
                          always_train: bool = False,
                          recompute_bo: bool = False,
                          recompute_non_nn_only: bool = False) -> Tuple[int, str, str]:
        """Run only the training part of an experiment."""
        try:
            args = self.registry.get_experiment_command_args(name)

            # Build the training command
            cmd = self._build_command(
                "train_acqf.py", args,
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

            # Note: recompute_bo and recompute_non_nn_only don't apply to training-only mode

            if dry_run:
                print("Dry run - would execute command:")
                print(" ".join(cmd))
                return 0, " ".join(cmd), ""
            
            return self._run_command_with_streaming(cmd)
            
        except Exception as e:
            return 1, "", f"Error running training: {str(e)}"
    
    def generate_plots(self, name: str, plot_type: str = "bo_experiments",
                      dry_run: bool = False, **plot_kwargs) -> Tuple[int, str, str]:
        """
        Generate plots for an experiment.

        Args:
            name: Experiment name
            plot_type: Type of plot to generate
            dry_run: If True, show command without executing
            **plot_kwargs: Additional plot arguments (passed directly to plotting script)
                Common arguments: n_iterations, center_stat, variant, max_iterations_to_plot,
                add_grid, add_markers, min_regret_for_plot, plot_mode, alpha,
                interval_of_center, assume_normal, etc.
        """
        try:
            args = self.registry.get_experiment_command_args(name)

            # Build base command based on plot type
            if plot_type == "bo_experiments":
                cmd = self._build_command(
                    "bo_experiments_gp_plot.py", args,
                    include_nn_config=True,
                    include_bo_config=True,
                    include_seed=True
                )
                # Add seeds and plots configuration
                cmd.extend(args['SEEDS_CFG'].strip('"').split())
                cmd.extend(args['PLOTS_CFG'].strip('"').split())

                # Set plots_name from experiment config
                if 'plots_name' not in plot_kwargs:
                    plot_kwargs['plots_name'] = args['BO_PLOTS_NAME'].strip('"')

                # Always add interval_of_center for BO plots unless explicitly False
                if 'interval_of_center' not in plot_kwargs:
                    plot_kwargs['interval_of_center'] = True

            elif plot_type == "train_acqf":
                cmd = self._build_command(
                    "train_acqf_plot.py", args,
                    include_nn_config=True,
                    include_bo_config=False,
                    include_seed=False
                )
                # Add plots configuration
                cmd.extend(args['PLOTS_CFG'].strip('"').split())

            elif plot_type == "combined_plot":
                cmd = self._build_command(
                    "bo_experiments_combined_plot.py", args,
                    include_nn_config=True,
                    include_bo_config=True,
                    include_seed=True
                )
                # Add seeds and plots configuration
                cmd.extend(args['SEEDS_CFG'].strip('"').split())
                cmd.extend(args['PLOTS_CFG'].strip('"').split())

                # Set plots_name from experiment config (with _combined suffix)
                if 'plots_name' not in plot_kwargs:
                    bo_plots_name = args['BO_PLOTS_NAME'].strip('"')
                    plot_kwargs['plots_name'] = f"{bo_plots_name}_combined"

                # Always add interval_of_center for combined plots unless explicitly False
                if 'interval_of_center' not in plot_kwargs:
                    plot_kwargs['interval_of_center'] = True

                # Map max_iterations_to_plot to iteration_to_plot for combined_plot
                if 'max_iterations_to_plot' in plot_kwargs and 'iteration_to_plot' not in plot_kwargs:
                    plot_kwargs['iteration_to_plot'] = plot_kwargs.pop('max_iterations_to_plot')

            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            # Add all plot-specific arguments dynamically
            cmd.extend(dict_to_cmd_args(plot_kwargs))

            if dry_run:
                print("Dry run - would execute command:")
                print(" ".join(cmd))
                return 0, " ".join(cmd), ""

            return self._run_command_with_streaming(cmd)
            
        except Exception as e:
            return 1, "", f"Error generating plots: {str(e)}"
    
    def print_experiment_commands(self, name: str):
        """Print the commands that would be executed for an experiment (like commands.txt)."""
        try:
            args = self.registry.get_experiment_command_args(name)
            exp_config = self.registry.get_experiment(name)
            
            print(f"# {exp_config.get('description', name)}")
            print(f"SEED={args['SEED']};")
            print(f"TRAIN_EXPERIMENT_CFG={args['TRAIN_EXPERIMENT_CFG']};")
            print(f"RUN_EXPERIMENT_CFG={args['RUN_EXPERIMENT_CFG']};")
            print(f"SWEEP_NAME={args['SWEEP_NAME']};")
            print(f"SEEDS_CFG={args['SEEDS_CFG']};")
            print(f"PLOTS_GROUP_NAME={args['PLOTS_GROUP_NAME']};")
            print(f"BO_PLOTS_NAME={args['BO_PLOTS_NAME']};")
            print()
            print("# Commands:")
            print(f"python bo_experiments_gp_status.py {args['TRAIN_CFG']} {args['RUN_CFG']}")
            print(f"python bo_experiments_gp.py {args['TRAIN_CFG']} {args['RUN_CFG']} {args['SLURM_CFG']} --no_submit")
            print(f"python bo_experiments_gp.py {args['TRAIN_CFG']} {args['RUN_CFG']} {args['SLURM_CFG']}")
            print(f"python train_acqf.py {args['TRAIN_CFG']} {args['SLURM_CFG']}")
            print(f"python bo_experiments_gp_plot.py {args['TRAIN_CFG']} {args['RUN_CFG']} {args['PLOTS_CFG']} --center_stat mean --interval_of_center --plots_name {args['BO_PLOTS_NAME']} --n_iterations 30")
            print(f"python train_acqf_plot.py {args['TRAIN_CFG']} {args['PLOTS_CFG']}")
            
        except Exception as e:
            print(f"Error generating commands: {str(e)}")