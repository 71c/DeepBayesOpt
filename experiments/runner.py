"""
Experiment Runner Module

Handles execution of experiments defined in the registry.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .registry import ExperimentRegistry


class ExperimentRunner:
    """Handles execution of experiments from the registry."""
    
    def __init__(self, registry: Optional[ExperimentRegistry] = None):
        """Initialize the runner."""
        self.registry = registry or ExperimentRegistry()
    
    def _run_command_with_streaming(self, cmd: List[str]) -> Tuple[int, str, str]:
        """Run command with real-time output streaming."""
        stdout_lines = []
        stderr_lines = []
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read stdout and stderr in real-time
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break
                
                # Read stdout
                stdout_line = process.stdout.readline()
                if stdout_line:
                    print(stdout_line.rstrip())
                    stdout_lines.append(stdout_line)
                
                # Read stderr
                stderr_line = process.stderr.readline()
                if stderr_line:
                    print(stderr_line.rstrip(), file=sys.stderr)
                    stderr_lines.append(stderr_line)
            
            # Read any remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                print(remaining_stdout.rstrip())
                stdout_lines.append(remaining_stdout)
            if remaining_stderr:
                print(remaining_stderr.rstrip(), file=sys.stderr)
                stderr_lines.append(remaining_stderr)
            
            return process.returncode, ''.join(stdout_lines), ''.join(stderr_lines)
            
        except Exception as e:
            return 1, "", f"Error running command: {str(e)}"
    
    def run_status_check(self, name: str, dry_run: bool = False) -> Tuple[int, str, str]:
        """Run status check for an experiment using existing status script."""
        try:
            args = self.registry.get_experiment_command_args(name)

            # Build the status command
            cmd = [
                sys.executable, "bo_experiments_gp_status.py",
                "--nn_base_config", "config/train_acqf.yml",
                "--nn_experiment_config", args['NN_EXPERIMENT_CFG'].strip('"'),
                "--bo_base_config", "config/bo_config.yml",
                "--bo_experiment_config", args['BO_EXPERIMENT_CFG'].strip('"'),
                "--seed", args['SEED']
            ]

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
    
    def run_experiment(self, name: str, dry_run: bool = False, no_submit: bool = False) -> Tuple[int, str, str]:
        """Run an experiment."""
        try:
            args = self.registry.get_experiment_command_args(name)
            
            # Build the experiment command
            cmd = [
                sys.executable, "bo_experiments_gp.py",
                "--nn_base_config", "config/train_acqf.yml",
                "--nn_experiment_config", args['NN_EXPERIMENT_CFG'].strip('"'),
                "--bo_base_config", "config/bo_config.yml",
                "--bo_experiment_config", args['BO_EXPERIMENT_CFG'].strip('"'),
                "--seed", args['SEED']
            ]
            
            # Add seeds configuration
            seeds_cfg = args['SEEDS_CFG'].strip('"').split()
            cmd.extend(seeds_cfg)
            
            # Add SLURM configuration
            slurm_cfg = args['SLURM_CFG'].strip('"').split()
            cmd.extend(slurm_cfg)
            
            # Add optional flags
            if no_submit:
                cmd.append("--no_submit")
                
            if dry_run:
                print("Dry run - would execute command:")
                print(" ".join(cmd))
                return 0, " ".join(cmd), ""
            
            return self._run_command_with_streaming(cmd)
            
        except Exception as e:
            return 1, "", f"Error running experiment: {str(e)}"
    
    def run_training_only(self, name: str,
                          dry_run: bool = False,
                          no_submit: bool = False) -> Tuple[int, str, str]:
        """Run only the training part of an experiment."""
        try:
            args = self.registry.get_experiment_command_args(name)
            
            # Build the training command
            cmd = [
                sys.executable, "train_acqf.py",
                "--nn_base_config", "config/train_acqf.yml",
                "--nn_experiment_config", args['NN_EXPERIMENT_CFG'].strip('"')
            ]
            
            # Add SLURM configuration
            slurm_cfg = args['SLURM_CFG'].strip('"').split()
            cmd.extend(slurm_cfg)

            # Add optional flags
            if no_submit:
                cmd.append("--no_submit")
            
            if dry_run:
                print("Dry run - would execute command:")
                print(" ".join(cmd))
                return 0, " ".join(cmd), ""
            
            return self._run_command_with_streaming(cmd)
            
        except Exception as e:
            return 1, "", f"Error running training: {str(e)}"
    
    def generate_plots(self, name: str, plot_type: str = "bo_experiments", n_iterations: int = 30, 
                      center_stat: str = "mean", variant: str = "default", dry_run: bool = False) -> Tuple[int, str, str]:
        """Generate plots for an experiment."""
        try:
            args = self.registry.get_experiment_command_args(name)
            
            if plot_type == "bo_experiments":
                cmd = [
                    sys.executable, "bo_experiments_gp_plot.py",
                    "--nn_base_config", "config/train_acqf.yml",
                    "--nn_experiment_config", args['NN_EXPERIMENT_CFG'].strip('"'),
                    "--bo_base_config", "config/bo_config.yml",
                    "--bo_experiment_config", args['BO_EXPERIMENT_CFG'].strip('"'),
                    "--seed", args['SEED']
                ]
                
                # Add seeds configuration
                seeds_cfg = args['SEEDS_CFG'].strip('"').split()
                cmd.extend(seeds_cfg)
                
                # Add plots configuration
                plots_cfg = args['PLOTS_CFG'].strip('"').split()
                cmd.extend(plots_cfg)
                
                # Add BO plots configuration
                bo_plots_name = args['BO_PLOTS_NAME'].strip('"')
                cmd.extend([
                    "--center_stat", center_stat,
                    "--interval_of_center",
                    "--plots_name", bo_plots_name,
                    "--n_iterations", str(n_iterations)
                ])
                
            elif plot_type == "train_acqf":
                cmd = [
                    sys.executable, "train_acqf_plot.py",
                    "--nn_base_config", "config/train_acqf.yml",
                    "--nn_experiment_config", args['NN_EXPERIMENT_CFG'].strip('"')
                ]
                
                # Add plots configuration
                plots_cfg = args['PLOTS_CFG'].strip('"').split()
                cmd.extend(plots_cfg)
                
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
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
            print(f"NN_EXPERIMENT_CFG={args['NN_EXPERIMENT_CFG']};")
            print(f"BO_EXPERIMENT_CFG={args['BO_EXPERIMENT_CFG']};")
            print(f"SWEEP_NAME={args['SWEEP_NAME']};")
            print(f"SEEDS_CFG={args['SEEDS_CFG']};")
            print(f"PLOTS_GROUP_NAME={args['PLOTS_GROUP_NAME']};")
            print(f"BO_PLOTS_NAME={args['BO_PLOTS_NAME']};")
            print()
            print("# Commands:")
            print(f"python bo_experiments_gp_status.py {args['NN_CFG']} {args['BO_CFG']}")
            print(f"python bo_experiments_gp.py {args['NN_CFG']} {args['BO_CFG']} {args['SLURM_CFG']} --no_submit")
            print(f"python bo_experiments_gp.py {args['NN_CFG']} {args['BO_CFG']} {args['SLURM_CFG']}")
            print(f"python train_acqf.py {args['NN_CFG']} {args['SLURM_CFG']}")
            print(f"python bo_experiments_gp_plot.py {args['NN_CFG']} {args['BO_CFG']} {args['PLOTS_CFG']} --center_stat mean --interval_of_center --plots_name {args['BO_PLOTS_NAME']} --n_iterations 30")
            print(f"python train_acqf_plot.py {args['NN_CFG']} {args['PLOTS_CFG']}")
            
        except Exception as e:
            print(f"Error generating commands: {str(e)}")