#!/usr/bin/env python3
"""
Experiment Manager CLI

Command-line interface for managing Bayesian optimization experiments
using the centralized experiment registry.
"""
import argparse
import subprocess
import sys
import threading
from typing import List, Tuple
from utils_general.plot_utils import add_plot_args
from utils_general.experiments.runner import ExperimentRunnerBase


def add_recompute_args(parser):
    """Add recompute options to argument parser."""
    recompute_group = parser.add_argument_group("Recompute options")
    recompute_group.add_argument(
        '--recompute-run',
        action='store_true',
        help='Recompute/overwrite existing run results (all types)'
    )
    recompute_group.add_argument(
        '--recompute-non-train-only',
        action='store_true',
        help='Recompute/overwrite only non-NN run results'
    )
    return recompute_group


class _ExperimentManagerCLI:
    def __init__(self, runner: ExperimentRunnerBase):
        self.runner = runner
        self.registry = runner.registry
    
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
    
    def _run_command(self, cmd, dry_run=False):
        if dry_run:
            print("DRY RUN MODE - no commands will be executed")
            print(f"Would execute: {' '.join(cmd)}")
        else:
            self._run_command_with_streaming(cmd)

    def cmd_list(self, args):
        """List all available experiments."""
        experiments = self.registry.list_experiments()
        
        print("Available experiments:")
        for exp_name in sorted(experiments):
            exp_config = self.registry.get_experiment(exp_name)
            desc = exp_config.get('description', 'No description')
            print(f"  {exp_name}: {desc}")
        
        print(f"\nTotal: {len(experiments)} experiments")

    def cmd_list_templates(self, args):
        """List all available templates."""
        templates = self.registry.list_templates()
        
        print("Available templates:")
        for template_name in sorted(templates):
            template_config = self.registry.get_template(template_name)
            desc = template_config.get('description', 'No description')
            print(f"  {template_name}: {desc}")
        
        print(f"\nTotal: {len(templates)} templates")

    def cmd_show(self, args):
        """Show details of a specific experiment."""        
        exp_config = self.registry.get_experiment(args.name)
        
        print(f"Experiment: {args.name}")
        print(f"Description: {exp_config.get('description', 'No description')}")
        print(f"Created: {exp_config.get('created_date', 'Unknown')}")
        print()
        
        if args.commands:
            print("Commands that would be executed:")
            print("=" * 50)
            self.runner.print_experiment_commands(args.name)
        else:
            print("Parameters:")
            params = exp_config.get('parameters', {})
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            if exp_config.get('plotting'):
                print("\nPlotting configuration:")
                plotting = exp_config['plotting']
                for plot_type, config in plotting.items():
                    print(f"  {plot_type}:")
                    for key, value in config.items():
                        print(f"    {key}: {value}")

    def cmd_status(self, args):
        """Check status of an experiment."""
        if args.name:
            print(f"Checking status of experiment: {args.name}")
            cmd = self.runner.run_status_check(args.name)
            self._run_command(cmd, dry_run=args.dry_run)
        else:
            # Previously this was implemented, but we don't need it
            raise NotImplementedError(
                "Status check for all experiments is not implemented.")

    def cmd_run(self, args):
        """Run an experiment."""
        print(f"Running experiment: {args.name}")

        # Validate recompute arguments
        if getattr(args, 'recompute_run', False) and getattr(args, 'recompute_non_train_only', False):
            raise ValueError("Error: Cannot specify both --recompute-run and --recompute-non-train-only.")

        if getattr(args, 'no_train', False) and getattr(args, 'always_train', False):
            raise ValueError("Error: Cannot specify both --no-train and --always-train.")

        tmp = dict(
            no_submit=args.no_submit,
            always_train=getattr(args, 'always_train', False),
            recompute_run=getattr(args, 'recompute_run', False),
            recompute_non_train_only=getattr(args, 'recompute_non_train_only', False)
        )

        if args.training_only:
            cmd = self.runner.run_training_only(args.name, **tmp)
        else:
            cmd = self.runner.run_experiment(args.name, **tmp,
                                             no_train=getattr(args, 'no_train', False))

        self._run_command(cmd, dry_run=args.dry_run)

    def cmd_plot(self, args):
        """Generate plots for an experiment."""
        print(f"Generating {args.type} plots for experiment: {args.name}")

        # Extract all plot-specific arguments from args namespace
        # Exclude the command-specific args (name, type, dry_run)
        # Note: argparse automatically converts hyphens to underscores in attribute names,
        # so we keep them as underscores (the plotting scripts expect underscores)
        excluded_args = {'name', 'type', 'dry_run', 'command'}
        plot_kwargs = {
            key: value
            for key, value in vars(args).items()
            if key not in excluded_args and value is not None
        }

        cmd = self.runner.generate_plots(args.name, plot_type=args.type, **plot_kwargs)

        self._run_command(cmd, dry_run=args.dry_run)

    def cmd_commands(self, args):
        """Show the commands that would be executed for an experiment (like commands.txt)."""
        self.runner.print_experiment_commands(args.name)

    def cmd_plot_variants(self, args):
        """List available plot variants for an experiment."""
        if args.type:
            # Show variants for specific plot type
            variants = self.registry.list_plot_variants(args.name, args.type)
            print(f"Available plot variants for experiment '{args.name}' ({args.type}):")
            for variant in variants:
                print(f"  {variant}")
        else:
            # Show variants for all plot types
            for plot_type in ['train_plot', 'run_plot']:
                variants = self.registry.list_plot_variants(args.name, plot_type)
                if variants:
                    print(f"Available plot variants for {plot_type}:")
                    for variant in variants:
                        print(f"  {variant}")
                    print()

    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="Experiment Manager for ML experiments",
            epilog="Use 'experiment_manager.py <command> --help' for command-specific help."
        )
        
        subparsers = parser.add_subparsers(
            dest='command', help='Available commands', required=True)
        
        # List command
        parser_list = subparsers.add_parser('list', help='List all available experiments')
        
        # List templates command
        parser_list_templates = subparsers.add_parser('list-templates', help='List all available templates')
        
        # Show command
        parser_show = subparsers.add_parser('show', help='Show details of a specific experiment')
        parser_show.add_argument('name', help='Name of the experiment')
        parser_show.add_argument('--commands', action='store_true', 
                            help='Show commands instead of configuration')
        
        # Status command
        parser_status = subparsers.add_parser('status', help='Check status of experiments')
        parser_status.add_argument('name', nargs='?', help='Name of specific experiment (optional)')
        parser_status.add_argument('--dry-run', action='store_true',
                                help='Show what would be executed without running')
        
        # Run command
        parser_run = subparsers.add_parser('run', help='Run an experiment')
        parser_run.add_argument('name', help='Name of the experiment')
        parser_run.add_argument('--dry-run', action='store_true',
                            help='Show what would be executed without running')
        parser_run.add_argument('--no-submit', action='store_true',
                            help='Prepare jobs but do not submit to SLURM')
        parser_run.add_argument('--training-only', action='store_true',
                            help='Run only the training part of the experiment')

        # Use shared recompute argument definitions
        add_recompute_args(parser_run)

        # Add training-specific options
        run_training_group = parser_run.add_argument_group('Training options')
        run_training_group.add_argument('--always-train', action='store_true',
                                    help='Recompute/overwrite existing NN training results')
        run_training_group.add_argument('--no-train', action='store_true',
                                    help='If specified, do not train any NNs; '
                                         'only run the experiment runs.')
        
        # Plot command
        parser_plot = subparsers.add_parser('plot', help='Generate plots for an experiment')
        parser_plot.add_argument('name', help='Name of the experiment')
        parser_plot.add_argument('--type', choices=['run_plot', 'train_plot'],
                            default='run_plot', help='Type of plots to generate')
        parser_plot.add_argument('--dry-run', action='store_true',
                            help='Show what would be executed without running')

        # Note: plots_name and plots_group_name are set automatically from experiment config
        add_plot_args(parser_plot, add_plot_name_args=False)
        if self.runner.add_extra_plot_args_func is not None:
            self.runner.add_extra_plot_args_func(parser_plot)
        
        # Commands command (show commands like commands.txt)
        parser_commands = subparsers.add_parser('commands', help='Show commands for an experiment')
        parser_commands.add_argument('name', help='Name of the experiment')
        
        # Plot variants command
        parser_variants = subparsers.add_parser('plot-variants', help='List available plot variants for an experiment')
        parser_variants.add_argument('name', help='Name of the experiment')
        parser_variants.add_argument('--type', choices=['run_plot', 'train_plot'], 
                                    help='Type of plots (show variants for specific type or all if not specified)')
        
        args = parser.parse_args()
        
        # Map commands to functions
        command_map = {
            'list': self.cmd_list,
            'list-templates': self.cmd_list_templates,
            'show': self.cmd_show,
            'status': self.cmd_status,
            'run': self.cmd_run,
            'plot': self.cmd_plot,
            'commands': self.cmd_commands,
            'plot-variants': self.cmd_plot_variants,
        }

        command_map[args.command](args)


def run_experiment_manager(runner: ExperimentRunnerBase):
    """Run the Experiment Manager CLI with the given runner."""
    cli = _ExperimentManagerCLI(runner)
    cli.main()
