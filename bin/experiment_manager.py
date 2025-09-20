#!/usr/bin/env python3
"""
Experiment Manager CLI

Command-line interface for managing Bayesian optimization experiments
using the centralized experiment registry.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import from experiments
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.registry import ExperimentRegistry
from experiments.runner import ExperimentRunner


def cmd_list(args):
    """List all available experiments."""
    registry = ExperimentRegistry()
    experiments = registry.list_experiments()
    
    print("Available experiments:")
    for exp_name in sorted(experiments):
        exp_config = registry.get_experiment(exp_name)
        desc = exp_config.get('description', 'No description')
        print(f"  {exp_name}: {desc}")
    
    print(f"\nTotal: {len(experiments)} experiments")


def cmd_list_templates(args):
    """List all available templates."""
    registry = ExperimentRegistry()
    templates = registry.list_templates()
    
    print("Available templates:")
    for template_name in sorted(templates):
        template_config = registry.get_template(template_name)
        desc = template_config.get('description', 'No description')
        print(f"  {template_name}: {desc}")
    
    print(f"\nTotal: {len(templates)} templates")


def cmd_show(args):
    """Show details of a specific experiment."""
    registry = ExperimentRegistry()
    runner = ExperimentRunner(registry)
    
    try:
        exp_config = registry.get_experiment(args.name)
        
        print(f"Experiment: {args.name}")
        print(f"Description: {exp_config.get('description', 'No description')}")
        print(f"Created: {exp_config.get('created_date', 'Unknown')}")
        print()
        
        if args.commands:
            print("Commands that would be executed:")
            print("=" * 50)
            runner.print_experiment_commands(args.name)
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
    
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_status(args):
    """Check status of an experiment."""
    registry = ExperimentRegistry()
    runner = ExperimentRunner(registry)

    try:
        if args.dry_run:
            print("DRY RUN MODE - no commands will be executed")

        if args.name:
            # Check specific experiment
            print(f"Checking status of experiment: {args.name}")
            returncode, stdout, stderr = runner.run_status_check(args.name, dry_run=args.dry_run)

            if stdout:
                print(stdout)
            if stderr:
                print(f"Errors:\n{stderr}")

            return returncode
        else:
            # Check all experiments
            experiments = registry.list_experiments()
            print(f"Checking status of {len(experiments)} experiments...")
            print("=" * 60)

            for exp_name in sorted(experiments):
                print(f"\n{exp_name}:")
                returncode, stdout, stderr = runner.run_status_check(exp_name, dry_run=args.dry_run)

                if returncode == 0:
                    # Parse and summarize stdout
                    lines = stdout.strip().split('\n')
                    if lines:
                        print(f"  Status: {lines[-1] if lines else 'Unknown'}")
                else:
                    print(f"  Error checking status")

    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_run(args):
    """Run an experiment."""
    registry = ExperimentRegistry()
    runner = ExperimentRunner(registry)
    
    try:
        print(f"Running experiment: {args.name}")
        
        if args.dry_run:
            print("DRY RUN MODE - no commands will be executed")
        
        if args.training_only:
            returncode, stdout, stderr = runner.run_training_only(
                args.name,
                dry_run=args.dry_run,
                no_submit=args.no_submit
            )
        else:
            returncode, stdout, stderr = runner.run_experiment(
                args.name, 
                dry_run=args.dry_run, 
                no_submit=args.no_submit
            )
        
        if stdout:
            print(stdout)
        if stderr:
            print(f"Errors:\n{stderr}")
        
        return returncode
    
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_plot(args):
    """Generate plots for an experiment."""
    registry = ExperimentRegistry()
    runner = ExperimentRunner(registry)
    
    try:
        print(f"Generating {args.type} plots for experiment: {args.name}")
        
        if args.dry_run:
            print("DRY RUN MODE - no commands will be executed")
        
        returncode, stdout, stderr = runner.generate_plots(
            args.name,
            plot_type=args.type,
            n_iterations=args.n_iterations,
            center_stat=args.center_stat,
            variant=args.variant,
            dry_run=args.dry_run
        )
        
        if stdout:
            print(stdout)
        if stderr:
            print(f"Errors:\n{stderr}")
        
        return returncode
    
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_commands(args):
    """Show the commands that would be executed for an experiment (like commands.txt)."""
    registry = ExperimentRegistry()
    runner = ExperimentRunner(registry)
    
    try:
        runner.print_experiment_commands(args.name)
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_plot_variants(args):
    """List available plot variants for an experiment."""
    registry = ExperimentRegistry()
    
    try:
        if args.type:
            # Show variants for specific plot type
            variants = registry.list_plot_variants(args.name, args.type)
            print(f"Available plot variants for experiment '{args.name}' ({args.type}):")
            for variant in variants:
                print(f"  {variant}")
        else:
            # Show variants for all plot types
            for plot_type in ['train_acqf', 'bo_experiments']:
                variants = registry.list_plot_variants(args.name, plot_type)
                if variants:
                    print(f"Available plot variants for {plot_type}:")
                    for variant in variants:
                        print(f"  {variant}")
                    print()
    
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment Manager for Bayesian Optimization experiments",
        epilog="Use 'experiment_manager.py <command> --help' for command-specific help."
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
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
    
    # Plot command
    parser_plot = subparsers.add_parser('plot', help='Generate plots for an experiment')
    parser_plot.add_argument('name', help='Name of the experiment')
    parser_plot.add_argument('--type', choices=['bo_experiments', 'train_acqf'], 
                           default='bo_experiments', help='Type of plots to generate')
    parser_plot.add_argument('--variant', default='default',
                           help='Plot configuration variant to use (default: default)')
    parser_plot.add_argument('--n-iterations', type=int, default=30,
                           help='Number of iterations for BO plots')
    parser_plot.add_argument('--center-stat', choices=['mean', 'median'], default='mean',
                           help='Center statistic for plots')
    parser_plot.add_argument('--dry-run', action='store_true',
                           help='Show what would be executed without running')
    
    # Commands command (show commands like commands.txt)
    parser_commands = subparsers.add_parser('commands', help='Show commands for an experiment')
    parser_commands.add_argument('name', help='Name of the experiment')
    
    # Plot variants command
    parser_variants = subparsers.add_parser('plot-variants', help='List available plot variants for an experiment')
    parser_variants.add_argument('name', help='Name of the experiment')
    parser_variants.add_argument('--type', choices=['bo_experiments', 'train_acqf'], 
                                help='Type of plots (show variants for specific type or all if not specified)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Map commands to functions
    command_map = {
        'list': cmd_list,
        'list-templates': cmd_list_templates,
        'show': cmd_show,
        'status': cmd_status,
        'run': cmd_run,
        'plot': cmd_plot,
        'commands': cmd_commands,
        'plot-variants': cmd_plot_variants,
    }
    
    try:
        return command_map[args.command](args) or 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())