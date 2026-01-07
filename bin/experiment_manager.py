#!/usr/bin/env python3
"""
Experiment Manager CLI

Command-line interface for managing Bayesian optimization experiments
using the centralized experiment registry.
"""
import sys
from pathlib import Path
# Add the project root to the path so we can import from experiments
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.registry import get_registry
from experiments.runner import ExperimentRunner

from plot_run import add_plot_interval_args, add_plot_formatting_args, add_plot_iterations_args
from utils_general.experiments.experiment_manager import run_experiment_manager


def add_extra_plot_args(parser_plot):
    add_plot_interval_args(parser_plot)
    add_plot_formatting_args(parser_plot)
    add_plot_iterations_args(parser_plot)


if __name__ == '__main__':
    registry = get_registry()
    runner = ExperimentRunner(registry)
    run_experiment_manager(runner, add_extra_plot_args)
