"""
Plot Helper Module

Provides automatic plot configuration detection and application for experiment scripts.
This eliminates the need for hardcoded PRE, ATTR_A, ATTR_B configurations in plot scripts.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.registry import ExperimentRegistry


def detect_experiment_from_config(nn_experiment_config: str) -> Optional[str]:
    """
    Detect which experiment is being used based on the nn_experiment_config path.
    Returns the experiment name if found, None otherwise.
    """
    registry = ExperimentRegistry()
    experiments = registry.list_experiments()
    
    # Normalize the config path for comparison
    config_path = nn_experiment_config.strip('"').strip("'")
    
    for exp_name in experiments:
        try:
            exp_config = registry.get_experiment(exp_name)
            exp_nn_config = exp_config['parameters']['nn_experiment_config']
            
            if exp_nn_config == config_path:
                return exp_name
        except:
            continue
    
    return None


def get_plotting_config_for_experiment(experiment_name: str, plot_type: str, variant: str = 'default') -> Optional[Dict[str, Any]]:
    """
    Get the plotting configuration for a specific experiment and plot type.
    
    Args:
        experiment_name: Name of the experiment
        plot_type: 'train_acqf' or 'bo_experiments'
        variant: Plot configuration variant (default: 'default')
    
    Returns:
        Dictionary with plotting configuration or None if not found
    """
    try:
        registry = ExperimentRegistry()
        plot_config = registry.get_plotting_config(experiment_name, plot_type, variant)
        return plot_config
    except:
        return None


def auto_configure_plotting(nn_experiment_config: str, plot_type: str, variant: str = 'default') -> Tuple[List, List, List]:
    """
    Automatically configure plotting parameters based on experiment configuration.
    
    Args:
        nn_experiment_config: Path to the NN experiment config file
        plot_type: 'train_acqf' or 'bo_experiments'
        variant: Plot configuration variant (default: 'default')
    
    Returns:
        Tuple of (PRE, ATTR_A, ATTR_B) lists
    """
    # Try to detect the experiment
    experiment_name = detect_experiment_from_config(nn_experiment_config)
    
    if experiment_name:
        plot_config = get_plotting_config_for_experiment(experiment_name, plot_type, variant)
        
        if plot_config:
            pre = plot_config.get('pre', [])
            attr_a = plot_config.get('attr_a', [])
            attr_b = plot_config.get('attr_b', [])
            
            print(f"Auto-detected experiment: {experiment_name}")
            print(f"Using plotting configuration for {plot_type} (variant: {variant}):")
            print(f"  PRE = {pre}")
            print(f"  ATTR_A = {attr_a}")
            print(f"  ATTR_B = {attr_b}")
            
            return pre, attr_a, attr_b
    
    # Fallback to default configuration if no experiment found
    print(f"Could not auto-detect experiment for config: {nn_experiment_config}")
    print("Using default plotting configuration")
    
    # Return sensible defaults
    if plot_type == "train_acqf":
        return [], ["train_samples_size"], ["learning_rate"]
    else:  # bo_experiments
        return [], ["nn.train_samples_size"], ["nn.learning_rate"]


def apply_auto_plotting(nn_experiment_config: str, plot_type: str, 
                       globals_dict: Dict[str, Any],
                       variant: str = 'default') -> None:
    """
    Apply automatic plotting configuration to a script's global variables.
    
    This function modifies the global PRE, ATTR_A, ATTR_B variables in place.
    
    Args:
        nn_experiment_config: Path to the NN experiment config file
        plot_type: 'train_acqf' or 'bo_experiments'  
        globals_dict: The global namespace dictionary of the calling script
    """
    pre, attr_a, attr_b = auto_configure_plotting(nn_experiment_config, plot_type, variant)
    
    # Update the global variables
    globals_dict['PRE'] = pre
    globals_dict['ATTR_A'] = attr_a
    globals_dict['ATTR_B'] = attr_b


def get_experiment_post_config(experiment_name: str, plot_type: str, variant: str = 'default') -> Optional[List]:
    """
    Get the POST configuration for an experiment if it exists.
    
    Args:
        experiment_name: Name of the experiment
        plot_type: 'train_acqf' or 'bo_experiments'
        variant: Plot configuration variant (default: 'default')
    
    Returns:
        POST configuration list or None if not found
    """
    plot_config = get_plotting_config_for_experiment(experiment_name, plot_type, variant)
    if plot_config:
        return plot_config.get('post', None)
    return None


# Convenience function for scripts to use
def setup_plotting_from_args(args, plot_type: str, globals_dict: Dict[str, Any]) -> None:
    """
    Set up plotting configuration from command line arguments.
    
    This is designed to be called from plotting scripts that use argparse.
    
    Args:
        args: Parsed arguments from argparse
        plot_type: 'train_acqf' or 'bo_experiments'
        globals_dict: The global namespace dictionary of the calling script
    """
    nn_config = getattr(args, 'nn_experiment_config', None)
    variant = getattr(args, 'variant', 'default')
    if nn_config:
        apply_auto_plotting(nn_config, plot_type, globals_dict, variant=variant)

        # Also try to set POST if available
        experiment_name = detect_experiment_from_config(nn_config)
        if experiment_name:
            post_config = get_experiment_post_config(experiment_name, plot_type, variant)
            if post_config:
                globals_dict['POST'] = post_config
            else:
                globals_dict['POST'] = []  # Default empty if not specified
