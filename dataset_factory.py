"""
Unified dataset factory for creating acquisition datasets across different types.

This module provides a single entry point for creating acquisition datasets
regardless of the underlying dataset type (GP, logistic regression, etc.).
"""

import argparse
from typing import Tuple, Any

from gp_acquisition_dataset_manager import create_train_test_gp_acq_datasets_from_args
from lr_acquisition_dataset_manager import create_train_test_lr_acq_datasets_from_args


def create_train_test_acquisition_datasets_from_args(
        args: argparse.Namespace, 
        check_cached: bool = False, 
        load_dataset: bool = True
    ) -> Tuple[Any, Any, Any]:
    """
    Create train/test acquisition datasets based on dataset_type in args.
    
    This is the unified entry point that delegates to the appropriate dataset-specific
    creation function based on the dataset_type parameter.
    
    Args:
        args: Namespace containing dataset configuration including dataset_type
        check_cached: Whether to only check if datasets are cached
        load_dataset: Whether to load existing datasets from disk
        
    Returns:
        Tuple of (train_dataset, test_dataset, small_test_dataset)
    """
    dataset_type = getattr(args, 'dataset_type', 'gp')  # Default to GP for backward compatibility
    
    # Validate required arguments for each dataset type
    _validate_args_for_dataset_type(args, dataset_type)
    
    if dataset_type == 'gp':
        return create_train_test_gp_acq_datasets_from_args(
            args, check_cached=check_cached, load_dataset=load_dataset)
    elif dataset_type == 'logistic_regression':
        return create_train_test_lr_acq_datasets_from_args(
            args, check_cached=check_cached, load_dataset=load_dataset)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. "
                        f"Supported types: ['gp', 'logistic_regression']")


def _validate_args_for_dataset_type(args: argparse.Namespace, dataset_type: str):
    """Validate that required arguments are present for the specified dataset type."""
    if dataset_type == 'gp':
        required_gp_args = ['dimension', 'kernel', 'lengthscale']
        missing_args = [arg for arg in required_gp_args if getattr(args, arg, None) is None]
        if missing_args:
            raise ValueError(f"Missing required GP arguments: {missing_args}")
    elif dataset_type == 'logistic_regression':
        # LR arguments all have defaults, so just check that they are reasonable
        pass  # No strict validation needed since all LR args have sensible defaults


def add_unified_dataset_args(parser: argparse.ArgumentParser):
    """
    Add unified dataset arguments that support all dataset types.
    
    This adds common arguments and optional dataset-specific arguments.
    The dataset_type parameter determines which ones are actually used.
    """
    # Add dataset type selector
    parser.add_argument(
        '--dataset_type',
        choices=['gp', 'logistic_regression'],
        default='gp',
        help='Type of dataset for hyperparameter optimization'
    )
    
    # Add common arguments
    from acquisition_dataset_base import add_lamda_args
    add_lamda_args(parser)
    
    # Add common acquisition dataset arguments that all datasets need
    from lr_acquisition_dataset_manager import add_common_acquisition_dataset_args
    add_common_acquisition_dataset_args(parser)
    
    # Add GP arguments (made optional)
    add_optional_gp_args(parser)
    
    # Add logistic regression arguments  
    add_optional_lr_args(parser)


def add_optional_gp_args(parser):
    """Add GP-specific arguments as optional for unified interface."""
    try:
        parser.add_argument('--dimension', type=int, help='Dimension of the optimization problem')
        parser.add_argument('--kernel', choices=['RBF', 'Matern32', 'Matern52'], help='Kernel function')
        parser.add_argument('--lengthscale', type=float, help='Lengthscale parameter for the kernel')
        parser.add_argument('--outcome_transform', choices=['exp'], help='Outcome transformation')
        parser.add_argument('--sigma', type=float, help='Noise level')
        parser.add_argument('--randomize_params', action='store_true', help='Randomize GP parameters')
    except argparse.ArgumentError:
        pass  # Argument already exists


def add_optional_lr_args(parser):
    """Add logistic regression-specific arguments as optional."""
    try:
        parser.add_argument('--lr_n_samples_range', nargs=2, type=int, default=[50, 2000])
        parser.add_argument('--lr_n_features_range', nargs=2, type=int, default=[5, 100]) 
        parser.add_argument('--lr_bias_range', nargs=2, type=float, default=[-2.0, 2.0])
        parser.add_argument('--lr_coefficient_std', type=float, default=1.0)
        parser.add_argument('--lr_noise_range', nargs=2, type=float, default=[0.01, 1.0])
        parser.add_argument('--lr_log_lambda_range', nargs=2, type=float, default=[-6, 2])
        parser.add_argument('--lr_log_uniform_sampling', action='store_true')
    except argparse.ArgumentError:
        pass  # Argument already exists