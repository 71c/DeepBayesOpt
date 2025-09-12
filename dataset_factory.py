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
    
    if dataset_type == 'gp':
        return create_train_test_gp_acq_datasets_from_args(
            args, check_cached=check_cached, load_dataset=load_dataset)
    elif dataset_type == 'logistic_regression':
        return create_train_test_lr_acq_datasets_from_args(
            args, check_cached=check_cached, load_dataset=load_dataset)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. "
                        f"Supported types: ['gp', 'logistic_regression']")


def add_unified_dataset_args(parser: argparse.ArgumentParser):
    """
    Add unified dataset arguments that support all dataset types.
    
    This adds all arguments for both GP and logistic regression datasets.
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
    
    # Add all possible dataset arguments - unused ones will be ignored
    # GP arguments
    try:
        from gp_acquisition_dataset_manager import add_gp_acquisition_dataset_args
        add_gp_acquisition_dataset_args(parser)
    except:
        pass  # If this fails, fall back to basic arguments
    
    # Logistic regression arguments (additional ones not in GP)
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