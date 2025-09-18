"""
Logistic regression-specific acquisition dataset manager.

This module implements the logistic regression-specific functionality for creating acquisition datasets,
building on the abstract base class.
"""

import argparse

from acquisition_dataset_base import (
    AcquisitionDatasetManager, add_common_acquisition_dataset_args,
    create_dataset_factory_function
)
from datasets.logistic_regression_dataset import LogisticRegressionRandomDataset


class LogisticRegressionAcquisitionDatasetManager(AcquisitionDatasetManager):
    """Logistic regression-specific acquisition dataset manager."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("logistic_regression", device)
    
    def create_function_samples_dataset(self, **kwargs):
        """Create logistic regression function samples dataset."""
        return LogisticRegressionRandomDataset(**kwargs)
    
    def get_function_samples_config(self, args: argparse.Namespace, device=None):
        """Get logistic regression-specific function samples configuration."""
        # Convert list args to tuples
        lr_n_samples_range = tuple(getattr(args, 'lr_n_samples_range', [50, 2000]))
        lr_n_features_range = tuple(getattr(args, 'lr_n_features_range', [5, 100]))
        lr_bias_range = tuple(getattr(args, 'lr_bias_range', [-2.0, 2.0]))
        lr_noise_range = tuple(getattr(args, 'lr_noise_range', [0.01, 1.0]))
        lr_log_lambda_range = tuple(getattr(args, 'lr_log_lambda_range', [-6, 2]))
        lr_coefficient_std = getattr(args, 'lr_coefficient_std', 1.0)
        lr_log_uniform_sampling = getattr(args, 'lr_log_uniform_sampling', True)

        return dict(
            #### Logistic regression settings
            n_samples_range=lr_n_samples_range,
            n_features_range=lr_n_features_range,
            bias_range=lr_bias_range,
            coefficient_std=lr_coefficient_std,
            noise_range=lr_noise_range,
            log_lambda_range=lr_log_lambda_range,
            log_uniform_sampling=lr_log_uniform_sampling,

            #### Dimension for LR is always 1 (single hyperparameter lambda)
            # dimension=1,

            # #### No models for LR (procedural generation), but needed for compatibility
            # models=[],
            # model_probabilities=None,
            # randomize_params=True,  # LR always randomizes parameters
            # model_sampler=None,  # LR doesn't use model sampling

            #### Dataset size
            train_samples_size=args.train_samples_size,
            test_samples_size=args.test_samples_size,
        )
    
    def get_outcome_transform(self, args: argparse.Namespace, device=None):
        """Get LR-specific outcome transform (always None)."""
        return None  # No outcome transform for logistic regression
    
    def add_dataset_args(self, parser: argparse.ArgumentParser):
        """Add logistic regression-specific arguments to parser."""
        add_logistic_regression_acquisition_dataset_args(parser)
    
    def get_train_test_true_stats_flags(self):
        """LR-specific stats flags - no GP stats."""
        return False, False


def add_lr_args(parser):
    """Add logistic regression-specific arguments to parser."""
    parser.add_argument(
        '--lr_n_samples_range',
        nargs=2,
        type=int,
        default=[50, 2000],
        help='Range (min, max) for number of samples in each logistic regression dataset'
    )
    parser.add_argument(
        '--lr_n_features_range',
        nargs=2,
        type=int,
        default=[5, 100],
        help='Range (min, max) for number of features in each logistic regression dataset'
    )
    parser.add_argument(
        '--lr_bias_range',
        nargs=2,
        type=float,
        default=[-2.0, 2.0],
        help='Range (min, max) for bias term b'
    )
    parser.add_argument(
        '--lr_coefficient_std',
        type=float,
        default=1.0,
        help='Standard deviation for coefficient vector elements'
    )
    parser.add_argument(
        '--lr_noise_range',
        nargs=2,
        type=float,
        default=[0.01, 1.0],
        help='Range (min, max) for noise standard deviation'
    )
    parser.add_argument(
        '--lr_log_lambda_range',
        nargs=2,
        type=float,
        default=[-6, 2],
        help='Range (min, max) for log(lambda) mapping from [0,1] hyperparameter space'
    )
    parser.add_argument(
        '--lr_log_uniform_sampling',
        action='store_true',
        help='Use log-uniform sampling for ranges in logistic regression datasets'
    )


def add_logistic_regression_acquisition_dataset_args(parser):
    """Add logistic regression acquisition dataset arguments to parser."""
    add_lr_args(parser)
    add_common_acquisition_dataset_args(parser)


create_train_test_lr_acq_datasets_from_args = create_dataset_factory_function(
    LogisticRegressionAcquisitionDatasetManager)