"""
GP-specific acquisition dataset manager.

This module implements the GP-specific functionality for creating acquisition datasets,
building on the abstract base class.
"""

import argparse
from typing import Optional

from datasets.acquisition_dataset_manager import AcquisitionDatasetManager
from datasets.function_samples_dataset import GaussianProcessRandomDataset
from utils.utils import get_gp, get_kernel, get_standardized_exp_transform
from botorch.utils.types import DEFAULT


# GP-specific constants
GP_GEN_DEVICE = "cpu"
GET_TRAIN_TRUE_GP_STATS = False
GET_TEST_TRUE_GP_STATS = True


class GPAcquisitionDatasetManager(AcquisitionDatasetManager):
    """GP-specific acquisition dataset manager."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("gp", device)
    
    def create_function_samples_dataset(self, **kwargs):
        """Create GP function samples dataset."""
        kwargs.pop('name')
        return GaussianProcessRandomDataset(**kwargs)
    
    def get_function_samples_config(self, args: argparse.Namespace, device=None):
        """Get GP-specific function samples configuration."""
        models = [
            get_gp_model_from_args_no_outcome_transform(
                dimension=args.dimension,
                kernel=args.kernel,
                lengthscale=args.lengthscale,
                add_priors=args.randomize_params,
                add_standardize=False,
                device=device
            )
        ]
        
        return dict(
            #### GP settings
            dimension=args.dimension,
            randomize_params=args.randomize_params,
            xvalue_distribution="uniform",
            observation_noise=False,
            models=models,
            model_probabilities=None,

            #### Dataset size
            train_samples_size=args.train_samples_size,
            test_samples_size=args.test_samples_size,
        )
    
    def get_outcome_transform(self, args: argparse.Namespace, device=None):
        """Get GP-specific outcome transform."""
        outcome_transform, _ = get_outcome_transform_from_args(args, device=device)
        return outcome_transform
    
    def get_train_test_true_stats_flags(self):
        """GP-specific stats flags."""
        return GET_TRAIN_TRUE_GP_STATS, GET_TEST_TRUE_GP_STATS


def get_gp_model_from_args_no_outcome_transform(
        dimension: int,
        kernel: str,
        lengthscale: Optional[float] = None,
        add_priors: bool = True,
        add_standardize: bool = True,
        device=None):
    """Create GP model from arguments without outcome transform."""
    kernel = get_kernel(
        dimension=dimension,
        kernel=kernel,
        add_priors=add_priors,
        lengthscale=lengthscale,
        device=device
    )
    return get_gp(dimension=dimension, observation_noise=False,
                  covar_module=kernel, device=device,
                  outcome_transform=DEFAULT if add_standardize else None)


def get_outcome_transform_from_args(args: argparse.Namespace, name_prefix="", device=None):
    """Get outcome transform from arguments."""
    if name_prefix:
        name_prefix = f"{name_prefix}_"
    octf = getattr(args, 'outcome_transform', None)
    sigma = getattr(args, 'sigma', None)
    octf_name = f'{name_prefix}outcome_transform'
    sigma_name = f'{name_prefix}sigma'
    if octf == 'exp':
        if sigma is None:
            raise ValueError(f"{sigma_name} should be specified if {octf_name}=exp")
        if sigma <= 0:
            raise ValueError(f"{sigma_name} should be positive if {octf_name}=exp")
        outcome_transform = get_standardized_exp_transform(sigma, device=device)
        outcome_transform_args = {'outcome_transform': octf, 'sigma': sigma}
    elif octf is None:
        if sigma is not None:
            raise ValueError(f"{sigma_name} should not be specified if {octf_name}=None")
        outcome_transform = None
        outcome_transform_args = None
    else:
        raise ValueError(f"Invalid outcome_transform: {octf}")
    return outcome_transform, outcome_transform_args


def add_gp_args(parser,
                thing_gp_used_for: str,
                name_prefix="",
                add_randomize_params=False):
    """Add GP-specific arguments to parser."""
    if name_prefix:
        name_prefix = f"{name_prefix}_"
    parser.add_argument(
        f'--{name_prefix}kernel',
        choices=['RBF', 'Matern32', 'Matern52'],
        help=f'Kernel to use for the GP for the {thing_gp_used_for}',
    )
    parser.add_argument(
        f'--{name_prefix}lengthscale', 
        type=float, 
        help=f'Lengthscale of the GP for the {thing_gp_used_for}',
    )
    parser.add_argument(
        f'--{name_prefix}outcome_transform', 
        choices=['exp'], 
        help=f'Outcome transform to apply to the GP for the {thing_gp_used_for}. '
        'E.g., if outcome_transform=exp, then all the y values of the GP are '
        'exponented. If unspecified, then no outcome transform is applied.'
    )
    parser.add_argument(
        f'--{name_prefix}sigma',
        type=float,
        help=f'Value of sigma for exp outcome transform of the GP for the '
        f'{thing_gp_used_for}. Only used if {name_prefix}outcome_transform=exp.'
    )
    if add_randomize_params:
        parser.add_argument(
            f'--{name_prefix}randomize_params', 
            action='store_true',
            help='Set this to randomize the parameters of the GP for the '
                f'{thing_gp_used_for}. Default is False.'
        )
