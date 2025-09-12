"""
GP-specific acquisition dataset manager.

This module implements the GP-specific functionality for creating acquisition datasets,
building on the abstract base class.
"""

import argparse
from typing import List, Optional, Any
import torch
from torch.distributions import Distribution
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.gp_regression import SingleTaskGP

from acquisition_dataset_base import AcquisitionDatasetManager
from datasets.function_samples_dataset import GaussianProcessRandomDataset
from utils.utils import get_gp, get_kernel, get_standardized_exp_transform


# GP-specific constants
GET_TRAIN_TRUE_GP_STATS = False
GET_TEST_TRUE_GP_STATS = True


class GPAcquisitionDatasetManager(AcquisitionDatasetManager):
    """GP-specific acquisition dataset manager."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("gp", device)
    
    def create_function_samples_dataset(self, **kwargs):
        """Create GP function samples dataset."""
        return GaussianProcessRandomDataset(**kwargs)
    
    def get_dataset_configs(self, args: argparse.Namespace, device=None):
        """Get GP-specific dataset configuration."""
        return get_gp_acquisition_dataset_configs(args, device)
    
    def add_dataset_args(self, parser: argparse.ArgumentParser):
        """Add GP-specific arguments to parser."""
        add_gp_acquisition_dataset_args(parser)
    
    def create_train_test_datasets_helper(
            self,
            args: argparse.Namespace,
            dataset_configs: dict,
            check_cached: bool = False,
            load_dataset: bool = True
        ):
        """GP-specific helper that sets GP stats flags."""
        
        dataset_kwargs = {
            **dataset_configs["function_samples_config"],
            **dataset_configs["acquisition_dataset_config"],
            **dataset_configs["n_points_config"],
            **dataset_configs["dataset_transform_config"]
        }

        from acquisition_dataset_base import get_lamda_min_max, CACHE_DATASETS, LAZY_TRAIN, LAZY_TEST, FIX_TRAIN_ACQUISITION_DATASET
        from utils.utils import dict_to_str
        
        lamda_min, lamda_max = get_lamda_min_max(args)
        
        # When load_dataset=False, disable caching to avoid None dataset issue
        use_cache = CACHE_DATASETS if load_dataset else False
        
        other_kwargs = dict(        
            get_train_true_stats=GET_TRAIN_TRUE_GP_STATS,
            get_test_true_stats=GET_TEST_TRUE_GP_STATS,
            cache_datasets=use_cache,
            lazy_train=LAZY_TRAIN,
            lazy_test=LAZY_TEST,
            
            batch_size=getattr(args, "batch_size", 64),
            fix_train_acquisition_dataset=FIX_TRAIN_ACQUISITION_DATASET,
            
            y_cand_indices=[0],
            lambda_min=lamda_min,
            lambda_max=lamda_max
        )

        all_kwargs = dict(
            **dataset_kwargs, **other_kwargs,
            check_cached=check_cached, load_dataset=load_dataset
        )

        info_str = dict_to_str(all_kwargs, include_space=False)
        if info_str in self._data_cache:
            return self._data_cache[info_str]

        ret = self.create_train_and_test_acquisition_datasets(**all_kwargs)
        self._data_cache[info_str] = ret

        train_aq_dataset, test_aq_dataset, small_test_aq_dataset = ret

        if not check_cached:
            from datasets.acquisition_dataset import CostAwareAcquisitionDataset, FunctionSamplesAcquisitionDataset
            
            if train_aq_dataset is not None:
                if isinstance(train_aq_dataset, CostAwareAcquisitionDataset):
                    fs_dataset = train_aq_dataset.base_dataset.base_dataset
                elif isinstance(train_aq_dataset, FunctionSamplesAcquisitionDataset):
                    fs_dataset = train_aq_dataset.base_dataset
                else:
                    fs_dataset = None
                if fs_dataset is not None:
                    print("Train function samples dataset size:", len(fs_dataset))
                print("Train      acquisition dataset size:", len(train_aq_dataset),
                    "number of batches:",
                    len(train_aq_dataset) // args.batch_size,
                    len(train_aq_dataset) % args.batch_size)

            if test_aq_dataset is not None:
                print("Test acquisition dataset size:", len(test_aq_dataset),
                      "number of batches:",
                      len(test_aq_dataset) // args.batch_size,
                      len(test_aq_dataset) % args.batch_size)
            
            if small_test_aq_dataset != test_aq_dataset and small_test_aq_dataset is not None:
                print("Small test acquisition dataset size:", len(small_test_aq_dataset),
                      "number of batches:",
                      len(small_test_aq_dataset) // args.batch_size,
                      len(small_test_aq_dataset) % args.batch_size)

        return ret


def get_gp_model_from_args_no_outcome_transform(
        dimension: int,
        kernel: str,
        lengthscale: float,
        add_priors: bool,
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
                  covar_module=kernel, device=device)


def get_outcome_transform(args: argparse.Namespace, name_prefix="", device=None):
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


def get_gp_acquisition_dataset_configs(args: argparse.Namespace, device=None):
    """Get GP acquisition dataset configuration."""
    ###################### GP realization characteristics ##########################
    models = [
        get_gp_model_from_args_no_outcome_transform(
            dimension=args.dimension,
            kernel=args.kernel,
            lengthscale=args.lengthscale,
            add_priors=args.randomize_params,
            device=device
        )
    ]
    outcome_transform, outcome_transform_args = get_outcome_transform(
        args, device=device)

    function_samples_config = dict(
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

    acquisition_dataset_config = dict(
        train_acquisition_size=args.train_acquisition_size,
        test_expansion_factor=args.test_expansion_factor,
        fix_train_samples_dataset=True,
        small_test_proportion_of_test=1.0,
        replacement=args.replacement,
        fix_test_samples_dataset=False,
        fix_test_acquisition_dataset=True,
    )
    if not acquisition_dataset_config['fix_train_samples_dataset']:
        acquisition_dataset_config['replacement'] = False

    ########## Set number of history and candidate points generation ###############
    n_points_config = dict(
        loguniform=False,
        fix_n_candidates=True,
        fix_n_samples=True
    )
    if n_points_config['loguniform']:
        n_points_config['pre_offset'] = 3.0
    if n_points_config['fix_n_candidates']:
        n_points_config = dict(
            train_n_candidates=args.train_n_candidates,
            test_n_candidates=args.test_n_candidates,
            min_history=args.min_history,
            max_history=args.max_history,
            **n_points_config
        )
        if args.samples_addition_amount is None:
            args.samples_addition_amount = 5
        if args.samples_addition_amount != 5:
            n_points_config['samples_addition_amount'] = args.samples_addition_amount
    else:
        n_points_config = dict(
            min_n_candidates=2,
            max_points=30,
            **n_points_config
        )

    dataset_transform_config = dict(
        outcome_transform=outcome_transform,
        standardize_outcomes=args.standardize_dataset_outcomes
    )

    return {
        "function_samples_config": function_samples_config,
        "acquisition_dataset_config": acquisition_dataset_config,
        "n_points_config": n_points_config,
        "dataset_transform_config": dataset_transform_config
    }


def add_gp_args(parser, thing_gp_used_for: str,
                name_prefix="", required=False,
                add_randomize_params=False):
    """Add GP-specific arguments to parser."""
    if name_prefix:
        name_prefix = f"{name_prefix}_"
    parser.add_argument(
        f'--{name_prefix}kernel',
        choices=['RBF', 'Matern32', 'Matern52'],
        help=f'Kernel to use for the GP for the {thing_gp_used_for}',
        required=required
    )
    parser.add_argument(
        f'--{name_prefix}lengthscale', 
        type=float, 
        help=f'Lengthscale of the GP for the {thing_gp_used_for}',
        required=required
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


def add_gp_acquisition_dataset_args(parser):
    """Add GP acquisition dataset arguments to parser."""
    ############################# GP-specific settings #############################
    parser.add_argument(
        '--dimension', 
        type=int, 
        help='Dimension of the optimization problem',
        required=True
    )
    add_gp_args(parser, "dataset", required=True, add_randomize_params=True)
    
    # Import the common function
    from lr_acquisition_dataset_manager import add_common_acquisition_dataset_args
    add_common_acquisition_dataset_args(parser)


def create_train_test_gp_acq_datasets_from_args(
        args, check_cached=False, load_dataset=True):
    """Create GP acquisition datasets from command line arguments."""
    if getattr(args, 'samples_addition_amount', None) is None:
        args.samples_addition_amount = 5
    
    manager = GPAcquisitionDatasetManager(device="cpu")
    return manager.create_train_test_datasets_from_args(
        args, check_cached=check_cached, load_dataset=load_dataset)