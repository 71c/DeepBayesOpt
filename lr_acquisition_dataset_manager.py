"""
Logistic regression-specific acquisition dataset manager.

This module implements the logistic regression-specific functionality for creating acquisition datasets,
building on the abstract base class.
"""

import argparse
from typing import Optional

from acquisition_dataset_base import AcquisitionDatasetManager, add_lamda_args
from datasets.logistic_regression_dataset import LogisticRegressionRandomDataset


class LogisticRegressionAcquisitionDatasetManager(AcquisitionDatasetManager):
    """Logistic regression-specific acquisition dataset manager."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("logistic_regression", device)
    
    def create_function_samples_dataset(self, **kwargs):
        """Create logistic regression function samples dataset."""
        return LogisticRegressionRandomDataset(**kwargs)
    
    def get_dataset_configs(self, args: argparse.Namespace, device=None):
        """Get logistic regression-specific dataset configuration."""
        return get_logistic_regression_acquisition_dataset_configs(args, device)
    
    def add_dataset_args(self, parser: argparse.ArgumentParser):
        """Add logistic regression-specific arguments to parser."""
        add_logistic_regression_acquisition_dataset_args(parser)

    def create_train_test_datasets_helper(
            self,
            args: argparse.Namespace,
            dataset_configs: dict,
            check_cached: bool = False,
            load_dataset: bool = True
        ):
        """LR-specific helper that disables GP stats and handles caching."""
        
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
            get_train_true_stats=False,  # No GP stats for logistic regression
            get_test_true_stats=False,   # No GP stats for logistic regression
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


def get_logistic_regression_acquisition_dataset_configs(args: argparse.Namespace, device=None):
    """Get logistic regression acquisition dataset configuration."""
    
    # Convert list args to tuples
    lr_n_samples_range = tuple(getattr(args, 'lr_n_samples_range', [50, 2000]))
    lr_n_features_range = tuple(getattr(args, 'lr_n_features_range', [5, 100]))
    lr_bias_range = tuple(getattr(args, 'lr_bias_range', [-2.0, 2.0]))
    lr_noise_range = tuple(getattr(args, 'lr_noise_range', [0.01, 1.0]))
    lr_log_lambda_range = tuple(getattr(args, 'lr_log_lambda_range', [-6, 2]))
    lr_coefficient_std = getattr(args, 'lr_coefficient_std', 1.0)
    lr_log_uniform_sampling = getattr(args, 'lr_log_uniform_sampling', False)

    function_samples_config = dict(
        #### Logistic regression settings
        n_samples_range=lr_n_samples_range,
        n_features_range=lr_n_features_range,
        bias_range=lr_bias_range,
        coefficient_std=lr_coefficient_std,
        noise_range=lr_noise_range,
        log_lambda_range=lr_log_lambda_range,
        log_uniform_sampling=lr_log_uniform_sampling,

        #### Dataset size
        train_samples_size=args.train_samples_size,
        test_samples_size=args.test_samples_size,
    )

    acquisition_dataset_config = dict(
        train_acquisition_size=args.train_acquisition_size,
        test_expansion_factor=getattr(args, 'test_expansion_factor', 1),
        fix_train_samples_dataset=True,
        small_test_proportion_of_test=1.0,
        replacement=getattr(args, 'replacement', False),
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
        if getattr(args, 'samples_addition_amount', None) is None:
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
        outcome_transform=None,  # No outcome transform for logistic regression
        standardize_outcomes=getattr(args, 'standardize_dataset_outcomes', False)
    )

    return {
        "function_samples_config": function_samples_config,
        "acquisition_dataset_config": acquisition_dataset_config,
        "n_points_config": n_points_config,
        "dataset_transform_config": dataset_transform_config
    }


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


def add_common_acquisition_dataset_args(parser):
    """Add common acquisition dataset arguments shared across dataset types."""
    ## Dataset Train and Test Size
    parser.add_argument(
        '--train_samples_size', 
        type=int, 
        help='Size of the train samples dataset',
        required=True
    )
    parser.add_argument(
        '--test_samples_size',
        type=int, 
        help='Size of the test samples dataset',
        required=True
    )

    ############################ Acquisition dataset settings ##########################
    parser.add_argument(
        '--train_acquisition_size', 
        type=int, 
        help='Size of the train acqusition dataset that is based on the train samples dataset',
        required=True
    )
    parser.add_argument(
        '--test_expansion_factor',
        type=int,
        default=1,
        help='The factor that the test dataset samples is expanded to get the test acquisition dataset'
    )
    parser.add_argument(
        '--replacement',
        action='store_true',
        help='Whether to sample with replacement for the acquisition dataset. Default is False.'
    )
    parser.add_argument(
        '--train_n_candidates',
        type=int,
        default=15,
        help='Number of candidate points for each item in the train dataset'
    )
    parser.add_argument(
        '--test_n_candidates',
        type=int,
        default=50,
        help='Number of candidate points for each item in the test dataset'
    )
    parser.add_argument(
        '--min_history',
        type=int,
        help='Minimum number of history points.',
        required=True
    )
    parser.add_argument(
        '--max_history',
        type=int,
        help='Maximum number of history points.',
        required=True
    )
    parser.add_argument(
        '--samples_addition_amount',
        type=int,
        help='Number of samples to add to the history points.'
    )
    parser.add_argument(
        '--standardize_dataset_outcomes', 
        action='store_true', 
        help='Whether to standardize the outcomes of the dataset (independently for each item). Default is False'
    )


def add_logistic_regression_acquisition_dataset_args(parser):
    """Add logistic regression acquisition dataset arguments to parser."""
    add_lr_args(parser)
    add_common_acquisition_dataset_args(parser)


def create_train_test_lr_acq_datasets_from_args(
        args, check_cached=False, load_dataset=True):
    """Create logistic regression acquisition datasets from command line arguments."""
    if getattr(args, 'samples_addition_amount', None) is None:
        args.samples_addition_amount = 5
    
    manager = LogisticRegressionAcquisitionDatasetManager(device="cpu")
    return manager.create_train_test_datasets_from_args(
        args, check_cached=check_cached, load_dataset=load_dataset)