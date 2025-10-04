"""
Unified dataset factory for creating acquisition datasets across different types.

This module provides a single entry point for creating acquisition datasets
regardless of the underlying dataset type (GP, logistic regression, etc.).
"""

import argparse
from typing import Optional, Tuple, Any, Type

from datasets.gp_acquisition_dataset_manager import GPAcquisitionDatasetManager, add_gp_args
from datasets.hpob_acquisition_dataset_manager import HPOBAcquisitionDatasetManager, add_hpob_args
from datasets.lr_acquisition_dataset_manager import LogisticRegressionAcquisitionDatasetManager, add_lr_args
from datasets.cancer_dosage_acquisition_dataset_manager import CancerDosageAcquisitionDatasetManager, add_cancer_dosage_args
from datasets.acquisition_dataset_manager import AcquisitionDatasetManager
from run_bo import get_arg_names 


# Mapping of dataset types to their respective manager classes
MANAGER_CLASS_MAP: dict[str, Type[AcquisitionDatasetManager]] = {
    'gp': GPAcquisitionDatasetManager,
    'logistic_regression': LogisticRegressionAcquisitionDatasetManager,
    'hpob': HPOBAcquisitionDatasetManager,
    'cancer_dosage': CancerDosageAcquisitionDatasetManager
}


def get_dataset_manager(dataset_type: str, device: str = "cpu") -> AcquisitionDatasetManager:
    """Get the appropriate dataset manager class based on dataset_type."""
    try:
        manager_cls = MANAGER_CLASS_MAP[dataset_type]
    except KeyError:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. "
                         f"Supported types: {list(MANAGER_CLASS_MAP.keys())}")
    return manager_cls(device=device)


def create_train_test_acquisition_datasets_from_args(
        args: argparse.Namespace,
        groups_arg_names=None,
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
    _validate_args_for_dataset_type(args, dataset_type, groups_arg_names)

    if getattr(args, 'samples_addition_amount', None) is None:
        args.samples_addition_amount = 5
    
    manager = get_dataset_manager(dataset_type, device="cpu")
    
    return manager.create_train_test_datasets_from_args(
            args, check_cached=check_cached, load_dataset=load_dataset)


def _validate_args_for_dataset_type(
        args: argparse.Namespace,
        dataset_type: str,
        groups_arg_names: Optional[dict[str, list[str]]]=None):
    """Validate that required arguments are present for the specified dataset type."""
    if dataset_type in {'gp', 'logistic_regression', 'cancer_dosage'}:
        # make sure has test_samples_size and train_samples_size
        if getattr(args, 'train_samples_size', None) is None:
            raise ValueError("Missing required argument: train_samples_size")
        if getattr(args, 'test_samples_size', None) is None:
            raise ValueError("Missing required argument: test_samples_size")
    else:
        # For HPO-B, these args are not needed
        if getattr(args, 'train_samples_size', None) is not None:
            raise ValueError("Argument 'train_samples_size' should not be set for HPO-B datasets")
        if getattr(args, 'test_samples_size', None) is not None:
            raise ValueError("Argument 'test_samples_size' should not be set for HPO-B datasets")

    if dataset_type == 'gp':
        required_gp_args = ['dimension', 'kernel', 'lengthscale']
        missing_args = [arg for arg in required_gp_args if getattr(args, arg, None) is None]
        if missing_args:
            raise ValueError(f"Missing required GP arguments: {missing_args}")
    elif dataset_type == 'cancer_dosage':
        if getattr(args, 'dimension', None) is None:
            raise ValueError("Missing required argument for dataset_type=cancer_dosage: dimension")
    else:
        if getattr(args, 'dimension', None) is not None:
            raise ValueError(f"Argument 'dimension' should not be set for dataset_type '{dataset_type}'")
    
    # TODO: finish implementing this function;
    # ensure proper usage of this function passing in groups_arg_names;
    # make this function have an option of whether to check train_samples_size
    # and test_samples_size (depending on if it is used for dataset or objective
    # function), and make this function be used in the run_bo.py pre_run_bo.


def _add_common_acquisition_dataset_args(parser):
    """Add common acquisition dataset arguments shared across dataset types."""
    ## Dataset Train and Test Size
    parser.add_argument(
        '--train_samples_size', 
        type=int, 
        help='Size of the train samples dataset',
        required=False
    )
    parser.add_argument(
        '--test_samples_size',
        type=int, 
        help='Size of the test samples dataset',
        required=False
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


def add_lamda_args(parser):
    """Add lambda arguments to argument parser."""
    parser.add_argument(
        '--lamda_min',
        type=float,
        help=('Minimum value of lambda (if using variable lambda). '
            'Only used if method=gittins.')
    )
    parser.add_argument(
        '--lamda_max',
        type=float,
        help=('Maximum value of lambda (if using variable lambda). '
            'Only used if method=gittins.')
    )
    parser.add_argument(
        '--lamda',
        type=float,
        help='Value of lambda (if using constant lambda). Only used if method=gittins.'
    )


def add_unified_function_dataset_args(
        parser: argparse.ArgumentParser,
        thing_used_for: str,
        name_prefix: str = "",
        dataset_group: Optional[argparse._ArgumentGroup] = None):
    if dataset_group is None:
        dataset_group = parser
    
    name_prefix_ = f"{name_prefix}_" if name_prefix else name_prefix
    
    # Add dataset type selector
    dataset_group.add_argument(
        f'--{name_prefix_}dataset_type',
        choices=list(MANAGER_CLASS_MAP),
        default='gp',
        help=f'Type of {thing_used_for}'
    )
    
    ret = {}
    
    dataset_group.add_argument(
        f'--{name_prefix_}dimension', type=int, help='(Only for dataset_type=gp or dataset_type=cancer_dosage) Dimension of the optimization problem')
    
    gp_group = parser.add_argument_group('GP dataset arguments')
    add_gp_args(gp_group, thing_used_for, name_prefix=name_prefix, add_randomize_params=True)
    ret['gp'] = get_arg_names(gp_group)
    
    lr_group = parser.add_argument_group('Logistic Regression dataset arguments')
    add_lr_args(lr_group)
    ret['logistic_regression'] = get_arg_names(lr_group)

    hpob_group = parser.add_argument_group('HPO-B arguments')
    add_hpob_args(hpob_group, thing_used_for, name_prefix=name_prefix)
    ret['hpob'] = get_arg_names(hpob_group)

    cancer_dosage_group = parser.add_argument_group('Cancer Dosage dataset arguments')
    add_cancer_dosage_args(cancer_dosage_group, thing_used_for, name_prefix=name_prefix)
    ret['cancer_dosage'] = get_arg_names(cancer_dosage_group)

    return ret


def add_unified_acquisition_dataset_args(
        parser: argparse.ArgumentParser,
        dataset_group: Optional[argparse._ArgumentGroup] = None,
        add_lamda_args_flag: bool = True):
    """
    Add unified dataset arguments that support all dataset types.
    
    This adds common arguments and optional dataset-specific arguments.
    The dataset_type parameter determines which ones are actually used.
    """
    if dataset_group is None:
        dataset_group = parser
    
    if add_lamda_args_flag:
        add_lamda_args(dataset_group)
    
    # Add common acquisition dataset arguments that all datasets need
    _add_common_acquisition_dataset_args(dataset_group)

    add_unified_function_dataset_args(
        parser=parser, thing_used_for="function samples",
        name_prefix="",
        dataset_group=dataset_group
    )


def main():
    """CLI interface for dataset creation."""
    parser = argparse.ArgumentParser(description='Create train/test acquisition datasets')
    groups_arg_names = add_unified_acquisition_dataset_args(parser)
    
    # Add batch size argument for compatibility with old interface
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for the acquisition dataset.'
    )
    
    args = parser.parse_args()
    create_train_test_acquisition_datasets_from_args(
        args, groups_arg_names=groups_arg_names,
        check_cached=False, load_dataset=False)


if __name__ == "__main__":
    main()
