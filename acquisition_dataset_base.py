"""
Abstract base class for acquisition dataset creation and management.

This module contains the core functionality for creating, caching, and managing
acquisition datasets that is shared across different dataset types (GP, logistic regression, etc.).
"""

import argparse
import os
import sys
import traceback
import math
from typing import Any, List, Optional, Union
from abc import ABC, abstractmethod

from botorch.models.transforms.outcome import OutcomeTransform

from utils.utils import (
    dict_to_hash, dict_to_str, 
    get_uniform_randint_generator, get_loguniform_randint_generator,
    get_lengths_from_proportions)
from utils.constants import DATASETS_DIR

from datasets.function_samples_dataset import ListMapFunctionSamplesDataset
from datasets.acquisition_dataset import (
    AcquisitionDataset, CostAwareAcquisitionDataset, FunctionSamplesAcquisitionDataset)

from nn_af.train_acquisition_function_net import train_or_test_loop


# Global configuration constants
CACHE_DATASETS = True
LAZY_TRAIN = True
LAZY_TEST = True
FIX_TRAIN_ACQUISITION_DATASET = False
DEBUG = False


def add_lamda_args(parser):
    """Add lambda arguments to argument parser."""
    try:
        parser.add_argument(
            '--lamda_min',
            type=float,
            help=('Minimum value of lambda (if using variable lambda). '
                'Only used if method=gittins.')
        )
    except argparse.ArgumentError:
        pass  # Argument already exists
    
    try:
        parser.add_argument(
            '--lamda_max',
            type=float,
            help=('Maximum value of lambda (if using variable lambda). '
                'Only used if method=gittins.')
        )
    except argparse.ArgumentError:
        pass  # Argument already exists
    
    try:
        parser.add_argument(
            '--lamda',
            type=float,
            help='Value of lambda (if using constant lambda). Only used if method=gittins.'
        )
    except argparse.ArgumentError:
        pass  # Argument already exists


def get_lamda_min_max(args: argparse.Namespace):
    """Extract lambda min/max values from arguments."""
    lamda = getattr(args, 'lamda', None)
    lamda_min = lamda if lamda is not None else args.lamda_min
    lamda_max = getattr(args, 'lamda_max', None)
    return lamda_min, lamda_max


def get_n_datapoints_random_gen_fixed_n_candidates(
        loguniform=True, pre_offset=None,
        min_history=1, max_history=8, n_candidates=50):
    """Generate function for random number of datapoints with fixed candidate count."""
    if loguniform:
        if pre_offset is None:
            pre_offset = 3.0
        return get_loguniform_randint_generator(
            min_history, max_history,
            pre_offset=pre_offset, offset=n_candidates)
    else:
        if pre_offset is not None:
            raise ValueError(
                "pre_offset should not be specified for uniform randint.")
        return get_uniform_randint_generator(
            n_candidates+min_history, n_candidates+max_history)


def get_n_datapoints_random_gen_variable_n_candidates(
        loguniform=True, pre_offset=None,
        min_n_candidates=2, max_points=30):
    """Generate function for random number of datapoints with variable candidate count."""
    if min_n_candidates is None:
        min_n_candidates = 2
    if max_points is None:
        max_points = 30
    min_points = min_n_candidates + 1
    if loguniform:
        if pre_offset is None:
            pre_offset = 3.0
        return get_loguniform_randint_generator(
            min_points, max_points, pre_offset=pre_offset, offset=0)
    else:
        if pre_offset is not None:
            raise ValueError(
                "pre_offset should not be specified for uniform randint.")
        return get_uniform_randint_generator(min_points, max_points)


class AcquisitionDatasetManager(ABC):
    """
    Abstract base class for managing acquisition dataset creation across different dataset types.
    
    This class provides the core functionality for caching, creating, and managing acquisition
    datasets while allowing subclasses to implement dataset-specific functionality.
    """
    
    def __init__(self, dataset_type: str, device: str = "cpu"):
        self.dataset_type = dataset_type
        self.device = device
        self._data_cache = {}
    
    @abstractmethod
    def create_function_samples_dataset(self, **kwargs):
        """Create the underlying function samples dataset (GP, logistic regression, etc.)."""
        pass
    
    @abstractmethod
    def get_dataset_configs(self, args: argparse.Namespace, device=None):
        """Get dataset configuration dictionary for this dataset type."""
        pass
    
    @abstractmethod
    def add_dataset_args(self, parser: argparse.ArgumentParser):
        """Add dataset-specific arguments to argument parser.""" 
        pass
    
    def create_acquisition_dataset(
            self,
            samples_size: int,
            acquisition_size: int,
            
            # Dataset-specific parameters will be in kwargs
            outcome_transform: Optional[OutcomeTransform] = None,
            standardize_outcomes: bool = False,
            
            # n_datapoints_kwargs
            loguniform: bool = True, 
            pre_offset: Optional[float] = None,
            min_history: Optional[int] = None, 
            max_history: Optional[int] = None,
            samples_addition_amount: Optional[int] = None,
            n_candidates: Optional[int] = None,
            min_n_candidates: Optional[int] = None, 
            max_points: Optional[int] = None,

            # Only used for Gittins index training
            lambda_min: Optional[float] = None,
            lambda_max: Optional[float] = None,

            # Whether to fix the number of samples and draw random history
            fix_n_samples: bool = True,

            y_cand_indices: Union[str, List[int]] = "all",
            give_improvements: bool = False,
            
            fix_function_samples: bool = False, 
            fix_acquisition_samples: bool = False,
            lazy: bool = True, 
            cache: bool = True,
            
            # For caching
            batch_size: Optional[int] = None, 
            get_true_stats: Optional[bool] = None,
            name: str = "",

            replacement: bool = False,
            
            # Control flags
            check_cached: bool = False,
            load_dataset: bool = True,
            
            **dataset_specific_kwargs
        ):
        """
        Create acquisition dataset with caching and persistence.
        
        This method contains the core logic for dataset creation, caching, and saving
        that is shared across all dataset types.
        """
        if check_cached:
            if load_dataset:
                raise ValueError("load_dataset should be False if check_cached is True.")
            if not cache:
                raise ValueError("cache should be True if check_cached is True.")

        # Validation
        for param, value in [
            ("cache", cache), ("lazy", lazy), ("fix_function_samples", fix_function_samples),
            ("fix_acquisition_samples", fix_acquisition_samples), 
            ("standardize_outcomes", standardize_outcomes), ("give_improvements", give_improvements)
        ]:
            if not isinstance(value, bool):
                raise ValueError(f"{param} should be a boolean.")
        
        # Determine n_datapoints configuration
        if min_n_candidates is None and max_points is None:
            fix_n_candidates = True
            if min_history is None:
                min_history = 1
            if max_history is None:
                max_history = 8
            if samples_addition_amount is None:
                samples_addition_amount = 5
            if n_candidates is None:
                n_candidates = 50
            if fix_n_samples:
                n_datapoints_kwargs = dict(
                    loguniform=False, pre_offset=None,
                    min_history=max_history + samples_addition_amount,
                    max_history=max_history + samples_addition_amount,
                    n_candidates=n_candidates)
            else:
                n_datapoints_kwargs = dict(
                    loguniform=loguniform, pre_offset=pre_offset,
                    min_history=min_history, max_history=max_history,
                    n_candidates=n_candidates)
        elif min_history is None and max_history is None and n_candidates is None:
            fix_n_candidates = False
            if min_n_candidates is None:
                min_n_candidates = 2
            n_datapoints_kwargs = dict(
                loguniform=loguniform, pre_offset=pre_offset,
                min_n_candidates=min_n_candidates, max_points=max_points)
        else:
            raise ValueError(
                "Either min_n_candidates and max_points or min_history, " 
                "max_history and n_candidates should be specified.")
        
        fix_n_samples = fix_n_candidates and fix_n_samples
        
        # Prepare dataset configuration
        function_dataset_kwargs = dict(
            **dataset_specific_kwargs, **n_datapoints_kwargs)
        
        aq_dataset_already_saved = False
        aq_dataset = None
        
        # Define paths and names for caching
        if cache:
            name_ = name + "_" if name != "" else name

            function_dataset_hash = dict_to_hash(function_dataset_kwargs)
            base_name = f"{name_}base_size={samples_size}_{function_dataset_hash}"

            aq_dataset_extra_info = dict(
                fix_acquisition_samples=fix_acquisition_samples,
                acquisition_size=acquisition_size,
                outcome_transform=outcome_transform,
                standardize_outcomes=standardize_outcomes,
                give_improvements=give_improvements,
                fix_n_samples=fix_n_samples,
                y_cand_indices=y_cand_indices,
                lambda_min=lambda_min,
                lambda_max=lambda_max)
            if fix_n_samples:
                aq_dataset_extra_info = dict(
                    **aq_dataset_extra_info,
                    min_history=min_history, max_history=max_history)
            
            aq_dataset_extra_info_str = dict_to_hash(aq_dataset_extra_info)
            aq_dataset_name = f"{base_name}_acquisition_{aq_dataset_extra_info_str}"
            aq_dataset_path = os.path.join(DATASETS_DIR, aq_dataset_name)

            aq_dataset_already_saved = os.path.exists(aq_dataset_path)
            
            if load_dataset and aq_dataset_already_saved:
                try:
                    aq_dataset = AcquisitionDataset.load(aq_dataset_path)
                except:
                    print("Error when loading acquisition function dataset:",
                          file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    aq_dataset_already_saved = False
                if aq_dataset_already_saved:
                    assert aq_dataset.data_is_fixed == fix_acquisition_samples
            
            if fix_acquisition_samples:
                if batch_size is None or get_true_stats is None:
                    raise ValueError(
                        "Both batch_size and get_true_stats must be specified " 
                        "if cache=True and fix_acquisition_samples=True")
                fix_function_samples = True
            
            function_dataset_name = f"{base_name}_{self.dataset_type}_samples"
            function_dataset_path = os.path.join(DATASETS_DIR, function_dataset_name)
        
        # Check whether function samples dataset already exists
        function_dataset_already_saved = cache and os.path.exists(function_dataset_path)

        # check_cached=True means to only check whether the datasets are already cached
        if check_cached:
            if fix_acquisition_samples:
                return aq_dataset_already_saved
            return function_dataset_already_saved or aq_dataset_already_saved

        # If we don't have the acquisition dataset saved, generate it
        if not aq_dataset_already_saved:
            # Generate or load the function samples dataset
            create_function_samples_dataset = True
            if cache and fix_function_samples and function_dataset_already_saved:
                create_function_samples_dataset = False
                if load_dataset or fix_acquisition_samples:
                    try:
                        function_samples_dataset = ListMapFunctionSamplesDataset.load(
                            function_dataset_path)
                    except:
                        print("Error when loading function samples dataset:",
                              file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)
                        create_function_samples_dataset = True
                else:
                    function_samples_dataset = None
            
            if create_function_samples_dataset:
                f = get_n_datapoints_random_gen_fixed_n_candidates if fix_n_candidates \
                    else get_n_datapoints_random_gen_variable_n_candidates
                n_datapoints_random_gen = f(**n_datapoints_kwargs)
                
                # Use subclass method to create dataset-specific function samples dataset
                function_samples_dataset = self.create_function_samples_dataset(
                    dataset_size=samples_size,
                    n_datapoints_random_gen=n_datapoints_random_gen,
                    device=self.device,
                    **dataset_specific_kwargs)
                
                if fix_function_samples:
                    function_samples_dataset = function_samples_dataset.fix_samples(
                        lazy=False if cache else lazy)
            
            if function_samples_dataset is None:
                aq_dataset = None
            else:
                # Save the function samples dataset
                if cache and fix_function_samples and not function_dataset_already_saved:
                    os.makedirs(DATASETS_DIR, exist_ok=True)
                    print(f"Saving {function_dataset_name}")
                    function_samples_dataset.save(function_dataset_path, verbose=True)
                
                # Transform the function samples dataset
                if outcome_transform is not None:
                    function_samples_dataset = function_samples_dataset.transform_outcomes(
                        outcome_transform)
                if standardize_outcomes:
                    function_samples_dataset = function_samples_dataset.standardize_outcomes()

                # Create the acquisition dataset
                if fix_n_candidates:
                    extra_kwargs = dict(n_candidate_points=n_candidates)
                    if fix_n_samples:
                        extra_kwargs = dict(
                            **extra_kwargs,
                            min_history=min_history, max_history=max_history)
                else:
                    extra_kwargs = dict(n_candidate_points="uniform",
                                        min_n_candidates=min_n_candidates)
                aq_dataset = FunctionSamplesAcquisitionDataset(
                    function_samples_dataset,
                    n_samples="uniform" if fix_n_samples else "all",
                    give_improvements=give_improvements,
                    acquisition_size=acquisition_size,
                    replacement=replacement,
                    y_cand_indices=y_cand_indices,
                    **extra_kwargs)

                # Add the lambdas if Gittins index
                if not (lambda_min is None and lambda_max is None):
                    if lambda_max is not None and lambda_min is None:
                        raise ValueError(
                            "lambda_min must be specified if lambda_max is specified.")
                    aq_dataset = CostAwareAcquisitionDataset(
                        aq_dataset, lambda_min=lambda_min, lambda_max=lambda_max)

        # Fix the acquisition samples and save the acquisition dataset
        if aq_dataset is not None and fix_acquisition_samples:
            if not aq_dataset.data_is_fixed:
                aq_dataset = aq_dataset.fix_samples(lazy=False if cache else lazy)
            if cache:
                dataloader = aq_dataset.get_dataloader(
                    batch_size=batch_size, drop_last=False)
                desc = f"Getting{' ' if name != '' else ''}{name} dataset stats to cache"
                # result==None iff all the requested dataset stats are already cached
                result = train_or_test_loop(dataloader, verbose=True, desc=desc,
                                            get_true_gp_stats=get_true_stats,
                                            return_none=True)
                if not (aq_dataset_already_saved and result is None):
                    os.makedirs(DATASETS_DIR, exist_ok=True)
                    print(f"Saving AF dataset: {aq_dataset_name}")
                    aq_dataset.save(aq_dataset_path, verbose=True)
        
        return aq_dataset

    def create_train_and_test_acquisition_datasets(
            self,
            train_samples_size: int,
            train_acquisition_size: int,
            fix_train_samples_dataset: bool,

            test_samples_size: int,
            small_test_proportion_of_test: float,
            fix_test_samples_dataset: bool,
            fix_test_acquisition_dataset: bool,
            
            get_train_true_stats: bool,
            get_test_true_stats: bool,
            cache_datasets: bool,
            lazy_train: bool,
            lazy_test: bool,
            
            batch_size: int,
            fix_train_acquisition_dataset: bool,

            replacement: bool = False,
            outcome_transform: Optional[OutcomeTransform] = None,
            standardize_outcomes: bool = False,

            # Only used for Gittins index training
            lambda_min: Optional[float] = None,
            lambda_max: Optional[float] = None,

            test_expansion_factor: int = 1,
            
            loguniform: bool = True, 
            pre_offset: Optional[float] = None, 
            fix_n_candidates: bool = True,
            train_n_candidates: Optional[int] = None, 
            test_n_candidates: Optional[int] = None,
            min_history: Optional[int] = None, 
            max_history: Optional[int] = None,
            samples_addition_amount: Optional[int] = None,
            min_n_candidates: Optional[int] = None, 
            max_points: Optional[int] = None,
            
            fix_n_samples: Optional[bool] = None,
            y_cand_indices: Union[str, List[int]] = "all",
            
            check_cached: bool = False,
            load_dataset: bool = True,
            
            **dataset_specific_kwargs
        ):
        """Create training and test acquisition datasets."""
        
        if fix_n_candidates:
            if samples_addition_amount is None:
                samples_addition_amount = 5
            train_n_points_kwargs = dict(min_history=min_history, max_history=max_history,
                                         samples_addition_amount=samples_addition_amount,
                                         n_candidates=train_n_candidates)
            test_n_points_kwargs = dict(min_history=min_history, max_history=max_history,
                                        samples_addition_amount=samples_addition_amount,
                                        n_candidates=test_n_candidates)
        else:
            train_n_points_kwargs = dict(min_n_candidates=min_n_candidates, max_points=max_points)
            test_n_points_kwargs = train_n_points_kwargs

        common_kwargs = dict(
            outcome_transform=outcome_transform,
            standardize_outcomes=standardize_outcomes,
            lambda_min=lambda_min, lambda_max=lambda_max,
            loguniform=loguniform,
            pre_offset=pre_offset if loguniform else None, 
            batch_size=batch_size,
            cache=cache_datasets,
            give_improvements=False,
            fix_n_samples=fix_n_samples,
            y_cand_indices=y_cand_indices,
            replacement=replacement,
            check_cached=check_cached,
            load_dataset=load_dataset,
            **dataset_specific_kwargs)

        train_aq_dataset = self.create_acquisition_dataset(
            train_samples_size, lazy=lazy_train,
            fix_function_samples=fix_train_samples_dataset,
            fix_acquisition_samples=fix_train_acquisition_dataset,
            get_true_stats=get_train_true_stats,
            name="train",
            acquisition_size=train_acquisition_size,
            **common_kwargs, **train_n_points_kwargs)

        test_dataset_kwargs = dict(lazy=lazy_test,
            fix_function_samples=fix_test_samples_dataset,
            fix_acquisition_samples=fix_test_acquisition_dataset,
            get_true_stats=get_test_true_stats,
            **common_kwargs, **test_n_points_kwargs)

        small_test_size, small_test_complement_size = get_lengths_from_proportions(
            test_samples_size,
            [small_test_proportion_of_test, 1 - small_test_proportion_of_test])

        if test_samples_size != small_test_size \
                and fix_test_acquisition_dataset and cache_datasets:
            print("Making small test acquisition dataset and complement")
            small_test_aq_dataset = self.create_acquisition_dataset(
                small_test_size, name="small-test",
                acquisition_size=small_test_size * test_expansion_factor,
                **test_dataset_kwargs)
            small_test_complement_aq_dataset = self.create_acquisition_dataset(
                small_test_complement_size, name="small-test-complement",
                acquisition_size=small_test_complement_size * test_expansion_factor,
                **test_dataset_kwargs)
            if isinstance(small_test_aq_dataset, AcquisitionDataset) and \
                isinstance(small_test_complement_aq_dataset, AcquisitionDataset):
                print("concatenating small test acquisition dataset and complement")
                test_aq_dataset = small_test_aq_dataset.concat(
                    small_test_complement_aq_dataset)
            elif isinstance(small_test_aq_dataset, bool):
                test_aq_dataset = small_test_aq_dataset and small_test_complement_aq_dataset
            else:
                test_aq_dataset = None
        else:
            test_aq_dataset = self.create_acquisition_dataset(
                test_samples_size, name="test",
                acquisition_size=test_samples_size * test_expansion_factor,
                **test_dataset_kwargs)
            if isinstance(test_aq_dataset, AcquisitionDataset):
                small_test_aq_dataset, _ = test_aq_dataset.random_split(
                    [small_test_proportion_of_test, 1 - small_test_proportion_of_test])
            elif isinstance(test_aq_dataset, bool):
                small_test_aq_dataset = test_aq_dataset
            else:
                small_test_aq_dataset = None
        
        return train_aq_dataset, test_aq_dataset, small_test_aq_dataset

    def create_train_test_datasets_helper(
            self,
            args: argparse.Namespace,
            dataset_configs: dict,
            check_cached: bool = False,
            load_dataset: bool = True
        ):
        """Helper function to create train/test datasets with caching."""
        
        dataset_kwargs = {
            **dataset_configs["function_samples_config"],
            **dataset_configs["acquisition_dataset_config"],
            **dataset_configs["n_points_config"],
            **dataset_configs["dataset_transform_config"]
        }

        lamda_min, lamda_max = get_lamda_min_max(args)
        other_kwargs = dict(        
            get_train_true_stats=False,  # Set appropriately per dataset type
            get_test_true_stats=False,   # Set appropriately per dataset type
            cache_datasets=CACHE_DATASETS,
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

    def create_train_test_datasets_from_args(
            self, 
            args: argparse.Namespace, 
            check_cached: bool = False, 
            load_dataset: bool = True
        ):
        """Create train/test datasets from command line arguments."""
        dataset_configs = self.get_dataset_configs(args, device=self.device)
        return self.create_train_test_datasets_helper(
            args, dataset_configs,
            check_cached=check_cached, load_dataset=load_dataset)