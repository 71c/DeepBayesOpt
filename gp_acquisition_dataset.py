import math
import os
from typing import Any, List, Optional, Union
from function_samples_dataset import GaussianProcessRandomDataset, ListMapFunctionSamplesDataset
from acquisition_dataset import AcquisitionDataset, FunctionSamplesAcquisitionDataset
from train_acquisition_function_net import train_or_test_loop
from utils import (dict_to_fname_str, dict_to_hash, get_uniform_randint_generator,
                   get_loguniform_randint_generator,
                   get_lengths_from_proportions)
from torch.distributions import Distribution
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.gp_regression import SingleTaskGP


def get_n_datapoints_random_gen_fixed_n_candidates(
        loguniform=True, pre_offset=None,
        min_history=1, max_history=8, n_candidates=50):
    if min_history is None:
        min_history = 1
    if max_history is None:
        max_history = 8
    if n_candidates is None:
        n_candidates = 50
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


script_dir = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(script_dir, "datasets")

def create_gp_acquisition_dataset(base_dataset_size,
        # gp_dataset_kwargs_non_datapoints
        dimension, randomize_params=False, xvalue_distribution="uniform",
        observation_noise=False, models=None, model_probabilities=None,

        outcome_transform: Optional[OutcomeTransform]=None,
        standardize_outcomes: bool=False,
        
        # n_datapoints_kwargs
        loguniform=True, pre_offset=None,
        min_history=None, max_history=None, n_candidates=None,
        min_n_candidates=None, max_points=None,

        expansion_factor=1,
        give_improvements:bool=True,
        
        fix_gp_samples=False, fix_acquisition_samples=False,
        device="cpu", lazy=True, cache=True,
        # For caching
        batch_size:Optional[int]=None, get_true_gp_stats:Optional[bool]=None,
        name=""):
    if type(cache) is not bool:
        raise ValueError("cache should be a boolean.")
    if type(lazy) is not bool:
        raise ValueError("lazy should be a boolean.")
    if type(fix_gp_samples) is not bool:
        raise ValueError("fix_gp_samples should be a boolean.")
    if type(fix_acquisition_samples) is not bool:
        raise ValueError("fix_acquisition_samples should be a boolean.")
    if type(standardize_outcomes) is not bool:
        raise ValueError("standardize_outcomes should be a boolean.")
    if type(give_improvements) is not bool:
        raise ValueError("give_improvements should be a boolean.")
    
    # Get the 
    if min_n_candidates is None and max_points is None:
        fix_n_candidates = True
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
            "Either min_n_candidates and max_points or min_history, " \
            "max_history and n_candidates should be specified.")
    
    gp_dataset_kwargs_non_datapoints = dict(
        dimension=dimension, randomize_params=randomize_params,
        observation_noise=observation_noise, models=models,
        model_probabilities=model_probabilities,
        xvalue_distribution=xvalue_distribution)
    
    if cache:
        name_ = name + "_" if name != "" else name

        gp_dataset_save_kwargs = dict(
            **n_datapoints_kwargs, **gp_dataset_kwargs_non_datapoints)
        function_dataset_hash = dict_to_hash(gp_dataset_save_kwargs)
        base_name = f"{name_}base_size={base_dataset_size}_{function_dataset_hash}"

        aq_dataset_extra_info_str = dict_to_fname_str(
            dict(fix_acquisition_samples=fix_acquisition_samples,
            expansion_factor=expansion_factor,
            outcome_transform=outcome_transform,
            standardize_outcomes=standardize_outcomes,
            give_improvements=give_improvements))
        aq_dataset_name = f"{base_name}_acquisition_{aq_dataset_extra_info_str}"
        aq_dataset_path = os.path.join(DATASETS_DIR, aq_dataset_name)

        aq_dataset_already_saved = os.path.exists(aq_dataset_path)
        
        if aq_dataset_already_saved:
            aq_dataset = AcquisitionDataset.load(aq_dataset_path)
            assert aq_dataset.data_is_fixed == fix_acquisition_samples
            # Won't just return it now because 
            # it is possible that the dataset currently doesn't have cached
            # data that we want to compute now.
            # Kind of redundant because we are loading and saving again.
        
        if fix_acquisition_samples:
            if batch_size is None or get_true_gp_stats is None:
                raise ValueError(
                    "Both batch_size and get_true_gp_stats must be specified " \
                    "if cache=True and fix_acquisition_samples=True")
            fix_gp_samples = True
        
        function_dataset_name = f"{base_name}_gp_samples"
        function_dataset_path = os.path.join(DATASETS_DIR, function_dataset_name)

    if not (cache and aq_dataset_already_saved):
        # Variable `function_dataset_path` doesn't exist if cache==False
        # but that's ok because "and" is lazy
        function_dataset_already_exists = cache and os.path.exists(function_dataset_path)
        if cache and fix_gp_samples and function_dataset_already_exists:
            function_samples_dataset = ListMapFunctionSamplesDataset.load(
                function_dataset_path)
        else:
            f = get_n_datapoints_random_gen_fixed_n_candidates if fix_n_candidates \
                else get_n_datapoints_random_gen_variable_n_candidates
            n_datapoints_random_gen = f(**n_datapoints_kwargs)
            function_samples_dataset = GaussianProcessRandomDataset(
                dataset_size=base_dataset_size,
                n_datapoints_random_gen=n_datapoints_random_gen,
                device=device, **gp_dataset_kwargs_non_datapoints)
            if fix_gp_samples:
                function_samples_dataset = function_samples_dataset.fix_samples(
                    lazy=False if cache else lazy)

        if cache and fix_gp_samples and not function_dataset_already_exists:
            os.makedirs(DATASETS_DIR, exist_ok=True)
            print(f"Saving {function_dataset_name}")
            function_samples_dataset.save(function_dataset_path, verbose=True)
        
        # Make sure to transform AFTER the dataset is saved because we want to
        # save the un-transformed values.
        if outcome_transform is not None:
            function_samples_dataset = function_samples_dataset.transform_outcomes(
                outcome_transform)
        if standardize_outcomes:
            function_samples_dataset = function_samples_dataset.standardize_outcomes()

        if fix_n_candidates:
            extra_kwargs = dict(n_candidate_points=n_candidates)
        else:
            extra_kwargs = dict(n_candidate_points="uniform",
                                min_n_candidates=min_n_candidates)
        aq_dataset = FunctionSamplesAcquisitionDataset(
            function_samples_dataset, n_samples="all",
            give_improvements=give_improvements,
            dataset_size_factor=expansion_factor, **extra_kwargs)
    
    if fix_acquisition_samples:
        if not aq_dataset.data_is_fixed:
            aq_dataset = aq_dataset.fix_samples(lazy=False if cache else lazy)
        if cache:
            dataloader = aq_dataset.get_dataloader(
                batch_size=batch_size, drop_last=False)
            desc = f"Getting{' ' if name != '' else ''}{name} dataset stats to cache"
            train_or_test_loop(dataloader, verbose=True, desc=desc,
                               get_true_gp_stats=get_true_gp_stats)
            os.makedirs(DATASETS_DIR, exist_ok=True)
            print(f"Saving {aq_dataset_name}")
            aq_dataset.save(aq_dataset_path, verbose=True)
    
    return aq_dataset


def create_train_and_test_gp_acquisition_datasets(
        train_acquisition_size:int,
        fix_train_samples_dataset:bool,

        test_factor:float,
        small_test_proportion_of_test:float,
        fix_test_samples_dataset:bool,
        fix_test_acquisition_dataset:bool,
        
        get_train_true_gp_stats:bool,
        get_test_true_gp_stats:bool,
        cache_datasets:bool,
        lazy_train:bool,
        lazy_test:bool,
        gp_gen_device,
        
        batch_size:int,
        fix_train_acquisition_dataset:bool,

        dimension:int,
        randomize_params:bool=False,
        xvalue_distribution: Union[Distribution,str]="uniform",
        observation_noise:bool=False,
        models:Optional[List[SingleTaskGP]]=None,
        model_probabilities:Optional[Any]=None,
        outcome_transform: Optional[OutcomeTransform]=None,
        standardize_outcomes:bool=False,

        expansion_factor:int=1,
        
        loguniform:bool=True, pre_offset:Optional[float]=None, fix_n_candidates:bool=True,
        train_n_candidates:Optional[int]=None, test_n_candidates:Optional[int]=None,
        min_history:Optional[int]=None, max_history:Optional[int]=None,
        min_n_candidates:Optional[int]=None, max_points:Optional[int]=None):
    train_samples_size = math.ceil(train_acquisition_size / expansion_factor)

    total_samples_dataset_size = math.ceil(train_samples_size * (1 + test_factor))

    ### Calculate test size
    test_samples_size = total_samples_dataset_size - train_samples_size
    ## Could alternatively calculate test size like this if going by
    ## proportions of an original dataset:
    # test_proportion = test_factor / (1 + test_factor)
    # train_proportion = 1 - test_proportion
    # train_samples_size, test_samples_size = get_lengths_from_proportions(
    #     total_samples_dataset_size, [train_proportion, test_proportion])

    print(f"Small test proportion of test: {small_test_proportion_of_test:.4f}")
    # small_test_proportion_of_test = 1 / ((1 / small_test_proportion_of_train_and_small_test - 1) * test_factor)
    small_test_proportion_of_train_and_small_test = 1 / (1 + 1 / (small_test_proportion_of_test * test_factor))
    print(f"Small test proportion of train + small test: {small_test_proportion_of_train_and_small_test:.4f}")

    if fix_n_candidates:
        train_n_points_kwargs = dict(min_history=min_history, max_history=max_history,
                                    n_candidates=train_n_candidates)
        test_n_points_kwargs = dict(min_history=min_history, max_history=max_history,
                                    n_candidates=test_n_candidates)
    else:
        train_n_points_kwargs = dict(min_n_candidates=min_n_candidates, max_points=max_points)
        test_n_points_kwargs = train_n_points_kwargs

    common_kwargs = dict(
        dimension=dimension,
        randomize_params=randomize_params,
        xvalue_distribution=xvalue_distribution,
        observation_noise=observation_noise,
        models=models,
        model_probabilities=model_probabilities,
        outcome_transform=outcome_transform,
        standardize_outcomes=standardize_outcomes,
        expansion_factor=expansion_factor, loguniform=loguniform,
        pre_offset=pre_offset if loguniform else None, batch_size=batch_size,
        device=gp_gen_device, cache=cache_datasets,
        give_improvements=False)

    train_aq_dataset = create_gp_acquisition_dataset(
        train_samples_size, lazy=lazy_train,
        fix_gp_samples=fix_train_samples_dataset,
        fix_acquisition_samples=fix_train_acquisition_dataset,
        get_true_gp_stats=get_train_true_gp_stats,
        name="train", **common_kwargs, **train_n_points_kwargs)

    test_dataset_kwargs = dict(lazy=lazy_test,
        fix_gp_samples=fix_test_samples_dataset,
        fix_acquisition_samples=fix_test_acquisition_dataset,
        get_true_gp_stats=get_test_true_gp_stats,
        **common_kwargs, **test_n_points_kwargs)

    small_test_size, small_test_complement_size = get_lengths_from_proportions(
        test_samples_size,
        [small_test_proportion_of_test, 1 - small_test_proportion_of_test])

    if test_samples_size != small_test_size \
            and fix_test_acquisition_dataset and cache_datasets:
        print("Making small test acquisition dataset and complement")
        small_test_aq_dataset = create_gp_acquisition_dataset(
            small_test_size, name="small-test", **test_dataset_kwargs)
        small_test_complement_aq_dataset = create_gp_acquisition_dataset(
            small_test_complement_size, name="small-test-complement",
            **test_dataset_kwargs)
        print("concatenating small test acquisition dataset and complement")
        test_aq_dataset = small_test_aq_dataset.concat(
            small_test_complement_aq_dataset)
    else:
        test_aq_dataset = create_gp_acquisition_dataset(
            test_samples_size, name="test", **test_dataset_kwargs)
        small_test_aq_dataset, _ = test_aq_dataset.random_split(
            [small_test_proportion_of_test, 1 - small_test_proportion_of_test])
    
    return train_aq_dataset, test_aq_dataset, small_test_aq_dataset
