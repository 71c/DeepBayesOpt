import hashlib
import os
from typing import Optional
from function_samples_dataset import GaussianProcessRandomDataset, ListMapFunctionSamplesDataset
from acquisition_dataset import AcquisitionDataset, FunctionSamplesAcquisitionDataset
from train_acquisition_function_net import train_or_test_loop
from utils import get_uniform_randint_generator, get_loguniform_randint_generator


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


def dict_to_str(d):
    return ','.join(f"{key}={value!r}" for key, value in sorted(d.items()))

def dict_to_hash(d):
    dict_bytes = dict_to_str(d).encode('ascii')
    return hashlib.sha256(dict_bytes).hexdigest()

script_dir = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(script_dir, "datasets")

def create_gp_acquisition_dataset(base_dataset_size,
        # gp_dataset_kwargs_non_datapoints
        dimension, randomize_params=False,
        observation_noise=False, models=None, model_probabilities=None,
        xvalue_distribution="uniform",
        
        # n_datapoints_kwargs
        loguniform=True, pre_offset=None,
        min_history=None, max_history=None, n_candidates=None,
        min_n_candidates=None, max_points=None,

        expansion_factor=1,
        
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

        gp_aq_dataset_extra_info_str = dict_to_str(
            dict(fix_acquisition_samples=fix_acquisition_samples,
            expansion_factor=expansion_factor))
        
        base_name = f"{name_}base_size={base_dataset_size}_{function_dataset_hash}"

        aq_dataset_name = f"{base_name}_acquisition_{gp_aq_dataset_extra_info_str}"
        aq_dataset_path = os.path.join(DATASETS_DIR, aq_dataset_name)
        
        if os.path.exists(aq_dataset_path):
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

    if not (cache and os.path.exists(aq_dataset_path)):
        if cache and os.path.exists(function_dataset_path):
            function_samples_dataset = ListMapFunctionSamplesDataset.load(
                function_dataset_path)
        else:
            f = get_n_datapoints_random_gen_fixed_n_candidates if fix_n_candidates \
                else get_n_datapoints_random_gen_variable_n_candidates
            n_datapoints_random_gen = f(**n_datapoints_kwargs)
            function_samples_dataset = GaussianProcessRandomDataset(
                dataset_size=base_dataset_size,
                n_datapoints_random_gen=n_datapoints_random_gen,
                device=device, set_random_model_train_data=False,
                **gp_dataset_kwargs_non_datapoints)
            if fix_gp_samples:
                function_samples_dataset = function_samples_dataset.fix_samples(
                    lazy=False if cache else lazy)
                if cache:
                    os.makedirs(DATASETS_DIR, exist_ok=True)
                    function_samples_dataset.save(function_dataset_path, verbose=True)
        
        if fix_n_candidates:
            extra_kwargs = dict(n_candidate_points=n_candidates)
        else:
            extra_kwargs = dict(n_candidate_points="uniform",
                                min_n_candidates=min_n_candidates)
        aq_dataset = FunctionSamplesAcquisitionDataset(
            function_samples_dataset, n_samples="all", give_improvements=True,
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
            aq_dataset.save(aq_dataset_path, verbose=True)
    
    return aq_dataset
