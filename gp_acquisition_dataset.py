import argparse
from functools import cache, lru_cache
import math
import os
import sys
import traceback
from typing import Any, List, Optional, Union
from function_samples_dataset import GaussianProcessRandomDataset, ListMapFunctionSamplesDataset
from acquisition_dataset import AcquisitionDataset, CostAwareAcquisitionDataset, FunctionSamplesAcquisitionDataset
from train_acquisition_function_net import train_or_test_loop
from utils import (dict_to_hash, get_gp, get_kernel, get_standardized_exp_transform, get_uniform_randint_generator,
                   get_loguniform_randint_generator,
                   get_lengths_from_proportions)
from torch.distributions import Distribution
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.gp_regression import SingleTaskGP
# from botorch.models.transforms.outcome import Power


CACHE_DATASETS = True

# The following two are not important.
LAZY_TRAIN = True
LAZY_TEST = True

GET_TRAIN_TRUE_GP_STATS = False
GET_TEST_TRUE_GP_STATS = True

# Generating the random GP realizations is faster on CPU than on GPU.
# This is likely because the random GP realizations are generated one-by-one
# rather than in batches since the number of points is random so it's difficult
# to batch this. Hence we set device="cpu".
# Also, making the padded batches (the creation of zeros, concatenating, and
# stacking) on CPU rather than on GPU is much faster.
GP_GEN_DEVICE = "cpu"

FIX_TRAIN_ACQUISITION_DATASET = False


def add_lamda_args(parser):
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


def get_lamda_min_max(args: argparse.Namespace):
    lamda = getattr(args, 'lamda', None)
    lamda_min = lamda if lamda is not None else args.lamda_min
    lamda_max = getattr(args, 'lamda_max', None)
    return lamda_min, lamda_max


def get_n_datapoints_random_gen_fixed_n_candidates(
        loguniform=True, pre_offset=None,
        min_history=1, max_history=8, n_candidates=50):
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
DATASETS_DIR = os.path.join(script_dir, "datasets_cache")

def create_gp_acquisition_dataset(
        samples_size:int,
        acquisition_size:int,
        # gp_dataset_kwargs_non_datapoints
        dimension, randomize_params=False, xvalue_distribution="uniform",
        observation_noise=False, models=None, model_probabilities=None,

        outcome_transform: Optional[OutcomeTransform]=None,
        standardize_outcomes: bool=False,
        
        # n_datapoints_kwargs
        loguniform=True, pre_offset=None,
        min_history=None, max_history=None, n_candidates=None,
        min_n_candidates=None, max_points=None,

        # Only used for Gittins index training
        lambda_min:Optional[float]=None,
        lambda_max:Optional[float]=None,

        # Whether to fix the number of samples and draw random history (rather than
        # random number of samples and use all remaining samples as history).
        # Setting this to True is more realistic.
        # Only used if fixing the number of candidates.
        fix_n_samples=True,

        y_cand_indices="all",

        give_improvements:bool=False,
        
        fix_gp_samples=False, fix_acquisition_samples=False,
        device="cpu", lazy=True, cache=True,
        # For caching
        batch_size:Optional[int]=None, get_true_gp_stats:Optional[bool]=None,
        name="",

        replacement=False,
        
        # True: return a bool indicating whether the dataset has already been cached.
        # False: works as normal.
        check_cached=False,

        # True: load the dataset if it exists on disk.
        # False: do not load the dataset, even if it exists on disk, and return None
        load_dataset=True
    ):
    if check_cached:
        if load_dataset:
            raise ValueError("load_dataset should be False if check_cached is True.")
        if not cache:
            raise ValueError("cache should be True if check_cached is True.")

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
    
    if min_n_candidates is None and max_points is None:
        fix_n_candidates = True
        if min_history is None:
            min_history = 1
        if max_history is None:
            max_history = 8
        if n_candidates is None:
            n_candidates = 50
        if fix_n_samples:
            n_datapoints_kwargs = dict(
                loguniform=False, pre_offset=None,
                min_history=max_history + 5, max_history=max_history + 5,
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
            "Either min_n_candidates and max_points or min_history, " \
            "max_history and n_candidates should be specified.")
    
    fix_n_samples = fix_n_candidates and fix_n_samples
    
    gp_dataset_kwargs_non_datapoints = dict(
        dimension=dimension, randomize_params=randomize_params,
        observation_noise=observation_noise, models=models,
        model_probabilities=model_probabilities,
        xvalue_distribution=xvalue_distribution)
    
    aq_dataset_already_saved = False
    aq_dataset = None
    #### Define the paths and names for the datasets; load the AF dataset if it exists
    if cache:
        name_ = name + "_" if name != "" else name

        gp_dataset_save_kwargs = dict(
            **n_datapoints_kwargs, **gp_dataset_kwargs_non_datapoints)
        function_dataset_hash = dict_to_hash(gp_dataset_save_kwargs)
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
        
        # aq_dataset_extra_info_str = dict_to_fname_str(aq_dataset_extra_info)
        # Can't have everything in the file name because it's too long, would give,
        # OSError: [Errno 36] File name too long
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
                # Won't just return it now because 
                # it is possible that the dataset currently doesn't have cached
                # data that we want to compute now.
                # If so, we compute the cached data and save the dataset again.
        
        if fix_acquisition_samples:
            if batch_size is None or get_true_gp_stats is None:
                raise ValueError(
                    "Both batch_size and get_true_gp_stats must be specified " \
                    "if cache=True and fix_acquisition_samples=True")
            fix_gp_samples = True
        
        function_dataset_name = f"{base_name}_gp_samples"
        function_dataset_path = os.path.join(DATASETS_DIR, function_dataset_name)
    
    #### Check whether function samples dataset already exists
    # Variable `function_dataset_path` doesn't exist if cache==False
    # but that's ok because "and" is lazy
    function_dataset_already_saved = cache and os.path.exists(function_dataset_path)

    #### check_cached=True means to only check whether the datasets are already cached
    if check_cached:
        if fix_acquisition_samples:
            return aq_dataset_already_saved
        return function_dataset_already_saved or aq_dataset_already_saved

    #### If we don't have the AF dataset saved, then we need to generate it
    if not aq_dataset_already_saved:
        #### Generate or load the function samples dataset
        if cache and fix_gp_samples and function_dataset_already_saved:
            create_function_samples_dataset = False
            # Even if load_dataset=False, then we still want to load the function
            # samples dataset when fix_acquisition_samples=True because in that case
            # we want to save the acquisition dataset.
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
        else:
            create_function_samples_dataset = True
        
        if create_function_samples_dataset:
            f = get_n_datapoints_random_gen_fixed_n_candidates if fix_n_candidates \
                else get_n_datapoints_random_gen_variable_n_candidates
            n_datapoints_random_gen = f(**n_datapoints_kwargs)
            function_samples_dataset = GaussianProcessRandomDataset(
                dataset_size=samples_size,
                n_datapoints_random_gen=n_datapoints_random_gen,
                device=device, **gp_dataset_kwargs_non_datapoints)
            if fix_gp_samples:
                function_samples_dataset = function_samples_dataset.fix_samples(
                    lazy=False if cache else lazy)
        
        if function_samples_dataset is None:
            aq_dataset = None
        else:
            #### Save the function samples dataset
            if cache and fix_gp_samples and not function_dataset_already_saved:
                os.makedirs(DATASETS_DIR, exist_ok=True)
                print(f"Saving {function_dataset_name}")
                function_samples_dataset.save(function_dataset_path, verbose=True)
            
            #### Transform the function samples dataset
            # Make sure to transform AFTER the dataset is saved because we want to
            # save the un-transformed values.
            if outcome_transform is not None:
                function_samples_dataset = function_samples_dataset.transform_outcomes(
                    outcome_transform)
            if standardize_outcomes:
                function_samples_dataset = function_samples_dataset.standardize_outcomes()

            #### Create the acquisition dataset
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

            #### Add the lambdas if Gittins index
            if not (lambda_min is None and lambda_max is None):
                if lambda_max is not None and lambda_min is None:
                    raise ValueError(
                        "lambda_min must be specified if lambda_max is specified.")
                aq_dataset = CostAwareAcquisitionDataset(
                    aq_dataset, lambda_min=lambda_min, lambda_max=lambda_max)

    #### Fix the AF samples and save the AF dataset
    if aq_dataset is not None and fix_acquisition_samples:
        if not aq_dataset.data_is_fixed:
            aq_dataset = aq_dataset.fix_samples(lazy=False if cache else lazy)
        if cache:
            dataloader = aq_dataset.get_dataloader(
                batch_size=batch_size, drop_last=False)
            desc = f"Getting{' ' if name != '' else ''}{name} dataset stats to cache"
            # result==None iff all the requested dataset stats are already cached
            result = train_or_test_loop(dataloader, verbose=True, desc=desc,
                                        get_true_gp_stats=get_true_gp_stats,
                                        return_none=True)
            if not (aq_dataset_already_saved and result is None):
                os.makedirs(DATASETS_DIR, exist_ok=True)
                print(f"Saving AF dataset: {aq_dataset_name}")
                aq_dataset.save(aq_dataset_path, verbose=True)
    
    return aq_dataset


def create_train_and_test_gp_acquisition_datasets(
        train_samples_size:int,
        train_acquisition_size:int,
        fix_train_samples_dataset:bool,

        test_samples_size:int,
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
        replacement:bool=False,
        randomize_params:bool=False,
        xvalue_distribution: Union[Distribution,str]="uniform",
        observation_noise:bool=False,
        models:Optional[List[SingleTaskGP]]=None,
        model_probabilities:Optional[Any]=None,
        outcome_transform: Optional[OutcomeTransform]=None,
        standardize_outcomes:bool=False,

        # Only used for Gittins index training
        lambda_min:Optional[float]=None,
        lambda_max:Optional[float]=None,

        test_expansion_factor:int=1,
        
        loguniform:bool=True, pre_offset:Optional[float]=None, fix_n_candidates:bool=True,
        train_n_candidates:Optional[int]=None, test_n_candidates:Optional[int]=None,
        min_history:Optional[int]=None, max_history:Optional[int]=None,
        min_n_candidates:Optional[int]=None, max_points:Optional[int]=None,
        
        fix_n_samples:Optional[bool]=None,
        
        y_cand_indices:Union[str,List[int]]="all",
        
        check_cached=False,
        load_dataset=True
    ):
    if not check_cached:
        print(f"Small test proportion of test: {small_test_proportion_of_test:.4f}")
        test_factor = (test_samples_size * test_expansion_factor) / train_acquisition_size
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
        lambda_min=lambda_min, lambda_max=lambda_max,
        loguniform=loguniform,
        pre_offset=pre_offset if loguniform else None, batch_size=batch_size,
        device=gp_gen_device, cache=cache_datasets,
        give_improvements=False,
        fix_n_samples=fix_n_samples,
        y_cand_indices=y_cand_indices,
        replacement=replacement,
        check_cached=check_cached,
        load_dataset=load_dataset)

    train_aq_dataset = create_gp_acquisition_dataset(
        train_samples_size, lazy=lazy_train,
        fix_gp_samples=fix_train_samples_dataset,
        fix_acquisition_samples=fix_train_acquisition_dataset,
        get_true_gp_stats=get_train_true_gp_stats,
        name="train",
        acquisition_size=train_acquisition_size,
        **common_kwargs, **train_n_points_kwargs)

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
            small_test_size, name="small-test",
            acquisition_size=small_test_size * test_expansion_factor,
            **test_dataset_kwargs)
        small_test_complement_aq_dataset = create_gp_acquisition_dataset(
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
        test_aq_dataset = create_gp_acquisition_dataset(
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


def get_gp_model_from_args_no_outcome_transform(
        dimension: int,
        kernel: str,
        lengthscale: float,
        add_priors: bool,
        device=None):
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
        # Dimension of the optimization problem
        dimension=args.dimension,
        # whether to randomize the GP parameters for training data
        randomize_params=args.randomize_params,
        # choose either "uniform" or "normal" (or a custom distribution)
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
        # Whether to fix the training dataset function samples
        # (as opposed to generating them randomly with each epoch)
        fix_train_samples_dataset=True,

        small_test_proportion_of_test=1.0,

        replacement=args.replacement,
        
        # The following two should be kept as they are -- ALWAYS want to fix the
        # test. As long as the acqisition dataset is fixed, then whether the
        # function samples dataset is fixed doesn't matter.
        fix_test_samples_dataset=False,
        fix_test_acquisition_dataset=True,
    )
    if not acquisition_dataset_config['fix_train_samples_dataset']:
        acquisition_dataset_config['replacement'] = False

    ########## Set number of history and candidate points generation ###############
    n_points_config = dict(
        # This means whether n history points (or whether the total number of
        # points) is log-uniform
        # (Note: This currently ONLY can make it log-uniform if fix_n_samples=False)
        loguniform=False,
        # Whether to fix the number of candidate points (as opposed to randomized)
        fix_n_candidates=True,
        # If this is True, we fix the number of samples in each item at
        # maximum when generating the function samples dataset -- which is more
        # realistic (Only used if # of candidates is fixed).
        # If this is False, then the number of samples in each item is
        # randomized to reflect random number of history points.
        fix_n_samples=True
    )
    if n_points_config['loguniform']:
        n_points_config['pre_offset'] = 3.0
    if n_points_config['fix_n_candidates']:
        # If fix_n_candidates is True, then the following are used:
        n_points_config = dict(
            # Number of candidate points for training. For MSE EI, could just set to 1.
            train_n_candidates=args.train_n_candidates,
            # Number of candidate points for testing.
            test_n_candidates=args.test_n_candidates,
            min_history=args.min_history,
            max_history=args.max_history,
            **n_points_config
        )
    else:
        # If fix_n_candidates is False, then the following are used:
        n_points_config = dict(
            min_n_candidates=2,
            max_points=30,
            **n_points_config
        )

    dataset_transform_config = dict(
        # Choose an outcome transform. Can be None if no outcome transform
        # TODO (bug): str(Power(2)) = "Power()" but we'd like it to be "Power(2)" so it
        # can be saved uniquely. Maybe use the attributes of the class or something
        # instead. Or alternateively, just don't save the acquisition datasets, or
        # transform the acquisition datasets directly. I think it would be easiest to
        # just not save the acquisition datasets anymore.
        outcome_transform=outcome_transform,
        standardize_outcomes=args.standardize_dataset_outcomes
    )

    return {
        "function_samples_config": function_samples_config,
        "acquisition_dataset_config": acquisition_dataset_config,
        "n_points_config": n_points_config,
        "dataset_transform_config": dataset_transform_config
    }


def create_train_test_gp_acq_datasets_helper(
        args: argparse.Namespace,
        gp_af_dataset_configs,
        check_cached=False,
        load_dataset=True
    ):
    dataset_kwargs = {
        **gp_af_dataset_configs["function_samples_config"],
        **gp_af_dataset_configs["acquisition_dataset_config"],
        **gp_af_dataset_configs["n_points_config"],
        **gp_af_dataset_configs["dataset_transform_config"]
    }

    lamda_min, lamda_max = get_lamda_min_max(args)
    other_kwargs = dict(        
        get_train_true_gp_stats=GET_TRAIN_TRUE_GP_STATS,
        get_test_true_gp_stats=GET_TEST_TRUE_GP_STATS,
        cache_datasets=CACHE_DATASETS,
        lazy_train=LAZY_TRAIN,
        lazy_test=LAZY_TEST,
        gp_gen_device=GP_GEN_DEVICE,
        
        batch_size=getattr(args, "batch_size", 64),
        fix_train_acquisition_dataset=FIX_TRAIN_ACQUISITION_DATASET,
        
        # For this particular, code, doing this works for both EI and Gittins index
        # training in all cases -- all good
        y_cand_indices=[0],

        lambda_min=lamda_min,
        lambda_max=lamda_max
    )

    (train_aq_dataset, test_aq_dataset,
     small_test_aq_dataset) = create_train_and_test_gp_acquisition_datasets(
        **dataset_kwargs, **other_kwargs,
        check_cached=check_cached, load_dataset=load_dataset)

    if not check_cached:
        if train_aq_dataset is not None:
            # print("Training function samples dataset size:", len(train_dataset))
            print("Original training acquisition dataset size parameter:",
                  gp_af_dataset_configs["acquisition_dataset_config"]["train_acquisition_size"])
            print("Training acquisition dataset size:", len(train_aq_dataset),
                "number of batches:", len(train_aq_dataset) // args.batch_size,
                len(train_aq_dataset) % args.batch_size)

        if test_aq_dataset is not None:
            # print("Test function samples dataset size:", len(test_dataset))
            print("Test acquisition dataset size:", len(test_aq_dataset),
                "number of batches:", len(test_aq_dataset) // args.batch_size,
                len(test_aq_dataset) % args.batch_size)
        
        if small_test_aq_dataset != test_aq_dataset and small_test_aq_dataset is not None:
            print("Small test acquisition dataset size:", len(small_test_aq_dataset),
                    "number of batches:", len(small_test_aq_dataset) // args.batch_size,
                    len(small_test_aq_dataset) % args.batch_size)
        
        for name, dataset in [("Train acquisition dataset", train_aq_dataset),
                ("Test acquisition dataset", test_aq_dataset),
                ("Small test acquisition dataset", small_test_aq_dataset)]:
            if dataset is None:
                continue
            print(f"{name}:")
            print(f"{name} type: {type(dataset)}")
            print(f"{name} n_out_cand: {next(iter(dataset)).vals_cand.size(-1)}")

    return train_aq_dataset, test_aq_dataset, small_test_aq_dataset

    # for item in train_aq_dataset.base_dataset:
    #     print(item.y_values.mean(), item.y_values.std(), item.y_values.shape)
    # exit()

    # print("Train acquisition dataset:")
    # print(train_aq_dataset)
    # print("\nTest acquisition dataset:")
    # print(test_aq_dataset)
    # if small_test_aq_dataset != test_aq_dataset:
    #     print("\nSmall test acquisition dataset:")
    #     print(small_test_aq_dataset)
    # print("\n")
    # exit()


def add_gp_args(parser, thing_gp_used_for: str,
                name_prefix="", required=False,
                add_randomize_params=False):
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
    ############################# Samples dataset settings #############################
    ## GP settings
    parser.add_argument(
        '--dimension', 
        type=int, 
        help='Dimension of the optimization problem',
        required=True
    )
    add_gp_args(parser, "dataset", required=True, add_randomize_params=True)
    parser.add_argument(
        '--standardize_dataset_outcomes', 
        action='store_true', 
        help='Whether to standardize the outcomes of the dataset (independently for each item). Default is False'
    )
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


def create_train_test_gp_acq_datasets_from_args(
        args, check_cached=False, load_dataset=True):
    gp_af_dataset_configs = get_gp_acquisition_dataset_configs(
         args, device=GP_GEN_DEVICE)
    (train_aq_dataset,
     test_aq_dataset,
     small_test_aq_dataset) = create_train_test_gp_acq_datasets_helper(
         args, gp_af_dataset_configs,
         check_cached=check_cached, load_dataset=load_dataset)
    return train_aq_dataset, test_aq_dataset, small_test_aq_dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    add_gp_acquisition_dataset_args(parser)
    add_lamda_args(parser)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for the acquisition dataset.'
    )

    args = parser.parse_args()

    create_train_test_gp_acq_datasets_from_args(
        args, check_cached=False, load_dataset=False)


if __name__ == "__main__":
    main()
