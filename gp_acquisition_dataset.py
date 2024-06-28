from function_samples_dataset import GaussianProcessRandomDataset
from acquisition_dataset import FunctionSamplesAcquisitionDataset
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


def create_gp_acquisition_dataset(
        base_dataset_size, dimension, randomize_params=False, device="cpu",
        observation_noise=False, models=None, model_probabilities=None,
        xvalue_distribution="uniform",
        
        # generate number of points
        loguniform=True, pre_offset=None,
        min_history=None, max_history=None, n_candidates=None,
        min_n_candidates=None, max_points=None,

        expansion_factor=1,
        fix_gp_samples=False, fix_acquisition_samples=False, lazy=True):
    
    if min_n_candidates is None and max_points is None:
        fix_n_candidates = True
        n_datapoints_random_gen = get_n_datapoints_random_gen_fixed_n_candidates(
            loguniform, pre_offset, min_history, max_history, n_candidates)
    elif min_history is None and max_history is None and n_candidates is None:
        fix_n_candidates = False
        if min_n_candidates is None:
            min_n_candidates = 2
        n_datapoints_random_gen = get_n_datapoints_random_gen_variable_n_candidates(
            loguniform, pre_offset, min_n_candidates, max_points)
    else:
        raise ValueError(
            "Either min_n_candidates and max_points or min_history, max_history and n_candidates should be specified.")
    
    function_samples_dataset = GaussianProcessRandomDataset(
        n_datapoints_random_gen=n_datapoints_random_gen,
        observation_noise=observation_noise,
        xvalue_distribution=xvalue_distribution,
        models=models, model_probabilities=model_probabilities,
        dimension=dimension, device=device,
        set_random_model_train_data=False,
        dataset_size=base_dataset_size,
        randomize_params=randomize_params)
    
    if fix_gp_samples:
        function_samples_dataset = function_samples_dataset.fix_samples(lazy=lazy)
    
    if fix_n_candidates:
        extra_kwargs = dict(n_candidate_points=n_candidates)
    else:
        extra_kwargs = dict(n_candidate_points="uniform",
                            min_n_candidates=min_n_candidates)
    aq_dataset = FunctionSamplesAcquisitionDataset(
        function_samples_dataset, n_samples="all", give_improvements=True,
        dataset_size_factor=expansion_factor, **extra_kwargs)
    
    if fix_acquisition_samples:
        aq_dataset = aq_dataset.fix_samples(lazy=lazy)
    
    return aq_dataset

