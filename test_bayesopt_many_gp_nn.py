import copy
import os
import matplotlib.pyplot as plt
import torch

from train_acquisition_function_net import load_configs, load_model
torch.set_default_dtype(torch.float64)
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement

from bayesopt import (GPAcquisitionOptimizer, LazyOptimizationResultsMultipleMethods,
                      NNAcquisitionOptimizer, RandomSearch,
                      get_random_gp_functions, plot_optimization_results_multiple_methods)

from utils import (concatenate_outcome_transforms, convert_to_json_serializable, get_dimension, get_gp,
                   dict_to_fname_str, combine_nested_dicts, DEVICE, Exp, invert_outcome_transform)
from dataset_with_models import RandomModelSampler
from acquisition_function_net import AcquisitionFunctionNet

from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.priors.torch_priors import GammaPrior

script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')
RESULTS_DIR = os.path.join(script_dir, 'bayesopt_results')
MODELS_DIR = os.path.join(script_dir, "saved_models")

# model_and_info_name = "model_20240713_012112" # untransformed
model_and_info_name = "model_20240713_014550" # transformed
MODEL_AND_INFO_PATH = os.path.join(MODELS_DIR, model_and_info_name)

nn_model = load_model(MODEL_AND_INFO_PATH).to(DEVICE)

gp_realization_config, dataset_size_config, n_points_config, \
        dataset_transform_config, gp_sampler = load_configs(MODEL_AND_INFO_PATH)

dimensions = set()
if 'dimension' in gp_realization_config:
    dimensions.add(gp_realization_config['dimension'])
if hasattr(nn_model, 'dimension'):
    dimensions.add(nn_model.dimension)
dimensions.add(get_dimension(gp_sampler.get_model(0)))
if len(dimensions) > 1:
    raise ValueError("Multiple dimensions found in configs")
dim = dimensions.pop()

n_functions = 1


DIFFERENT_GP = False
# These are the settings to use if DIFFERENT_GP is True:
RANDOMIZE_PARAMS = True
OUTCOME_TRANSFORM = Exp()

opt_config = {
    'n_iter': 10,
    'seed': 12,
    'n_trials_per_function': 5,
    'n_initial_samples': 2*(dim+1)
}
config = {
    'dim': dim,
    'observation_noise': gp_realization_config['observation_noise'],
    'randomize_params': gp_sampler.randomize_params,
    'model_and_info_name': model_and_info_name,
    'outcome_transform': dataset_transform_config['outcome_transform'],
    'different_gp': DIFFERENT_GP,
    **opt_config
}

if DIFFERENT_GP:
    # Construct random GP sampler
    kernel = ScaleKernel(
        base_kernel=RBFKernel(
            ard_num_dims=dim,
            batch_shape=torch.Size(),
            lengthscale_prior=GammaPrior(3.0, 6.0),
        ),
        batch_shape=torch.Size(),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
    gp_models = [get_gp(
        dimension=dim,
        observation_noise=config['observation_noise'],
        covar_module=kernel)]
    gp_sampler = RandomModelSampler(gp_models,
                                    randomize_params=RANDOMIZE_PARAMS)

    config['randomize_params'] = RANDOMIZE_PARAMS
    config['outcome_transform'] = OUTCOME_TRANSFORM


SEED = config['seed']
observation_noise = config['observation_noise']
n_trials = config['n_trials_per_function']
config_json = convert_to_json_serializable(config)
config_str = dict_to_fname_str(config_json)

bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

torch.manual_seed(SEED)
init_x = draw_sobol_samples(bounds=bounds, 
                            n=n_trials,
                            q=config['n_initial_samples'])

random_gps, gp_realizations, objective_names, function_plot_names = get_random_gp_functions(
    gp_sampler, SEED, n_functions, observation_noise)

outcome_transform = config['outcome_transform']
if outcome_transform is not None:
    objective_names = [
        f"{name}_{outcome_transform}" for name in objective_names]
    function_plot_names = [
        f"{name} ({outcome_transform.__class__.__name__} transform)"
        for name in function_plot_names]
    gp_realizations = [
        # remember it also takes optional argument Yvar and returns (Y, Yvar)
        lambda x: outcome_transform(gp_realization(x))[0]
        for gp_realization in gp_realizations
    ]

    random_gps_transformed = []
    for gp in random_gps:
        new_gp = copy.deepcopy(gp)
        outcome_transform_of_gp = invert_outcome_transform(outcome_transform)
        if hasattr(new_gp, 'outcome_transform'):
            new_gp.outcome_transform = concatenate_outcome_transforms(
                outcome_transform_of_gp, new_gp.outcome_transform)
        else:
            new_gp.outcome_transform = outcome_transform_of_gp
        random_gps_transformed.append(new_gp)


acquisition_functions = {
    'Log EI': LogExpectedImprovement,
    # 'EI': ExpectedImprovement
}
gp_options = {
    'True GP': {'fit_params': False},
    'MAP': {'fit_params': True, 'mle': False},
    # 'MLE': {'fit_params': True, 'mle': True}
}

acquisition_function_options = {
    name: {'acquisition_function_class': acq_func_class}
    for name, acq_func_class in acquisition_functions.items()}

kwargs_always = {
    'optimizer_class': GPAcquisitionOptimizer,
}

if outcome_transform is None:
    kwargs_always['optimizer_kwargs_per_function'] = [
        {'model': gp} for gp in random_gps]
    options_to_combine = [acquisition_function_options, gp_options]
else:
    transform_opt = {
            'optimizer_kwargs_per_function': [
                {'model': gp} for gp in random_gps_transformed]
        }
    nontransform_opt = {
            'optimizer_kwargs_per_function': [
                {'model': gp} for gp in random_gps]
        }

    gp_options_true = {
        'True GP params (with transform)': {**gp_options['True GP'], **transform_opt}
    }
    gp_transform_options = {
        'true GP model with transform': transform_opt,
        'untransformed GP model': nontransform_opt
    }
    gp_options_untrue = combine_nested_dicts(
        {k: v for k, v in gp_options.items() if k != 'True GP'},
        gp_transform_options)

    options_to_combine = [acquisition_function_options,
                          {**gp_options_true, **gp_options_untrue}]

options_dict_gp = {
    key: {
        **kwargs_always,
        **value
    } for key,value in combine_nested_dicts(*options_to_combine).items()
}

options_dict_random = {
    'Random Search': {'optimizer_class': RandomSearch}
}

options_dict_nn = {
    'NN': {
        'optimizer_class': NNAcquisitionOptimizer,
        'model': nn_model,
        'nn_model_name': model_and_info_name
    }
}

options_dict = {
    **options_dict_gp,
    **options_dict_random,
    **options_dict_nn
}


results_generator = LazyOptimizationResultsMultipleMethods(
    options_dict, gp_realizations, init_x, config['n_iter'],
    SEED, objective_names, RESULTS_DIR, dim=dim, bounds=bounds, maximize=True)

plot_optimization_results_multiple_methods(
    optimization_results=results_generator,
    max_n_functions_to_plot=5,
    alpha=0.05,
    sharey=False,
    aspect=1.,
    scale=0.5,
    objective_names_plot=function_plot_names,
    plots_fname_desc=None,
    plots_title=config_str,
    plots_dir=PLOTS_DIR
)
plt.show()
