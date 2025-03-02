from typing import Callable
import copy
import os
import matplotlib.pyplot as plt
import torch

from train_acquisition_function_net import load_configs, load_model
torch.set_default_dtype(torch.float64)
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement


from bayesopt import (GPAcquisitionOptimizer, OptimizationResultsMultipleMethods,
                      NNAcquisitionOptimizer, RandomSearch, generate_gp_acquisition_options,
                      get_random_gp_functions, plot_optimization_results_multiple_methods, transform_functions_and_names)

from utils import (dict_to_hash, dict_to_str, get_dimension,
                   get_gp, DEVICE, Exp, save_json)
from dataset_with_models import RandomModelSampler

from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.priors.torch_priors import GammaPrior

script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')
RESULTS_DIR = os.path.join(script_dir, 'bayesopt_results')
MODELS_DIR = os.path.join(script_dir, "saved_models")

# model_and_info_name = "model_20240715_204121_e6101c167831be592f2608748e4175bd77837853ed05e81c56ec7f9f3ee61695" # untransformed

# G2
# 6-dim, policy gradient, Exp, 14-50 history points
# outcomes standardized both dataset and history outcomes in forward
# model_and_info_name = "model_20240716_230558_f182344a0fd9ee8ac324de86db74ccf2f2b60ea2fa07f471cdfb3a9728a64d5d"

# Same as above but 64 instead of 32
# model_and_info_name = "model_20240717_141109_b7a5a9f189d98493d8f3eefef37e9bd5b4ed023742ddd6f60d4ae91e7c0350e9"

# 50K train size; MSE loss; 64 widths
# Best max EI: 0.3141892009482172
# model_and_info_name = "model_20240717_212151_f3ff357217b4bce861abb578d2b853eebad30b90e2345c4062393bfca2417a3e"

# Same as above but training size is 200K, and layer-width 128 instead of 64
# Best max EI: 0.35370628685185046
# model_and_info_name = "model_20240718_004436_e1c8afcd9487924a8dbedbb6675293c42d94e30a63eabcf0c35338d19e3b64f4"

# Same as above but training size is 800K, and layer-width 256 instead of 128
# Best max EI: 0.36729013620012463
# model_and_info_name = "model_20240718_030711_3b42b16944fa8b5d8affffdd7c130d4188d4d8f7335a4c99758399fa7efa79ec"


## Dataset size comparison
# model_and_info_names = [
#     # MSE, trained on 14-54 history and 50 candidate points, width 256, 6D, fixed params,
#     # no dataset or NN-history outcome transforms, training size 50K; ei_max: 0.08256165126391311
#     "model_20240720_032737_39e71d4622815b884867c11201f77b0c24488c2300408de3a1f7ec5478517bd2",
#     # MSE, trained on 14-54 history and 50 candidate points, width 256, 6D, fixed params,
#     # no dataset or NN-history outcome transforms, training size 300K; ei_max: 0.08693664313535554
#     "model_20240720_041525_7a99c1685961e1d895428262ca983b9883f1cb5ffe47dc944b34083d0183d389",
#     # MSE, trained on 14-54 history and 50 candidate points, width 256, 6D, fixed params,
#     # no dataset or NN-history outcome transforms, training size 1800K; ei_max: 0.0882794855344188
#     "model_20240720_073340_0a363bb237949dc076c8e1873cb00d0d4bb159bd255ed360d0dda90d92a5dddb"
# ]
# model_and_info_plot_names = [
#     "NN, 50K training size", "NN, 300K training size", "NN, 1800K training size"
# ]
# plots_title = "MSE with 50 candidates, layer-width 256, 6D, fixed params, no standardizations or transforms, 14 initial points, 40 iterations, 5 trial/func"


# ## Standardize history in NN or not comparison
# model_and_info_names = [
#     # MSE, trained on 14-54 history and 50 candidate points, width 256, 6D, fixed params,
#     # no dataset or NN-history outcome transforms, training size 300K; ei_max: 0.08693664313535554
#     "model_20240720_041525_7a99c1685961e1d895428262ca983b9883f1cb5ffe47dc944b34083d0183d389",
#     # MSE, trained on 14-54 history and 50 candidate points, width 256, 6D, fixed params,
#     # no dataset transforms, standardized history outcomes to NN, training size 300K; ei_max: 0.08552981475318698
#     "model_20240720_174324_a0827f9d932527b5b33f569b3f93d8eb07d42996593d533a22de6193ce386174",
# ]
# model_and_info_plot_names = [
#     "NN, history outcomes not standardized", "NN, history outcomes standardized"
# ]
# plots_title = "Compare history standardization: MSE with 50 candidates, layer-width 256, 6D, fixed params, no transforms, 300K training size, 14 initial points, 40 iterations, 5 trial/func"


## Standardize history in NN or not comparison
model_and_info_names = [
    "v1/model_588a669bcb9adcef51580f532b8c10b82b0733a28ebd885095ec8763a5085a91"
]
model_and_info_plot_names = [
    "NN"
]
plots_title = "Just a test"

nn_models = [
    load_model(model_and_info_name).to(DEVICE)
    for model_and_info_name in model_and_info_names
]

configs = [load_configs(model_and_info_name)
           for model_and_info_name in model_and_info_names]

methods = [
    config["training_config"]["method"] for config in configs
]

method_descriptions = {
    'policy_gradient': 'policy gradient',
    'mse_ei': 'MSE EI',
    'gittins': 'Gittins index'
}

model_and_info_plot_names = [
    f"{name} ({method_descriptions[method]})"
    for name, method in zip(model_and_info_plot_names, methods)
]

# Assume that all the configs are the same
function_samples_config = configs[0]["function_samples_config"]
dataset_transform_config = configs[0]["dataset_transform_config"]
gp_sampler = configs[0]["gp_sampler"]

dimensions = set()
if 'dimension' in function_samples_config:
    dimensions.add(function_samples_config['dimension'])
if hasattr(nn_models[0], 'dimension'):
    dimensions.add(nn_models[0].dimension)
dimensions.add(get_dimension(gp_sampler.get_model(0)))
if len(dimensions) > 1:
    raise ValueError("Multiple dimensions found in configs")
dim = dimensions.pop()
print("DIMENSION:", dim)

DIFFERENT_GP = False
# These are the settings to use if DIFFERENT_GP is True:
RANDOMIZE_PARAMS = True
OUTCOME_TRANSFORM = Exp()

n_functions = 3

opt_config = {
    'n_iter': 40,
    'seed': 12,
    'n_trials_per_function': 5,
    'n_initial_samples': 2*(dim+1)
}
config = {
    'dim': dim,
    'observation_noise': function_samples_config['observation_noise'],
    'randomize_params': gp_sampler.randomize_params,
    'model_and_info_names': model_and_info_names,
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

bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

torch.manual_seed(SEED)
init_x = draw_sobol_samples(bounds=bounds, 
                            n=n_trials,
                            q=config['n_initial_samples'])

(random_gps, gp_realizations,
 objective_names, function_plot_names) = get_random_gp_functions(
    gp_sampler, SEED, n_functions, observation_noise)

outcome_transform = config['outcome_transform']
if outcome_transform is not None:
    gp_realizations, objective_names, function_plot_names = transform_functions_and_names(
        gp_realizations, objective_names, function_plot_names, outcome_transform)

acquisition_functions = {
    'Log EI': LogExpectedImprovement,
    'EI': ExpectedImprovement
}

options_dict_gp = generate_gp_acquisition_options(
    acquisition_functions, random_gps, outcome_transform,
    fit_map=True, fit_mle=False)

options_dict_random = {
    'Random Search': {'optimizer_class': RandomSearch}
}

options_dict_nn = {}
for nn_plot_name, nn_model, model_and_info_name, method in zip(
    model_and_info_plot_names, nn_models, model_and_info_names, methods):
    option_dict = {
        'optimizer_class': NNAcquisitionOptimizer,
        'model': nn_model,
        'nn_model_name': model_and_info_name
    }
    if method == 'mse_ei':
        options_dict_nn[f"{nn_plot_name} (with softplus)"] = {
            'exponentiate': True,
            **option_dict
        }
        options_dict_nn[f"{nn_plot_name} (without softplus)"] = {
            'exponentiate': False,
            **option_dict
        }
    else:
        # defaults to exponentiate=False
        options_dict_nn[nn_plot_name] = option_dict

options_dict = {
    **options_dict_gp,
    **options_dict_random,
    **options_dict_nn
}

has_any_mse = any(method == 'mse_ei' for method in methods)

is_ei_acqf_dict = {
    LogExpectedImprovement: False,
    ExpectedImprovement: True
}
def is_ei_option(option_dict):
    if option_dict['optimizer_class'] is NNAcquisitionOptimizer:
        return option_dict.get('exponentiate', False)
    if option_dict['optimizer_class'] is GPAcquisitionOptimizer:
        acqf_cls = option_dict['acquisition_function_class']
        return is_ei_acqf_dict.get(acqf_cls, None)
    return None

combined_options_to_plot = {
    'no EI methods': [
        option_name for option_name, option_dict in options_dict.items()
        if is_ei_option(option_dict) in {False, None}
    ]
}
if has_any_mse:
    combined_options_to_plot['EI methods'] = [
        option_name for option_name, option_dict in options_dict.items()
        if is_ei_option(option_dict) in {True, None}
    ]


results_generator = OptimizationResultsMultipleMethods(
    options_dict, gp_realizations, init_x, config['n_iter'],
    SEED, objective_names, RESULTS_DIR, dim=dim, bounds=bounds, maximize=True)

config_json = {
    k: v.__class__.__name__ if k == 'outcome_transform' and v is not None else v
    for k, v in config.items()
}
config_str = dict_to_str(config_json)

config_with_n_functions_json = {**config_json, 'n_functions': n_functions}
config_with_n_functions_hash = dict_to_hash(config_with_n_functions_json)
plots_dir = os.path.join(PLOTS_DIR, config_with_n_functions_hash)
save_json(config_with_n_functions_json, os.path.join(plots_dir, 'config.json'))

plot_optimization_results_multiple_methods(
    optimization_results=results_generator,
    max_n_functions_to_plot=5,
    alpha=0.1,
    sharey=False,
    aspect=1.35,
    scale=1.,
    combined_options_to_plot=combined_options_to_plot,
    objective_names_plot=function_plot_names,
    plots_fname_desc=None,
    # plots_title=config_str,
    plots_title=plots_title,
    plots_dir=plots_dir
)
plt.show()
