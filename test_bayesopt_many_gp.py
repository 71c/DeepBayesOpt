import os
import matplotlib.pyplot as plt
import torch
torch.set_default_dtype(torch.float64)
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement

from bayesopt import (GPAcquisitionOptimizer, LazyOptimizationResultsMultipleMethods,
                      NNAcquisitionOptimizer, RandomSearch,
                      get_random_gp_functions, plot_optimization_results_multiple_methods)

from utils import (get_gp, dict_to_fname_str,
                   combine_nested_dicts)
from dataset_with_models import RandomModelSampler
from acquisition_function_net import AcquisitionFunctionNet

from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.priors.torch_priors import GammaPrior

script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')
RESULTS_DIR = os.path.join(script_dir, 'bayesopt_results')

dim = 3
n_functions = 6
opt_config = {
    'n_iter': 50,
    'seed': 12,
    'n_trials_per_function': 5,
    'n_initial_samples': 2*(dim+1)
}
config = {
    'dim': dim,
    'observation_noise': False,
    'randomize_params': True,
    **opt_config
}
SEED = config['seed']


observation_noise = config['observation_noise']
n_trials = config['n_trials_per_function']

config_str = dict_to_fname_str(config)

bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

torch.manual_seed(SEED)
init_x = draw_sobol_samples(bounds=bounds, 
                            n=n_trials,
                            q=config['n_initial_samples'])

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
models = [get_gp(
    dimension=dim,
    observation_noise=observation_noise,
    covar_module=kernel)]
gp_sampler = RandomModelSampler(models,
                                randomize_params=config['randomize_params'])


random_gps, gp_realizations, objective_names, function_plot_names = get_random_gp_functions(
    gp_sampler, SEED, n_functions, observation_noise)


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
options_dict_gp = {
    key: {
        'optimizer_class': GPAcquisitionOptimizer,
        'optimizer_kwargs_per_function': [{'model': gp} for gp in random_gps],
        **value
    } for key,value in combine_nested_dicts(
        acquisition_function_options, gp_options).items()
}

options_dict_random = {
    'Random Search': {'optimizer_class': RandomSearch}
}

options_dict = {**options_dict_gp,
                **options_dict_random
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
    plots_fname_desc=config_str,
    plots_title=None,
    plots_dir=PLOTS_DIR
)
plt.show()
