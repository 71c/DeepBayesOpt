import copy
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
torch.set_default_dtype(torch.float64)
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement

from bayesopt import (GPAcquisitionOptimizer, NNAcquisitionOptimizer, RandomSearch,
                      get_optimization_results,
                      plot_optimization_trajectories_error_bars,
                      plot_optimization_trajectories)
from random_gp_function import RandomGPFunction
from botorch.sampling.pathwise import draw_kernel_feature_paths
from utils import (get_gp, dict_to_fname_str, combine_nested_dicts,
                   convert_to_json_serializable, json_serializable_to_numpy)
from dataset_with_models import RandomModelSampler
from acquisition_function_net import AcquisitionFunctionNet

script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

dim = 6
config = {
    'dim': dim,
    'observation_noise': False,
    'n_initial_samples': 2*(dim+1),
    'n_functions': 4,
    'n_opt_trials_per_function': 3,
    'n_iter': 50
}

observation_noise = config['observation_noise']
n_functions = config['n_functions']

config_str = dict_to_fname_str(config)

def get_rff_function(gp):
    f = draw_kernel_feature_paths(
        copy.deepcopy(gp), sample_shape=torch.Size(), num_features=4096)
    return lambda x: f(x).detach()


# Construct random GP sampler
models = [get_gp(dimension=dim, observation_noise=observation_noise)]
gp_sampler = RandomModelSampler(models, randomize_params=False)
# sample n_functions random GPs and construct random realizations from them
random_gps = [gp_sampler.sample(deepcopy=True) for _ in range(n_functions)]
# Don't use RandomGPFunction because it gives numerical problems if you sample
# too many times.
# However, draw_kernel_feature_paths doesn't work with observation noise
# as far as I can tell.
# (But we're not even testing observation noise currently anyway)
if observation_noise:
    gp_realizations = [
        RandomGPFunction(copy.deepcopy(gp), observation_noise)
        for gp in random_gps]
else:
    gp_realizations = [get_rff_function(gp) for gp in random_gps]
function_names = [f'gp{i}' for i in range(1, n_functions+1)]

bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
init_x = draw_sobol_samples(bounds=bounds, 
                            n=config['n_opt_trials_per_function'],
                            q=config['n_initial_samples'])

experiment_name = 'EI_GP_realizations'
acquisition_functions = {
    'Log EI': LogExpectedImprovement,
    'EI': ExpectedImprovement
}
gp_options = {
    'True GP': {'fit_params': False},
    'MAP': {'fit_params': True, 'mle': False},
    'MLE': {'fit_params': True, 'mle': True}
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

nn_model: AcquisitionFunctionNet = None # TODO
options_dict_nn = {
    'NN': {
        'optimizer_class': NNAcquisitionOptimizer,
        'model': nn_model
    }
}

options_dict = {**options_dict_gp, **options_dict_random, **options_dict_nn}

results = {func_name: {} for func_name in function_names}
desc = (f"Running optimization of {n_functions} functions "
        f"{config['n_opt_trials_per_function']} times each "
        f"with {len(options_dict)} bayesian optimization methods.")
for options_name, options in tqdm(options_dict.items(), desc=desc):
    optimization_results = get_optimization_results(
        objectives=gp_realizations,
        initial_points=init_x,
        n_iter=config['n_iter'],
        objective_names=function_names,
        dim=dim,
        maximize=True,
        bounds=bounds,
        **options
    )
    it = tqdm(optimization_results, desc=f"Optimizing functions with {options_name}")
    for func_name, func_result in it:
        results[func_name][options_name] = func_result
        # print(f"Function {func_name} optimized with {options_name}.")
        # print(f"Best y: {func_result['best_y'][:, -1]}")


# Plot individual functions (up to max_n_functions_to_plot)
max_n_functions_to_plot = 5
n_functions_to_plot = min(n_functions, max_n_functions_to_plot)

scale = 0.5

fig, axes = plt.subplots(1, n_functions_to_plot,
                        figsize=(scale * 10 * n_functions_to_plot, scale * 5),
                        sharex=True, sharey=True)
if n_functions_to_plot == 1:
    axes = [axes]

for func_index in range(n_functions_to_plot):
    func_name = function_names[func_index]
    ax = axes[func_index]

    for options_name in options_dict:
        data = results[func_name][options_name]
        best_y = data['best_y']
        plot_optimization_trajectories_error_bars(ax, best_y, options_name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best function value')
    ax.set_title(f'Function {func_name}')
    ax.legend()
plt.title(config_str)
plt.tight_layout()
filename = f"individual_functions_optimization_{config_str}.pdf"

plt.savefig(os.path.join(PLOTS_DIR, filename),
            dpi=300, format='pdf', bbox_inches='tight')

for options_name in options_dict:
    fig, axes = plt.subplots(1, n_functions_to_plot,
                        figsize=(scale * 10 * n_functions_to_plot, scale * 5),
                        sharex=True, sharey=True)
    if n_functions_to_plot == 1:
        axes = [axes]
    
    for func_index in range(n_functions_to_plot):
        func_name = function_names[func_index]
        ax = axes[func_index]
        data = results[func_name][options_name]
        best_y = data['best_y']
        plot_optimization_trajectories(ax, best_y, "")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best function value')
        ax.set_title(f'Function {func_name}')
        ax.legend()
    plt.title(f'{config_str}, {options_name}')
    plt.tight_layout()
    filename = f"individual_functions_optimization_{config_str}_{options_name}.pdf"
    plt.savefig(os.path.join(PLOTS_DIR, filename),
                dpi=300, format='pdf', bbox_inches='tight')

plt.show()

# # Plot aggregate data (TODO) (might not be necessary)
# fig, ax = plt.subplots(figsize=(10, 6))

# all_data = np.concatenate(optimization_best_y_data)
# plot_optimization_trajectory(ax, all_data, 'Aggregate')

# plt.title('Aggregate Optimization Trajectory')
# filename = f"aggregate_optimization_{config_str}.pdf"
# plt.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')
# plt.show()


