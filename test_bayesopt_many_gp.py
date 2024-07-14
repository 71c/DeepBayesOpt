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
from utils import (get_gp, dict_to_fname_str, dict_to_hash,
                   combine_nested_dicts, convert_to_json_serializable,
                   json_serializable_to_numpy, remove_priors)
from dataset_with_models import RandomModelSampler
from acquisition_function_net import AcquisitionFunctionNet

from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.priors.torch_priors import GammaPrior

script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')
RESULTS_DIR = os.path.join(script_dir, 'bayesopt_results')
os.makedirs(PLOTS_DIR, exist_ok=True)

dim = 3
n_functions = 5
opt_config = {
    'n_initial_samples': 2*(dim+1),
    'n_trials_per_function': 3,
    'n_iter': 50,
    'seed': 1234
}
config = {
    'dim': dim,
    'observation_noise': False,
    **opt_config
}
SEED = config['seed']
torch.manual_seed(SEED)

observation_noise = config['observation_noise']
n_trials = config['n_trials_per_function']

config_str = dict_to_fname_str(config)

bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
init_x = draw_sobol_samples(bounds=bounds, 
                            n=n_trials,
                            q=config['n_initial_samples'])

def get_rff_function_and_name(gp):
    gp_copy = copy.deepcopy(gp)
    # Remove priors so that the name of the model doesn't depend on the priors.
    # The priors are not used in the function anyway.
    remove_priors(gp_copy)
    f = draw_kernel_feature_paths(
        gp_copy, sample_shape=torch.Size(), num_features=4096)
    function_hash = dict_to_hash(convert_to_json_serializable(f.state_dict()))
    return (lambda x: f(x).detach()), function_hash

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
gp_sampler = RandomModelSampler(models, randomize_params=True)


# Sample n_functions random GPs and construct random realizations from them
# Don't use RandomGPFunction because it gives numerical problems if you sample
# too many times.
# However, draw_kernel_feature_paths doesn't work with observation noise
# as far as I can tell.
# (But we're not even testing observation noise currently anyway)

# Set seed again for reproducibility
torch.manual_seed(SEED)
if observation_noise:
    random_gps = [gp_sampler.sample(deepcopy=True) for _ in range(n_functions)]
    gp_realizations = [
        RandomGPFunction(copy.deepcopy(gp), observation_noise)
        for gp in random_gps]
    function_names = [f'gp{i}' for i in range(1, n_functions+1)]
else:
    # To get reproducible results even if the number of functions changes,
    # we can sample in this way. (As opposed to sampling the possibly random
    # GPs all at once and then constructing all the random functions from them.)
    random_gps = []
    gp_realizations = []
    function_names = []
    for _ in range(n_functions):
        gp = gp_sampler.sample(deepcopy=True)
        random_gps.append(gp)
        gp_realization, realization_hash = get_rff_function_and_name(gp)
        gp_realizations.append(gp_realization)
        function_names.append(f'gp_{realization_hash}')

experiment_name = 'EI_GP_realizations'
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

# nn_model: AcquisitionFunctionNet = None # TODO
# options_dict_nn = {
#     'NN': {
#         'optimizer_class': NNAcquisitionOptimizer,
#         'model': nn_model
#     }
# }

options_dict = {**options_dict_gp,
                **options_dict_random,
                # **options_dict_nn
}


# Run optimization
# Set seed again for reproducibility
torch.manual_seed(SEED)
# Set a seed for each trial
seeds = torch.randint(0, 2**63-1, (n_trials,), dtype=torch.int64)

results = {func_name: {} for func_name in function_names}
desc = (f"Running optimization of {n_functions} functions "
        f"{n_trials} times each "
        f"with {len(options_dict)} bayesian optimization methods.")
for options_name, options in tqdm(options_dict.items(), desc=desc):
    optimization_results = get_optimization_results(
        objectives=gp_realizations,
        initial_points=init_x,
        n_iter=config['n_iter'],
        seeds=seeds,
        objective_names=function_names,
        save_dir=RESULTS_DIR,
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


