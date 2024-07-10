import copy
import torch
torch.set_default_dtype(torch.float64)
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement

from bayesopt import GPAcquisitionOptimizer
from random_gp_function import RandomGPFunction
from utils import get_gp, dict_to_fname_str
from dataset_with_models import RandomModelSampler

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

dim = 6
config = {
    'dim': dim,
    'observation_noise': False,
    'n_initial_samples': 2*(dim+1),
    'n_functions': 4,
    'n_opt_trials_per_function': 3,
    'n_iter': 30,
    'fit_params': False,
    'mle': False
}

observation_noise = config['observation_noise']
n_initial_samples = config['n_initial_samples']
n_functions = config['n_functions']
n_opt_trials_per_function = config['n_opt_trials_per_function']
n_iter = config['n_iter']
fit_params = config['fit_params']
mle = config['mle']

config_ = config.copy()
if not fit_params:
    config_.pop('mle')
config_str = dict_to_fname_str(config_)


max_n_functions_to_plot = 5
bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

models = [get_gp(
    dimension=dim, observation_noise=observation_noise,
    )]
gp_sampler = RandomModelSampler(models, randomize_params=False)


random_gps = [gp_sampler.sample(deepcopy=True) for _ in range(n_functions)]
gp_realizations = [
    RandomGPFunction(copy.deepcopy(gp), observation_noise)
    for gp in random_gps]

init_x = draw_sobol_samples(bounds=bounds, 
                            n=n_opt_trials_per_function,
                            q=n_initial_samples).squeeze(0)

optimization_best_y_data = [] # list of n_opt_trials_per_function x n_iter arrays
optimization_best_x_data = [] # list of n_opt_trials_per_function x n_iter x dim arrays
for func_index in trange(n_functions, desc="Optimizing functions"):
    objective = gp_realizations[func_index]
    gp = random_gps[func_index]
    function_best_y_data = []
    function_best_x_data = []

    for trial_index in trange(n_opt_trials_per_function, desc=f"Optimizing function {func_index+1}"):
        optimizer = GPAcquisitionOptimizer(
            dim, maximize=True,
            initial_points=init_x[trial_index],
            objective=objective,
            model=gp,
            acquisition_function_class=LogExpectedImprovement,
            fit_params=False,
            mle=mle,
            bounds=bounds
        )
        optimizer.optimize(n_iter)
        function_best_y_data.append(optimizer.best_y_history.numpy())
        function_best_x_data.append(optimizer.best_x_history.numpy())
    optimization_best_y_data.append(np.array(function_best_y_data))
    optimization_best_x_data.append(np.array(function_best_x_data))


def calculate_mean_and_ci(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ci = 1.96 * std  # 95% confidence interval
    return mean, mean - ci, mean + ci

def plot_optimization_trajectory(ax, data, label):
    mean, lower, upper = calculate_mean_and_ci(data)
    x = range(len(mean))
    ax.plot(x, mean, label=label)
    ax.fill_between(x, lower, upper, alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best function value')
    ax.legend()


# Plot individual functions (up to max_n_functions_to_plot)
n_functions_to_plot = min(n_functions, max_n_functions_to_plot)

scale = 0.5
fig, axes = plt.subplots(1, n_functions_to_plot,
                         figsize=(scale * 10 * n_functions_to_plot, scale * 5), sharex=True)
if n_functions_to_plot == 1:
    axes = [axes]

for func_index in range(n_functions_to_plot):
    ax = axes[func_index]
    plot_optimization_trajectory(ax, optimization_best_y_data[func_index], f'Function {func_index + 1}')

plt.tight_layout()
filename = f"individual_functions_optimization_{config_str}.pdf"
plt.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')
plt.show()

# Plot aggregate data
fig, ax = plt.subplots(figsize=(10, 6))

all_data = np.concatenate(optimization_best_y_data)
plot_optimization_trajectory(ax, all_data, 'Aggregate')

plt.title('Aggregate Optimization Trajectory')
filename = f"aggregate_optimization_{config_str}.pdf"
plt.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')
plt.show()

