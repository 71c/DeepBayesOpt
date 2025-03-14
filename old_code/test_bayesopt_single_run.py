import copy
import torch
torch.set_default_dtype(torch.float64)
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement
from bayesopt import GPAcquisitionOptimizer
from utils.random_gp_function import RandomGPFunction
from utils.utils import get_gp
from datasets.dataset_with_models import RandomModelSampler


dim = 6
bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
observation_noise = False
n_initial_samples = 3 #2*(dim+1)

models = [get_gp(
    dimension=dim, observation_noise=observation_noise,
    )]
gp_sampler = RandomModelSampler(models, randomize_params=False)

gp = gp_sampler.sample(deepcopy=True)
objective = RandomGPFunction(copy.deepcopy(gp), observation_noise)


init_x = draw_sobol_samples(bounds=bounds, n=1, q=n_initial_samples).squeeze(0)

optimizer = GPAcquisitionOptimizer(
    dim, maximize=True, initial_points=init_x,
    objective=objective,
    model=gp,
    acquisition_function_class=LogExpectedImprovement,
    fit_params=True,
    bounds=bounds
)

optimizer.optimize(20)

print("x:")
print(optimizer.x.shape) # shape (n_initial_samples+n_iter, dim)

print("\ny:")
print(optimizer.y.shape) # shape (n_initial_samples,)

print("\nbest y history:")
print(optimizer.best_y_history.shape) # shape (1+n_iter,)

print("\nbest x history:")
print(optimizer.best_x_history.shape) # shape (1+n_iter, dim)

print("\nacquisition values:")
print(optimizer.acq_history.shape) # shape (n_iter,)
