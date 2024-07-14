from abc import ABC, abstractmethod
import copy
from tqdm import tqdm, trange
from typing import Callable, Type, Optional, List
import numpy as np
import torch
from torch import Tensor
from botorch.optim import optimize_acqf
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from utils import remove_priors
from acquisition_function_net import AcquisitionFunctionNet, AcquisitionFunctionNetModel, LikelihoodFreeNetworkAcquisitionFunction


class BayesianOptimizer(ABC):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 bounds: Optional[Tensor]=None):
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer.")
        if not isinstance(maximize, bool):
            raise ValueError("maximize must be a boolean value.")
        if not (torch.is_tensor(initial_points) and initial_points.dim() == 2 and initial_points.size(1) == dim):
            raise ValueError("initial_points must be a 2D tensor of shape (n, dim)")
        if not callable(objective):
            raise ValueError("objective must be a callable function.")
            
        self.dim = dim
        self.maximize = maximize
        if bounds is None:
            bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        self.bounds = bounds
        self.objective = objective
        
        self.x = initial_points
        self.y = self.objective(self.x)
        
        self._best_y_history = []
        self._best_x_history = []
        self.update_best()
    
    @property
    def best_y_history(self):
        return torch.stack(self._best_y_history)
    
    @property
    def best_x_history(self):
        return torch.stack(self._best_x_history)
    
    @abstractmethod
    def get_new_point(self) -> Tensor:
        """Get the new point to sample.
        Return value should be either shape (self.dim,) or (1, self.dim)
        """
        pass  # pragma: no cover
    
    def optimize(self, n_iter: int):
        for i in range(n_iter):
            new_x = self.get_new_point()
            if new_x.dim() == 1:
                new_x = new_x.unsqueeze(0)
            else:
                assert new_x.dim() == 2
            
            new_y = self.objective(new_x)
            assert new_y.dim() == 1

            self.x = torch.cat([self.x, new_x])
            self.y = torch.cat([self.y, new_y])
            self.update_best()
    
    def update_best(self):
        y = self.y
        best_index = torch.argmax(y).item() if self.maximize else torch.argmin(y).item()
        self.best_f = y[best_index]
        self._best_y_history.append(self.best_f)
        self._best_x_history.append(self.x[best_index])


class RandomSearch(BayesianOptimizer):
    def get_new_point(self) -> Tensor:
        lb, ub = self.bounds[0], self.bounds[1]
        return torch.rand(self.dim) * (ub - lb) + lb


class SimpleAcquisitionOptimizer(BayesianOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 bounds: Optional[Tensor]=None):
        super().__init__(dim, maximize, initial_points, objective, bounds)
        self._acq_history = []
    
    @property
    def acq_history(self):
        return torch.tensor(self._acq_history)

    @abstractmethod
    def get_acquisition_function(self) -> AcquisitionFunction:
        """Get the acquisition function to be optimized for the next iteration
        """
        pass  # pragma: no cover

    def get_new_point(self):
        acq_function = self.get_acquisition_function()
        new_point, new_point_acquisition_val = optimize_acqf(
            acq_function=acq_function,
            bounds=self.bounds,
            q=1,
            num_restarts=10 * self.dim,
            raw_samples=200 * self.dim,
            # options={
            #     "batch_limit": 5,
            #     "maxiter": 200,
            #     "method": "L-BFGS-B",
            # }
        )
        self._acq_history.append(new_point_acquisition_val.item())
        return new_point


import inspect
def get_all_args(func):
    # Get the full argument specification
    sig = inspect.signature(func)
    # Extract the parameters from the signature
    params = sig.parameters
    # Create a list of all parameter names
    return [name for name in params]


class ModelAcquisitionOptimizer(SimpleAcquisitionOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 acquisition_function_class: Type[AcquisitionFunction],
                 bounds: Optional[Tensor]=None,
                 **acqf_kwargs):
        super().__init__(dim, maximize, initial_points, objective, bounds)
        self.acqf_kwargs = acqf_kwargs
        self.acquisition_function_class = acquisition_function_class
        self._acquisition_args = get_all_args(acquisition_function_class)
    
    @abstractmethod
    def get_model(self) -> Model:
        """Get the model for giving to the acquisition function"""
        pass  # pragma: no cover
    
    def get_acquisition_function(self) -> AcquisitionFunction:
        model = self.get_model()
        acqf_kwargs = {**self.acqf_kwargs}
        if 'best_f' in self._acquisition_args:
            acqf_kwargs['best_f'] = self.best_f
        if 'maximize' in self._acquisition_args:
            acqf_kwargs['maximize'] = self.maximize
        return self.acquisition_function_class(model, **acqf_kwargs)


class NNAcquisitionOptimizer(ModelAcquisitionOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 model: AcquisitionFunctionNet,
                 bounds: Optional[Tensor]=None):
        super().__init__(dim, maximize, initial_points, objective, 
                         LikelihoodFreeNetworkAcquisitionFunction, bounds)
        self.model = model
    
    def get_model(self):
        # Assumed that the NN was trained to maximize... so this hack should probably work
        y = self.y if self.maximize else -self.y
        return AcquisitionFunctionNetModel(self.model, self.x, y)


class GPAcquisitionOptimizer(ModelAcquisitionOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 model: SingleTaskGP,
                 acquisition_function_class: Type[AcquisitionFunction],
                 fit_params: bool,
                 mle: bool=False,
                 bounds: Optional[Tensor]=None,
                 **acqf_kwargs):
        super().__init__(dim, maximize, initial_points, objective,
                         acquisition_function_class, bounds, **acqf_kwargs)
        
        self.fit_params = fit_params
        if fit_params:
            # Just so we don't need to worry about anything,
            # we have our own model.
            # If not fitting paramters, then the model shouldn't change so don't
            # need to worry about this.
            model = copy.deepcopy(model)
            
            if mle:
                remove_priors(model)
        
        self.model = model
    
    def get_model(self):
        if self.fit_params:
            self.model.train()
        
        self.model.set_train_data_with_transforms(self.x, self.y.unsqueeze(-1), strict=False, train=self.fit_params)
        
        if self.fit_params:
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_model(mll)

        return self.model


class LazyOptimizationResults:
    def __init__(self,
                 objectives: List[Callable],
                 initial_points: Tensor,
                 n_iter: int,
                 optimizer_class: Type[BayesianOptimizer],
                 optimizer_kwargs_per_function: Optional[List[dict]]=None,
                 objective_names: Optional[list[str]]=None,
                 seeds: Optional[List[int]]=None,
                 **optimizer_kwargs):
        self.objectives = objectives
        self.n_functions = len(objectives)
        
        if initial_points.dim() != 3:
            raise ValueError("initial_points must be a 3D tensor of shape "
                            "(n_opt_trials_per_function, n_initial_samples, dim)")
        self.initial_points = initial_points
        self.n_opt_trials_per_function = initial_points.size(0)

        self.n_iter = n_iter
        self.optimizer_class = optimizer_class

        if optimizer_kwargs_per_function is None:
            optimizer_kwargs_per_function = [{} for _ in range(self.n_functions)]
        elif len(optimizer_kwargs_per_function) != self.n_functions:
            raise ValueError("optimizer_kwargs_per_function must have the same length as objectives.")
        self.optimizer_kwargs_per_function = optimizer_kwargs_per_function

        if objective_names is not None and len(objective_names) != self.n_functions:
            raise ValueError("objective_names must have the same length as objectives.")
        self.objective_names = objective_names
        
        if seeds is not None and len(seeds) != self.n_opt_trials_per_function:
            raise ValueError("seeds must be None or a list of length n_opt_trials_per_function.")
        self.seeds = seeds

        self.optimizer_kwargs = optimizer_kwargs

    def __len__(self):
        return self.n_functions

    def __iter__(self):
        for func_index in range(self.n_functions):
            objective = self.objectives[func_index]
            optimizer_kwargs_for_this_function = self.optimizer_kwargs_per_function[func_index]
            function_best_y_data = []
            function_best_x_data = []

            for trial_index in trange(self.n_opt_trials_per_function,
                                      desc=f"Optimizing function {func_index+1}"):
                if self.seeds is not None:
                    torch.manual_seed(self.seeds[trial_index])
                optimizer = self.optimizer_class(
                    initial_points=self.initial_points[trial_index],
                    objective=objective,
                    **self.optimizer_kwargs,
                    **optimizer_kwargs_for_this_function
                )
                optimizer.optimize(self.n_iter)
                function_best_y_data.append(optimizer.best_y_history.numpy())
                function_best_x_data.append(optimizer.best_x_history.numpy())

            result = {
                # n_opt_trials_per_function x 1+n_iter
                'best_y': np.array(function_best_y_data),
                # n_opt_trials_per_function x 1+n_iter x dim
                'best_x': np.array(function_best_x_data)
            }
            func_name = self.objective_names[func_index] if \
                self.objective_names else f"Function_{func_index}"
            yield func_name, result

def get_optimization_results(objectives: List[Callable],
                             initial_points: Tensor,
                             n_iter: int,
                             optimizer_class: Type[BayesianOptimizer],
                             optimizer_kwargs_per_function: Optional[List[dict]]=None,
                             objective_names: Optional[list[str]]=None,
                             seeds: Optional[List[int]]=None,
                             **optimizer_kwargs) -> LazyOptimizationResults:
    return LazyOptimizationResults(
        objectives, initial_points, n_iter, optimizer_class,
        optimizer_kwargs_per_function, objective_names, seeds,
        **optimizer_kwargs)


def calculate_mean_and_ci(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ci = 1.96 * std  # 95% confidence interval
    return mean, mean - ci, mean + ci


def plot_optimization_trajectories_error_bars(ax, data, label):
    mean, lower, upper = calculate_mean_and_ci(data)
    x = range(len(mean))
    ax.plot(x, mean, label=label)
    ax.fill_between(x, lower, upper, alpha=0.3)

def plot_optimization_trajectories(ax, data, label):
    for i in range(data.shape[0]):
        x = np.arange(data.shape[1])
        label_i = f'trial {i+1}'
        if label != "":
            label_i = f'{label}, {label_i}'
        ax.plot(x, data[i], label=label_i)
