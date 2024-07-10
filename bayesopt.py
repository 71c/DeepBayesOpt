from abc import ABC, abstractmethod
import copy
import torch
from torch import Tensor
from typing import Callable, Type, Optional
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
                 objective: Callable):
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
        self.bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        self.objective = objective
        
        self.x = initial_points
        self.y = self.objective(self.x)

        self.best_history = []
        self.update_best()
    
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
        self.best_f = self.y.max().item() if self.maximize else self.y.min().item()
        self.best_history.append(self.best_f)


class RandomSearch(BayesianOptimizer):
    def get_new_point(self) -> Tensor:
        return torch.rand(1, self.dim)


class SimpleAcquisitionOptimizer(BayesianOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable):
        super().__init__(dim, maximize, initial_points, objective)
        self.acq_history = []

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
            options={
                "batch_limit": 5,
                "maxiter": 200,
                "method": "L-BFGS-B",
            }
        )
        self.acq_history.append(new_point_acquisition_val.item())
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
                 **acqf_kwargs):
        super().__init__(dim, maximize, initial_points, objective)
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
                 model: AcquisitionFunctionNet):
        super().__init__(dim, maximize, initial_points, objective, LikelihoodFreeNetworkAcquisitionFunction)
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
                 **acqf_kwargs):
        super().__init__(dim, maximize, initial_points, objective,
                         acquisition_function_class, **acqf_kwargs)
        
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
