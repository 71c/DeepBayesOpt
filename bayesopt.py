from abc import ABC, abstractmethod
import copy
import os
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
from utils import convert_to_json_serializable, dict_to_hash, json_serializable_to_numpy, load_json, remove_priors, save_json
from json.decoder import JSONDecodeError
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
                 seeds: List[int],
                 optimizer_class: Type[BayesianOptimizer],
                 optimizer_kwargs_per_function: Optional[List[dict]]=None,
                 objective_names: Optional[list[str]]=None,
                 nn_model_name: Optional[str]=None,
                 save_dir: Optional[str]=None,
                 **optimizer_kwargs):
        self.objectives = objectives
        self.n_functions = len(objectives)
        
        if initial_points.dim() != 3:
            raise ValueError("initial_points must be a 3D tensor of shape "
                            "(n_trials_per_function, n_initial_samples, dim)")
        self.initial_points = initial_points
        self.n_trials_per_function = initial_points.size(0)

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
        
        if len(seeds) != self.n_trials_per_function:
            raise ValueError("seeds must be a list of length n_trials_per_function.")
        self.seeds = seeds

        self.optimizer_kwargs = optimizer_kwargs

        opt_config = {
            'n_initial_samples': initial_points.size(1),
            'n_iter': self.n_iter,
            **optimizer_kwargs
        }
        if optimizer_class is NNAcquisitionOptimizer:
            if save_dir is not None and nn_model_name is None:
                raise ValueError(
                    "nn_model_name must be provided when saving "
                    "NNAcquisitionOptimizer results (save_dir is not None).")
            opt_config.pop('model')
            opt_config['nn_model_name'] = nn_model_name
        elif nn_model_name is not None:
            raise ValueError(
                "nn_model_name must not be provided if not using NNAcquisitionOptimizer.")
        
        info_kwargs = dict(include_priors=False, hash_gpytorch_modules=False)
        opt_config_json = convert_to_json_serializable(opt_config, **info_kwargs)
        extra_fn_configs = [
            convert_to_json_serializable(fn_kwargs, **info_kwargs)
            for fn_kwargs in self.optimizer_kwargs_per_function]
        self.func_opt_configs = [
            {**opt_config_json, **fn_config} for fn_config in extra_fn_configs
        ]
        self.func_opt_configs_str = list(map(dict_to_hash, self.func_opt_configs))

        self.trial_configs = [
            convert_to_json_serializable(
                {'seed': seeds[i], 'initial_points': initial_points[i]}
            ) for i in range(self.n_trials_per_function)
        ]
        self.trial_configs_str = list(map(dict_to_hash, self.trial_configs))

        self.save_dir = save_dir
        if save_dir is not None:
            self._cached_results = {}
            os.makedirs(save_dir, exist_ok=True)

    def __len__(self):
        return self.n_functions

    def _results_fname(self, func_name: str):
        return os.path.join(self.save_dir, f'{func_name}.json')

    def _get_func_results(self, func_name: str, reload_result: bool) -> dict:
        if reload_result or func_name not in self._cached_results:
            try:
                func_results = load_json(self._results_fname(func_name))
            except FileNotFoundError:
                func_results = {}
            except JSONDecodeError as e:
                raise RuntimeError("Error decoding json!") from e
            self._cached_results[func_name] = func_results
        return self._cached_results[func_name]

    def _get_opt_trial_result(self, func_name: str,
                            func_opt_config_str: str,
                            trial_config_str: str):
        if self.save_dir is None:
            return None
        func_results = self._get_func_results(func_name, reload_result=False)
        if func_opt_config_str not in func_results:
            return None
        trials_dict = func_results[func_opt_config_str]['trials']
        if trial_config_str not in trials_dict:
            return None
        return json_serializable_to_numpy(
            trials_dict[trial_config_str]['result'])

    def _save_func_opt_trial_results(self,
            func_name: str, func_opt_config: dict, func_opt_config_str: str,
            function_trial_results_to_save: dict):
        if self.save_dir is None:
            return
        func_results = self._get_func_results(func_name, reload_result=True)
        function_trial_results_to_save = convert_to_json_serializable(
            function_trial_results_to_save)
        if func_opt_config_str not in func_results:
            func_results[func_opt_config_str] = {
                'opt_config': func_opt_config,
                'trials': function_trial_results_to_save
            }
        else:
            func_opt_res = func_results[func_opt_config_str]
            assert func_opt_res['opt_config'] == func_opt_config
            func_opt_res['trials'].update(function_trial_results_to_save)
        save_json(func_results, self._results_fname(func_name), indent=4)

    def __iter__(self):
        for func_index in range(self.n_functions):
            if self.objective_names is not None:
                func_name = self.objective_names[func_index]
            else:
                func_name = f"Function_{func_index}"

            objective = self.objectives[func_index]
            
            optimizer_kwargs_for_this_function = self.optimizer_kwargs_per_function[func_index]
            
            func_opt_config = self.func_opt_configs[func_index]
            func_opt_config_str = self.func_opt_configs_str[func_index]
            
            function_best_y_data = []
            function_best_x_data = []
            function_trial_results_to_save = {}

            for trial_index in trange(self.n_trials_per_function,
                                      desc=f"Optimizing function {func_index+1}"):
                trial_config = self.trial_configs[trial_index]
                trial_config_str = self.trial_configs_str[trial_index]
                trial_result = self._get_opt_trial_result(
                    func_name, func_opt_config_str,
                    trial_config_str)
                
                if trial_result is None:
                    torch.manual_seed(self.seeds[trial_index])
                    optimizer = self.optimizer_class(
                        initial_points=self.initial_points[trial_index],
                        objective=objective,
                        **self.optimizer_kwargs,
                        **optimizer_kwargs_for_this_function
                    )
                    optimizer.optimize(self.n_iter)
                    trial_result_save = {
                        **trial_config,
                        'result': {
                            'best_y': optimizer.best_y_history.numpy(),
                            'best_x': optimizer.best_x_history.numpy()
                        }
                    }
                    function_trial_results_to_save[trial_config_str] = trial_result_save
                    trial_result = trial_result_save['result']

                function_best_y_data.append(trial_result['best_y'])
                function_best_x_data.append(trial_result['best_x'])
            
            self._save_func_opt_trial_results(
                func_name, func_opt_config, func_opt_config_str,
                function_trial_results_to_save)

            result = {
                # n_trials_per_function x 1+n_iter
                'best_y': np.array(function_best_y_data),
                # n_trials_per_function x 1+n_iter x dim
                'best_x': np.array(function_best_x_data)
            }
            
            yield func_name, result


def get_optimization_results(objectives: List[Callable],
                             initial_points: Tensor,
                             n_iter: int,
                             seeds: List[int],
                             optimizer_class: Type[BayesianOptimizer],
                             optimizer_kwargs_per_function: Optional[List[dict]]=None,
                             objective_names: Optional[list[str]]=None,
                             nn_model_name: Optional[str]=None,
                             save_dir: Optional[str]=None,
                             **optimizer_kwargs) -> LazyOptimizationResults:
    return LazyOptimizationResults(
        objectives, initial_points, n_iter, seeds, optimizer_class,
        optimizer_kwargs_per_function, objective_names, nn_model_name,
        save_dir, **optimizer_kwargs)


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
