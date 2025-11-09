from abc import ABC, abstractmethod
from typing import Any, Callable, Type, Optional, List, Union
import copy
import os
import time
import math
import warnings

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import Tensor
import botorch
from botorch.optim import optimize_acqf
from botorch.generation.gen import TGenCandidates
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.pathwise import draw_kernel_feature_paths
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions import UnsupportedError
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement

from bayesopt.random_gp_function import RandomGPFunction
from nn_af.acquisition_function_net_save_utils import load_nn_acqf_configs
from utils.utils import (
    add_outcome_transform, aggregate_stats_list, combine_nested_dicts,
    convert_to_json_serializable, dict_to_hash, json_serializable_to_numpy,
    load_json, remove_priors, sanitize_file_name, save_json)
from utils.plot_utils import plot_optimization_trajectories_error_bars

from datasets.dataset_with_models import RandomModelSampler
from nn_af.acquisition_function_net import (
    AcquisitionFunctionNet, AcquisitionFunctionNetModel,
    AcquisitionFunctionNetAcquisitionFunction, ExpectedImprovementAcquisitionFunctionNet)


class BayesianOptimizer(ABC):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 bounds: Tensor):
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
        if not (bounds.dim() == 2 and bounds.size(0) == 2 and bounds.size(1) == dim):
            raise ValueError("Invalid bounds")
        self.bounds = bounds
        self.objective = objective
        
        self.x = initial_points
        self.y = self.objective(self.x)
        
        self._best_y_history = []
        self._best_x_history = []
        self._time_history = []
        self._process_time_history = []
        self.update_best(0.0, 0.0) # dummy time
    
    @property
    def best_y_history(self):
        return torch.stack(self._best_y_history)
    
    @property
    def best_x_history(self):
        return torch.stack(self._best_x_history)

    @property
    def time_history(self):
        return torch.tensor(self._time_history)
    
    @property
    def process_time_history(self):
        return torch.tensor(self._process_time_history)
    
    def get_stats(self):
        return {
            'y': self.y.numpy(),
            'best_y': self.best_y_history.numpy(),
            'x': self.x.numpy(),
            'best_x': self.best_x_history.numpy(),
            'time': self.time_history.numpy(),
            'process_time': self.process_time_history.numpy()
        }
    
    @abstractmethod
    def get_new_point(self) -> Tensor:
        """Get the new point to sample.
        Return value should be either shape (self.dim,) or (1, self.dim)
        """
        pass  # pragma: no cover
    
    def optimize(self, n_iter: int, verbose=False):
        it = trange(n_iter) if verbose else range(n_iter)
        for i in it:
            start_p = time.process_time()
            start = time.time()
            new_x = self.get_new_point()
            end_p = time.process_time()
            end = time.time()

            # Get to shape n x d (where n=1 here)
            if new_x.dim() == 1:
                new_x = new_x.unsqueeze(0)
            else:
                assert new_x.dim() == 2
            
            new_y = self.objective(new_x)

            # Make sure new_y has shape n x m
            assert new_y.dim() == 2
            assert new_y.size(0) == new_x.size(0) # n = n (# of data points)
            assert new_y.size(1) == self.y.size(1) # m = m (output dimension)

            self.x = torch.cat([self.x, new_x])
            self.y = torch.cat([self.y, new_y])
            self.update_best(end - start, end_p - start_p)
    
    def update_best(self, time, process_time):
        y = self.y
        best_index = torch.argmax(y).item() if self.maximize else torch.argmin(y).item()
        self.best_f = y[best_index]
        self._best_y_history.append(self.best_f)
        self._best_x_history.append(self.x[best_index])
        self._time_history.append(time)
        self._process_time_history.append(process_time)


class RandomSearch(BayesianOptimizer):
    def get_new_point(self) -> Tensor:
        lb, ub = self.bounds[0], self.bounds[1]
        return torch.rand(self.dim) * (ub - lb) + lb


class TimeLogAcquisitionFunction(AcquisitionFunction):
    def __init__(self, af: AcquisitionFunction) -> None:
        super().__init__(model=None)
        self.af: AcquisitionFunction = af
        self._total_eval_process_time = 0.0
        self._total_eval_time = 0.0
        self._n_evals = 0

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.af.set_X_pending(X_pending)

    def forward(self, X: Tensor) -> Tensor:
        if X.dim() != 3:
            raise UnsupportedError("Does not support")
        n_batches = X.size(0) # X has shape batch_size x q x d
        self._n_evals += n_batches

        start_p = time.process_time()
        start = time.time()
        
        ret = self.af.forward(X)
        
        end_p = time.process_time()
        end = time.time()
        
        self._total_eval_process_time += end_p - start_p
        self._total_eval_time += end - start
        
        return ret

    def get_stats(self):
        return {
            "n_evals": self._n_evals,
            "mean_eval_time": self._total_eval_time / self._n_evals,
            "mean_eval_process_time": self._total_eval_process_time / self._n_evals
        }


class SimpleAcquisitionOptimizer(BayesianOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 bounds: Tensor,
                 num_restarts: int,
                 raw_samples: int,
                 gen_candidates: Optional[TGenCandidates]=None,
                 options: Optional[dict[str, Union[bool, float, int, str]]]=None):
        super().__init__(dim, maximize, initial_points, objective, bounds)
        self._acq_history = []
        self._optimize_process_times = []
        self._optimize_times = []
        self._optimize_stats_history = []
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.gen_candidates = gen_candidates
        self.options = options
    
    @property
    def acq_history(self):
        return torch.tensor(self._acq_history)

    @property
    def acq_history_exponentiated(self):
        # Raises AttributeError if not exist
        return torch.tensor(self._acq_history_exponentiated)

    @property
    def optimize_process_times(self):
        return torch.tensor(self._optimize_process_times)

    @property
    def optimize_times(self):
        return torch.tensor(self._optimize_times)
    
    def get_stats(self):
        optimize_stats_ = aggregate_stats_list(self._optimize_stats_history)
        ret = {
            **super().get_stats(),
            **optimize_stats_,
            'acqf_value': self.acq_history.numpy(),
            'optimize_process_time': self.optimize_process_times.numpy(),
            'optimize_time': self.optimize_times.numpy()
        }
        try:
            ret['acqf_value_exponentiated'] = self.acq_history_exponentiated.numpy()
        except AttributeError:
            pass
        return ret

    @abstractmethod
    def get_acquisition_function(self, **extra_kwargs) -> AcquisitionFunction:
        """Get the acquisition function to be optimized for the next iteration
        """
        pass  # pragma: no cover

    def get_new_point(self):
        self._acq_function = TimeLogAcquisitionFunction(
            self.get_acquisition_function()
        )
        start_p = time.process_time()
        start = time.time()
        new_point, new_point_acquisition_val = optimize_acqf(
            acq_function=self._acq_function,
            bounds=self.bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            gen_candidates=self.gen_candidates,
            options=self.options
        )
        # num_restarts=10 * self.dim,
        # raw_samples=200 * self.dim,
        end_p = time.process_time()
        end = time.time()
        self._optimize_process_times.append(end_p - start_p)
        self._optimize_times.append(end - start)
        self._acq_history.append(new_point_acquisition_val.item())
        
        self._optimize_stats_history.append(self._acq_function.get_stats())
        return new_point.detach()


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
                 bounds: Tensor,
                 acquisition_function_class: Type[AcquisitionFunction],
                 num_restarts: int,
                 raw_samples: int,
                 gen_candidates: Optional[TGenCandidates]=None,
                 options: Optional[dict[str, Union[bool, float, int, str]]]=None,
                 **acqf_kwargs):
        super().__init__(dim, maximize, initial_points, objective, bounds,
                         num_restarts, raw_samples, gen_candidates, options)
        self.acqf_kwargs = acqf_kwargs
        self.acquisition_function_class = acquisition_function_class
        self._acquisition_args = get_all_args(acquisition_function_class)
    
    @abstractmethod
    def get_model(self) -> Model:
        """Get the model for giving to the acquisition function"""
        pass  # pragma: no cover
    
    def get_acquisition_function(self, **extra_kwargs) -> AcquisitionFunction:
        model = self.get_model()
        acqf_kwargs = {**self.acqf_kwargs}
        if 'best_f' in self._acquisition_args:
            acqf_kwargs['best_f'] = self.best_f
        if 'maximize' in self._acquisition_args:
            acqf_kwargs['maximize'] = self.maximize
        acqf_kwargs.update(extra_kwargs)
        return self.acquisition_function_class(model, **acqf_kwargs)


class NNAcquisitionOptimizer(ModelAcquisitionOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 bounds: Tensor,
                 model: AcquisitionFunctionNet,
                 num_restarts: int,
                 raw_samples: int,
                 gen_candidates: Optional[TGenCandidates]=None,
                 options: Optional[dict[str, Union[bool, float, int, str]]]=None,
                 **acqf_kwargs):
        super().__init__(
            dim, maximize, initial_points, objective, bounds,
            acquisition_function_class=AcquisitionFunctionNetAcquisitionFunction,
            num_restarts=num_restarts, raw_samples=raw_samples,
            gen_candidates=gen_candidates, options=options,
            **acqf_kwargs
        )
        self.model = model
        
        if isinstance(model, ExpectedImprovementAcquisitionFunctionNet) \
            and not model.includes_alpha:
            # If it is ExpectedImprovementAcquisitionFunctionNet, then we could either
            # be using MSE EI method or policy EI method.
            # If the former, we want to keep track of the "exponentiated" AF values;
            # if the latter, we don't need to. But since it's easier, let's just still
            # keep track of it regardless as long as it doesn't give error because it's
            # ok if we don't end up using it.
            self._is_ei = True
            self._acq_history_exponentiated = []
        else:
            self._is_ei = False
    
    def get_new_point(self):
        new_point = super().get_new_point()
        if self._is_ei:
            exponentiated_af = self.get_acquisition_function(exponentiate=True)
            new_point_acquisition_val = exponentiated_af(new_point)
            self._acq_history_exponentiated.append(new_point_acquisition_val.item())
        return new_point

    def get_model(self):
        # Assumed that the NN was trained to maximize... so this hack should probably work
        y = self.y if self.maximize else -self.y

        # Need to put it on the GPU
        nn_device = next(self.model.parameters()).device
        return AcquisitionFunctionNetModel(
            self.model, self.x.to(nn_device), y.to(nn_device))


class GPAcquisitionOptimizer(ModelAcquisitionOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 bounds: Tensor,
                 model: SingleTaskGP,
                 acquisition_function_class: Type[AcquisitionFunction],
                 num_restarts: int,
                 raw_samples: int,
                 fit_params: bool,
                 mle: bool=False,
                 gen_candidates: Optional[TGenCandidates]=None,
                 options: Optional[dict[str, Union[bool, float, int, str]]]=None,
                 **acqf_kwargs):
        self.model_fitting_errors_count = 0  # Keep track of total count
        self.model_fitting_errors_history = []  # Track cumulative count per iteration

        super().__init__(
            dim, maximize, initial_points, objective, bounds,
            acquisition_function_class=acquisition_function_class,
            num_restarts=num_restarts, raw_samples=raw_samples,
            gen_candidates=gen_candidates, options=options,
            **acqf_kwargs
        )
        
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

        if issubclass(acquisition_function_class, LogExpectedImprovement):
            self._is_ei = True
            self._is_log = True
            self._acq_history_exponentiated = []
        elif issubclass(acquisition_function_class, ExpectedImprovement):
            self._is_ei = True
            self._is_log = False
            self._acq_history_exponentiated = []
        else:
            self._is_ei = False
    
    def get_new_point(self):
        new_point = super().get_new_point()
        if self._is_ei:
            new_point_acquisition_val = self._acq_history[-1]
            if self._is_log:
                new_point_acquisition_val = math.exp(new_point_acquisition_val)
            self._acq_history_exponentiated.append(new_point_acquisition_val)
        return new_point

    def update_best(self, time, process_time):
        super().update_best(time, process_time)
        # Append current cumulative count to history after each iteration
        self.model_fitting_errors_history.append(self.model_fitting_errors_count)

    def get_stats(self):
        ret = super().get_stats()
        ret['model_fitting_errors'] = torch.tensor(self.model_fitting_errors_history).numpy()
        return ret

    def get_model(self):
        if self.fit_params:
            self.model.train()
        
        self.model.set_train_data_with_transforms(
            self.x, self.y, strict=False, train=self.fit_params)
        
        if self.fit_params:
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            try:
                fit_gpytorch_mll(mll)
            except botorch.exceptions.errors.ModelFittingError as e:
                self.model_fitting_errors_count += 1
                warnings.warn(
                    "Model fitting error: " + str(e) +
                    " Proceeding with unfitted model.",
                    RuntimeWarning
                )

        return self.model


class OptimizationResultsSingleMethod:
    def __init__(self,
                 objectives: List[Callable],
                 initial_points: Tensor,
                 n_iter: int,
                 seeds: List[int],
                 optimizer_class: Type[BayesianOptimizer],
                 y_mins: Optional[List[Tensor]]=None,
                 y_maxs: Optional[List[Tensor]]=None,
                 optimizer_kwargs_per_function: Optional[List[dict]]=None,
                 objective_names: Optional[list[str]]=None,
                 nn_model_name: Optional[str]=None,
                 save_dir: Optional[str]=None,
                 results_name: Optional[str]=None,
                 verbose=False,
                 result_cache={},
                 recompute_results=False,
                 **optimizer_kwargs):
        self.objectives = objectives
        self.n_functions = len(objectives)
        self.verbose = verbose

        if initial_points.dim() != 3:
            raise ValueError("initial_points must be a 3D tensor of shape "
                             "(n_trials_per_function, n_initial_samples, dim) "
                             f"but got shape {tuple(initial_points.shape)}")
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
        if 'gen_candidates' in opt_config:
            # Need to convert the function to a string
            # in such a way that it is the same every time
            opt_config['gen_candidates'] = opt_config['gen_candidates'].__name__
        if optimizer_class is NNAcquisitionOptimizer:
            if save_dir is not None and nn_model_name is None:
                raise ValueError(
                    "nn_model_name must be provided when saving "
                    "NNAcquisitionOptimizer results (save_dir is not None).")
            opt_config.pop('model')

            # opt_config['nn_model_name'] = nn_model_name

            opt_config['nn_acqf'] = {
                'name': nn_model_name,
                'model': optimizer_kwargs['model'].get_info_dict(),
                **load_nn_acqf_configs(nn_model_name)
            }
            
        elif nn_model_name is not None:
            raise ValueError(
                "nn_model_name must not be provided if not using NNAcquisitionOptimizer.")

        self.results_name = results_name
        self.recompute_results = recompute_results
        
        self.save_dir = save_dir
        if save_dir is not None:
            if objective_names is None:
                raise ValueError(
                    "objective_names must be provided when saving results.")
            self._cached_results = result_cache
            self._func_results_to_save = [{} for _ in range(self.n_functions)]
            os.makedirs(save_dir, exist_ok=True)

            info_kwargs = dict(include_priors=True, hash_gpytorch_modules=False)
            
            opt_config_json = convert_to_json_serializable(opt_config, **info_kwargs)
            extra_fn_configs = [
                convert_to_json_serializable(fn_kwargs, **info_kwargs)
                for fn_kwargs in self.optimizer_kwargs_per_function]
            func_opt_configs_list = [
                {**opt_config_json, **fn_config,
                 # This is a fix to make incomplete code work. TODO: remove this check of whether it's None
                 'optimizer_class': None if optimizer_class is None else optimizer_class.__name__}
                for fn_config in extra_fn_configs
            ]
            # self.func_opt_configs_str is fucked up
            self.func_opt_configs_str = list(map(dict_to_hash, func_opt_configs_list))
            self.func_opt_configs = dict(zip(self.func_opt_configs_str, func_opt_configs_list))

            trial_configs_list = [
                convert_to_json_serializable(
                    {'seed': seeds[i], 'initial_points': initial_points[i]}
                ) for i in range(self.n_trials_per_function)
            ]
            self.trial_configs_str = list(map(dict_to_hash, trial_configs_list))
            self.trial_configs = dict(zip(self.trial_configs_str, trial_configs_list))

        self._trial_indices_not_cached = [[] for _ in range(self.n_functions)]
        self._results = [[None for _ in range(self.n_trials_per_function)]
                         for _ in range(self.n_functions)]
        for func_index in range(self.n_functions):
            for trial_index in range(self.n_trials_per_function):
                if recompute_results:
                    cached_trial_result = None
                else:
                    cached_trial_result = self._get_cached_trial_result(
                        func_index, trial_index, return_result=False)
                if cached_trial_result is None:
                    self._trial_indices_not_cached[func_index].append(trial_index)
                else:
                    self._results[func_index][trial_index] = cached_trial_result

        self.n_trials_to_run_per_func = [
            len(trial_indices_not_cached_func)
            for trial_indices_not_cached_func in self._trial_indices_not_cached]

        self.n_funcs_to_optimize = sum(x > 0 for x in self.n_trials_to_run_per_func)

        for name in ['y_mins', 'y_maxs']:
            var = locals()[name]
            if var is not None:
                if len(var) != self.n_functions:
                    raise ValueError(f"{name} must be a list of length n_functions.")
                for i in range(self.n_functions):
                    v = var[i]
                    if not torch.is_tensor(v) or v.dim() != 1 or v.size(0) != 1:
                        raise ValueError(f"Each element of {name} must be a tensor of shape (1,).")
                    var[i] = v.numpy()
        self.y_mins: Optional[List[np.ndarray]] = y_mins
        self.y_maxs: Optional[List[np.ndarray]] = y_maxs

    def __len__(self):
        return self.n_functions
    
    def _add_regret_to_result(self, result: dict, func_index: int):
        y_min = None if self.y_mins is None else self.y_mins[func_index]
        y_max = None if self.y_maxs is None else self.y_maxs[func_index]
        if self.optimizer_kwargs['maximize']:
            if y_max is not None:
                result['regret'] = y_max - result['best_y']
        else:
            if y_min is not None:
                result['regret'] = result['best_y'] - y_min
        if y_min is not None and y_max is not None:
            result['normalized_regret'] = result['regret'] / (y_max - y_min)

    def _get_cached_trial_result(self, func_index: int, trial_index: int,
                                 return_result: bool=True):
        if self.save_dir is None:
            return None
        func_name = self.objective_names[func_index]
        func_opt_config_str = self.func_opt_configs_str[func_index]
        trial_config_str = self.trial_configs_str[trial_index]
        key = (func_name, func_opt_config_str, trial_config_str)
        try:
            trial_result = self._cached_results[key]
        except KeyError:
            data_path = os.path.join(
                self.save_dir, func_name, "results",
                func_opt_config_str, "trials", trial_config_str + ".json")
            if return_result:
                try:
                    trial_result = load_json(data_path)
                    self._cached_results[key] = trial_result
                except FileNotFoundError:
                    return None
            else:
                return True if os.path.exists(data_path) else None
        if return_result:
            result = json_serializable_to_numpy(trial_result)
            self._add_regret_to_result(result, func_index)
            return result
        else:
            return True

    def _get_trial_result(self, func_index: int, trial_index: int, verbose=False):
        trial_result = None
        if not self.recompute_results and self.save_dir is not None:
            trial_result = self._get_cached_trial_result(func_index, trial_index)

        if trial_result is None:
            torch.manual_seed(self.seeds[trial_index])
            optimizer = self.optimizer_class(
                initial_points=self.initial_points[trial_index],
                objective=self.objectives[func_index],
                **self.optimizer_kwargs,
                **self.optimizer_kwargs_per_function[func_index]
            )
            optimizer.optimize(self.n_iter, verbose=verbose)
            trial_result = optimizer.get_stats()
            self._add_regret_to_result(trial_result, func_index)
            if self.save_dir is not None:
                trial_config_str = self.trial_configs_str[trial_index]
                self._func_results_to_save[func_index][trial_config_str] = \
                    convert_to_json_serializable(trial_result)

        return trial_result

    def _save_func_results(self, func_index: int):
        if self.save_dir is None:
            return

        new_trial_results_dict = self._func_results_to_save[func_index]
        if not new_trial_results_dict:
            return

        func_name = self.objective_names[func_index]
        func_opt_config_str = self.func_opt_configs_str[func_index]

        func_dir = os.path.join(self.save_dir, func_name)
        os.makedirs(func_dir, exist_ok=True)
        
        # Save any new trial configs
        trial_configs_dir = os.path.join(func_dir, "trial_configs")
        os.makedirs(trial_configs_dir, exist_ok=True)
        for trial_config_str in new_trial_results_dict:
            trial_config_path = os.path.join(trial_configs_dir, trial_config_str + ".json")
            if not os.path.exists(trial_config_path):
                save_json(self.trial_configs[trial_config_str], trial_config_path, indent=4)
        
        # Save results in separate files under results directory
        results_dir = os.path.join(func_dir, "results")
        opt_config_dir = os.path.join(results_dir, func_opt_config_str)
        os.makedirs(opt_config_dir, exist_ok=True)
        
        # Save opt_config if it doesn't exist
        opt_config_path = os.path.join(opt_config_dir, "opt_config.json")
        opt_config = self.func_opt_configs[func_opt_config_str]
        try:
            existing_opt_config = load_json(opt_config_path)
            assert existing_opt_config == opt_config
        except FileNotFoundError:
            save_json(opt_config, opt_config_path, indent=4)
        
        # Save new trial results
        trials_dir = os.path.join(opt_config_dir, "trials")
        os.makedirs(trials_dir, exist_ok=True)
        for trial_config_str, trial_result in new_trial_results_dict.items():
            trial_file_path = os.path.join(trials_dir, trial_config_str + ".json")
            save_json(trial_result, trial_file_path, indent=4)
            print(f"Saved trial result to {trial_file_path}")
        
        self._func_results_to_save[func_index] = {}

    def add_pbar(self, pbar):
        self.pbar = pbar

    def n_opts_to_run(self):
        ret = 0
        for func_index in range(self.n_functions):
            trial_indices_not_cached_func = self._trial_indices_not_cached[func_index]
            for trial_index in trial_indices_not_cached_func:
                if self.save_dir is None or self.recompute_results:
                    is_cached = False
                else:
                    result = self._get_cached_trial_result(
                        func_index, trial_index, return_result=False)
                    is_cached = result is not None
                if not is_cached:
                    ret += 1
        return ret

    def __iter__(self):
        n_funcs_to_optimize = self.n_funcs_to_optimize
        trial_indices_not_cached = self._trial_indices_not_cached
        results = self._results

        prefix = f"{self.results_name}: " if self.results_name is not None else ""

        if n_funcs_to_optimize > 0:
            if n_funcs_to_optimize > 1:
                desc = f"{prefix}Optimizing {n_funcs_to_optimize} functions"
                pbar = tqdm(total=n_funcs_to_optimize, desc=desc)
                verbose = self.verbose
            else:
                print(f"{prefix}: Optimizing 1 function")
                verbose = True

        for func_index in range(self.n_functions):
            trial_indices_not_cached_func = trial_indices_not_cached[func_index]
            results_func = results[func_index]

            for trial_index in range(self.n_trials_per_function):
                if results_func[trial_index] == True:
                    # We know it's there but didn't load the JSON yet
                    results_func[trial_index] = self._get_cached_trial_result(
                        func_index, trial_index, return_result=True)

            if trial_indices_not_cached_func:
                n_cached = self.n_trials_per_function - len(trial_indices_not_cached_func)
                if n_cached > 0:
                    desc = (f"{prefix}  Function {func_index+1}: {n_cached} trials cached, "
                            f"{len(trial_indices_not_cached_func)} trials to optimize.")
                else:
                    desc = f"{prefix}  Optimizing function {func_index+1}"

                it = trial_indices_not_cached_func
                if len(it) > 1:
                    it = tqdm(it, desc=desc)
                else:
                    print(desc)
                for trial_index in it:
                    results_func[trial_index] = self._get_trial_result(
                        func_index, trial_index, verbose=verbose)
                    if hasattr(self, 'pbar'):
                        self.pbar.update(1)
                
                if self.save_dir is not None:
                    self._save_func_results(func_index)

                if n_funcs_to_optimize > 1:
                    pbar.update(1)
                
                trial_indices_not_cached[func_index] = []
            
            if self.objective_names is not None:
                func_name = self.objective_names[func_index]
            else:
                func_name = f"Function_{func_index}"
            
            if self.save_dir is not None:
                func_dir = os.path.join(self.save_dir, func_name)
                results_dir = os.path.join(func_dir, "results")
                func_opt_config_str = self.func_opt_configs_str[func_index]
                opt_config_dir = os.path.join(results_dir, func_opt_config_str)
                trials_dir = os.path.join(opt_config_dir, "trials")
            else:
                trials_dir = None

            result = aggregate_stats_list(results_func)

            yield func_name, trials_dir, result

        if n_funcs_to_optimize > 1:
            pbar.close()


class OptimizationResultsMultipleMethods:
    def __init__(self,
            options_dict: dict[str, dict[str, Any]],
            objectives: List[Callable],
            initial_points: Tensor,
            n_iter: int,
            seed: int,
            objective_names: Optional[list[str]]=None,
            save_dir: Optional[str]=None,
            **universal_optimizer_kwargs):
        n_trials = initial_points.size(0)
        self.objective_names = objective_names
        self.options_dict = options_dict

        # Set seed again for reproducibility
        torch.manual_seed(seed)
        # Set a seed for each trial
        seeds = torch.randint(0, 2**63-1, (n_trials,), dtype=torch.int64)

        todo = []
        n_funcs_to_optimize_per_method = []
        n_trials_list = []
        for options_name, options in options_dict.items():
            optimization_results = OptimizationResultsSingleMethod(
                objectives=objectives,
                initial_points=initial_points,
                n_iter=n_iter,
                seeds=seeds,
                objective_names=objective_names,
                save_dir=save_dir,
                results_name=options_name,
                **universal_optimizer_kwargs,
                **options
            )
            todo.append((options_name, optimization_results))
            n_funcs_to_optimize = optimization_results.n_funcs_to_optimize
            if n_funcs_to_optimize > 0:
                n_trials_per_func = optimization_results.n_trials_to_run_per_func
                n_trials_list.extend([n for n in n_trials_per_func if n > 0])
                n_funcs_to_optimize_per_method.append(n_funcs_to_optimize)
        self.todo = todo
        
        n_methods_to_optimize = len(n_funcs_to_optimize_per_method)
        total_n_funcs_to_optimize = sum(n_funcs_to_optimize_per_method)
        
        self.desc0 = None
        self.desc = None
        if total_n_funcs_to_optimize > 0:
            n_functions = len(objectives)
            n_methods = len(options_dict)
            total_n_trials_to_get = n_methods * n_functions * n_trials
            total_n_trials_to_optimize = sum(n_trials_list)

            if total_n_trials_to_get != total_n_trials_to_optimize:
                total_n_cached_trials = total_n_trials_to_get - total_n_trials_to_optimize
                self.desc0 = (f"Getting optimization results for {n_methods} "
                    f"bayesian optimization methods, on {n_functions} "
                    f"functions, {n_trials} trials each function, "
                    f"with {total_n_trials_to_get} total trials. "
                    f"{total_n_cached_trials} trials already ran.")                

            descs = [f"Running {n_methods_to_optimize} bayesian optimization methods"]
            
            min_funcs = min(n_funcs_to_optimize_per_method)
            max_funcs = max(n_funcs_to_optimize_per_method)
            if min_funcs == max_funcs:
                descs.append(f"on {min_funcs} functions per method")
            else:
                descs.append(f"on {min_funcs}-{max_funcs} functions per method")
            
            min_n, max_n = min(n_trials_list), max(n_trials_list)
            if min_n == max_n:
                descs.append(f"{min_n} times per method+function")
            else:
                descs.append(f"{min_n}-{max_n} times per method+function")
            
            descs.append(f"with {total_n_trials_to_optimize} total trials")

            self.desc = ", ".join(descs)
            self.total_n_trials_to_optimize = total_n_trials_to_optimize
    
    def __len__(self):
        return sum(
            len(optimization_results) for _, optimization_results in self.todo)

    def __iter__(self):
        desc0, desc = self.desc0, self.desc
        if desc0 is not None:
            print(desc0)
        if desc is not None:
            print(desc)
            pbar = tqdm(total=self.total_n_trials_to_optimize, desc=desc)
        
        for options_name, optimization_results in self.todo:
            if desc is not None:
                optimization_results.add_pbar(pbar)
            for func_name, func_result in optimization_results:
                yield func_name, options_name, func_result

        if desc is not None:
            pbar.close()


def plot_optimization_results_multiple_methods(
        optimization_results: OptimizationResultsMultipleMethods,
        max_n_functions_to_plot: int=5,
        alpha=0.05,
        sharey=True,
        aspect=2.0,
        scale=0.5,
        combined_options_to_plot: Optional[dict[str, list[str]]]=None,
        objective_names_plot: Optional[list[str]]=None,
        plots_fname_desc: Optional[str]=None,
        plots_title: Optional[str]=None,
        plots_dir:Optional[str]=None):
    # Run (or load) optimization results and save them
    objective_names = optimization_results.objective_names
    results = {func_name: {} for func_name in objective_names}
    for func_name, options_name, func_result in optimization_results:
        results[func_name][options_name] = func_result
    
    objective_names_plot = objective_names if objective_names_plot is None else objective_names_plot
    n_objectives_to_plot = min(len(objective_names), max_n_functions_to_plot)
    options_dict = optimization_results.options_dict
    
    if plots_title is None:
        plots_title = 'Optimization results' if plots_fname_desc is None else plots_fname_desc
    _plots_fname_desc = '' if plots_fname_desc is None else '_' + sanitize_file_name(
        plots_fname_desc)
    
    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
    
    area = 50 * scale**2
    height = np.sqrt(area / aspect)
    width = aspect * height
    
    # Plot all runs of each optimization method in its own plot
    for options_name in options_dict:
        fig, axes = plt.subplots(1, n_objectives_to_plot,
                            figsize=(width * n_objectives_to_plot, height),
                            sharex=True, sharey=sharey)
        if n_objectives_to_plot == 1:
            axes = [axes]
        
        for func_index in range(n_objectives_to_plot):
            func_name = objective_names[func_index]
            func_plot_name = objective_names_plot[func_index]
            ax = axes[func_index]
            data = results[func_name][options_name]
            best_y = data['best_y']
            plot_optimization_trajectories(ax, best_y, "")
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best function value')
            ax.set_title(f'Function {func_plot_name}')
            ax.legend()
        fig.suptitle(f'{plots_title}\n{options_name}')
        fig.tight_layout()

        if plots_dir is not None:
            filename = sanitize_file_name(
                f"functions_optimization{_plots_fname_desc}_{options_name}.pdf")
            plt.savefig(os.path.join(plots_dir, filename),
                        dpi=300, format='pdf', bbox_inches='tight')
    
    # Plot all optimizatoins means with error bars in one plot
    if combined_options_to_plot is None:
        combined_options_to_plot = {}
    combined_options_to_plot = {
        'all': list(options_dict),
        **combined_options_to_plot
    }

    for options_list_name, options_list in combined_options_to_plot.items():
        fig, axes = plt.subplots(1, n_objectives_to_plot,
                            figsize=(width * n_objectives_to_plot, height),
                            sharex=True, sharey=sharey)
        if n_objectives_to_plot == 1:
            axes = [axes]

        for func_index in range(n_objectives_to_plot):
            func_name = objective_names[func_index]
            func_plot_name = objective_names_plot[func_index]
            ax = axes[func_index]

            for options_name in options_list:
                data = results[func_name][options_name]
                best_y = data['best_y']
                plot_optimization_trajectories_error_bars(
                    ax, best_y, options_name, alpha)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best function value')
            ax.set_title(f'Function {func_plot_name}')
            ax.legend()
        fig.suptitle(f'{plots_title}, {options_list_name}')
        fig.tight_layout()
        
        if plots_dir is not None:
            filename = sanitize_file_name(
                f"functions_optimization{_plots_fname_desc}_{options_list_name}.pdf")
            plt.savefig(os.path.join(plots_dir, filename),
                        dpi=300, format='pdf', bbox_inches='tight')
    
    ## Plot aggregate data (TODO) (might not be necessary)
    ## Old code:
    # fig, ax = plt.subplots(figsize=(10, 6))

    # all_data = np.concatenate(optimization_best_y_data)
    # plot_optimization_trajectory(ax, all_data, 'Aggregate')

    # plt.title('Aggregate Optimization Trajectory')
    # filename = f"aggregate_optimization_{config_str}.pdf"
    # plt.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')


def plot_optimization_trajectories(ax, data, label):
    for i in range(data.shape[0]):
        x = np.arange(data.shape[1])
        label_i = f'trial {i+1}'
        if label != "":
            label_i = f'{label}, {label_i}'
        ax.plot(x, data[i], label=label_i)


def get_rff_function(gp, deepcopy=True, dimension=None, get_hash=False):
    if deepcopy:
        gp = copy.deepcopy(gp)
    # Remove priors so that the name of the model doesn't depend on the priors.
    # The priors are not used in the function anyway.
    remove_priors(gp)
    f = draw_kernel_feature_paths(
        gp, sample_shape=torch.Size(), num_features=4096)

    
    if get_hash:
        j = convert_to_json_serializable(f.state_dict())
        # save_json(j, "test-gpu.json", indent=2)
        function_hash = dict_to_hash(j)

    def func(x):
        if x.dim() == 2:
            # n x d (n_datapoints x dimension)
            if dimension is not None and x.size(1) != dimension:
                raise ValueError(
                    f"Incorrect input {x.shape}: dimension does not match {dimension}")
        else:
            raise ValueError(
                f"Incorrect input {x.shape}: should be of shape n x "
                + (f"{dimension}" if dimension is not None else "d"))
        out = f(x) #.detach()
        assert out.dim() == 1 and out.size(0) == x.size(0)
        return out

    if get_hash:
        return func, function_hash
    return func


def outcome_transform_function(objective_fn, outcome_transform):
    # Need to make a new function for this, otherwise it doesn't work right
    def transformed_gp(x):
        # remember it also takes optional argument Yvar and returns (Y, Yvar)
        return outcome_transform(objective_fn(x))[0]
    return transformed_gp


def transform_functions_and_names(functions,
                                  function_names, function_plot_names,
                                  outcome_transform):
    function_names_tr = [
        sanitize_file_name(f"{name}_{outcome_transform}")
        for name in function_names]
    function_plot_names_tr = [
        f"{name} ({outcome_transform.__class__.__name__} transform)"
        for name in function_plot_names]

    functions_tr = [
        outcome_transform_function(fn, outcome_transform)
        for fn in functions
    ]
    
    return functions_tr, function_names_tr, function_plot_names_tr


def get_random_gp_functions(gp_sampler:RandomModelSampler,
                            seed:int,
                            n_functions:int,
                            observation_noise:bool):
    """Sample n_functions random GPs and construct random realizations from them
    Don't use RandomGPFunction because it gives numerical problems if you
    sample too many times.
    However, draw_kernel_feature_paths doesn't work with observation noise
    as far as I can tell.
    (But we're not even testing observation noise currently anyway)"""

    # Set seed again for reproducibility
    torch.manual_seed(seed)
    function_plot_names = [f'gp{i}' for i in range(1, n_functions+1)]
    if observation_noise:
        random_gps = [
            gp_sampler.sample(deepcopy=True).eval() for _ in range(n_functions)]
        gp_realizations = [
            RandomGPFunction(copy.deepcopy(gp), observation_noise)
            for gp in random_gps]
        function_names = function_plot_names
    else:
        # To get reproducible results even if the number of functions changes,
        # we can sample in this way. (As opposed to sampling the possibly random
        # GPs all at once and then constructing all the random functions from them.)
        random_gps = []
        gp_realizations = []
        function_names = []
        for _ in range(n_functions):
            gp = gp_sampler.sample(deepcopy=True).eval()
            random_gps.append(gp)
            gp_realization, realization_hash = get_rff_function(gp, get_hash=True)
            gp_realizations.append(gp_realization)
            function_names.append(f'gp_{realization_hash}')
    
    return random_gps, gp_realizations, function_names, function_plot_names


def generate_gp_acquisition_options(
        acquisition_functions:dict[str, AcquisitionFunction],
        gps, outcome_transform=None,
        fit_map=False, fit_mle=False):
    gp_params_options = {
        'True GP': {'fit_params': False}
    }
    if fit_map:
        gp_params_options['MAP'] = {'fit_params': True, 'mle': False}
    if fit_mle:
        gp_params_options['MLE'] = {'fit_params': True, 'mle': True}

    kwargs_always_gp = {
        'optimizer_class': GPAcquisitionOptimizer,
    }

    if outcome_transform is None:
        kwargs_always_gp['optimizer_kwargs_per_function'] = [
            {'model': gp} for gp in gps]
        gp_options_to_combine = gp_params_options
    else:
        # Transform the random GPs
        random_gps_tr = []
        for gp in gps:
            new_gp = copy.deepcopy(gp)
            add_outcome_transform(new_gp, outcome_transform)
            random_gps_tr.append(new_gp)
        
        transform_opt = {
            'optimizer_kwargs_per_function': [
                {'model': gp} for gp in random_gps_tr]
        }
        nontransform_opt = {
            'optimizer_kwargs_per_function': [
                {'model': gp} for gp in gps]
        }
        
        nontransformed_gps_with_standardize = []
        for gp in gps:
            new_gp = copy.deepcopy(gp)
            new_gp.outcome_transform = Standardize(m=1)
            nontransformed_gps_with_standardize.append(new_gp)
        nontransform_opt_outcome_standardize = {
            'optimizer_kwargs_per_function': [
                {'model': gp} for gp in nontransformed_gps_with_standardize]
        }

        gp_options_true = {
            'True GP params (with transform)': {**gp_params_options['True GP'], **transform_opt}
        }
        gp_transform_options = {
            'true GP model with transform': transform_opt,
            'untransformed GP model': nontransform_opt,
            'untransformed GP model with outcome standardize': nontransform_opt_outcome_standardize
        }
        gp_options_untrue = combine_nested_dicts(
            {k: v for k, v in gp_params_options.items() if k != 'True GP'},
            gp_transform_options)

        gp_options_to_combine = {**gp_options_true, **gp_options_untrue}

    acquisition_function_options = {
        name: {'acquisition_function_class': acq_func_class}
        for name, acq_func_class in acquisition_functions.items()}

    options_to_combine = [acquisition_function_options, gp_options_to_combine]

    return {
        key: {
            **kwargs_always_gp,
            **value
        } for key, value in combine_nested_dicts(*options_to_combine).items()
    }
