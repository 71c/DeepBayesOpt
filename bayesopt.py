from abc import ABC, abstractmethod
import copy
import os
from tqdm import tqdm, trange
from typing import Any, Callable, Type, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from botorch.optim import optimize_acqf
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.pathwise import draw_kernel_feature_paths
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from dataset_with_models import RandomModelSampler
from random_gp_function import RandomGPFunction
from utils import add_outcome_transform, combine_nested_dicts, concatenate_outcome_transforms, convert_to_json_serializable, dict_to_hash, invert_outcome_transform, json_serializable_to_numpy, load_json, remove_priors, sanitize_file_name, save_json
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
                 bounds: Optional[Tensor]=None,
                 **acqf_kwargs):
        super().__init__(dim, maximize, initial_points, objective, 
                         LikelihoodFreeNetworkAcquisitionFunction, bounds,
                         **acqf_kwargs)
        self.model = model
    
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
        
        self.model.set_train_data_with_transforms(
            self.x, self.y, strict=False, train=self.fit_params)
        
        if self.fit_params:
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)

        return self.model


class OptimizationResultsSingleMethod:
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
                 results_name: Optional[str]=None,
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

        self.results_name = results_name
    
        self.save_dir = save_dir
        if save_dir is not None:
            if objective_names is None:
                raise ValueError(
                    "objective_names must be provided when saving results.")
            self._cached_results = {}
            self._func_results_to_save = [{} for _ in range(self.n_functions)]
            os.makedirs(save_dir, exist_ok=True)
        
            info_kwargs = dict(include_priors=False, hash_gpytorch_modules=False)
            opt_config_json = convert_to_json_serializable(opt_config, **info_kwargs)
            extra_fn_configs = [
                convert_to_json_serializable(fn_kwargs, **info_kwargs)
                for fn_kwargs in self.optimizer_kwargs_per_function]
            func_opt_configs_list = [
                {**opt_config_json, **fn_config,
                 'optimizer_class': str(optimizer_class)}
                for fn_config in extra_fn_configs
            ]
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
                cached_trial_result = self._get_cached_trial_result(
                    func_index, trial_index)
                if cached_trial_result is None:
                    self._trial_indices_not_cached[func_index].append(trial_index)
                else:
                    self._results[func_index][trial_index] = cached_trial_result
        
        self.n_trials_to_run_per_func = [
            len(trial_indices_not_cached_func)
            for trial_indices_not_cached_func in self._trial_indices_not_cached]
        
        self.n_funcs_to_optimize = sum(x > 0 for x in self.n_trials_to_run_per_func)

    def __len__(self):
        return self.n_functions

    def _results_fname(self, func_name: str):
        return os.path.join(self.save_dir, f'{func_name}.json')

    def _validate_loaded_data(self, func_results: dict):
        """Make sure the loaded results are consistent with the current
        configuration"""
        try:
            # Check that the trial configs are consistent
            trial_configs_dict = func_results['trial_configs']
            for trial_config_str, trial_config in trial_configs_dict.items():
                if trial_config_str in self.trial_configs:
                    assert trial_config == self.trial_configs[trial_config_str]
                else:
                    assert trial_config_str == dict_to_hash(trial_config)

            # Check that the optimization configurations are consistent
            results_dict = func_results['results']
            for func_opt_config_str, opt_results in results_dict.items():
                opt_config = opt_results['opt_config']
                if func_opt_config_str in self.func_opt_configs:
                    assert opt_config == self.func_opt_configs[func_opt_config_str]
                else:
                    assert func_opt_config_str == dict_to_hash(opt_config)
                
                for trial_config_str in opt_results['trials']:
                    assert trial_config_str in trial_configs_dict
        except AssertionError:
            raise RuntimeError("Loaded results are inconsistent with current configuration.")
        except KeyError as e:
            raise RuntimeError("Loaded results are missing keys!") from e

    def _get_func_results(self, func_name: str, reload_result: bool) -> dict:
        if reload_result or func_name not in self._cached_results:
            try:
                func_results = load_json(self._results_fname(func_name))
                self._validate_loaded_data(func_results)
                self._cached_results[func_name] = func_results
            except FileNotFoundError:
                if func_name not in self._cached_results:
                    self._cached_results[func_name] = {
                        'trial_configs': {}, 'results': {}}
            except JSONDecodeError as e:
                raise RuntimeError("Error decoding json!") from e

        return self._cached_results[func_name]

    def _get_cached_trial_result(self, func_index: int, trial_index: int):
        if self.save_dir is None:
            return None
        func_name = self.objective_names[func_index]
        func_results = self._get_func_results(func_name, reload_result=False)
        results_dict = func_results['results']
        
        func_opt_config_str = self.func_opt_configs_str[func_index]
        if func_opt_config_str not in results_dict:
            return None
        trials_dict = results_dict[func_opt_config_str]['trials']
        
        trial_config_str = self.trial_configs_str[trial_index]
        if trial_config_str not in trials_dict:
            return None
        return json_serializable_to_numpy(trials_dict[trial_config_str])
    
    def _get_trial_result(self, func_index: int, trial_index: int):
        trial_result = None
        if self.save_dir is not None:
            trial_result = self._get_cached_trial_result(func_index, trial_index)
        
        if trial_result is None:
            torch.manual_seed(self.seeds[trial_index])
            optimizer = self.optimizer_class(
                initial_points=self.initial_points[trial_index],
                objective=self.objectives[func_index],
                **self.optimizer_kwargs,
                **self.optimizer_kwargs_per_function[func_index]
            )
            optimizer.optimize(self.n_iter)
            trial_result = {
                'best_y': optimizer.best_y_history.numpy(),
                'best_x': optimizer.best_x_history.numpy()
            }
            if self.save_dir is not None:
                h = self.trial_configs_str[trial_index]
                self._func_results_to_save[func_index][h] = convert_to_json_serializable(trial_result)
        
        return trial_result

    def _save_func_results(self, func_index: int):
        if self.save_dir is None:
            return
        
        new_trial_results_dict = self._func_results_to_save[func_index]
        if not new_trial_results_dict:
            return
        
        func_name = self.objective_names[func_index]
        func_results = self._get_func_results(func_name, reload_result=True)
        
        results_dict = func_results['results']
        func_opt_config_str = self.func_opt_configs_str[func_index]
        if func_opt_config_str not in results_dict:
            results_dict[func_opt_config_str] = {
                'opt_config': self.func_opt_configs[func_opt_config_str],
                'trials': new_trial_results_dict
            }
        else:
            func_opt_res = results_dict[func_opt_config_str]
            func_opt_res['trials'].update(new_trial_results_dict)
        
        for trial_config_str in new_trial_results_dict:
            if trial_config_str not in func_results['trial_configs']:
                func_results['trial_configs'][trial_config_str] = self.trial_configs[trial_config_str]
        
        # Probably doesn't matter but it doesn't hurt
        self._func_results_to_save[func_index] = {}

        save_json(func_results, self._results_fname(func_name), indent=4)
    
    def add_pbar(self, pbar):
        self.pbar = pbar

    def __iter__(self):
        n_funcs_to_optimize = self.n_funcs_to_optimize
        trial_indices_not_cached = self._trial_indices_not_cached
        results = self._results

        prefix = f"{self.results_name}: " if self.results_name is not None else ""
        
        if n_funcs_to_optimize > 0:
            desc = f"{prefix}Optimizing {n_funcs_to_optimize} functions"
            pbar = tqdm(total=n_funcs_to_optimize, desc=desc)
        
        for func_index in range(self.n_functions):
            trial_indices_not_cached_func = trial_indices_not_cached[func_index]
            results_func = results[func_index]
            
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
                        func_index, trial_index)
                    if hasattr(self, 'pbar'):
                        self.pbar.update(1)
                    
                self._save_func_results(func_index)

                pbar.update(1)

            # n_trials_per_function x 1+n_iter
            function_best_y_data = np.array([r['best_y'] for r in results_func])
            # n_trials_per_function x 1+n_iter x dim
            function_best_x_data = np.array([r['best_x'] for r in results_func])

            result = {
                'best_y': function_best_y_data,
                'best_x': function_best_x_data
            }

            if self.objective_names is not None:
                func_name = self.objective_names[func_index]
            else:
                func_name = f"Function_{func_index}"
            
            yield func_name, result
        
        if n_funcs_to_optimize > 0:
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


import scipy.stats as stats

def calculate_mean_and_interval(data, alpha=0.05, use_std=False):
    mean = np.mean(data, axis=0)
    
    if use_std:
        std = np.std(data, axis=0)
        x = stats.norm.ppf(1 - alpha / 2)
        ci = x * std 
        lo, hi = mean - ci, mean + ci
    else:
        lo = np.quantile(data, alpha / 2, axis=0)
        hi = np.quantile(data, 1 - alpha / 2, axis=0)

    return mean, lo, hi


def plot_optimization_trajectories_error_bars(ax, data, label, alpha):
    mean, lower, upper = calculate_mean_and_interval(
        data, alpha, use_std=False)
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


def get_rff_function_and_name(gp, deepcopy=True, dimension=None):
    if deepcopy:
        gp = copy.deepcopy(gp)
    # Remove priors so that the name of the model doesn't depend on the priors.
    # The priors are not used in the function anyway.
    remove_priors(gp)
    f = draw_kernel_feature_paths(
        gp, sample_shape=torch.Size(), num_features=4096)
    function_hash = dict_to_hash(convert_to_json_serializable(f.state_dict()))

    def func(x):
        if x.dim() == 2:
            # n x d (n_datapoints x dimension)
            if dimension is not None and x.size(1) != dimension:
                raise ValueError(
                    f"Incorrect input {x.shape}: dimension does not match {dimension}")
        else:
            raise ValueError(
                f"Incorrect input {x.shape}: should be of shape n x {dimension}")
        out = f(x).detach()
        assert out.dim() == 1 and out.size(0) == x.size(0)
        out = out.unsqueeze(-1) # get to shape n x m where m=1 (m = number of outputs)
        return out

    return func, function_hash


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
            gp_realization, realization_hash = get_rff_function_and_name(gp)
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
