"""Just keeping the old version because idk why not"""

import os
from typing import Callable, List, Type, Optional
import torch
from torch import Tensor
from tqdm import tqdm

from bayesopt.bayesopt import BayesianOptimizer, NNAcquisitionOptimizer
from utils.utils import (aggregate_stats_list, convert_to_json_serializable,
                         dict_to_hash, json_serializable_to_numpy, load_json, save_json)


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
                 verbose=False,
                 result_cache={},
                 **optimizer_kwargs):
        self.objectives = objectives
        self.n_functions = len(objectives)
        self.verbose = verbose

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
            self._cached_results = result_cache
            self._func_results_to_save = [{} for _ in range(self.n_functions)]
            os.makedirs(save_dir, exist_ok=True)

            info_kwargs = dict(include_priors=False, hash_gpytorch_modules=False)
            opt_config_json = convert_to_json_serializable(opt_config, **info_kwargs)
            extra_fn_configs = [
                convert_to_json_serializable(fn_kwargs, **info_kwargs)
                for fn_kwargs in self.optimizer_kwargs_per_function]
            func_opt_configs_list = [
                {**opt_config_json, **fn_config,
                 'optimizer_class': optimizer_class.__name__}
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
        except AssertionError as e:
            raise RuntimeError(
                "Loaded results are inconsistent with current configuration.") from e
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

        return self._cached_results[func_name]

    def _get_cached_trial_result(self, func_index: int, trial_index: int,
                                 return_result: bool=True):
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

        if return_result:
            return json_serializable_to_numpy(trials_dict[trial_config_str])
        else:
            return True

    def _get_trial_result(self, func_index: int, trial_index: int, verbose=False):
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
            optimizer.optimize(self.n_iter, verbose=verbose)
            trial_result = optimizer.get_stats()
            if self.save_dir is not None:
                h = self.trial_configs_str[trial_index]
                self._func_results_to_save[func_index][h] = \
                    convert_to_json_serializable(trial_result)

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
            results_dict[func_opt_config_str]['trials'].update(new_trial_results_dict)

        for trial_config_str in new_trial_results_dict:
            if trial_config_str not in func_results['trial_configs']:
                func_results['trial_configs'][trial_config_str] = self.trial_configs[trial_config_str]

        self._func_results_to_save[func_index] = {}

        save_json(func_results, self._results_fname(func_name), indent=4)

    def add_pbar(self, pbar):
        self.pbar = pbar

    def n_opts_to_run(self):
        ret = 0
        for func_index in range(self.n_functions):
            trial_indices_not_cached_func = self._trial_indices_not_cached[func_index]
            for trial_index in trial_indices_not_cached_func:
                if self.save_dir is None:
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

                self._save_func_results(func_index)

                if n_funcs_to_optimize > 1:
                    pbar.update(1)

            result = aggregate_stats_list(results_func)

            if self.objective_names is not None:
                func_name = self.objective_names[func_index]
            else:
                func_name = f"Function_{func_index}"

            yield func_name, result

        if n_funcs_to_optimize > 1:
            pbar.close()
