import os
from typing import Type
from abc import ABC, abstractmethod
from utils_general.basic_model_save_utils import BasicModelSaveUtils
import torch
from torch import nn
from utils_general.io_utils import load_json
from utils_general.saveable_object import SaveableObject
from utils_general.utils import check_subclass


class TorchModuleSaveUtils(ABC):
    def __init__(self,
                 basic_save_utils: BasicModelSaveUtils,
                 models_version_name: str,
                 module_class: Type[nn.Module]):
        if not isinstance(basic_save_utils, BasicModelSaveUtils):
            raise ValueError(
                "basic_save_utils must be an instance of BasicModelSaveUtils")
        self.basic_save_utils = basic_save_utils
        self.models_path = basic_save_utils.models_path
        self.models_subdir_name = basic_save_utils.models_subdir_name
        
        self.models_version_name = models_version_name

        check_subclass(module_class, "module_class", nn.Module)
        check_subclass(module_class, "module_class", SaveableObject)
        self.module_class = module_class

        self._empty_modules_cache = {}
        self._weights_cache = {}
        self._configs_cache = {}

    def _load_empty(self, model_and_info_path: str):
        # Loads empty model (without weights)
        if model_and_info_path in self._empty_modules_cache:
            return self._empty_modules_cache[model_and_info_path]
        models_path = os.path.join(model_and_info_path, self.models_subdir_name)
        ret = self.module_class.load_init(models_path)
        self._empty_modules_cache[model_and_info_path] = ret
        return ret

    def _get_state_dict(self, weights_path: str, verbose: bool=True):
        if weights_path in self._weights_cache:
            return self._weights_cache[weights_path]
        if verbose:
            print(f"Loading best weights from {weights_path}")
        ret = torch.load(weights_path)
        self._weights_cache[weights_path] = ret
        return ret

    def load_module(
            self,
            model_and_info_folder_name: str,
            return_model_path=False,
            load_weights=True,
            verbose=True):
        model_and_info_path = os.path.join(self.models_path, model_and_info_folder_name)
        model = self._load_empty(model_and_info_path)

        if return_model_path or load_weights:
            model_path = self.basic_save_utils.get_latest_model_path(model_and_info_path)

        if load_weights:
            # print(f"Loading model from {model_path}")
            # Load best weights
            best_model_fname_json_path = os.path.join(model_path, "best_model_fname.json")
            try:
                best_model_fname = load_json(best_model_fname_json_path)["best_model_fname"]
            except FileNotFoundError:
                raise ValueError(f"No best model found: {best_model_fname_json_path} not found")
            best_model_path = os.path.join(model_path, best_model_fname)
            
            state_dict = self._get_state_dict(best_model_path, verbose=verbose)
            model.load_state_dict(state_dict)

        if return_model_path:
            return model, model_path
        return model

    def load_module_configs(self, model_and_info_folder_name: str):
        if model_and_info_folder_name in self._configs_cache:
            return self._configs_cache[model_and_info_folder_name]
        model_and_info_path = os.path.join(self.models_path, model_and_info_folder_name)
        return self.load_module_configs_from_path(model_and_info_path)
    
    @abstractmethod
    def load_module_configs_from_path(self, model_and_info_path: str) -> dict:
        pass
