import argparse
import os
from typing import Any, Optional, Sequence, Tuple, Type
from abc import ABC, abstractmethod
from utils_general.basic_model_save_utils import BasicModelSaveUtils
import torch
from torch import nn
from utils_general.io_utils import load_json
from utils_general.saveable_object import SaveableObject
from utils_general.utils import check_subclass, dict_to_cmd_args, dict_to_str


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
        self._single_train_parser_and_info = None
        self.MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN = {}
        self._cmd_opts_nn_to_model_and_info_name_cache = {}

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

    @abstractmethod
    def get_single_train_parser_and_info(self) -> Tuple[argparse.ArgumentParser, Any]:
        pass

    @abstractmethod
    def validate_single_train_args(self, args: argparse.Namespace, additional_info: Any):
        pass

    @abstractmethod
    def initialize_module_from_args(self, args: argparse.Namespace) -> nn.Module:
        pass

    def _get_single_train_parser_and_info(self):
        if self._single_train_parser_and_info is None:
            self._single_train_parser_and_info = self.get_single_train_parser_and_info()
        return self._single_train_parser_and_info

    def get_args_module_paths_from_cmd_args(self, cmd_args:Optional[Sequence[str]]=None):
        parser, additional_info = self._get_single_train_parser_and_info()
        args = parser.parse_args(args=cmd_args)
        self.validate_single_train_args(args, additional_info)

        # Get the untrained model
        # This wastes some resources, but need to do it to get the model's init dict to
        # obtain the correct path for saving the model because that is currently how the
        # model is uniquely identified.
        model = self.initialize_module_from_args(args)

        # Save the configs for the model and training and datasets
        model_and_info_folder_name, models_path = self._get_module_paths_and_save(model, args)

        return args, model, model_and_info_folder_name, models_path
    
    @abstractmethod
    def get_module_folder_name_and_configs(self, model, args) -> Tuple[str, dict]:
        pass

    @abstractmethod
    def save_module_configs_to_path(model_and_info_path, data):
        pass

    def _get_module_paths_and_save(self, model, args):
        model_and_info_folder_name, data = self.get_module_folder_name_and_configs(model, args)
        model_and_info_path = os.path.join(self.models_path, model_and_info_folder_name)
        models_path = os.path.join(model_and_info_path, self.models_subdir_name)

        already_saved = os.path.isdir(model_and_info_path)

        # Assume that all the json files are already saved if the directory exists
        if args.save_model and not already_saved:
            print(f"Saving model and configs to new directory {model_and_info_folder_name}")
            os.makedirs(model_and_info_path, exist_ok=False)
            model.save_init(models_path) # Save model config
            self.save_module_configs_to_path(model_and_info_path, data)

        return model_and_info_folder_name, models_path

    def cmd_opts_train_to_model_and_info_name(self, cmd_opts_nn):
        s = dict_to_str(cmd_opts_nn)
        if s in self._cmd_opts_train_to_model_and_info_name_cache:
            return self._cmd_opts_train_to_model_and_info_name_cache[s]
        cmd_args_list_nn = dict_to_cmd_args({**cmd_opts_nn, 'no-save-model': True})
        ret = self.get_args_module_paths_from_cmd_args(cmd_args_list_nn)
        (args_nn, model, model_and_info_name, models_path) = ret
        self._cmd_opts_train_to_model_and_info_name_cache[s] = ret
        self.MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN[model_and_info_name] = cmd_opts_nn
        return ret
