import argparse
import os
from typing import Any, ClassVar, Optional, Sequence, Tuple, Type
from abc import ABC, abstractmethod

import torch
from torch import nn
from utils_general.io_utils import load_json
from utils_general.saveable_object import SaveableObject
from utils_general.utils import check_subclass, dict_to_cmd_args, dict_to_str
from utils_general.basic_model_save_utils import BasicModelSaveUtils


class TorchModuleSaveUtils(ABC):
    def __init__(self, basic_save_utils: BasicModelSaveUtils):
        if not isinstance(basic_save_utils, BasicModelSaveUtils):
            raise ValueError(
                "basic_save_utils must be an instance of BasicModelSaveUtils")
        self.basic_save_utils = basic_save_utils
        self.models_path = basic_save_utils.models_path
        self.models_subdir_name = basic_save_utils.models_subdir_name

        self._empty_modules_cache = {}
        self._weights_cache = {}
        self._configs_cache = {}
        self._single_train_parser_and_info = None
        self.MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN = {}
        self._cmd_opts_train_to_args_module_paths_cache = {}
    
    module_class: ClassVar[Type[nn.Module]]

    @classmethod
    @abstractmethod
    def load_module_configs_from_path(cls, model_and_info_path: str) -> dict:
        """Load all configuration files from a saved model directory.

        Read configuration files from model_and_info_path and return a complete
        configuration dictionary. Inverse of save_module_configs_to_path.
        """
        pass

    @classmethod
    @abstractmethod
    def add_single_train_args_and_return_info(
        cls, parser: argparse.ArgumentParser) -> Any:
        """Define all command-line arguments needed to train your module.

        Add argument groups to the parser for dataset settings, architecture hyperparameters,
        training options, etc. Return metadata about argument groupings (used by validation).
        """
        pass

    @classmethod
    @abstractmethod
    def validate_single_train_args(cls, args: argparse.Namespace, additional_info: Any):
        """Validate parsed arguments for consistency and set defaults.

        Check required arguments, mutually exclusive options, and conditional requirements.
        Raise ValueError with helpful messages on validation failure. Set sensible defaults.
        """
        pass

    @classmethod
    @abstractmethod
    def initialize_module_from_args(cls, args: argparse.Namespace) -> nn.Module:
        """Create and return an instance of your PyTorch module from validated arguments.

        Extract parameters from args, construct initialization dictionaries, and instantiate
        the module. Returns an empty/untrained model (weights loaded separately).
        """
        pass

    @classmethod
    @abstractmethod
    def get_module_folder_name_and_configs(
        cls, model: nn.Module, args: argparse.Namespace) -> Tuple[str, dict]:
        """Generate unique folder name for this configuration and prepare data for saving.

        Create configuration dictionaries from args and model. Compute content-based hash
        for folder name. Return (folder_name, data_dict) for save_module_configs_to_path.
        """
        pass

    @classmethod
    @abstractmethod
    def save_module_configs_to_path(cls, model_and_info_path: str, data: dict):
        """Persist all configuration data to disk in the specified directory.

        Extract configs from data dict (from get_module_folder_name_and_configs) and save
        as JSON files and/or PyTorch checkpoints. Inverse of load_module_configs_from_path.
        """
        pass

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
        ret = self.load_module_configs_from_path(model_and_info_path)
        self._configs_cache[model_and_info_folder_name] = ret
        return ret

    def get_args_module_paths_from_cmd_args(self, cmd_args:Optional[Sequence[str]]=None):
        parser, additional_info = self.get_single_train_parser_and_info()
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

    def cmd_opts_train_to_args_module_paths(self, cmd_opts_nn):
        s = dict_to_str(cmd_opts_nn)
        if s in self._cmd_opts_train_to_args_module_paths_cache:
            return self._cmd_opts_train_to_args_module_paths_cache[s]
        cmd_args_list_nn = dict_to_cmd_args({**cmd_opts_nn, 'no-save-model': True})
        ret = self.get_args_module_paths_from_cmd_args(cmd_args_list_nn)
        (args_nn, model, model_and_info_name, models_path) = ret
        self._cmd_opts_train_to_args_module_paths_cache[s] = ret
        self.MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN[model_and_info_name] = cmd_opts_nn
        return ret
    
    def get_single_train_parser_and_info(self):
        if self._single_train_parser_and_info is None:
            self._single_train_parser_and_info = self._get_single_train_parser_and_info_impl()
        return self._single_train_parser_and_info
    
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

    def _get_single_train_parser_and_info_impl(self) -> Tuple[argparse.ArgumentParser, Any]:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--no-train',
            action='store_false',
            dest='train',
            help='If set, do not train the model. Default is to train the model.'
        )
        parser.add_argument(
            '--no-save-model',
            action='store_false',
            dest='save_model',
            help=('If set, do not save the model. Default is to save the model. '
                'Only applicable if training the model.')
        )
        parser.add_argument(
            '--load_saved_model',
            action='store_true',
            help='Whether to load a saved model. Set this flag to load the saved model.'
        )

        parser_info = self.add_single_train_args_and_return_info(parser)

        return parser, parser_info
    
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
    
    def __init_subclass__(cls, **kwargs):
        # Validate that subclasses define module_class correctly
        super().__init_subclass__(**kwargs)
        error_message = f"Subclasses of {cls.__name__} must define a valid " \
            f"'module_class' class variable"
        if not hasattr(cls, "module_class"):
            raise TypeError(error_message)
        try:
            check_subclass(cls.module_class, "module_class", nn.Module)
            check_subclass(cls.module_class, "module_class", SaveableObject)
        except ValueError as e:
            raise TypeError(f"{error_message}: {e}") from e
