from functools import cache
import os
from typing import Any, Sequence, Optional
import torch
import argparse

from datasets.hpob_dataset import get_hpob_dataset_dimension
from utils.basic_model_save_utils import BASIC_SAVING
from utils.utils import convert_to_json_serializable_gpytorch
from utils_general.io_utils import load_json, save_json
from utils.constants import MODELS_DIR, MODELS_SUBDIR, MODELS_VERSION

from utils_train.acquisition_function_net import AcquisitionFunctionBodyPointnetV1and2, AcquisitionFunctionBodyTransformerNP, AcquisitionFunctionNet, AcquisitionFunctionNetFinalMLP, ExpectedImprovementAcquisitionFunctionNet, GittinsAcquisitionFunctionNet, TwoPartAcquisitionFunctionNetFixedHistoryOutputDim
from utils_train.train_acquisition_function_net import GI_NORMALIZATIONS, METHODS
from utils_train.acquisition_function_net_constants import POINTNET_ACQF_PARAMS_INPUT_DEFAULT, POINTNET_ACQF_PARAMS_INPUT_OPTIONS

from datasets.dataset_with_models import RandomModelSampler
from datasets.acquisition_dataset_manager import FIX_TRAIN_ACQUISITION_DATASET, get_lamda_min_max
from datasets.gp_acquisition_dataset_manager import GP_GEN_DEVICE
from dataset_factory import add_unified_acquisition_dataset_args, get_dataset_manager, add_lamda_args, validate_args_for_dataset_type
from utils_general.utils import dict_to_cmd_args, dict_to_hash, dict_to_str


@cache
def _load_empty_model(model_and_info_path: str):
    # Loads empty model (without weights)
    models_path = os.path.join(model_and_info_path, MODELS_SUBDIR)
    return AcquisitionFunctionNet.load_init(models_path)


_weights_cache = {}
def _get_state_dict(weights_path: str, verbose: bool=True):
    if weights_path in _weights_cache:
        return _weights_cache[weights_path]
    if verbose:
        print(f"Loading best weights from {weights_path}")
    ret = torch.load(weights_path)
    _weights_cache[weights_path] = ret
    return ret


def load_module(
        model_and_info_folder_name: str,
        return_model_path=False,
        load_weights=True,
        verbose=True):
    model_and_info_path = os.path.join(MODELS_DIR, model_and_info_folder_name)
    model = _load_empty_model(model_and_info_path)

    if return_model_path or load_weights:
        model_path = BASIC_SAVING.get_latest_model_path(model_and_info_path)

    if load_weights:
        # print(f"Loading model from {model_path}")
        # Load best weights
        best_model_fname_json_path = os.path.join(model_path, "best_model_fname.json")
        try:
            best_model_fname = load_json(best_model_fname_json_path)["best_model_fname"]
        except FileNotFoundError:
            raise ValueError(f"No best model found: {best_model_fname_json_path} not found")
        best_model_path = os.path.join(model_path, best_model_fname)
        
        model.load_state_dict(_get_state_dict(best_model_path, verbose=verbose))

    if return_model_path:
        return model, model_path
    return model


@cache
def load_module_configs(model_and_info_folder_name: str):
    model_and_info_path = os.path.join(MODELS_DIR, model_and_info_folder_name)
    return _load_module_configs_from_path(model_and_info_path)


def _load_module_configs_from_path(model_and_info_path: str):
    function_samples_config = load_json(
        os.path.join(model_and_info_path, "function_samples_config.json"))
    acquisition_dataset_config = load_json(
        os.path.join(model_and_info_path, "acquisition_dataset_config.json"))
    n_points_config = load_json(
        os.path.join(model_and_info_path, "n_points_config.json"))
    
    dataset_transform_config = load_json(
        os.path.join(model_and_info_path, "dataset_transform_config.json"))
    if dataset_transform_config['outcome_transform'] is not None:
        dataset_transform_config['outcome_transform'] = torch.load(
            os.path.join(model_and_info_path, "outcome_transform.pt"))
    
    try:
        model_sampler = RandomModelSampler.load(
            os.path.join(model_and_info_path, "model_sampler"))
        if function_samples_config['models'] is not None:
            function_samples_config['models'] = model_sampler.initial_models
            function_samples_config['model_probabilities'] = model_sampler.model_probabilities
            assert model_sampler.randomize_params == function_samples_config['randomize_params']
            function_samples_config['model_sampler'] = model_sampler
    except FileNotFoundError:
        assert not any(
            k in function_samples_config for k in
            ['models', 'model_probabilities', 'randomize_params', 'model_sampler'])
    
    training_config = load_json(
        os.path.join(model_and_info_path, "training_config.json"))
    
    af_dataset_config = {
        "function_samples_config": function_samples_config,
        "acquisition_dataset_config": acquisition_dataset_config,
        "n_points_config": n_points_config,
        "dataset_transform_config": dataset_transform_config
    }
    all_info_json, model_sampler = _json_serialize_nn_acqf_configs(
        training_config=training_config,
        af_dataset_config=af_dataset_config,
        hash_gpytorch_modules=False
    )
    if model_sampler is not None:
        del all_info_json['function_samples_config']['models']
        del all_info_json['function_samples_config']['model_probabilities']
        del all_info_json['function_samples_config']['randomize_params']
    return all_info_json


def get_args_module_paths_from_cmd_args(cmd_args:Optional[Sequence[str]]=None):
    parser, additional_info = get_single_train_parser_and_info()
    args = parser.parse_args(args=cmd_args)
    validate_single_train_args(args, additional_info)

    ################################### Get NN model ###############################
    #### Get the untrained model
    # This wastes some resources, but need to do it to get the model's init dict to
    # obtain the correct path for saving the model because that is currently how the
    # model is uniquely identified.
    model = _initialize_module_from_args(args)

    ############ Save the configs for the model and training and datasets
    model_and_info_folder_name, models_path = _get_module_paths_and_save(model, args)

    return args, model, model_and_info_folder_name, models_path


def _get_module_folder_name_and_configs(model, args):
    training_config = _get_training_config(args)
    manager = get_dataset_manager(getattr(args, 'dataset_type', 'gp'), device="cpu")
    af_dataset_config = manager.get_dataset_configs(args, device=GP_GEN_DEVICE)

    all_info_json, model_sampler = _json_serialize_nn_acqf_configs(
        training_config=training_config,
        af_dataset_config=af_dataset_config
    )
    all_info_json['model'] = model.get_info_dict()
    
    all_info_hash = dict_to_hash(all_info_json)
    model_and_info_folder_name = os.path.join(MODELS_VERSION, f"model_{all_info_hash}")
    
    data = {
        'training_config': training_config,
        'af_dataset_config': af_dataset_config,
        'model_sampler': model_sampler,
        'all_info_json': all_info_json
    }

    return model_and_info_folder_name, data


def _save_module_configs_to_path(model_and_info_path: str, data: dict):
    # Save training config
    save_json(data['training_config'],
            os.path.join(model_and_info_path, "training_config.json"))
    
    all_info_json = data['all_info_json']
    model_sampler = data['model_sampler']
    af_dataset_config = data['af_dataset_config']

    # Save GP dataset config
    function_samples_config_json = all_info_json['function_samples_config']
    acquisition_dataset_config = all_info_json['acquisition_dataset_config']
    n_points_config = all_info_json['n_points_config']
    save_json(function_samples_config_json,
            os.path.join(model_and_info_path, "function_samples_config.json"))
    save_json(acquisition_dataset_config,
            os.path.join(model_and_info_path, "acquisition_dataset_config.json"))
    save_json(n_points_config,
            os.path.join(model_and_info_path, "n_points_config.json"))
    if model_sampler is not None:
        model_sampler.save(
            os.path.join(model_and_info_path, "model_sampler"))

    # Save dataset transform config
    dataset_transform_config_json = all_info_json['dataset_transform_config']
    dataset_transform_config = af_dataset_config['dataset_transform_config']
    save_json(dataset_transform_config_json,
            os.path.join(model_and_info_path, "dataset_transform_config.json"))
    outcome_transform = dataset_transform_config['outcome_transform']
    if outcome_transform is not None:
        torch.save(outcome_transform,
                os.path.join(model_and_info_path, "outcome_transform.pt"))


def _get_module_paths_and_save(model, args):
    model_and_info_folder_name, data = _get_module_folder_name_and_configs(model, args)
    model_and_info_path = os.path.join(MODELS_DIR, model_and_info_folder_name)
    models_path = os.path.join(model_and_info_path, MODELS_SUBDIR)

    already_saved = os.path.isdir(model_and_info_path)

    # Assume that all the json files are already saved if the directory exists
    if args.save_model and not already_saved:
        print(f"Saving model and configs to new directory {model_and_info_folder_name}")
        os.makedirs(model_and_info_path, exist_ok=False)
        model.save_init(models_path) # Save model config
        _save_module_configs_to_path(model_and_info_path, data)

    return model_and_info_folder_name, models_path


def _get_training_config(args: argparse.Namespace):
    training_config = dict(
        method=args.method,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        fix_train_acquisition_dataset=FIX_TRAIN_ACQUISITION_DATASET,
        early_stopping=args.early_stopping,
        use_maxei=args.use_maxei
    )
    if args.method == 'policy_gradient':
        training_config = dict(
            **training_config,
            include_alpha=args.include_alpha,
            learn_alpha=args.learn_alpha,
            initial_alpha=args.initial_alpha,
            alpha_increment=args.alpha_increment)
    elif args.method == 'gittins':
        lamda_min, lamda_max = get_lamda_min_max(args)
        training_config = dict(
            **training_config,
            lamda_min=lamda_min,
            lamda_max=lamda_max,
            gi_loss_normalization=args.gi_loss_normalization)
    elif args.method == 'mse_ei':
        initial_tau = getattr(args, "initial_tau", None)
        initial_tau = 1.0 if initial_tau is None else initial_tau
        training_config = dict(
            **training_config,
            learn_tau=args.learn_tau,
            initial_tau=initial_tau,
            softplus_batchnorm=args.softplus_batchnorm,
            softplus_batchnorm_momentum=args.softplus_batchnorm_momentum,
            positive_linear_at_end=args.positive_linear_at_end,
            gp_ei_computation=args.gp_ei_computation)
    if args.early_stopping:
        training_config = dict(
            **training_config,
            patience=args.patience,
            min_delta=args.min_delta,
            cumulative_delta=args.cumulative_delta)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        training_config = dict(
            **training_config,
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_patience=args.lr_scheduler_patience,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_min_lr=args.lr_scheduler_min_lr,
            lr_scheduler_cooldown=args.lr_scheduler_cooldown)
    elif args.lr_scheduler == 'power':
        training_config = dict(
            **training_config,
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_power=args.lr_scheduler_power,
            lr_scheduler_burnin=args.lr_scheduler_burnin)
    if args.weight_decay is not None:
        training_config = dict(
            **training_config,
            weight_decay=args.weight_decay)
    return training_config


MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN = {}
_cmd_opts_train_to_args_module_paths_cache = {}
def cmd_opts_train_to_args_module_paths(cmd_opts_nn):
    s = dict_to_str(cmd_opts_nn)
    if s in _cmd_opts_train_to_args_module_paths_cache:
        return _cmd_opts_train_to_args_module_paths_cache[s]
    cmd_args_list_nn = dict_to_cmd_args({**cmd_opts_nn, 'no-save-model': True})
    ret = get_args_module_paths_from_cmd_args(cmd_args_list_nn)
    (args_nn, model, model_and_info_name, models_path) = ret
    _cmd_opts_train_to_args_module_paths_cache[s] = ret
    MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN[model_and_info_name] = cmd_opts_nn
    return ret


def _json_serialize_nn_acqf_configs(
        training_config: dict[str, Any],
        af_dataset_config: dict[str, dict[str, Any]],
        hash_gpytorch_modules:bool=True
    ):
    function_samples_config = af_dataset_config["function_samples_config"]
    acquisition_dataset_config = af_dataset_config["acquisition_dataset_config"]
    n_points_config = af_dataset_config["n_points_config"]
    dataset_transform_config = af_dataset_config["dataset_transform_config"]

    all_info_json = {
        'acquisition_dataset_config': acquisition_dataset_config,
        'n_points_config': n_points_config,
        'dataset_transform_config': convert_to_json_serializable_gpytorch(dataset_transform_config),
        'training_config': training_config
    }

    dataset_type = function_samples_config.get("dataset_type", "gp")
    if dataset_type == 'gp':
        if 'model_sampler' in function_samples_config:
            model_sampler = function_samples_config.pop('model_sampler')
        else:
            model_sampler = RandomModelSampler(
                models=function_samples_config["models"],
                model_probabilities=function_samples_config["model_probabilities"],
                randomize_params=function_samples_config["randomize_params"]
            )
    
        all_info_json['model_sampler'] = convert_to_json_serializable_gpytorch({
                '_models': model_sampler._models,
                '_initial_params_list': model_sampler._initial_params_list,
                'model_probabilities': model_sampler.model_probabilities,
                'randomize_params': model_sampler.randomize_params
            },
            include_priors=True, hash_gpytorch_modules=hash_gpytorch_modules,
            hash_include_str=False, hash_str=True)
    else:
        model_sampler = None

    all_info_json['function_samples_config'] = convert_to_json_serializable_gpytorch(
        function_samples_config)

    return all_info_json, model_sampler


def validate_single_train_args(args: argparse.Namespace, additional_info: Any):
    all_groups_arg_names = additional_info

    # Extract dataset groups for validation
    dataset_groups_arg_names = all_groups_arg_names['dataset']
    validate_args_for_dataset_type(args, dataset_groups_arg_names)

    # Only have include_alpha=True when method=policy_gradient
    if args.method != 'policy_gradient' and args.include_alpha:
        raise ValueError("include_alpha should be True only if method=policy_gradient")

    if args.train:
        if args.learning_rate is None:
            raise ValueError("learning_rate should be specified if training the model")
        if args.epochs is None:
            raise ValueError("epochs should be specified if training the model")
    else:
        args.save_model = False

    for reason, reason_desc in [(args.method != 'mse_ei', 'method != mse_ei')]:
        if reason:
            if args.learn_tau:
                raise ValueError(f"learn_tau should be False if {reason_desc}")
            if args.initial_tau is not None:
                raise ValueError(f"initial_tau should not be specified if {reason_desc}")
            if args.softplus_batchnorm:
                raise ValueError(f"softplus_batchnorm should be False if {reason_desc}")
            if args.positive_linear_at_end:
                raise ValueError(f"positive_linear_at_end should be False if {reason_desc}")
            if args.gp_ei_computation:
                raise ValueError(f"gp_ei_computation should be False if {reason_desc}")
    
    if args.early_stopping:
        if args.patience is None:
            raise ValueError("patience should be specified if early_stopping is True")
        if args.min_delta is None:
            raise ValueError("min_delta should be specified if early_stopping is True")

    if args.lr_scheduler == 'ReduceLROnPlateau':
        if args.lr_scheduler_patience is None:
            raise ValueError("lr_scheduler_patience should be specified if lr_scheduler=ReduceLROnPlateau")
        if args.lr_scheduler_factor is None:
            raise ValueError("lr_scheduler_factor should be specified if lr_scheduler=ReduceLROnPlateau")
        if args.lr_scheduler_min_lr is None:
            args.lr_scheduler_min_lr = 0.0
        if args.lr_scheduler_cooldown is None:
            args.lr_scheduler_cooldown = 0
    else:
        if args.lr_scheduler_patience is not None:
            raise ValueError("lr_scheduler_patience should not be specified if lr_scheduler != ReduceLROnPlateau")
        if args.lr_scheduler_factor is not None:
            raise ValueError("lr_scheduler_factor should not be specified if lr_scheduler != ReduceLROnPlateau")
        if args.lr_scheduler_min_lr is not None:
            raise ValueError("lr_scheduler_min_lr should not be specified if lr_scheduler != ReduceLROnPlateau")
        if args.lr_scheduler_cooldown is not None:
            raise ValueError("lr_scheduler_cooldown should not be specified if lr_scheduler != ReduceLROnPlateau")
    
    if args.lr_scheduler == 'power':
        if args.lr_scheduler_power is None:
            raise ValueError("lr_scheduler_power should be specified if lr_scheduler=power")
        if args.lr_scheduler_burnin is None:
            raise ValueError("lr_scheduler_burnin should be specified if lr_scheduler=power")
    else:
        if args.lr_scheduler_power is not None:
            raise ValueError("lr_scheduler_power should not be specified if lr_scheduler != power")
        if args.lr_scheduler_burnin is not None:
            raise ValueError("lr_scheduler_burnin should not be specified if lr_scheduler != power")
        
    if args.architecture == 'transformer':
        if args.num_heads is None:
            args.num_heads = 4
        if args.x_cand_input is not None:
            raise ValueError("x_cand_input should not be specified for transformer architecture")
    elif args.architecture == 'pointnet':
        if args.x_cand_input is None:
            args.x_cand_input = 'local_and_final'
        if args.encoded_history_dim is None:
            args.encoded_history_dim = args.layer_width

    lamda_given = args.lamda is not None
    lamda_min_given = args.lamda_min is not None
    lamda_max_given = args.lamda_max is not None
    if args.method == 'gittins':
        if lamda_given:
            if lamda_min_given or lamda_max_given:
                raise ValueError(
                    "If method=gittins, should specify only either lamda, or both lamda_min and lamda_max")
        else:
            if not (lamda_min_given or lamda_max_given):
                # No lamda anything is specified
                raise ValueError(
                    "If method=gittins, need to specify either lamda, or both lamda_min and lamda_max")
            if not lamda_min_given:
                # lamda_max is given but lamda_min is not given
                raise ValueError(
                    "If method=gittins and lamda_max is specified, then lamda_min must be "
                    "specified (or give lamda instead)")
            if not lamda_max_given:
                # lamda_min is given but lamda_max is not given
                raise ValueError(
                    "If method=gittins and lamda_min is specified, then lamda_max must be "
                    "specified (or give lamda instead)")
    else: # method = 'mse_ei' or 'policy_gradient'
        if args.initial_tau is None:
            args.initial_tau = 1.0
        if lamda_given or lamda_min_given or lamda_max_given:
            raise ValueError(
                "If method != gittins, then lamda, lamda_min, and lamda_max should not be specified")
        if args.gi_loss_normalization:
            raise ValueError("gi_loss_normalization should be None if method != gittins")


_POINTNET_X_CAND_INPUT_OPTIONS = {
    "local_and_final": dict(
        input_xcand_to_local_nn=True,
        input_xcand_to_final_mlp=True
    ),
    "local_only": dict(
        input_xcand_to_local_nn=True,
        input_xcand_to_final_mlp=False
    ),
    "final_only": dict(
        input_xcand_to_local_nn=False,
        input_xcand_to_final_mlp=True
    ),
    "subtract-lossy": dict(
        input_xcand_to_local_nn=False,
        input_xcand_to_final_mlp=False,
        subtract_x_cand_from_x_hist=True
    ),
    "subtract-local_only": dict(
        input_xcand_to_local_nn=True,
        input_xcand_to_final_mlp=False,
        subtract_x_cand_from_x_hist=True
    ),
    "subtract-final_only": dict(
        input_xcand_to_local_nn=False,
        input_xcand_to_final_mlp=True,
        subtract_x_cand_from_x_hist=True
    )
}


def _initialize_module_from_args(args: argparse.Namespace):
    ### Get dimension based on dataset type
    dataset_type = getattr(args, 'dataset_type', 'gp')
    if dataset_type in {'gp', 'cancer_dosage'}:
        dimension = None # already in args.dimension
    elif dataset_type == 'logistic_regression':
        dimension = 1
    elif dataset_type == 'hpob':
        # Make sure to set the dimension for non-GP datasets so that it is available
        # for model creation
        dimension = get_hpob_dataset_dimension(args.hpob_search_space_id)

    # Exp technically works, but Power does not
    # Make sure to set these appropriately depending on whether the transform
    # supports mean transform
    # if dataset_transform_config['outcome_transform'] is not None:
    #     GET_TRAIN_TRUE_GP_STATS = False
    #     GET_TEST_TRUE_GP_STATS = False

    architecture = args.architecture
    hidden_dims = [args.layer_width] * args.num_layers
    if architecture == "pointnet":
        body_cls = AcquisitionFunctionBodyPointnetV1and2
        af_body_init_params_base = dict(
            dimension=dimension if dimension is not None else args.dimension,

            history_enc_hidden_dims=hidden_dims,
            pooling=args.pooling,
            encoded_history_dim=args.encoded_history_dim,

            activation_at_end_pointnet=True,
            layer_norm_pointnet=False,
            dropout_pointnet=args.dropout,
            activation_pointnet="relu",

            include_best_y=args.include_best_y,
            n_pointnets=1)
        # Only add this if it is True; that way it is backwards compatible
        # with subtract_best_y not being an option (being False) previously.
        # (subtract_best_y is False by default.)
        if args.subtract_best_y:
            af_body_init_params_base['subtract_best_y'] = True
        # Similarly for max_history_input:
        if args.max_history_input is not None:
            af_body_init_params_base['max_history_input'] = args.max_history_input
        
        try:
            extra_params = _POINTNET_X_CAND_INPUT_OPTIONS[args.x_cand_input]
        except KeyError:
            raise ValueError(
                f"Unknown x_cand_input option '{args.x_cand_input}' for PointNet. "
                f"Available options are: {list(_POINTNET_X_CAND_INPUT_OPTIONS)}")
        
        if args.acqf_params_input == POINTNET_ACQF_PARAMS_INPUT_DEFAULT:
            args.acqf_params_input = None
        if args.acqf_params_input is not None:
            try:
                extra_params_acqf = POINTNET_ACQF_PARAMS_INPUT_OPTIONS[args.acqf_params_input]
            except KeyError:
                raise ValueError(
                    f"Unknown acqf_params_input option '{args.acqf_params_input}' for PointNet. "
                    f"Available options are: {list(POINTNET_ACQF_PARAMS_INPUT_OPTIONS)}")
            extra_params = dict(**extra_params, **extra_params_acqf)

        af_body_init_params = dict(**af_body_init_params_base, **extra_params)
    elif architecture == "transformer":
        body_cls = AcquisitionFunctionBodyTransformerNP
        af_body_init_params = dict(
            dimension=args.dimension,
            hidden_dim=args.layer_width,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=0.0 if args.dropout is None else args.dropout,
            include_best_y=args.include_best_y,
            input_xcand_to_final_mlp=True,
        )
    else:
        raise ValueError(f"Unknown architecture {architecture}")
    
    af_head_init_params = dict(
        hidden_dims=hidden_dims,
        activation="relu",
        layer_norm_before_end=False,
        layer_norm_at_end=False,
        dropout=args.dropout,
    )

    if args.method == 'gittins':
        model = GittinsAcquisitionFunctionNet(
            af_class=TwoPartAcquisitionFunctionNetFixedHistoryOutputDim,
            variable_lambda=args.lamda is None,
            costs_in_history=False,
            cost_is_input=False,
            af_body_class=body_cls,
            af_head_class=AcquisitionFunctionNetFinalMLP,
            af_body_init_params=af_body_init_params,
            af_head_init_params=af_head_init_params,
            standardize_outcomes=args.standardize_nn_history_outcomes
        )
    elif args.method == 'policy_gradient' or args.method == 'mse_ei':
        initial_tau = getattr(args, "initial_tau", None)
        initial_tau = 1.0 if initial_tau is None else initial_tau
        af_head_init_params = dict(
            **af_head_init_params,
            include_alpha=args.include_alpha,
            learn_alpha=args.learn_alpha,
            initial_alpha=args.initial_alpha,
            initial_beta=1.0 / initial_tau,
            learn_beta=args.learn_tau,
            softplus_batchnorm=args.softplus_batchnorm,
            softplus_batchnorm_momentum=args.softplus_batchnorm_momentum,
            positive_linear_at_end=args.positive_linear_at_end,
            gp_ei_computation=args.gp_ei_computation
        )
        model = ExpectedImprovementAcquisitionFunctionNet(
            af_body_class=body_cls,
            af_body_init_params=af_body_init_params,
            af_head_init_params=af_head_init_params,
            standardize_outcomes=args.standardize_nn_history_outcomes
        )
    
    return model

    # model = AcquisitionFunctionNetV4(args.dimension,
    #                                 history_enc_hidden_dims=[32, 32], pooling="max",
    #                 include_local_features=True,
    #                 encoded_history_dim=4, include_mean=False,
    #                 mean_enc_hidden_dims=[32, 32], mean_dim=1,
    #                 std_enc_hidden_dims=[32, 32], std_dim=32,
    #                 aq_func_hidden_dims=[32, 32], layer_norm=True,
    #                 layer_norm_at_end_mlp=False,
    #                 include_alpha=args.include_alpha and policy_gradient_flag,
    #                                 learn_alpha=args.learn_alpha,
    #                                 initial_alpha=args.initial_alpha)

    # model = AcquisitionFunctionNetV3(args.dimension, pooling="max",
    #                 history_enc_hidden_dims=[args.layer_width, args.layer_width],
    #                 encoded_history_dim=args.layer_width,
    #                 mean_enc_hidden_dims=[args.layer_width, args.layer_width], mean_dim=1,
    #                 std_enc_hidden_dims=[args.layer_width, args.layer_width], std_dim=16,
    #                 aq_func_hidden_dims=[args.layer_width, args.layer_width], layer_norm=False,
    #                 layer_norm_at_end_mlp=False, include_y=True,
    #                 include_alpha=args.include_alpha and policy_gradient_flag,
    #                                 learn_alpha=args.learn_alpha,
    #                                 initial_alpha=args.initial_alpha)

    # model = AcquisitionFunctionNetDense(args.dimension, MAX_HISTORY,
    #                                     hidden_dims=[128, 128, 64, 32],
    #                                     include_alpha=args.include_alpha and policy_gradient_flag,
    #                                     learn_alpha=args.learn_alpha,
    #                                     initial_alpha=args.initial_alpha)


@cache
def get_single_train_parser_and_info():
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

    parser_info = _add_single_train_args_and_return_info(parser)

    return parser, parser_info


def _add_single_train_args_and_return_info(parser: argparse.ArgumentParser):
    ################################ Dataset settings ##################################
    dataset_group = parser.add_argument_group("Dataset options")
    groups_arg_names = add_unified_acquisition_dataset_args(
        parser, dataset_group, add_lamda_args_flag=False)

    ############################ NN architecture settings ##############################
    nn_architecture_group = parser.add_argument_group("NN Architecture options")
    nn_architecture_group.add_argument(
        '--layer_width',
        type=int,
        required=True,
        help='The width of the NN layers.'
    )
    nn_architecture_group.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='Number of hidden layers. Default is 2.'
    )
    nn_architecture_group.add_argument(
        '--standardize_nn_history_outcomes',
        action='store_true',
        help=('Whether to standardize the history outcomes when computing the NN '
            'acquisition function. Default is False.')
    )
    nn_architecture_group.add_argument(
        '--architecture',
        type=str,
        choices=['pointnet', 'transformer'],
        required=True,
        help='Type of NN architecture to use.'
    )
    nn_architecture_group.add_argument(
        '--x_cand_input',
        type=str,
        choices=list(_POINTNET_X_CAND_INPUT_OPTIONS),
        help='(Only for PointNet) How to use x_cand as input to the NN. '
             'Default is "local_and_final".'
    )
    nn_architecture_group.add_argument(
        '--acqf_params_input',
        type=str,
        choices=list(POINTNET_ACQF_PARAMS_INPUT_OPTIONS),
        help=('(Only for PointNet) How to use the acquisition function parameters '
              f'as input to the NN. Default is "{POINTNET_ACQF_PARAMS_INPUT_DEFAULT}".'),
    )
    nn_architecture_group.add_argument(
        '--encoded_history_dim',
        type=int,
        help=('(Only for PointNet) The feature dimension of the input to the NN. '
            'Default is the same as layer_width.')
    )
    nn_architecture_group.add_argument(
        '--include_best_y',
        action='store_true',
        help='Whether to include the best y in the input to the NN. Default is False.'
    )
    nn_architecture_group.add_argument(
        '--subtract_best_y',
        action='store_true',
        help=('Whether to subtract the best y from the history outcomes. '
              'Default is False.')
    )
    nn_architecture_group.add_argument(
        '--max_history_input',
        type=int,
        help=('Maximum history size that the NN can take as input. Default is None, '
                'which means no limit.')
    )
    nn_architecture_group.add_argument(
        '--pooling',
        type=str,
        choices=['max', 'mean', 'sum'],
        help=('(Only for PointNet) The pooling method to use in the history encoder. '
              'Default is "max".'),
        default='max'
    )
    nn_architecture_group.add_argument(
        '--dropout',
        type=float,
        help='Dropout rate. Default is None.'
    )
    # Optional transformer-specific arguments
    nn_architecture_group.add_argument(
        '--num_heads',
        type=int,
        help='(Transformer only) Number of attention heads. Default is 4.'
    )

    ############################ Training settings #####################################
    training_group = parser.add_argument_group("Training options")
    # Which AF training loss function to use
    training_group.add_argument(
        '--method',
        choices=METHODS,
        required=True,
    )
    training_group.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate for training the model'
    )
    training_group.add_argument(
        '--batch_size',
        type=int,
        required=True,
        help='Batch size for training the model'
    )
    training_group.add_argument(
        '--weight_decay',
        type=float,
        help='Weight decay for training the model'
    )
    training_group.add_argument(
        '--epochs',
        type=int,
        help='Maximum number of epochs for training the model',
    )
    ### Early stopping
    training_group.add_argument(
        '--early_stopping',
        action='store_true',
        help=('Whether to use early stopping. Default is False.')
    )
    training_group.add_argument(
        '--patience',
        type=int,
        help=('Number of epochs with no improvement after which training will be stopped. '
            'Only used if early_stopping=True.')
    )
    training_group.add_argument(
        '--min_delta',
        type=float,
        help=('Minimum change in the monitored quantity to qualify as an improvement. '
            'Only used if early_stopping=True.')
    )
    training_group.add_argument(
        '--cumulative_delta',
        action='store_true',
        help=('Whether to use cumulative delta for early stopping. Default is False. '
            'Only used if early_stopping=True.')
    )
    ### Learning rate scheduler
    training_group.add_argument(
        '--lr_scheduler',
        choices=['ReduceLROnPlateau', 'power'],
        help='Use a learning rate scheduler. Default is to not use any.'
    )
    training_group.add_argument(
        '--lr_scheduler_patience',
        type=int,
        help=('Number of epochs with no improvement after which learning rate will be '
              'reduced. Only used if lr_scheduler=ReduceLROnPlateau.')
    )
    training_group.add_argument(
        '--lr_scheduler_factor',
        type=float,
        help=('Factor by which the learning rate will be reduced. new_lr = lr * factor. '
              'Only used if lr_scheduler=ReduceLROnPlateau.')
    )
    training_group.add_argument(
        '--lr_scheduler_min_lr',
        type=float,
        help=('A lower bound on the learning rate. Only used if '
              'lr_scheduler=ReduceLROnPlateau. Default is 0.0.')
    )
    training_group.add_argument(
        '--lr_scheduler_cooldown',
        type=int,
        help='Number of epochs to wait before resuming normal operation after lr has '
             'been reduced. Only used if lr_scheduler=ReduceLROnPlateau. Default is 0.'
    )
    training_group.add_argument(
        '--lr_scheduler_power',
        type=float,
        help=('Power for the power learning rate scheduler. Only used if '
                'lr_scheduler=power. Default is 0.6.')
    )
    training_group.add_argument(
        '--lr_scheduler_burnin',
        type=int,
        help=('Number of epochs to wait before starting to apply the power learning rate '
                'scheduler. Only used if lr_scheduler=power. Default is 1.')
    )
    ### Evaluation metric
    training_group.add_argument(
        '--use_maxei',
        action='store_true',
        help='Use --use_maxei to use the "max ei" statistic as the evaluation metric '
        'on the test dataset. Otherwise, the evaluation metric is the same as the loss '
        'function used to train the NN.'
    )

    #### Options when method=policy_gradient
    policy_gradient_group = parser.add_argument_group(
        "Training options when method=policy_gradient")
    policy_gradient_group.add_argument(
        '--include_alpha',
        action='store_true',
        help='Whether to include alpha. Only used if method=policy_gradient.'
    )
    policy_gradient_group.add_argument(
        '--learn_alpha',
        action='store_true',
        help=('Whether to learn alpha. Default is True. Only used if '
            'method=policy_gradient and include_alpha=true.')
    )
    policy_gradient_group.add_argument(
        '--initial_alpha',
        type=float,
        help=('Initial value of alpha. Default is 1.0. Only used if '
              'method=policy_gradient and include_alpha=true.'),
        default=1.0
    )
    policy_gradient_group.add_argument( # default is None, equivalent to 0.0
        '--alpha_increment',
        type=float,
        help=('Increment for alpha. Default is 0.0. Only used if method=policy_gradient'
            ' and include_alpha=true.')
    )

    #### Options when method=gittins
    gittins_group = parser.add_argument_group(
        "Training options when method=gittins")
    gittins_group.add_argument(
        '--gi_loss_normalization',
        choices=GI_NORMALIZATIONS,
        help=('Normalization of the Gittins index loss function. to use. '
            'Default is to not use any. Only used if method=gittins.')
    )
    add_lamda_args(gittins_group)

    #### Options for NN when method=mse_ei
    mse_ei_group = parser.add_argument_group(
        "Training options when method=mse_ei")
    mse_ei_group.add_argument(
        '--learn_tau',
        action='store_true',
        help=('Set this flag to enable learning of tau=1/beta which is the parameter for softplus'
            ' applied at the end of the MSE acquisition function. Default is False. '
            'Only used if method=mse_ei.')
    )
    mse_ei_group.add_argument(
        '--initial_tau',
        type=float,
        help='Initial value of tau. Default is 1.0. Only used if method=mse_ei.'
    )
    mse_ei_group.add_argument(
        '--softplus_batchnorm',
        action='store_true',
        help=('Set this flag to apply positive-batchnorm after softplus in the MSE acquisition function. '
            'Default is False. Only used if method=mse_ei.')
    )
    mse_ei_group.add_argument(
        '--softplus_batchnorm_momentum',
        type=float,
        default=0.1,
        help=('Momentum for the batchnorm after softplus in the MSE acquisition function. Default is 0.1. '
            'Only used if method=mse_ei.')
    )
    mse_ei_group.add_argument(
        '--positive_linear_at_end',
        action='store_true',
        help=('Set this flag to apply positive linear at end technique. Default is False. '
            'Only used if method=mse_ei.')
    )
    mse_ei_group.add_argument(
        '--gp_ei_computation',
        action='store_true',
        help=('Set this flag to apply gp_ei_computation at end technique. Default is False. '
            'Only used if method=mse_ei.')
    )

    # Extract argument names from all groups
    from utils_general.utils import get_arg_names

    return {
        'dataset': groups_arg_names,  # Dataset-specific groups (gp, hpob, cancer_dosage, etc.)
        'architecture': get_arg_names(nn_architecture_group),
        'training': get_arg_names(training_group),
        'policy_gradient': get_arg_names(policy_gradient_group),
        'gittins': get_arg_names(gittins_group),
        'mse_ei': get_arg_names(mse_ei_group),
    }
