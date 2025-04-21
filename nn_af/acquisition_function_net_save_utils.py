from functools import cache
import math
import os
from typing import Any, Sequence, Optional
import torch
import argparse

from utils.utils import convert_to_json_serializable, dict_to_hash, load_json, save_json
from utils.constants import MODELS_DIR, MODELS_VERSION

from nn_af.acquisition_function_net import AcquisitionFunctionBodyPointnetV1and2, AcquisitionFunctionBodyTransformerNP, AcquisitionFunctionNet, AcquisitionFunctionNetFinalMLP, ExpectedImprovementAcquisitionFunctionNet, GittinsAcquisitionFunctionNet, TwoPartAcquisitionFunctionNetFixedHistoryOutputDim
from nn_af.train_acquisition_function_net import GI_NORMALIZATIONS, METHODS

from datasets.dataset_with_models import RandomModelSampler
from gp_acquisition_dataset import FIX_TRAIN_ACQUISITION_DATASET, GP_GEN_DEVICE, add_gp_acquisition_dataset_args, add_lamda_args, get_gp_acquisition_dataset_configs, get_lamda_min_max


MODELS_SUBDIR = "models"


def get_latest_model_path(model_and_info_folder_name: str):
    model_and_info_path = os.path.join(MODELS_DIR, model_and_info_folder_name)
    already_saved = os.path.isdir(model_and_info_path)
    if not already_saved:
        raise FileNotFoundError(f"Models path {model_and_info_path} does not exist")

    models_path = os.path.join(model_and_info_path, MODELS_SUBDIR)

    latest_model_path = os.path.join(models_path, "latest_model.json")
    try:
        latest_model_name = load_json(latest_model_path)["latest_model"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Latest model path {latest_model_path} does not exist."
                                " i.e., no models have been fully trained yet.")
    model_path = os.path.join(models_path, latest_model_name)
    return model_path


def safe_get_latest_model_path(model_and_info_folder_name: str):
    try:
        return get_latest_model_path(model_and_info_folder_name)
    except FileNotFoundError:
        return None


def nn_acqf_is_trained(model_and_info_folder_name: str):
    return safe_get_latest_model_path(model_and_info_folder_name) is not None


def load_nn_acqf(
        model_and_info_folder_name: str,
        return_model_path=False,
        load_weights=True,
        verbose=True):
    model_and_info_path = os.path.join(MODELS_DIR, model_and_info_folder_name)
    model = _load_empty_nn_acqf(model_and_info_path)

    if return_model_path or load_weights:
        model_path = get_latest_model_path(model_and_info_path)

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


def load_nn_acqf_configs(model_and_info_folder_name: str):
    model_and_info_path = os.path.join(MODELS_DIR, model_and_info_folder_name)
    
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
    
    model_sampler = RandomModelSampler.load(
        os.path.join(model_and_info_path, "model_sampler"))
    if function_samples_config['models'] is not None:
        function_samples_config['models'] = model_sampler.initial_models
        function_samples_config['model_probabilities'] = model_sampler.model_probabilities
        assert model_sampler.randomize_params == function_samples_config['randomize_params']
    
    training_config = load_json(
        os.path.join(model_and_info_path, "training_config.json"))
    
    af_dataset_config = {
        "function_samples_config": function_samples_config,
        "acquisition_dataset_config": acquisition_dataset_config,
        "n_points_config": n_points_config,
        "dataset_transform_config": dataset_transform_config
    }

    all_info_json, model_sampler = json_serialize_nn_acqf_configs(
        training_config=training_config,
        af_dataset_config=af_dataset_config,
        hash_gpytorch_modules=False
    )
    del all_info_json['function_samples_config']['models']
    del all_info_json['function_samples_config']['model_probabilities']
    del all_info_json['function_samples_config']['randomize_params']
    return all_info_json


def get_nn_af_args_configs_model_paths_from_cmd_args(
        cmd_args:Optional[Sequence[str]]=None):
    args = _parse_af_train_cmd_args(cmd_args=cmd_args)

    #### Get AF dataset configs
    gp_af_dataset_configs = get_gp_acquisition_dataset_configs(
        args, device=GP_GEN_DEVICE)

    # Exp technically works, but Power does not
    # Make sure to set these appropriately depending on whether the transform
    # supports mean transform
    # if dataset_transform_config['outcome_transform'] is not None:
    #     GET_TRAIN_TRUE_GP_STATS = False
    #     GET_TEST_TRUE_GP_STATS = False

    ################################### Get NN model ###############################
    #### Get the untrained model
    # This wastes some resources, but need to do it to get the model's init dict to
    # obtain the correct path for saving the model because that is currently how the
    # model is uniquely identified.
    model = _get_model(args)

    #### Save the configs for the model and training and datasets
    training_config = _get_training_config(args)
    model_and_info_folder_name, models_path = _save_nn_acqf_configs(
        model,
        training_config=training_config,
        af_dataset_config=gp_af_dataset_configs,
        save=args.save_model
    )

    return (args, gp_af_dataset_configs,
            model, model_and_info_folder_name, models_path)


def json_serialize_nn_acqf_configs(
        training_config: dict[str, Any],
        af_dataset_config: dict[str, dict[str, Any]],
        hash_gpytorch_modules:bool=True
    ):
    function_samples_config = af_dataset_config["function_samples_config"]
    acquisition_dataset_config = af_dataset_config["acquisition_dataset_config"]
    n_points_config = af_dataset_config["n_points_config"]
    dataset_transform_config = af_dataset_config["dataset_transform_config"]

    model_sampler = RandomModelSampler(
        models=function_samples_config["models"],
        model_probabilities=function_samples_config["model_probabilities"],
        randomize_params=function_samples_config["randomize_params"]
    )

    function_samples_config_json = convert_to_json_serializable(function_samples_config)
    dataset_transform_config_json = convert_to_json_serializable(dataset_transform_config)
    model_sampler_json = convert_to_json_serializable({
            '_models': model_sampler._models,
            '_initial_params_list': model_sampler._initial_params_list,
            'model_probabilities': model_sampler.model_probabilities,
            'randomize_params': model_sampler.randomize_params
        },
        include_priors=True, hash_gpytorch_modules=hash_gpytorch_modules,
        hash_include_str=False, hash_str=True)

    all_info_json = {
        # af_dataset_config
        'function_samples_config': function_samples_config_json,
        'acquisition_dataset_config': acquisition_dataset_config,
        'n_points_config': n_points_config,
        'dataset_transform_config': dataset_transform_config_json,
        'model_sampler': model_sampler_json,
        
        'training_config': training_config
    }
    # all_info_json = _remove_none_and_false(all_info_json)

    return all_info_json, model_sampler


def get_lamda_for_bo_of_nn(lamda, lamda_min, lamda_max):
    if lamda is not None:
        # Then it is trained with a fixed value of lamda
        return lamda
    if lamda_min is None or lamda_max is None:
        assert lamda_min is None and lamda_max is None
        return None
    # Trained with a range of lamda values
    log_min, log_max = math.log10(lamda_min), math.log10(lamda_max)
    # We will test with the average
    log_lamda = 0.5 * (log_min + log_max)
    return 10**log_lamda


def _save_nn_acqf_configs(
        model: AcquisitionFunctionNet,
        training_config: dict[str, Any],
        af_dataset_config: dict[str, dict[str, Any]],
        save:bool=True
    ):

    all_info_json, model_sampler = json_serialize_nn_acqf_configs(
        training_config=training_config,
        af_dataset_config=af_dataset_config
    )
    all_info_json['model'] = model.get_info_dict()
    
    all_info_hash = dict_to_hash(all_info_json)
    model_and_info_folder_name = os.path.join(MODELS_VERSION, f"model_{all_info_hash}")
    model_and_info_path = os.path.join(MODELS_DIR, model_and_info_folder_name)
    models_path = os.path.join(model_and_info_path, MODELS_SUBDIR)

    already_saved = os.path.isdir(model_and_info_path)

    # Assume that all the json files are already saved if the directory exists
    if save and not already_saved:
        print(f"Saving model and configs to new directory {model_and_info_folder_name}")

        os.makedirs(model_and_info_path, exist_ok=False)

        # Save model config
        model.save_init(models_path)

        # Save training config
        save_json(training_config,
                os.path.join(model_and_info_path, "training_config.json"))

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

    return model_and_info_folder_name, models_path


_CACHED_WEIGHTS = {}
def _get_state_dict(weights_path: str, verbose: bool=True):
    if weights_path in _CACHED_WEIGHTS:
        return _CACHED_WEIGHTS[weights_path]
    if verbose:
        print(f"Loading best weights from {weights_path}")
    ret = torch.load(weights_path)
    _CACHED_WEIGHTS[weights_path] = ret
    return ret


@cache
def _load_empty_nn_acqf(model_and_info_path: str):
    # Loads empty model (without weights)
    models_path = os.path.join(model_and_info_path, MODELS_SUBDIR)
    return AcquisitionFunctionNet.load_init(models_path)


def _parse_af_train_cmd_args(cmd_args:Optional[Sequence[str]]=None):
    parser = _get_run_train_parser()
    args = parser.parse_args(args=cmd_args)

    ######################## Check the arguments ###############################
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
        # lr_scheduler_min_lr and lr_scheduler_cooldown are optional (have defaults)
    
    if args.architecture == 'transformer':
        if args.num_heads is None:
            args.num_heads = 4
        if args.num_layers is None:
            args.num_layers = 2
        if args.dropout is None:
            args.dropout = 0.0

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

    return args


def _get_model(args: argparse.Namespace):
    if args.architecture == "pointnet":
        body_cls = AcquisitionFunctionBodyPointnetV1and2
        af_body_init_params = dict(
            dimension=args.dimension,

            history_enc_hidden_dims=[args.layer_width, args.layer_width],
            pooling="max",
            encoded_history_dim=args.layer_width,

            input_xcand_to_local_nn=True,
            input_xcand_to_final_mlp=True,

            activation_at_end_pointnet=True,
            layer_norm_pointnet=False,
            dropout_pointnet=None,
            activation_pointnet="relu",

            include_best_y=False,
            n_pointnets=1)
    elif args.architecture == "transformer":
        body_cls = AcquisitionFunctionBodyTransformerNP
        af_body_init_params = dict(
            dimension=args.dimension,
            hidden_dim=args.layer_width,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            include_best_y=False,
            input_xcand_to_final_mlp=True,
        )
    else:
        ValueError(f"Unknown architecture {args.architecture}")
    
    af_head_init_params = dict(
        hidden_dims=[args.layer_width, args.layer_width],
        activation="relu",
        layer_norm_before_end=False,
        layer_norm_at_end=False,
        dropout=None,
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
    if args.weight_decay is not None:
        training_config = dict(
            **training_config,
            weight_decay=args.weight_decay)
    return training_config


@cache
def _get_run_train_parser():
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

    ################################ Dataset settings ##################################
    dataset_group = parser.add_argument_group("Dataset options")
    add_gp_acquisition_dataset_args(dataset_group)

    ############################ NN architecture settings ##############################
    nn_architecture_group = parser.add_argument_group("NN Architecture options")
    nn_architecture_group.add_argument(
        '--layer_width',
        type=int,
        required=True,
        help='The width of the NN layers.'
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
        help='Type of NN architecture to use: "pointnet" or "transformer".'
    )
    # Optional transformer-specific arguments
    nn_architecture_group.add_argument(
        '--num_heads',
        type=int,
        help='(Transformer only) Number of attention heads. Default is 4.'
    )
    nn_architecture_group.add_argument(
        '--num_layers',
        type=int,
        help='(Transformer only) Number of transformer encoder layers. Default is 2.'
    )
    nn_architecture_group.add_argument(
        '--dropout',
        type=float,
        help='(Transformer only) Dropout rate for transformer layers. Default is None.'
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
        choices=['ReduceLROnPlateau'],
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
        default=0.0,
        help=('A lower bound on the learning rate. Only used if '
              'lr_scheduler=ReduceLROnPlateau. Default is 0.0.')
    )
    training_group.add_argument(
        '--lr_scheduler_cooldown',
        type=int,
        default=0,
        help='Number of epochs to wait before resuming normal operation after lr has '
             'been reduced. Only used if lr_scheduler=ReduceLROnPlateau. Default is 0.'
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

    return parser
