import argparse
from typing import Any, Optional
import os
from webbrowser import get

from run_train_transfer_bo_baseline import get_dataset_hash_for_transfer_bo_baselines, transfer_bo_baseline_is_trained
from utils.utils import dict_to_cmd_args, dict_to_str, save_json
from utils.experiments.experiment_config_utils import CONFIG_DIR, add_config_args, get_config_options_list
from utils.experiments.submit_dependent_jobs import add_slurm_args, submit_jobs_sweep_from_args

from nn_af.acquisition_function_net_save_utils import (
    get_nn_af_args_configs_model_paths_from_cmd_args, nn_acqf_is_trained)
from dataset_factory import create_train_test_acquisition_datasets_from_args


def get_cmd_options_sample_dataset(options: dict[str, Any]):
    # Extract dataset_type to determine which parameters to include
    dataset_type = options.get('dataset_type', 'gp')
    
    # Base dataset parameters common to all types
    cmd_opts_sample_dataset = {
        'dataset_type': dataset_type,
        'train_samples_size': options.get('train_samples_size'),
        'test_samples_size': options.get('test_samples_size'),
        'standardize_outcomes': options['standardize_outcomes']
    }
    
    # Add dataset-specific parameters
    if dataset_type == 'gp':
        cmd_opts_sample_dataset.update({
            k: options.get(k)
            for k in ['dimension', 'kernel', 'lengthscale',
                      'randomize_params', 'outcome_transform', 'sigma']
        })
    elif dataset_type == 'logistic_regression':
        cmd_opts_sample_dataset.update({
            k: options.get(k)
            for k in ['lr_n_samples_range', 'lr_n_features_range', 'lr_bias_range',
                      'lr_coefficient_std', 'lr_noise_range', 'lr_log_lambda_range',
                      'lr_log_uniform_sampling']
        })
    elif dataset_type == 'hpob':
        cmd_opts_sample_dataset.update({
            'hpob_search_space_id': options.get('hpob_search_space_id')
        })
    elif dataset_type == 'cancer_dosage':
        cmd_opts_sample_dataset.update({
            k: options.get(k)
            for k in ['dimension', 'dim_features', 'nnz_per_row',
                      'scale_intercept', 'scale_coef', 'noise_std',
                      'is_simplex', 'matrix_seed']
        })
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    return cmd_opts_sample_dataset


def get_cmd_options_train_acqf(options: dict[str, Any]):
    # TODO: In the future, could do this more automatically rather than hard-coding
    # everything.
    options = {k.split('.')[-1]: v for k, v in options.items()}
    
    cmd_opts_sample_dataset = get_cmd_options_sample_dataset(options)

    cmd_opts_acquisition_dataset = {
        'train_acquisition_size': options['train_acquisition_size'],
        'test_expansion_factor': options['test_expansion_factor'],
        'replacement': options['replacement'],
        'train_n_candidates': options['n_candidates'],
        'test_n_candidates': options['n_candidates'],
        **{
            k: options.get(k)
            for k in [
                'min_history', 'max_history', 'samples_addition_amount',
                'lamda_min', 'lamda_max', 'lamda'
            ]
        }
    }

    cmd_opts_dataset = {**cmd_opts_sample_dataset, **cmd_opts_acquisition_dataset}
    cmd_args_dataset = dict_to_cmd_args(cmd_opts_dataset)
    cmd_dataset = "python dataset_factory.py " + " ".join(cmd_args_dataset)

    transfer_bo_method = options.get('transfer_bo_method', None)
    if transfer_bo_method is not None:
        # Baseline transfer BO method
        cmd_opts_dataset_no_lamda = {k: v for k, v in cmd_opts_dataset.items()
                                     if k not in ['lamda', 'lamda_min', 'lamda_max']}
        cmd_opts_nn = {'transfer_bo_method': transfer_bo_method,
                       **cmd_opts_dataset_no_lamda}
        cmd_nn_train = " ".join(["python run_train_transfer_bo_baseline.py",
                                *dict_to_cmd_args(cmd_opts_nn)])
    else:
        cmd_opts_architecture = {
            k: options.get(k)
            for k in [
                'layer_width', 'standardize_nn_history_outcomes',
                'architecture', 'include_best_y', 'subtract_best_y',
                'x_cand_input', 'encoded_history_dim', 'pooling',
                'num_heads', 'num_layers',
                'dropout', 'max_history_input', 'acqf_params_input'
            ]
        }

        cmd_opts_training = {
            k: options.get(k)
            for k in [
                'method', 'learning_rate', 'batch_size', 'epochs', 'use_maxei',
                # early stopping
                'early_stopping', 'patience', 'min_delta', 'cumulative_delta',
                # learning rate scheduler
                ## ReduceLROnPlateau
                'lr_scheduler', 'lr_scheduler_patience', 'lr_scheduler_factor',
                'lr_scheduler_min_lr', 'lr_scheduler_cooldown',
                ## power
                'lr_scheduler_power', 'lr_scheduler_burnin',
                # weight decay
                'weight_decay',
                # method=policy_gradient
                'include_alpha', 'learn_alpha', 'initial_alpha', 'alpha_increment',
                # method=gittins
                'gi_loss_normalization',
                # 'lamda_min', 'lamda_max', 'lamda', # already in cmd_opts_acquisition_dataset
                # method=mse_ei
                'learn_tau', 'initial_tau', 'softplus_batchnorm',
                'softplus_batchnorm_momentum', 'positive_linear_at_end',
                'gp_ei_computation'
            ]
        }

        cmd_opts_nn_no_dataset = {**cmd_opts_architecture, **cmd_opts_training}
        cmd_nn_train = " ".join(["python run_train.py",
                                *cmd_args_dataset,
                                *dict_to_cmd_args(cmd_opts_nn_no_dataset)])
        cmd_opts_nn = {**cmd_opts_nn_no_dataset, **cmd_opts_dataset}
    return cmd_dataset, cmd_opts_dataset, cmd_nn_train, cmd_opts_nn


MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN = {}
_cache = {}
def cmd_opts_nn_to_model_and_info_name(cmd_opts_nn):
    s = dict_to_str(cmd_opts_nn)
    if s in _cache:
        return _cache[s]
    cmd_args_list_nn = dict_to_cmd_args({**cmd_opts_nn, 'no-save-model': True})
    ret = get_nn_af_args_configs_model_paths_from_cmd_args(cmd_args_list_nn)
    (args_nn, af_dataset_configs,
     model, model_and_info_name, models_path) = ret
    _cache[s] = ret
    MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN[model_and_info_name] = cmd_opts_nn
    return ret


DATASETS_JOB_ID = "datasets"
NO_DATASET_ID = 0
NO_NN_ID = "dep-nn"

def create_dependency_structure_train_acqf(
        options_list:list[dict[str, Any]],
        dependents_list:Optional[list[list[str]]]=None,
        always_train=False,
        dependents_slurm_options:dict[str, Any]={}):
    r"""Create a command dependency structure for training acquisition function NNs.
    Includes dataset generation commands and NN training commands.
    Only includes dataset generation commands for datasets that are not cached."""
    if dependents_list is None:
        dependents_list = [[] for _ in options_list]
    else:
        if len(dependents_list) != len(options_list):
            raise ValueError(
                "Length of dependents_list must match length of options_list.")

    datasets_cached_dict = {}
    dataset_command_ids = {}
    ret = {}
    
    for options, depdendent_commands in zip(options_list, dependents_list):
        (cmd_dataset, cmd_opts_dataset,
         cmd_nn_train, cmd_opts_nn) = get_cmd_options_train_acqf(options)
        
        # Determine whether to train the NN
        if always_train:
            train_nn = True
        else:
            # Train the NN iff it has not already been trained
            transfer_bo_method = options.get('transfer_bo_method', None)
            if transfer_bo_method is not None:
                dataset_hash = get_dataset_hash_for_transfer_bo_baselines(options)
                train_nn = not transfer_bo_baseline_is_trained(
                    transfer_bo_method, dataset_hash)
            else:
                (args_nn, af_dataset_configs, pre_model, model_and_info_name, models_path
                ) = cmd_opts_nn_to_model_and_info_name(cmd_opts_nn)
                train_nn = not nn_acqf_is_trained(model_and_info_name)
        
        if train_nn:
            # Determine whether or not the dataset is already cached
            if cmd_dataset in datasets_cached_dict:
                dataset_already_cached = datasets_cached_dict[cmd_dataset]
            else:
                args_dataset = argparse.Namespace(**cmd_opts_dataset)
                whether_cached = create_train_test_acquisition_datasets_from_args(
                    args_dataset, check_cached=True, load_dataset=False)
                dataset_already_cached = True
                for cached in whether_cached:
                    dataset_already_cached = dataset_already_cached and cached
                datasets_cached_dict[cmd_dataset] = dataset_already_cached
            
            # Determine the dataset id
            if dataset_already_cached:
                # dataset already cached; assign special id
                dataset_id = NO_DATASET_ID
            else:
                # dataset not cached
                # Only store cmd_dataset in dataset_command_ids if dataset is not cached
                if cmd_dataset in dataset_command_ids:
                    # dataset id already assigned
                    dataset_id = dataset_command_ids[cmd_dataset]
                else:
                    # assign a new dataset id
                    dataset_id = len(dataset_command_ids) + 1 
                    dataset_command_ids[cmd_dataset] = dataset_id
            
            nn_list_id = f"nn{dataset_id}"
            
            # Create the job array of NN training commands that are dependent on the
            # dataset generation command
            if nn_list_id not in ret:
                tmp = {
                    "commands": [],
                    "gpu": True
                }
                # Only add prerequisites if the dataset is not cached
                if dataset_id != NO_DATASET_ID:
                    tmp["prerequisites"] = [{
                        "job_name": DATASETS_JOB_ID,
                        "index": dataset_id
                    }]

                ret[nn_list_id] = tmp

            # Add the NN command to the appropriate list corresponding to the dataset id
            commands = ret[nn_list_id]["commands"]
            commands.append(cmd_nn_train)

            # Add commands that are dependent on the NN training command
            if len(depdendent_commands) > 0:
                nn_train_cmd_index = len(commands)
                dependent_jobs_name = f"nn{dataset_id}-{nn_train_cmd_index}"
                ret[dependent_jobs_name] = {
                    "commands": depdendent_commands,
                    "prerequisites": [{
                        "job_name": nn_list_id,
                        "index": nn_train_cmd_index
                    }],
                    **dependents_slurm_options
                }
        else:
            # Do not add the NN train command

            # Add the commands that are dependent on the NN training command
            # (but the NN was already trained so they are not dependent on anything)
            if len(depdendent_commands) > 0:
                if NO_NN_ID not in ret:
                    ret[NO_NN_ID] = {
                        "commands": [],
                        **dependents_slurm_options
                    }
                for cmd in depdendent_commands:
                    ret[NO_NN_ID]["commands"].append(cmd)

    #### Add the dataset generation commands
    # Recall that dataset_command_ids only contains commands for datasets not cached
    # Note that dicts are guaranteed to retain insertion order since Python 3.7
    dataset_commands_not_cached = list(dataset_command_ids)
    if len(dataset_commands_not_cached) != 0:
        ret[DATASETS_JOB_ID] = {
            "commands": dataset_commands_not_cached,
            "gpu": False
        }

    return ret


ALWAYS_TRAIN_NAME = 'always_train'


def add_train_acqf_args(parser, train=True, prefix='nn'):
    nn_base_config_name, nn_experiment_config_name = add_config_args(
        parser, prefix=prefix, experiment_name='NN acqf')

    if train:
        parser.add_argument(
            f'--{ALWAYS_TRAIN_NAME}',
            action='store_true',
            help=('If this flag is set, train all acquisition function NNs regardless of '
                'whether they have already been trained. Default is to only train '
                'acquisition function NNs that have not already been trained.')
        )
    
    return nn_base_config_name, nn_experiment_config_name


def main():
    ## Create parser
    parser = argparse.ArgumentParser()
    nn_base_config_name, nn_experiment_config_name = add_train_acqf_args(parser,
                                                                         train=True)
    add_slurm_args(parser)

    ## Parse arguments
    args = parser.parse_args()

    options_list, refined_config = get_config_options_list(
        getattr(args, nn_base_config_name), getattr(args, nn_experiment_config_name))
    
    jobs_spec = create_dependency_structure_train_acqf(
        options_list, always_train=getattr(args, ALWAYS_TRAIN_NAME))
    save_json(jobs_spec, os.path.join(CONFIG_DIR, "dependencies.json"), indent=4)
    submit_jobs_sweep_from_args(jobs_spec, args)


if __name__ == "__main__":
    main()
