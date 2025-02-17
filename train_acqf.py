import argparse
from collections import defaultdict
import itertools
from datetime import datetime
from typing import Union
import yaml
import os
from gp_acquisition_dataset import create_train_test_gp_acq_datasets_from_args
from run_train import get_configs_and_model_and_paths
from train_acquisition_function_net import model_is_trained
from utils import dict_to_cmd_args, save_json
from submit_dependent_jobs import CONFIG_DIR, SWEEPS_DIR, submit_dependent_jobs


def check_dict_has_keys(d: dict, keys: list[str], error_msg=None):
    for kk in d:
        if kk not in keys:
            if error_msg is None:
                msg = f"Invalid key '{kk}' in dictionary."
            else:
                msg = error_msg(kk)
            raise ValueError(msg)


def refine_config(params_value: Union[dict, list[dict]],
                  experiment_config: dict,
                  params_names:list[str]=[]) -> dict:
    if not isinstance(experiment_config, dict):
        raise ValueError('experiment_config must be a dictionary.')
    
    vary_params = any(param_name in experiment_config['parameters']
                      for param_name in params_names)
        
    if isinstance(params_value, list): # an OR
        if not vary_params:
            # If the parameter is not varying, we just take the first value
            params_value = [params_value[0]]
        return [
            refine_config(params_dict, experiment_config, params_names)
            for params_dict in params_value
        ]
    elif isinstance(params_value, dict): # an AND
        tmp = {}
        for param_name, param_config in params_value.items():
            this_params_names = params_names
            this_vary_params = vary_params or param_name in experiment_config['parameters']
            if param_name in experiment_config['parameters']:
                info = experiment_config['parameters'][param_name]
                check_dict_has_keys(
                    info,
                    ['values', 'value', 'recurse'],
                    lambda kk: (f"Invalid key '{kk}' in experiment parameter "
                                f"dictionary for parameter '{param_name}'.")
                )
                if 'values' in info and 'value' in info:
                    raise ValueError(
                        f"Cannot specify both 'values' and 'value' for parameter '{param_name}'.")
                if 'value' in info:
                    allowed_values = [info['value']]
                else:
                    allowed_values = info.get('values', 'all')
                if info.get('recurse', False):
                    this_params_names = params_names + [param_name]
            
            if 'parameters' in param_config:
                check_dict_has_keys(
                    param_config,
                    ['parameters'],
                    lambda kk: f"Invalid key '{kk}' in parameter dictionary."
                )
                tmp[param_name] = {
                    'parameters': refine_config(
                    param_config['parameters'], experiment_config, this_params_names)
                }
            else:
                check_dict_has_keys(
                    param_config,
                    ['value', 'values'],
                    lambda kk: f"Invalid key '{kk}' in parameter dictionary.")
                if this_vary_params:
                    if 'values' in param_config:
                        values = param_config['values']
                    else:
                        values = [param_config['value']]
                    
                    if param_name in experiment_config['parameters']:
                        if allowed_values != 'all':
                            # Take the set of values in the base config
                            # that are also in the experiment config
                            new_values = []
                            value_names_already_have = set()
                            for value in values:
                                if type(value) is dict:
                                    check_dict_has_keys(
                                        value, ['value', 'parameters'],
                                        lambda kk: f"Invalid key '{kk}' in parameter value dictionary.")
                                    val_name = value['value']
                                else:
                                    val_name = value
                                value_names_already_have.add(val_name)
                                if val_name in allowed_values:
                                    new_values.append(value)
                            
                            # Now add the values from the experiment config
                            # that are not already in the base config
                            for value in allowed_values:
                                if value not in value_names_already_have:
                                    new_values.append(value)

                            values = new_values
                else:
                    if 'value' in param_config:
                        values = [param_config['value']]
                    else:
                        values = [param_config['values'][0]]
                
                for i, value in enumerate(values):
                    if type(value) is dict:
                        check_dict_has_keys(
                            value, ['value', 'parameters'],
                            lambda kk: f"Invalid key '{kk}' in parameter value dictionary.")
                        if 'parameters' in value:
                            values[i]['parameters'] = refine_config(
                                value['parameters'], experiment_config, this_params_names)
                        else:
                            values[i] = value['value']
                tmp[param_name] = {'values': values} if len(values) > 1 else {'value': values[0]}
        return tmp
    else:
        raise ValueError(
            f'params_value must be a dictionary or list, got {(params_value)}.')


def generate_options(params_value: Union[dict, list[dict]], prefix=''):
    if isinstance(params_value, list): # an OR
        return [d for p in params_value for d in generate_options(p, prefix=prefix)]
    elif isinstance(params_value, dict): # an AND
        options_per_param = []
        for param_name, cfg in params_value.items():
            new_prefix = param_name if prefix == '' else prefix + '.' + param_name
            
            if 'parameters' in cfg:
                check_dict_has_keys(
                    cfg,
                    ['parameters'],
                    lambda kk: f"Invalid key '{kk}' in parameter dictionary."
                )
                param_options = generate_options(cfg['parameters'], prefix=new_prefix)
            else:
                check_dict_has_keys(
                    cfg,
                    ['value', 'values'],
                    lambda kk: f"Invalid key '{kk}' in parameter dictionary.")
                values = cfg['values'] if 'values' in cfg else [cfg['value']]
                param_options = []
                for value in values:
                    if type(value) is dict:
                        check_dict_has_keys(
                            value, ['value', 'parameters'],
                            lambda kk: f"Invalid key '{kk}' in parameter value dictionary.")
                        if 'parameters' in value:
                            options_this_value = [
                                {new_prefix: value['value'], **params}
                                for params in generate_options(
                                    value['parameters'], prefix=prefix)
                            ]
                        else:
                            options_this_value = [{new_prefix: value['value']}]
                    else:
                        options_this_value = [{new_prefix: value}]
                    param_options.extend(options_this_value)
            options_per_param.append(param_options)
        result = []
        for dicts in itertools.product(*options_per_param):
            result.append(
                {k: v for d in dicts for k, v in d.items()}
            )
        return result
    else:
        raise ValueError(
            f'params_value must be a dictionary or list, got {(params_value)}.')


def get_command_line_options(options: dict):
    options = {k.split('.')[-1]: v for k, v in options.items()}
    expansion_factor = options['expansion_factor']
    cmd_opts_sample_dataset = {
        k: options.get(k)
        for k in ['dimension', 'kernel', 'lengthscale',
                  'randomize_params', 'outcome_transform', 'sigma']
    }
    cmd_opts_sample_dataset['standardize_dataset_outcomes'] = options['standardize_outcomes']
    cmd_opts_sample_dataset['train_acquisition_size'] = options['train_samples_size'] * expansion_factor
    cmd_opts_sample_dataset['test_acquisition_size'] = options['test_samples_size'] * expansion_factor

    cmd_opts_acquisition_dataset = {
        'expansion_factor': expansion_factor,
        'train_n_candidates': options['n_candidates'],
        'test_n_candidates': options['n_candidates'],
        **{
            k: options.get(k)
            for k in [
                'min_history', 'max_history',
                'lamda_min', 'lamda_max', 'lamda'
            ]
        }
    }

    cmd_opts_architecture = {
        'layer_width': options['layer_width'],
        'standardize_nn_history_outcomes': options['standardize_nn_history_outcomes'],
    }

    cmd_opts_training = {
        k: options.get(k)
        for k in [
            'method', 'learning_rate', 'batch_size', 'epochs',
            # early stopping
            'early_stopping', 'patience', 'min_delta', 'cumulative_delta',
            # method=policy_gradient
            'include_alpha', 'learn_alpha', 'initial_alpha', 'alpha_increment',
            # method=gittins
            'normalize_gi_loss', 'lamda_min', 'lamda_max', 'lamda',
            # method=mse_ei
            'learn_tau', 'initial_tau', 'softplus_batchnorm',
            'softplus_batchnorm_momentum', 'positive_linear_at_end',
            'gp_ei_computation'
        ]
    }

    cmd_opts_dataset = {
        **cmd_opts_sample_dataset, **cmd_opts_acquisition_dataset
    }
    cmd_args_dataset = dict_to_cmd_args(cmd_opts_dataset)
    cmd_dataset = "python gp_acquisition_dataset.py " + cmd_args_dataset

    cmd_opts_nn_no_dataset = {
        **cmd_opts_architecture, **cmd_opts_training
    }
    cmd_nn_train = " ".join(["python run_train.py",
                             cmd_args_dataset,
                             dict_to_cmd_args(cmd_opts_nn_no_dataset)])

    cmd_opts_nn = {**cmd_opts_nn_no_dataset, **cmd_opts_dataset}
    return cmd_dataset, cmd_opts_dataset, cmd_nn_train, cmd_opts_nn


DATASETS_JOB_ID = "datasets"
NO_DATASET_ID = 0

def create_dependency_structure_train_acqf(options_list, always_train=False):
    r"""Create a command dependency structure for training acquisition function NNs.
    Includes dataset generation commands and NN training commands.
    Only includes dataset generation commands for datasets that are not cached."""
    datasets_cached_dict = {}
    dataset_command_ids = {}
    nn_job_arrays = defaultdict(list)
    for option_dict in options_list:
        (cmd_dataset, cmd_opts_dataset,
         cmd_nn_train, cmd_opts_nn) = get_command_line_options(option_dict)
        
        # Determine whether to train the NN
        if always_train:
            train_nn = True
        else:
            # Determine whether or not the NN is already cached
            args_nn = argparse.Namespace(**cmd_opts_nn)
            (af_dataset_configs, model,
            model_and_info_name, models_path) = get_configs_and_model_and_paths(args_nn)
            model_already_trained = model_is_trained(model_and_info_name)
            # Train the NN iff it has not already been trained
            train_nn = not model_already_trained
        
        # Skip the command(s) if the NN has already been trained
        if not train_nn:
            continue

        # Determine whether or not the dataset is already cached
        if cmd_dataset in datasets_cached_dict:
            dataset_already_cached = datasets_cached_dict[cmd_dataset]
        else:
            args_dataset = argparse.Namespace(**cmd_opts_dataset)
            whether_cached = create_train_test_gp_acq_datasets_from_args(
                args_dataset, check_cached=True, load_dataset=False)
            dataset_already_cached = True
            for cached in whether_cached:
                dataset_already_cached = dataset_already_cached and cached
            datasets_cached_dict[cmd_dataset] = dataset_already_cached
        
        # Determine the dataset id
        if dataset_already_cached:
            # dataset already cached; assign a dummy id
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

        # Add the NN command to the appropriate list corresponding to the dataset id
        nn_job_arrays[dataset_id].append(cmd_nn_train)
    
    ret = {}
    # Recall that dataset_command_ids only contains commands for datasets not cached
    # Note that dicts are guaranteed to retain insertion order since Python 3.7
    dataset_commands_not_cached = list(dataset_command_ids)
    if len(dataset_commands_not_cached) != 0:
        ret[DATASETS_JOB_ID] = {
            "commands": dataset_commands_not_cached,
            "gpu": False
        }
    for dataset_id, job_array in nn_job_arrays.items():
        tmp = {
            "commands": job_array,
            "gpu": True
        }
        # Only add dependencies if the dataset is not cached
        if dataset_id != NO_DATASET_ID:
            tmp["dependencies"] = [{
                "job_name": DATASETS_JOB_ID,
                "index": dataset_id
            }]
        ret[f"nn{dataset_id}"] = tmp
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_config',
        type=str,
        required=True,
        help='YAML file containing the base configuration for the experiment.'
    )
    parser.add_argument(
        '--experiment_config',
        type=str,
        required=True,
        help='YAML file containing the experiment configuration.'
    )
    parser.add_argument(
        '--gpu_gres',
        type=str,
        help=('GPU resource specification for Slurm. e.g., "gpu:a100:1" or "gpu:1". '
              'Default is "gpu:a100:1".'),
        default="gpu:a100:1"
    )
    parser.add_argument(
        '--mail',
        type=str,
        help=('email address to send Slurm notifications to. '
              'If not specified, no notifications are sent.')
    )
    parser.add_argument(
        '--always_train',
        action='store_true',
        help=('If this flag is set, train all acquisition function NNs regardless of '
              'whether they have already been trained. Default is to only train '
              'acquisition function NNs that have not already been trained.')
    )

    args = parser.parse_args()

    # Load the base configuration
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)

    # Load the experiment configuration
    with open(args.experiment_config, 'r') as f:
        experiment_config = yaml.safe_load(f)

    check_dict_has_keys(
        base_config,
        ['parameters'],
        lambda kk: f"Invalid key '{kk}' in base configuration.")
    
    check_dict_has_keys(
        experiment_config,
        ['parameters'],
        lambda kk: f"Invalid key '{kk}' in experiment configuration.")

    if 'parameters' not in experiment_config:
        experiment_config['parameters'] = {}

    # Refine the configuration
    refined_config = {
        'parameters': refine_config(base_config['parameters'], experiment_config)
    }

    # Save the refined configuration
    with open(os.path.join(CONFIG_DIR, 'refined_config.yml'), 'w') as f:
        yaml.dump(refined_config, f)

    # Generate the options
    options_list = generate_options(refined_config['parameters'])
    with open(os.path.join(CONFIG_DIR, 'options.yml'), 'w') as f:
        yaml.dump(options_list, f)
    
    jobs_spec = create_dependency_structure_train_acqf(options_list, args.always_train)
    save_json(jobs_spec, os.path.join(CONFIG_DIR, "dependencies.json"), indent=4)

    sweep_name = "test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(SWEEPS_DIR, f"{sweep_name}_{timestamp}")
    submit_dependent_jobs(
        sweep_dir=sweep_dir,
        jobs_spec=jobs_spec,
        gpu_gres=args.gpu_gres,
        mail=args.mail
    )

if __name__ == "__main__":
    main()
