import argparse
import itertools
from datetime import datetime
from typing import Any, Optional, Union
import yaml
import os
from gp_acquisition_dataset import create_train_test_gp_acq_datasets_from_args
from run_train import get_configs_and_model_and_paths, get_run_train_parser
from train_acquisition_function_net import model_is_trained
from utils import dict_to_cmd_args, dict_to_str, group_by_nested_attrs, save_json
from submit_dependent_jobs import CONFIG_DIR, SWEEPS_DIR, submit_dependent_jobs


def check_dict_has_keys(d: dict, keys: list[str], error_msg=None):
    for kk in d:
        if kk not in keys:
            if error_msg is None:
                msg = f"Invalid key '{kk}' in dictionary."
            else:
                msg = error_msg(kk)
            raise ValueError(msg)


def refine_config(params_value: Union[dict[str, dict[str, Any]], list[dict[str, Any]]],
                  experiment_config: dict[str, dict[str, Union[bool, list, int, float, type[None]]]],
                  params_names:list[str]=[]):
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
        tmp: dict[str, Any] = {}
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


def generate_options(params_value: Union[dict[str, dict[str, Any]], list[dict[str, dict[str, Any]]]],
                     prefix='') -> list[dict[str, Any]]:
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


def get_command_line_options(options: dict[str, Any]):
    # TODO: In the future, could do this more automatically rather than hard-coding
    # everything.
    options = {k.split('.')[-1]: v for k, v in options.items()}
    cmd_opts_sample_dataset = {
        k: options.get(k)
        for k in ['dimension', 'kernel', 'lengthscale',
                  'randomize_params', 'outcome_transform', 'sigma',
                  'train_samples_size', 'test_samples_size']
    }
    cmd_opts_sample_dataset['standardize_dataset_outcomes'] = options['standardize_outcomes']

    cmd_opts_acquisition_dataset = {
        'train_acquisition_size': options['train_acquisition_size'],
        'test_expansion_factor': options['test_expansion_factor'],
        'replacement': options['replacement'],
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
            'method', 'learning_rate', 'batch_size', 'epochs', 'use_maxei',
            # early stopping
            'early_stopping', 'patience', 'min_delta', 'cumulative_delta',
            # method=policy_gradient
            'include_alpha', 'learn_alpha', 'initial_alpha', 'alpha_increment',
            # method=gittins
            'gi_loss_normalization', 'lamda_min', 'lamda_max', 'lamda',
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
    cmd_dataset = "python gp_acquisition_dataset.py " + " ".join(cmd_args_dataset)

    cmd_opts_nn_no_dataset = {
        **cmd_opts_architecture, **cmd_opts_training
    }
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
    args_nn = get_run_train_parser().parse_args(cmd_args_list_nn)
    (af_dataset_configs, model,
    model_and_info_name, models_path) = get_configs_and_model_and_paths(args_nn)
    _cache[s] = model_and_info_name
    MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN[model_and_info_name] = cmd_opts_nn
    return model_and_info_name


DATASETS_JOB_ID = "datasets"
NO_DATASET_ID = 0
NO_NN_ID = "dep-nn"

def create_dependency_structure_train_acqf(
        options_list:list[dict[str, Any]],
        dependents_list:Optional[list[list[str]]]=None,
        always_train=False):
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
         cmd_nn_train, cmd_opts_nn) = get_command_line_options(options)
        
        # Determine whether to train the NN
        if always_train:
            train_nn = True
        else:
            # Determine whether or not the NN is already cached
            model_and_info_name = cmd_opts_nn_to_model_and_info_name(cmd_opts_nn)
            
            model_already_trained = model_is_trained(model_and_info_name)
            # Train the NN iff it has not already been trained
            train_nn = not model_already_trained
        
        if train_nn:
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
                    "gpu": True,
                    "gpu_gres": "gpu:1"
                }
        else:
            # Do not add the NN train command

            # Add the commands that are dependent on the NN training command
            # (but the NN was already trained so they are not dependent on anything)
            if len(depdendent_commands) > 0:
                if NO_NN_ID not in ret:
                    ret[NO_NN_ID] = {
                        "commands": [],
                        "gpu": True,
                        "gpu_gres": "gpu:1"
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


def add_slurm_args(parser):
    parser.add_argument(
        '--sweep_name',
        type=str,
        default='test',
        help='Name of the sweep.'
    )
    parser.add_argument(
        '--gpu_gres',
        type=str,
        help='GPU resource specification for Slurm. e.g., "gpu:a100:1" or "gpu:1". '
              'Default is "gpu:a100:1".',
        default="gpu:a100:1"
    )
    parser.add_argument(
        '--mail',
        type=str,
        help='email address to send Slurm notifications to. '
              'If not specified, no notifications are sent.'
    )


def add_train_acqf_args(parser, train=True):
    parser.add_argument(
        '--base_config',
        type=str,
        required=True,
        help='YAML file containing the base configuration for the NN acqf experiment.'
    )
    parser.add_argument(
        '--experiment_config',
        type=str,
        required=True,
        help='YAML file containing the specific experiment configuration '
             'for the NN acqf experiment.'
    )
    if train:
        parser.add_argument(
            '--always_train',
            action='store_true',
            help=('If this flag is set, train all acquisition function NNs regardless of '
                'whether they have already been trained. Default is to only train '
                'acquisition function NNs that have not already been trained.')
        )


def get_train_acqf_options_list(args: argparse.Namespace):
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
    
    # TEST
    # b = generate_options(base_config['parameters'])
    # keys = sorted(list(set().union(*[set(x) for x in b])))
    # print(len(b), len(keys))
    # with open(os.path.join(CONFIG_DIR, 'keys.yml'), 'w') as f:
    #     yaml.dump(keys, f)
    # exit()

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
    
    # # TEST
    # # options_list[0]['odf'] = 3
    # combined = group_by_nested_attrs(
    #     options_list,
    #     [{"function_samples_dataset.gp.dimension"},
    #      {"training.method", "training.gi_loss_normalization", "training.lamda_config.lamda_min", "training.lamda_config.lamda_max", "training.lamda_config.lamda"},
    #      {"function_samples_dataset.train_samples_size"}]
    # )
    # print(combined.keys())
    # # exit()
    # with open(os.path.join(CONFIG_DIR, 'combined.yml'), 'w') as f:
    #     yaml.dump(combined, f)
    # exit()
    
    return options_list, refined_config


def submit_jobs_sweep_from_args(jobs_spec, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(SWEEPS_DIR, f"{args.sweep_name}_{timestamp}")
    submit_dependent_jobs(
        sweep_dir=sweep_dir,
        jobs_spec=jobs_spec,
        args=args,
        gpu_gres=args.gpu_gres,
        mail=args.mail
    )


def main():
    parser = argparse.ArgumentParser()
    add_train_acqf_args(parser, train=True)
    add_slurm_args(parser)

    args = parser.parse_args()

    options_list, refined_config = get_train_acqf_options_list(args)
    
    jobs_spec = create_dependency_structure_train_acqf(
        options_list, always_train=args.always_train)
    save_json(jobs_spec, os.path.join(CONFIG_DIR, "dependencies.json"), indent=4)
    submit_jobs_sweep_from_args(jobs_spec, args)


if __name__ == "__main__":
    main()
