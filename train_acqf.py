import argparse
from collections import defaultdict
import itertools
from datetime import datetime
from typing import Union
import yaml
import os
from gp_acquisition_dataset import create_train_test_gp_acq_datasets_from_args
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
                            new_values = []
                            for value in values:
                                if type(value) is dict:
                                    check_dict_has_keys(
                                        value, ['value', 'parameters'],
                                        lambda kk: f"Invalid key '{kk}' in parameter value dictionary.")
                                    val_name = value['value']
                                else:
                                    val_name = value
                                if val_name in allowed_values:
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
                  'outcome_transform', 'sigma', 'randomize_params']
    }
    cmd_opts_sample_dataset['standardize_dataset_outcomes'] = options['standardize_outcomes']
    cmd_opts_sample_dataset['train_acquisition_size'] = options['train_samples_size'] * expansion_factor
    cmd_opts_sample_dataset['test_acquisition_size'] = options['test_samples_size'] * expansion_factor

    cmd_opts_acquisition_dataset = {
        'expansion_factor': expansion_factor,
        'train_n_candidates': options['n_candidates'],
        'test_n_candidates': options['n_candidates'],
        'min_history': options['min_history'],
        'max_history': options['max_history']
    }

    cmd_opts_architecture = {
        'layer_width': options['layer_width'],
        'standardize_nn_history_outcomes': options['standardize_nn_history_outcomes'],
    }

    cmd_opts_training = {
        k: options.get(k)
        for k in [
            'learning_rate', 'batch_size', 'method',
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


def create_dependency_structure_train_acqf(options_list):
    dataset_commands = []
    nn_commands = []
    for option_dict in options_list:
        (cmd_dataset, cmd_opts_dataset,
         cmd_nn_train, cmd_opts_nn) = get_command_line_options(option_dict)

        if cmd_dataset not in dataset_commands:
            dataset_commands.append((cmd_dataset, cmd_opts_dataset))
        
        nn_commands.append(cmd_nn_train)
    
    datasets_created = []
    for cmd_dataset, cmd_opts_dataset in dataset_commands:
        args_dataset = argparse.Namespace(**cmd_opts_dataset)
        whether_cached = create_train_test_gp_acq_datasets_from_args(
            args_dataset, check_cached=True, load_dataset=False)
        dataset_already_cached = True
        for cached in whether_cached:
            dataset_already_cached = dataset_already_cached and cached
        datasets_created.append(dataset_already_cached)

    dataset_command_ids = {}
    nn_job_arrays = defaultdict(list)
    for ((cmd_dataset, cmd_opts_dataset),
         dataset_already_cached,
         cmd_nn_train) in zip(dataset_commands, datasets_created, nn_commands):
        if dataset_already_cached:
            dataset_id = -1
        else:
            if cmd_dataset in dataset_command_ids:
                dataset_id = dataset_command_ids[cmd_dataset]
            else:
                dataset_id = len(dataset_command_ids) + 1
                dataset_command_ids[cmd_dataset] = dataset_id

        nn_job_arrays[dataset_id].append(cmd_nn_train)
    
    datasets_job_id = "datasets"
    ret = {
        datasets_job_id: {
            "commands": list(dataset_command_ids),
            "gpu": False
        }
    }
    for dataset_id, job_array in nn_job_arrays.items():
        tmp = {
            "commands": job_array,
            "gpu": True
        }
        if dataset_id != -1:
            tmp["dependencies"] = [{
                "job_name": datasets_job_id,
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
        '--mail',
        type=str,
        help='email address to send Slurm notifications to'
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
    
    jobs_spec = create_dependency_structure_train_acqf(options_list)
    save_json(jobs_spec, os.path.join(CONFIG_DIR, "dependencies.json"), indent=4)

    sweep_name = "test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(SWEEPS_DIR, f"{sweep_name}_{timestamp}")
    submit_dependent_jobs(sweep_dir=sweep_dir,
                            jobs_spec=jobs_spec,
                            mail=args.mail)

if __name__ == "__main__":
    main()
