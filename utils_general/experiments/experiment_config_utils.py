from typing import Any, Union, Optional
import itertools
import yaml
import os


CONFIG_DIR = "config"


def add_config_args(parser, prefix='', experiment_name=None):
    prefix = prefix + '_' if prefix else ''
    suffix = f' for the {experiment_name} experiment' if experiment_name else ''
    base_config_name = f'{prefix}base_config'
    experiment_config_name = f'{prefix}experiment_config'
    parser.add_argument(
        f'--{base_config_name}',
        type=str,
        required=True,
        help=f'YAML file containing the base configuration{suffix}.'
    )
    parser.add_argument(
        f'--{experiment_config_name}',
        type=str,
        required=False,
        help=f'YAML file containing the specific experiment configuration{suffix}.'
    )
    return base_config_name, experiment_config_name


def add_submit_train_args(parser, train=True, prefix='train'):
    train_base_config_name, train_experiment_config_name = add_config_args(
        parser, prefix=prefix, experiment_name='train')

    if train:
        parser.add_argument(
            f'--always_train',
            action='store_true',
            help=('If this flag is set, train all models regardless of '
                'whether they have already been trained. Default is to only train '
                'models that have not already been trained.')
        )
    
    return train_base_config_name, train_experiment_config_name


def get_config_options_list(base_config: str,
                            experiment_config: Optional[str]=None):
    # Load the base configuration
    with open(base_config, 'r') as f:
        base_config = yaml.safe_load(f)

    _check_dict_has_keys(
        base_config,
        ['parameters'],
        lambda kk: f"Invalid key '{kk}' in base configuration.")

    # TEST
    # b = generate_options(base_config['parameters'])
    # keys = sorted(list(set().union(*[set(x) for x in b])))
    # print(len(b), len(keys))
    # with open(os.path.join(CONFIG_DIR, 'keys.yml'), 'w') as f:
    #     yaml.dump(keys, f)
    # exit()

    if experiment_config:
        # Load the experiment configuration
        with open(experiment_config, 'r') as f:
            experiment_config = yaml.safe_load(f)

        _check_dict_has_keys(
            experiment_config,
            ['parameters'],
            lambda kk: f"Invalid key '{kk}' in experiment configuration.")

        if 'parameters' not in experiment_config:
            experiment_config['parameters'] = {}

        # Refine the configuration
        refined_config = {
            'parameters': _refine_config(base_config['parameters'], experiment_config)
        }

        # Save the refined configuration
        with open(os.path.join(CONFIG_DIR, 'refined_config.yml'), 'w') as f:
            yaml.dump(refined_config, f)
    else:
        refined_config = base_config

    # Generate the options
    options_list = _generate_options(refined_config['parameters'])
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


def _check_dict_has_keys(d: dict, keys: list[str], error_msg=None):
    for kk in d:
        if kk not in keys:
            if error_msg is None:
                msg = f"Invalid key '{kk}' in dictionary."
            else:
                msg = error_msg(kk)
            msg += f" Expected one of {keys} but got keys {list(d.keys())}."
            raise ValueError(msg)


def _refine_config(
        params_value: Union[dict[str, dict[str, Any]], list[dict[str, Any]]],
        experiment_config: dict[str, dict[str, Union[bool, list, int, float, type[None]]]],
        params_names:list[str]=[],
        apply_defaults=True):
    if not isinstance(experiment_config, dict):
        raise ValueError('experiment_config must be a dictionary.')

    vary_params = any(param_name in experiment_config['parameters']
                      for param_name in params_names)

    if isinstance(params_value, list): # an OR
        ret = [
            _refine_config(params_dict, experiment_config, params_names,
                          apply_defaults=vary_params)
            for params_dict in params_value
        ]
        if vary_params:
            return ret
        nonempty_indices = [i for i, x in enumerate(ret) if x]
        if len(nonempty_indices) == 0:
            # If the parameter is not varying, we just take the first value
            params_value = [params_value[0]]
            return [
                _refine_config(params_dict, experiment_config, params_names,
                            apply_defaults=True)
                for params_dict in params_value
            ]
        return [
            _refine_config(params_value[i], experiment_config, params_names,
                          apply_defaults=True)
            for i in nonempty_indices
        ]
    elif isinstance(params_value, dict): # an AND
        tmp: dict[str, Any] = {}
        for param_name, param_config in params_value.items():
            this_params_names = params_names
            this_vary_params = vary_params or param_name in experiment_config['parameters']
            if param_name in experiment_config['parameters']:
                info = experiment_config['parameters'][param_name]
                _check_dict_has_keys(
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
                _check_dict_has_keys(
                    param_config,
                    ['parameters'],
                    lambda kk: f"Invalid key '{kk}' in parameter dictionary."
                )
                tmp[param_name] = {
                    'parameters': _refine_config(
                    param_config['parameters'], experiment_config, this_params_names,
                    apply_defaults=apply_defaults)
                }
            else:
                _check_dict_has_keys(
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
                                    _check_dict_has_keys(
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
                    if apply_defaults:
                        if 'value' in param_config:
                            values = [param_config['value']]
                        else:
                            values = [param_config['values'][0]]
                    else:
                        values = []

                for i, value in enumerate(values):
                    if type(value) is dict:
                        _check_dict_has_keys(
                            value, ['value', 'parameters'],
                            lambda kk: f"Invalid key '{kk}' in parameter value dictionary.")
                        if 'parameters' in value:
                            values[i]['parameters'] = _refine_config(
                                value['parameters'], experiment_config, this_params_names,
                                apply_defaults=apply_defaults)
                        else:
                            values[i] = value['value']
                if len(values) > 0:
                    tmp[param_name] = {'values': values} if len(values) != 1 else {'value': values[0]}
        return tmp
    else:
        raise ValueError(
            f'params_value must be a dictionary or list, got {(params_value)}.')


def _generate_options(
        params_value: Union[dict[str, dict[str, Any]], list[dict[str, dict[str, Any]]]],
        prefix=''
    ) -> list[dict[str, Any]]:
    if isinstance(params_value, list): # an OR
        return [d for p in params_value for d in _generate_options(p, prefix=prefix)]
    elif isinstance(params_value, dict): # an AND
        options_per_param = []
        for param_name, cfg in params_value.items():
            new_prefix = param_name if prefix == '' else prefix + '.' + param_name

            if 'parameters' in cfg:
                _check_dict_has_keys(
                    cfg,
                    ['parameters'],
                    lambda kk: f"Invalid key '{kk}' in parameter dictionary."
                )
                param_options = _generate_options(cfg['parameters'], prefix=new_prefix)
            else:
                _check_dict_has_keys(
                    cfg,
                    ['value', 'values'],
                    lambda kk: f"Invalid key '{kk}' in parameter dictionary.")
                values = cfg['values'] if 'values' in cfg else [cfg['value']]
                param_options = []
                for value in values:
                    if type(value) is dict:
                        _check_dict_has_keys(
                            value, ['value', 'parameters'],
                            lambda kk: f"Invalid key '{kk}' in parameter value dictionary.")
                        if 'parameters' in value:
                            options_this_value = [
                                {new_prefix: value['value'], **params}
                                for params in _generate_options(
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
