import argparse
from typing import Any, Optional
import cProfile, pstats
import torch

from nn_af.acquisition_function_net_save_utils import get_lamda_for_bo_of_nn
from utils.utils import dict_to_str, group_by, save_json
from utils.experiments.experiment_config_utils import CONFIG_DIR, add_config_args, get_config_options_list
from utils.experiments.submit_dependent_jobs import add_slurm_args, submit_jobs_sweep_from_args

from run_bo import GP_AF_DICT, bo_loop_dicts_to_cmd_args_list, run_bo
from train_acqf import add_train_acqf_args, cmd_opts_nn_to_model_and_info_name, create_dependency_structure_train_acqf, get_cmd_options_train_acqf


CPROFILE = False


def _generate_bo_commands(
        seeds: list[int], objective_args, bo_policy_args, gp_af_args,
        single_objective=False):
    new_bo_comma = []
    new_bo_conf = []
    existing_bo_conf_and_results = []
    seed0 = seeds[0]
    for seed in seeds:
        bo_seed = seed
        gp_seed = seed0 if single_objective else seed
        objective_args_ = {**objective_args, 'gp_seed': gp_seed}
        bo_policy_args_ = {**bo_policy_args, 'bo_seed': bo_seed}
        bo_config = dict(
            objective_args=objective_args_,
            bo_policy_args=bo_policy_args_,
            gp_af_args=gp_af_args
        )
        cmd_args_list = bo_loop_dicts_to_cmd_args_list(**bo_config, validate=False)
        # Only run the BO loop if it has not been run before
        opt_results = run_bo(**bo_config, load_weights=False)
        
        does_not_have_result = opt_results is None or opt_results.n_opts_to_run() > 0
        if does_not_have_result:
            # Need to get the result
            new_bo_comma.append("python run_bo.py " + " ".join(cmd_args_list))
            new_bo_conf.append(bo_config)
        else:
            # Store the configs corresponding to results we already have
            existing_bo_conf_and_results.append((bo_config, opt_results))

    return new_bo_comma, new_bo_conf, existing_bo_conf_and_results


def _gp_bo_jobs_spec_and_cfgs(
        options_list, bo_loop_args_list, bo_loop_args_list_random_search,
        seeds, single_objective=False, always_train=False,
        dependents_slurm_options:dict[str, Any]={}):
    gp_options_dict = {}

    new_bo_configs = []
    existing_bo_configs_and_results = []

    included = []

    for nn_options in options_list:
        gp_options = {
            k.split('.')[-1]: v for k, v in nn_options.items()
            if k.startswith("function_samples_dataset.gp.")
        }

        gp_options_str = dict_to_str(gp_options)

        if gp_options_str not in gp_options_dict:
            gp_options_dict[gp_options_str] = {
                'gp_options': gp_options,
                'lamda_vals': set()
            }

        if nn_options['training.method'] == 'gittins':
            # then there will be lambda value(s) specified
            lamda = nn_options.get('training.lamda_config.lamda')
            lamda_min = nn_options.get('training.lamda_config.lamda_min')
            lamda_max = nn_options.get('training.lamda_config.lamda_max')
            
            lamda = get_lamda_for_bo_of_nn(lamda, lamda_min, lamda_max)
            gp_options_dict[gp_options_str]['lamda_vals'].add(lamda)
        else:
            lamda = None

        (cmd_dataset, cmd_opts_dataset,
         cmd_nn_train, cmd_opts_nn) = get_cmd_options_train_acqf(nn_options)
        
        (args_nn, af_dataset_configs, pre_model, model_and_info_name, models_path
        ) = cmd_opts_nn_to_model_and_info_name(cmd_opts_nn)

        all_new_cmds_this_nn = []
        for bo_loop_args in bo_loop_args_list:
            new_cmds, new_cfgs, existing_cfgs_and_results = _generate_bo_commands(
                seeds,
                objective_args=gp_options,
                bo_policy_args={**bo_loop_args, 'lamda': lamda,
                                'nn_model_name': model_and_info_name},
                gp_af_args={},
                single_objective=single_objective
            )
            all_new_cmds_this_nn.extend(new_cmds)
            new_bo_configs.extend(new_cfgs)
            existing_bo_configs_and_results.extend(existing_cfgs_and_results)
        if all_new_cmds_this_nn:
            included.append((nn_options, all_new_cmds_this_nn))

    if included:
        options_list, nn_bo_loop_commands_list = zip(*included)
        jobs_spec = create_dependency_structure_train_acqf(
            options_list,
            dependents_list=nn_bo_loop_commands_list,
            always_train=always_train,
            dependents_slurm_options=dependents_slurm_options)
    else:
        jobs_spec = {}

    # Add the GP AF commands & random search
    non_nn_bo_commands = []
    for options in gp_options_dict.values():
        gp_options = options['gp_options']
        lamda_vals = options['lamda_vals']
        if len(lamda_vals) == 0:
            lamda_vals = {1e-4}

        if gp_options['randomize_params']:
            gp_af_fit_args = {
                k: v for k, v in gp_options
                if not (k == 'dimension' or k == 'randomize_params')
            }
            gp_af_fit_args['fit'] = 'map'
        else:
            gp_af_fit_args = {'fit': 'exact'}

        # Add all the GP AF commands
        for gp_af_name in GP_AF_DICT:
            gp_af_args = {'gp_af': gp_af_name, **gp_af_fit_args}
            
            lamda_vals_this_af = lamda_vals if gp_af_name == 'gittins' else [None]
            for lamda in lamda_vals_this_af:
                extra_bo_policy_args = {'lamda': lamda}
                for bo_loop_args in bo_loop_args_list:
                    new_cmds, new_cfgs, existing_cfgs_and_results = _generate_bo_commands(
                        seeds,
                        objective_args=gp_options,
                        bo_policy_args={**bo_loop_args, **extra_bo_policy_args},
                        gp_af_args=gp_af_args,
                        single_objective=single_objective
                    )
                    non_nn_bo_commands.extend(new_cmds)
                    new_bo_configs.extend(new_cfgs)
                    existing_bo_configs_and_results.extend(existing_cfgs_and_results)
        
        # Add random search
        for bo_loop_args in bo_loop_args_list_random_search:
            new_cmds, new_cfgs, existing_cfgs_and_results = _generate_bo_commands(
                seeds,
                objective_args=gp_options,
                bo_policy_args={**bo_loop_args, 'random_search': True},
                gp_af_args={},
                single_objective=single_objective
            )
            non_nn_bo_commands.extend(new_cmds)
            new_bo_configs.extend(new_cfgs)
            existing_bo_configs_and_results.extend(existing_cfgs_and_results)
    
    if non_nn_bo_commands:
        jobs_spec['no_nn'] = {
            'commands': non_nn_bo_commands,
            'gpu': False
        }
    return jobs_spec, new_bo_configs, existing_bo_configs_and_results


def get_bo_experiments_parser(train=True):
    parser = argparse.ArgumentParser()
    
    objectives_group = parser.add_argument_group("Objective functions and seed")
    objectives_group.add_argument(
        '--n_seeds', 
        type=int,
        required=True,
        help='The number of replicates to run per BO method'
    )
    objectives_group.add_argument(
        '--single_objective',
        action='store_true',
        help='If single_objective=True, then only one GP objective function is drawn '
        'per set of GP params, and n_seeds replicates are run for that same objective '
        'function, each with a different BO seed. Otherwise, n_seeds GP objective '
        'functions are drawn per set of GP params, and for each of these, the BO '
        'seed is set to be the same as the GP seed.'
    )
    objectives_group.add_argument(
        '--seed',
        type=int,
        required=True,
        help='The seed for the random number generator.'
    )
    bo_loop_group = parser.add_argument_group("BO loops")
    # add_bo_loop_args(bo_loop_group) # n_iter, n_initial_samples

    bo_base_config_name, bo_experiment_config_name = add_config_args(
        bo_loop_group, prefix='bo', experiment_name='BO loops')

    nn_train_group = parser.add_argument_group("NN experiments")
    nn_base_config_name, nn_experiment_config_name = add_train_acqf_args(nn_train_group,
                                                                         train=train)
    return (parser,
            nn_base_config_name, nn_experiment_config_name,
            bo_base_config_name, bo_experiment_config_name)


def generate_gp_bo_job_specs(args: argparse.Namespace,
                             nn_base_config: str,
                             bo_base_config: str,
                             nn_experiment_config: Optional[str]=None,
                             bo_experiment_config: Optional[str]=None,
                             dependents_slurm_options:dict[str, Any]={}):
    bo_options_list, bo_refined_config = get_config_options_list(
        bo_base_config, bo_experiment_config)

    # I admit, this is really hacky and dumb, but what the hell,
    # much of my code is like that.
    bo_options_list_random_search = [
        {k: v for k, v in o.items() if not k.startswith('optimizer.')}
        for o in bo_options_list
    ]
    g = group_by(bo_options_list_random_search, dict_to_str)
    bo_options_list_random_search = [next(iter(v)) for v in g.values()]

    bo_options_list = [
        {k.split('.')[-1]: v for k, v in options.items()}
        for options in bo_options_list
    ]

    nn_options_list, nn_refined_config = get_config_options_list(
        nn_base_config, nn_experiment_config)

    # Set seed again for reproducibility
    torch.manual_seed(args.seed)
    # Set a seed for each round of GP function draw + BO loop
    seeds = torch.randint(0, 2**63-1, (args.n_seeds,), dtype=torch.int64).tolist()

    if CPROFILE:
        pr = cProfile.Profile()
        pr.enable()
    
    jobs_spec, new_cfgs, existing_cfgs_and_results = _gp_bo_jobs_spec_and_cfgs(
        nn_options_list, bo_options_list, bo_options_list_random_search,
        seeds,
        single_objective=args.single_objective,
        always_train=getattr(args, 'always_train', False),
        dependents_slurm_options=dependents_slurm_options
    )

    if CPROFILE:
        pr.disable()
        with open('stats_output-_gp_bo_jobs_spec_and_cfgs.txt', 'w') as s:
            ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
    
    refined_config = {
        'parameters': {
            **nn_refined_config['parameters'],
            **bo_refined_config['parameters']
        }
    }

    return jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config


def main():
    (parser, nn_base_config_name, nn_experiment_config_name, bo_base_config_name,
     bo_experiment_config_name) = get_bo_experiments_parser(train=True)

    slurm_group = parser.add_argument_group("Slurm and logging")
    add_slurm_args(slurm_group)

    args = parser.parse_args()

    jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config \
        = generate_gp_bo_job_specs(
            args,
            nn_base_config=getattr(args, nn_base_config_name),
            nn_experiment_config=getattr(args, nn_experiment_config_name),
            bo_base_config=getattr(args, bo_base_config_name),
            bo_experiment_config=getattr(args, bo_experiment_config_name),
            dependents_slurm_options={
                "gpu": True,
                "gres": "gpu:1",
                "time": "2:00:00",
            }
        )
    
    print(f"Number of new BO configs: {len(new_cfgs)}")
    print(f"Number of existing BO configs: {len(existing_cfgs_and_results)}")

    submit_jobs_sweep_from_args(jobs_spec, args)


if __name__ == "__main__":
    main()
