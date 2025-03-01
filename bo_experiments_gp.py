import argparse
import math
import os
from submit_dependent_jobs import CONFIG_DIR
import torch
from run_bo import GP_AF_DICT, add_bo_loop_args, bo_loop_dicts_to_cmd_args_list, get_arg_names, run_bo
from train_acqf import add_slurm_args, add_train_acqf_args, cmd_opts_nn_to_model_and_info_name, create_dependency_structure_train_acqf, get_command_line_options, get_train_acqf_options_list, submit_jobs_sweep_from_args
from utils import dict_to_str, save_json
import cProfile, pstats


CPROFILE = True


def _generate_bo_commands(
        seeds: list[int], objective_args, bo_policy_args, gp_af_args):
    new_bo_comma = []
    new_bo_conf = []
    existing_bo_conf_and_results = []
    for seed in seeds:
        objective_args_ = {**objective_args, 'gp_seed': seed}
        bo_policy_args_ = {**bo_policy_args, 'bo_seed': seed}
        bo_config = dict(
            objective_args=objective_args_,
            bo_policy_args=bo_policy_args_,
            gp_af_args=gp_af_args
        )
        cmd_args_list = bo_loop_dicts_to_cmd_args_list(**bo_config)
        # Only run the BO loop if it has not been run before
        opt_results = run_bo(**bo_config)
        
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
        options_list, bo_loop_args, seeds, always_train=False):    
    gp_options_dict = {}
    nn_bo_loop_commands_list = []

    new_bo_configs = []
    existing_bo_configs_and_results = []

    for options in options_list:
        gp_options = {
            k.split('.')[-1]: v for k, v in options.items()
            if k.startswith("function_samples_dataset.gp.")
        }

        gp_options_str = dict_to_str(gp_options)

        if gp_options_str not in gp_options_dict:
            gp_options_dict[gp_options_str] = {
                'gp_options': gp_options,
                'lamda_vals': set()
            }

        if options['training.method'] == 'gittins':
            # then there will be lambda value(s) specified
            lamda = options.get('training.lamda_config.lamda')
            if lamda is None:
                # then it is trained with a range of lamda values
                lamda_min = options['training.lamda_config.lamda_min']
                lamda_max = options['training.lamda_config.lamda_max']
                log_min, log_max = math.log10(lamda_min), math.log10(lamda_max)
                # We will test with the average
                log_lamda = 0.5 * (log_min + log_max)
                lamda = 10**log_lamda
                gp_options_dict[gp_options_str]['lamda_vals'].add(lamda)
            else:
                # then it is trained with a fixed value of lamda
                pass
        else:
            lamda = None

        (cmd_dataset, cmd_opts_dataset,
         cmd_nn_train, cmd_opts_nn) = get_command_line_options(options)
        
        model_and_info_name = cmd_opts_nn_to_model_and_info_name(cmd_opts_nn)
        new_cmds, new_cfgs, existing_cfgs_and_results = _generate_bo_commands(
            seeds,
            objective_args=gp_options,
            bo_policy_args={**bo_loop_args, 'lamda': lamda,
                            'nn_model_name': model_and_info_name},
            gp_af_args={}
        )
        nn_bo_loop_commands_list.append(new_cmds)
        new_bo_configs.extend(new_cfgs)
        existing_bo_configs_and_results.extend(existing_cfgs_and_results)
    
    # Only train the NNs that are needed for the BO loops
    included = [
        (nn_options, nn_bo_loop_cmds)
        for nn_options, nn_bo_loop_cmds in zip(options_list, nn_bo_loop_commands_list)
        if nn_bo_loop_cmds
    ]
    options_list, nn_bo_loop_commands_list = zip(*included)

    jobs_spec = create_dependency_structure_train_acqf(
        options_list,
        dependents_list=nn_bo_loop_commands_list,
        always_train=always_train)

    # Add the commands for the GP-BO loops
    gp_bo_commands = []
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

        for gp_af_name in GP_AF_DICT:
            gp_af_args = {'gp_af': gp_af_name, **gp_af_fit_args}
            
            lamda_vals_this_af = lamda_vals if gp_af_name == 'gittins' else [None]
            for lamda in lamda_vals_this_af:
                extra_bo_policy_args = {'lamda': lamda}
                bo_policy_args = {**bo_loop_args, **extra_bo_policy_args}
                new_cmds, new_cfgs, existing_cfgs_and_results = _generate_bo_commands(
                    seeds,
                    objective_args=gp_options,
                    bo_policy_args=bo_policy_args,
                    gp_af_args=gp_af_args
                )
                gp_bo_commands.extend(new_cmds)
                new_bo_configs.extend(new_cfgs)
                existing_bo_configs_and_results.extend(existing_cfgs_and_results)
    if gp_bo_commands:
        jobs_spec['gp_bo'] = {
            'commands': gp_bo_commands,
            'gpu': False
        }
    return jobs_spec, new_bo_configs, existing_bo_configs_and_results


def get_bo_experiments_parser(train=True):
    parser = argparse.ArgumentParser()
    
    objectives_group = parser.add_argument_group("Objective functions and seed")
    objectives_group.add_argument(
        '--n_gp_draws', 
        type=int,
        required=True,
        help='The number of draws of GP objective functions per set of GP params.'
    )
    objectives_group.add_argument(
        '--seed',
        type=int,
        required=True,
        help='The seed for the random number generator.'
    )
    bo_loop_group = parser.add_argument_group("BO loops")
    add_bo_loop_args(bo_loop_group) # n_iter, n_initial_samples

    nn_train_group = parser.add_argument_group("NN experiments")
    add_train_acqf_args(nn_train_group, train=train)
    return parser, bo_loop_group


def gp_bo_jobs_spec_cfgs_from_args(args, bo_loop_group):
    bo_loop_arg_names = get_arg_names(bo_loop_group)
    bo_loop_args = {k: v for k, v in vars(args).items() if k in bo_loop_arg_names}

    options_list, refined_config = get_train_acqf_options_list(args)

    # Set seed again for reproducibility
    torch.manual_seed(args.seed)
    # Set a seed for each round of GP function draw + BO loop
    seeds = torch.randint(0, 2**63-1, (args.n_gp_draws,), dtype=torch.int64).tolist()

    if CPROFILE:
        pr = cProfile.Profile()
        pr.enable()
    
    jobs_spec, new_cfgs, existing_cfgs_and_results = _gp_bo_jobs_spec_and_cfgs(
        options_list, bo_loop_args, seeds,
        always_train=getattr(args, 'always_train', False)
    )

    if CPROFILE:
        pr.disable()
        with open('stats_output.txt', 'w') as s:
            ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
    
    return jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config


def main():
    parser, bo_loop_group = get_bo_experiments_parser(train=True)

    slurm_group = parser.add_argument_group("Slurm and logging")
    add_slurm_args(slurm_group)

    args = parser.parse_args()

    jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config \
        = gp_bo_jobs_spec_cfgs_from_args(args, bo_loop_group)

    save_json(jobs_spec, os.path.join(CONFIG_DIR, "dependencies.json"), indent=4)
    submit_jobs_sweep_from_args(jobs_spec, args)


# e.g. python bo_experiments_gp.py --base_config config/train_acqf.yml --experiment_config config/train_acqf_experiment_test2.yml --mail adj53@cornell.edu --n_gp_draws 4 --seed 8 --sweep_name test-alon --n_iter 30 --n_initial_samples 1

if __name__ == "__main__":
    main()
