import argparse
import math
import torch
from run_bo import GP_AF_DICT, add_bo_loop_args, bo_loop_dicts_to_cmd_args_list, get_arg_names, run_bo
from run_train import get_configs_and_model_and_paths, get_run_train_parser
from train_acqf import add_slurm_args, add_train_acqf_args, create_dependency_structure_train_acqf, get_command_line_options, get_train_acqf_options_list, submit_jobs_sweep_from_args
from utils import dict_to_cmd_args, dict_to_str
import cProfile, pstats


CPROFILE = False


def generate_bo_commands(
        seeds: list[int], objective_args, bo_policy_args, gp_af_args) -> list[str]:
    new_bo_commands = []
    for seed in seeds:
        objective_args_ = {**objective_args, 'gp_seed': seed}
        bo_policy_args_ = {**bo_policy_args, 'bo_seed': seed}
        cmd_args_list = bo_loop_dicts_to_cmd_args_list(
            objective_args=objective_args_,
            bo_policy_args=bo_policy_args_,
            gp_af_args=gp_af_args
        )
        # Only run the BO loop if it has not been run before
        optimization_results = run_bo(
            objective_args_, bo_policy_args_, gp_af_args)
        if optimization_results is None or optimization_results.n_opts_to_run() > 0:
            new_bo_commands.append("python run_bo.py " + " ".join(cmd_args_list))
    return new_bo_commands


def main():
    parser = argparse.ArgumentParser()

    nn_train_group = parser.add_argument_group("NN training")
    add_train_acqf_args(nn_train_group)
    
    parser.add_argument(
        '--n_gp_draws', 
        type=int,
        required=True,
        help='The number of draws of GP objective functions per set of GP params.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='The seed for the random number generator.'
    )
    add_slurm_args(parser)
    bo_loop_group = parser.add_argument_group("BO loops")
    add_bo_loop_args(bo_loop_group) # n_iter, n_initial_samples

    args = parser.parse_args()

    if CPROFILE:
        pr = cProfile.Profile()
        pr.enable()

    bo_loop_arg_names = get_arg_names(bo_loop_group)
    bo_loop_args = {k: v for k, v in vars(args).items() if k in bo_loop_arg_names}

    options_list = get_train_acqf_options_list(args)

    # Set seed again for reproducibility
    torch.manual_seed(args.seed)
    # Set a seed for each round of GP function draw + BO loop
    seeds = torch.randint(0, 2**63-1, (args.n_gp_draws,), dtype=torch.int64).tolist()

    gp_options_dict = {}
    nn_bo_loop_commands_list = []

    run_train_parser = get_run_train_parser()

    for options in options_list:
        gp_options = {
            k.split('.')[-1]: v for k, v in options.items()
            if k.startswith("function_samples_dataset.gp.")
        }

        gp_options_str = dict_to_str(gp_options)

        if gp_options_str not in gp_options_dict:
            gp_options_dict[gp_options_str] = {
                'gp_options': gp_options,
                'lamda_vals': []
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
                gp_options_dict[gp_options_str]['lamda_vals'].append(lamda)
            else:
                # then it is trained with a fixed value of lamda
                pass
        else:
            lamda = None

        (cmd_dataset, cmd_opts_dataset,
         cmd_nn_train, cmd_opts_nn) = get_command_line_options(options)
        
        cmd_args_list_nn = dict_to_cmd_args({**cmd_opts_nn, 'no-save-model': True})
        args_nn = run_train_parser.parse_args(cmd_args_list_nn)
        (af_dataset_configs, model,
        model_and_info_name, models_path) = get_configs_and_model_and_paths(args_nn)
        
        this_bo_loop_commands_list = generate_bo_commands(
            seeds,
            objective_args=gp_options,
            bo_policy_args={**bo_loop_args, 'lamda': lamda,
                            'nn_model_name': model_and_info_name},
            gp_af_args={}
        )
        nn_bo_loop_commands_list.append(this_bo_loop_commands_list)
    
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
        always_train=args.always_train)

    # Add the commands for the GP-BO loops
    gp_bo_commands = []
    for options in gp_options_dict.values():
        gp_options = options['gp_options']
        lamda_vals = options['lamda_vals']
        if lamda_vals == []:
            lamda_vals = [1e-4]

        objective_args = {**gp_options}

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
                new_bo_commands = generate_bo_commands(
                    seeds,
                    objective_args=objective_args,
                    bo_policy_args=bo_policy_args,
                    gp_af_args=gp_af_args
                )
                gp_bo_commands.extend(new_bo_commands)
    if gp_bo_commands:
        jobs_spec['gp_bo'] = {
            'commands': gp_bo_commands,
            'gpu': False
        }
    
    if CPROFILE:
        pr.disable()
        with open('stats_output.txt', 'w') as s:
            ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()

    submit_jobs_sweep_from_args(jobs_spec, args)

# e.g. python bo_experiments_gp.py --base_config config/train_acqf.yml --experiment_config config/train_acqf_experiment_test2.yml --mail adj53@cornell.edu --n_gp_draws 4 --seed 8 --sweep_name test-alon --n_iter 30 --n_initial_samples 1

if __name__ == "__main__":
    main()
