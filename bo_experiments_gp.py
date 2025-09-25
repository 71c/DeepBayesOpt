import argparse
from typing import Any, Optional
import cProfile, pstats
import warnings
import torch
from botorch.exceptions import UnsupportedError

from datasets.hpob_dataset import get_hpob_dataset_ids
from nn_af.acquisition_function_net_save_utils import get_lamda_for_bo_of_nn
from utils.utils import dict_to_str, group_by
from utils.experiments.experiment_config_utils import add_config_args, get_config_options_list
from utils.experiments.submit_dependent_jobs import add_slurm_args, submit_jobs_sweep_from_args

from run_bo import GP_AF_DICT, bo_loop_dicts_to_cmd_args_list, run_bo
from train_acqf import add_train_acqf_args, cmd_opts_nn_to_model_and_info_name, create_dependency_structure_train_acqf, get_cmd_options_train_acqf


CPROFILE = False

N_HPOB_SEEDS = 5 # They provide 5 predefined seeds for HPO-B experiments
HPOB_SEEDS = [f"test{i}" for i in range(N_HPOB_SEEDS)]


def _generate_bo_commands(
        seeds: list[int], objective_args, bo_policy_args, gp_af_args,
        n_objectives:Optional[int]=None, n_bo_seeds: Optional[int]=None,
        use_hpob_seeds=False, recompute_bo: bool = False,
        recompute_non_nn_only: bool = False):
    new_bo_comma = []
    new_bo_conf = []
    existing_bo_conf_and_results = []

    dataset_type = objective_args['dataset_type']
    if dataset_type == 'hpob':
        test_dataset_ids = get_hpob_dataset_ids(
            objective_args['hpob_search_space_id'], 'test')
        n_objectives = len(test_dataset_ids)
        if use_hpob_seeds:
            bo_seeds = seeds[:N_HPOB_SEEDS]
        else:
            assert n_bo_seeds is not None
            bo_seeds = seeds[:n_bo_seeds]
    elif dataset_type == 'gp':
        assert n_objectives is not None
        assert n_bo_seeds is not None
        bo_seeds = seeds[:n_bo_seeds]
    
    for objective_idx in range(n_objectives):
        objective_args_ = {**objective_args}
        if dataset_type == 'hpob':
            objective_args_['id'] = test_dataset_ids[objective_idx]
        else:
            objective_args_['id'] = seeds[objective_idx]
        
        for bo_seed_idx, bo_seed in enumerate(bo_seeds):
            bo_policy_args_ = {**bo_policy_args, 'bo_seed': bo_seed}
            if dataset_type == 'hpob' and use_hpob_seeds:
                bo_policy_args_['hpob_seed'] = HPOB_SEEDS[bo_seed_idx]
                # Don't need n_initial_samples for HPO-B if using predefined seeds
                bo_policy_args_.pop('n_initial_samples', None)
            bo_config = dict(
                objective_args=objective_args_,
                bo_policy_args=bo_policy_args_,
                gp_af_args=gp_af_args
            )

            cmd_args_list = bo_loop_dicts_to_cmd_args_list(**bo_config, validate=False)
            # Only run the BO loop if it has not been run before
            opt_results = run_bo(**bo_config, load_weights=False)
            
            does_not_have_result = opt_results is None or opt_results.n_opts_to_run() > 0

            # Check if we should recompute based on the flags
            should_recompute = False
            if recompute_bo:
                should_recompute = True
            elif recompute_non_nn_only:
                # Check if this is a non-NN result (GP or random search)
                is_nn = bo_policy_args.get('nn_model_name') is not None
                is_random_search = bo_policy_args.get('random_search', False)
                has_gp_af = len(gp_af_args) > 0 and gp_af_args.get('gp_af') is not None
                is_non_nn = is_random_search or has_gp_af
                should_recompute = is_non_nn and not is_nn

            if does_not_have_result or should_recompute:
                # Need to get the result
                cmd_str = "python run_bo.py " + " ".join(cmd_args_list)

                # Add recompute flag if we're forcing recomputation
                if should_recompute:
                    cmd_str += " --recompute-result"

                new_bo_comma.append(cmd_str)
                new_bo_conf.append(bo_config)
            else:
                # Store the configs corresponding to results we already have
                existing_bo_conf_and_results.append((bo_config, opt_results))

    return new_bo_comma, new_bo_conf, existing_bo_conf_and_results


def _gp_bo_jobs_spec_and_cfgs(
        options_list, bo_loop_args_list, bo_loop_args_list_random_search,
        seeds,
        n_objectives: Optional[int]=None, n_bo_seeds: Optional[int]=None,
        use_hpob_seeds=False, always_train=False,
        dependents_slurm_options:dict[str, Any]={},
        recompute_bo: bool = False,
        recompute_non_nn_only: bool = False):
    objective_args_dict = {}

    new_bo_configs = []
    existing_bo_configs_and_results = []

    included = []

    for nn_options in options_list:
        ## Determine dataset_type and get objective_args
        dataset_type = nn_options.get('function_samples_dataset.dataset_type', 'gp')
        if dataset_type not in {'gp', 'hpob'}:
            raise UnsupportedError(
                f"Unsupported dataset type: {dataset_type}. Must be 'gp' or 'hpob'.")
        gp_options = {
            k.split('.')[-1]: v for k, v in nn_options.items()
            if k.startswith("function_samples_dataset.gp.")
        }
        hpob_options = {
            k.split('.')[-1]: v for k, v in nn_options.items()
            if k.startswith("function_samples_dataset.hpob.")
        }
        if dataset_type == 'gp':
            assert len(gp_options) != 0 and len(hpob_options) == 0, \
                "If dataset_type is 'gp', then there must be some "\
                "gp options and no hpob options."
            objective_args = gp_options
        elif dataset_type == 'hpob':
            assert len(gp_options) == 0 and len(hpob_options) != 0, \
                "If dataset_type is 'hpob', then there must be some "\
                "hpob options and no gp options."
            objective_args = hpob_options

        objective_args['dataset_type'] = dataset_type
        
        objective_args_str = dict_to_str(objective_args)

        if objective_args_str not in objective_args_dict:
            objective_args_dict[objective_args_str] = {
                'objective_args': objective_args,
                'lamda_vals': set()
            }

        if nn_options['training.method'] == 'gittins':
            # then there will be lambda value(s) specified
            lamda = nn_options.get('training.lamda_config.lamda')
            lamda_min = nn_options.get('training.lamda_config.lamda_min')
            lamda_max = nn_options.get('training.lamda_config.lamda_max')
            
            lamda = get_lamda_for_bo_of_nn(lamda, lamda_min, lamda_max)
            objective_args_dict[objective_args_str]['lamda_vals'].add(lamda)
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
                objective_args=objective_args,
                bo_policy_args={**bo_loop_args, 'lamda': lamda,
                                'nn_model_name': model_and_info_name},
                gp_af_args={},
                n_objectives=n_objectives,
                n_bo_seeds=n_bo_seeds,
                use_hpob_seeds=use_hpob_seeds,
                recompute_bo=recompute_bo,
                recompute_non_nn_only=False  # This is NN-based, so exclude from non_nn_only
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
    for options in objective_args_dict.values():
        objective_args = options['objective_args']
        lamda_vals = options['lamda_vals']
        if len(lamda_vals) == 0:
            lamda_vals = {1e-4}

        dataset_type = objective_args['dataset_type']
        if dataset_type == 'hpob':
            # Use the default GP fit settings of the latest BoTorch version (0.15.1)
            gp_af_fit_args = {'fit': 'map', 'kernel': 'RBF'}
        elif dataset_type == 'gp':
            if objective_args['randomize_params']:
                gp_af_fit_args = {
                    k: v for k, v in objective_args
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
                        objective_args=objective_args,
                        bo_policy_args={**bo_loop_args, **extra_bo_policy_args},
                        gp_af_args=gp_af_args,
                        n_objectives=n_objectives,
                        n_bo_seeds=n_bo_seeds,
                        use_hpob_seeds=use_hpob_seeds,
                        recompute_bo=recompute_bo,
                        recompute_non_nn_only=recompute_non_nn_only
                    )
                    non_nn_bo_commands.extend(new_cmds)
                    new_bo_configs.extend(new_cfgs)
                    existing_bo_configs_and_results.extend(existing_cfgs_and_results)
        
        # Add random search
        for bo_loop_args in bo_loop_args_list_random_search:
            new_cmds, new_cfgs, existing_cfgs_and_results = _generate_bo_commands(
                seeds,
                objective_args=objective_args,
                bo_policy_args={**bo_loop_args, 'random_search': True},
                gp_af_args={},
                n_objectives=n_objectives,
                n_bo_seeds=n_bo_seeds,
                use_hpob_seeds=use_hpob_seeds,
                recompute_bo=recompute_bo,
                recompute_non_nn_only=recompute_non_nn_only
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
        required=False,
        help='The number of replicates to run per BO method and objective function.'
    )
    objectives_group.add_argument(
        '--use_hpob_seeds',
        action='store_true',
        help=f'If set, then the {N_HPOB_SEEDS} HPO-B predefined seeds will be used when optimizing '
                'HPO-B objective functions.'
    )
    objectives_group.add_argument(
        '--n_objectives',
        type=int,
        required=False,
        help='The number of objective functions to optimize. Only used for GP '
             'datasets; ignored for HPO-B datasets. Must be specified if any of '
             'the datasets are GP.'
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

    # Add recompute options
    recompute_group = parser.add_argument_group("Recompute options")
    recompute_group.add_argument(
        '--recompute-bo',
        action='store_true',
        help='Recompute/overwrite existing BO results (all types)'
    )
    recompute_group.add_argument(
        '--recompute-non-nn-only',
        action='store_true',
        help='Recompute/overwrite only non-NN BO results (GP and random search)'
    )

    return (parser,
            nn_base_config_name, nn_experiment_config_name,
            bo_base_config_name, bo_experiment_config_name)


def _validate_bo_experiments_args(args: argparse.Namespace, dataset_types):
    n_seeds = getattr(args, 'n_seeds', None)
    n_objectives = getattr(args, 'n_objectives', None)
    use_hpob_seeds = args.use_hpob_seeds
    
    if n_seeds is None:
        if 'gp' in dataset_types:
            raise ValueError("If any of the objective functions are GP, then --n_seeds "
                            "must be specified.")
        if not use_hpob_seeds:
            raise ValueError("If --n_seeds is not specified, then --use_hpob_seeds "
                             "must be set.")
    else:
        if 'gp' not in dataset_types and use_hpob_seeds:
            raise ValueError("Cannot set both --n_seeds and --use_hpob_seeds if none "
                             "of the objective functions are GP.")
        if n_seeds <= 0:
            raise ValueError("If specified, --n_seeds must be a positive integer.")
    
    if n_objectives is None:
        if 'gp' in dataset_types:
            raise ValueError("If any of the objective functions are GP, then "
                             "--n_objectives must be specified.")
        if 'hpob' not in dataset_types:
            # This should never happen, but just in case -- maybe more dataset types
            # will be added in the future.
            raise ValueError("If --n_objectives is not specified, then at least one of the "
                             "objective functions must be HPO-B.")
    else:
        if 'gp' not in dataset_types:
            raise ValueError("If --n_objectives is specified, then at least one of the "
                             "objective functions must be GP.")
        if 'hpob' in dataset_types:
            warnings.warn("Warning: --n_objectives is specified, but some of the objective "
                  "functions are HPO-B. The HPO-B functions will use their own "
                  "predefined objectives, and the number of HPO-B functions will not be "
                  "controlled by --n_objectives.")
        if n_objectives <= 0:
            raise ValueError("--n_objectives must be a positive integer.")


def generate_gp_bo_job_specs(args: argparse.Namespace,
                             nn_base_config: str,
                             bo_base_config: str,
                             nn_experiment_config: Optional[str]=None,
                             bo_experiment_config: Optional[str]=None,
                             dependents_slurm_options:dict[str, Any]={},
                             recompute_bo: bool = False,
                             recompute_non_nn_only: bool = False):
    bo_options_list, bo_refined_config = get_config_options_list(
        bo_base_config, bo_experiment_config)
    
    nn_options_list, nn_refined_config = get_config_options_list(
        nn_base_config, nn_experiment_config)
    
    # Determine the dataset type for purpose of args validation
    dataset_types = {
        item['function_samples_dataset.dataset_type'] for item in nn_options_list}

    # Since generate_gp_bo_job_specs is always called after get_bo_experiments_parser
    # and parser.parse_args(), we can do this here.
    _validate_bo_experiments_args(args, dataset_types)

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

    # Determine the number of seeds to generate
    if args.n_seeds is None:
        # then we are optimizing HPO-B, and use_hpob_seeds must be True
        n_seeds = N_HPOB_SEEDS
    else:
        n_seeds = max(N_HPOB_SEEDS, args.n_seeds) if args.use_hpob_seeds else args.n_seeds
        if args.n_objectives is not None:
            # We are using the seeds for BO loop seeds and also potentially
            # GP objective function seeds, so make sure we have enough seeds
            n_seeds = max(n_seeds, args.n_objectives)

    # Set the seeds for initial random state of the BO loops and/or GP objective functions
    torch.manual_seed(args.seed)
    seeds = torch.randint(0, 2**63-1, (n_seeds,), dtype=torch.int64).tolist()

    if CPROFILE:
        pr = cProfile.Profile()
        pr.enable()
    
    jobs_spec, new_cfgs, existing_cfgs_and_results = _gp_bo_jobs_spec_and_cfgs(
        nn_options_list, bo_options_list, bo_options_list_random_search,
        seeds,
        n_objectives=getattr(args, 'n_objectives', None),
        n_bo_seeds=args.n_seeds,
        use_hpob_seeds=args.use_hpob_seeds,
        always_train=getattr(args, 'always_train', False),
        dependents_slurm_options=dependents_slurm_options,
        recompute_bo=recompute_bo,
        recompute_non_nn_only=recompute_non_nn_only
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

    # Validate recompute arguments
    if getattr(args, 'recompute_bo', False) and getattr(args, 'recompute_non_nn_only', False):
        parser.error("Cannot specify both --recompute-bo and --recompute-non-nn-only. "
                     "--recompute-bo includes all BO results.")

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
            },
            recompute_bo=getattr(args, 'recompute_bo', False),
            recompute_non_nn_only=getattr(args, 'recompute_non_nn_only', False)
        )
    
    print(f"Number of new BO configs: {len(new_cfgs)}")
    print(f"Number of existing BO configs: {len(existing_cfgs_and_results)}")

    submit_jobs_sweep_from_args(jobs_spec, args)


if __name__ == "__main__":
    main()
