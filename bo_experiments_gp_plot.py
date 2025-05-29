import copy
import os
import cProfile, pstats

from nn_af.acquisition_function_net_save_utils import load_nn_acqf
from utils.plot_utils import create_plot_directory, get_plot_ax_af_iterations_func
from utils.utils import dict_to_str, group_by, group_by_nested_attrs, save_json
from utils.plot_utils import add_plot_args, get_plot_ax_bo_stats_vs_iteration_func, plot_dict_to_str, save_figures_from_nested_structure
from utils.experiments.experiment_config_utils import CONFIG_DIR

from bo_experiments_gp import get_bo_experiments_parser, generate_gp_bo_job_specs
from run_bo import GP_AF_DICT, get_arg_names, pre_run_bo
from train_acqf import MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN


CPROFILE = True



# PRE = [
#     ["nn.layer_width", "nn.train_samples_size", "gen_candidates"]
# ]
# ATTR_A = ["nn.batch_size"]
# ATTR_B = ["nn.learning_rate"]


# PRE = [
#     ["nn.train_samples_size"],
#     ["nn.learning_rate", "nn.lr_scheduler"]
# ]
# ATTR_A = ["gen_candidates"]
# ATTR_B = ["raw_samples", "num_restarts"]


# PRE = [
#     ["gen_candidates"],
#     ["nn.train_samples_size"],
#     ["nn.learning_rate", "nn.lr_scheduler"]
# ]
# ATTR_A = ["raw_samples"]
# ATTR_B = ["num_restarts"]


# PRE = [
#     ["nn.layer_width", "nn.train_samples_size", "gen_candidates"]
# ]
# ATTR_A = ["nn.lr_scheduler"]
# ATTR_B = ["nn.learning_rate"]


# PRE = [
#     ["nn.layer_width", "nn.train_samples_size", "gen_candidates", "nn.lr_scheduler"]
# ]
# ATTR_A = ["nn.lr_scheduler_patience", "nn.lr_scheduler_factor"]
# ATTR_B = ["nn.learning_rate"]


# For 8dim_maxhistory20_gittins_dataset_size
# PRE = []
# ATTR_A = ["nn.train_samples_size"]
# ATTR_B = ["nn.samples_addition_amount"]


# For 8dim_maxhistory20_big
## First version:
# PRE = [
#     ["objective.lengthscale"],
#     ["nn.learning_rate"]
# ]
# ATTR_A = ["nn.train_samples_size"]
# ATTR_B = ["nn.samples_addition_amount"]
# POST = [
#     ["lamda", "gp_af", "nn.method", "nn.lr_scheduler"],
#     ["bo_seed"]
# ]
## Second version:
# PRE = [
#     ["objective.lengthscale"],
#     ["nn.train_samples_size"]
# ]
# ATTR_A = ["nn.samples_addition_amount"]
# ATTR_B = ["nn.learning_rate"]
# POST = [
#     ["lamda", "gp_af", "nn.method", "nn.lr_scheduler"],
#     ["bo_seed"]
# ]


# # For 8dim_maxhistory20_gittins_regularization
# PRE = [
#     ["nn.samples_addition_amount"]
# ]
# ATTR_A = ["nn.layer_width"]
# ATTR_B = ["nn.weight_decay"]
# POST = [
#     ["lamda", "gp_af", "nn.method", "nn.learning_rate", "nn.lr_scheduler"],
#     ["bo_seed"]
# ]

# For 8dim_maxhistory20_gittins_regularization_2
# and 8dim_maxhistory20_regularization
PRE = [
    ["nn.layer_width"]
]
ATTR_A = ["nn.dropout"]
ATTR_B = ["nn.weight_decay"]
POST = [
    ["lamda", "gp_af", "nn.method", "nn.learning_rate", "nn.lr_scheduler"],
    ["bo_seed"]
]


# Default POST:
# POST = [
#     ["lamda", "gp_af", "nn.method"],
#     ["bo_seed"]
# ]

PER_ITERATION_DECISIONS_SPLIT_INTO_FOLDERS = True
ONE_FIGURE = True
PLOT_ALL_SEEDS = True

INCLUDE_TIMES = False

# "optimize_process_time"
ATTR_GROUPS = [
    # ["per_iteration_decisions"],
    ["best_y"],
    # ["x"],
    # ["best_y", "x"]
]
if INCLUDE_TIMES:
    ATTR_GROUPS += [
        ["best_y", "mean_eval_process_time", "process_time", "n_evals"],
        ["process_time"],
        ["mean_eval_process_time"]
    ]


ATTR_NAME_TO_TITLE = {
    "best_y": "Best function value",
    "process_time": "Process time",
    "time": "Time",
    "n_evals": "Number of AF evaluations by optimize_acqf",
    "mean_eval_process_time": "Mean time to evaluate AF in optimize_acqf",
    "optimize_process_time": "Time spent optimizing",
    "x": "x"
}


def add_plot_interval_args(parser):
    interval_group = parser.add_argument_group("Plotting intervals")
    interval_group.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='The significance level alpha for the interval '
        '(e.g., 0.05 for a 95%% interval)'
    )
    interval_group.add_argument(
        '--interval_of_center',
        action='store_true',
        help='If set, calculate the confidence interval for the center (mean/median); '
        'if not, calculate the prediction interval'
    )
    interval_group.add_argument(
        '--center_stat',
        type=str,
        choices=['mean', 'median'],
        help=('Specifies the statistic to use as the center. '
              'When --assume_normal is set, this must be "mean" '
              '(defaulting to "mean" if not provided). '
              'When --assume_normal is not set, you may choose "mean" or "median", '
              'with "median" as the default.')
    )
    interval_group.add_argument(
        '--assume_normal',
        action='store_true',
        help='If set, assume the data is normally distributed and use the '
        't-distribution for interval calculations; otherwise, use '
        'bootstrapping/quantiles. I recommend to use this option only if '
        '--interval_of_center is set, as a way to save compute, since '
        'this a good approximation due to the limit theorem. '
        '(But it is not that bad compute using bootstrapping.)'
    )
    return interval_group


def main():
    ## Create parser
    (parser, nn_base_config_name, nn_experiment_config_name, bo_base_config_name,
     bo_experiment_config_name) = get_bo_experiments_parser(train=False)
    add_plot_args(parser)
    interval_group = add_plot_interval_args(parser)
    parser.add_argument(
        '--n_iterations',
        type=int,
        default=40,
        help='Number of iterations to plot for the acquisition function animation'
    )
    
    ## Parse arguments
    args = parser.parse_args()

    interval_kwargs = {k: getattr(args, k) for k in get_arg_names(interval_group)}

    ## Get the configurations and corresponding results
    jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config \
        = generate_gp_bo_job_specs(
            args,
            nn_base_config=getattr(args, nn_base_config_name),
            nn_experiment_config=getattr(args, nn_experiment_config_name),
            bo_base_config=getattr(args, bo_base_config_name),
            bo_experiment_config=getattr(args, bo_experiment_config_name)
        )
    save_json(jobs_spec, os.path.join(CONFIG_DIR, "dependencies.json"), indent=4)
    
    print(f"Number of new configs: {len(new_cfgs)}")
    print(f"Number of existing configs: {len(existing_cfgs_and_results)}")
    
    if len(existing_cfgs_and_results) == 0:
        raise ValueError("There are no saved BO configs to plot.")
    
    existing_cfgs, existing_results = zip(*existing_cfgs_and_results)
    
    gr = group_by(existing_cfgs, dict_to_str)
    assert all(len(v) == 1 for v in gr.values())

    # Extract results
    results_list = [
        {k: v[0, :] for k, v in next(iter(r))[2].items()}
        for r in existing_results
    ]

    reformatted_configs = []
    for item in existing_cfgs:
        bo_policy_args = item['bo_policy_args']
        nn_model_name = bo_policy_args.get('nn_model_name')
        if nn_model_name is not None:
            bo_policy_args.update(
                {"nn." + k: v
                 for k, v in MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN[nn_model_name].items()
                 }
            )
        
        if 'random_search' in bo_policy_args:
            random_search = bo_policy_args['random_search']
            if random_search:
                item['gp_af_args']['gp_af'] = 'random search'

        reformatted_configs.append({
            **{k if k == 'dimension' else f'objective.{k}': v
               for k, v in item['objective_args'].items()},
            **{k: v for k, v in bo_policy_args.items() if k != 'random_search'},
            **{k if k == 'gp_af' else f'gp_af.{k}': v
               for k, v in item['gp_af_args'].items()}
        })

    # Folder name
    save_dir = create_plot_directory(args.plots_name, args.plots_group_name, is_bo=True)
    
    
    script_plot_kwargs = dict(
        sharey=True,
        aspect=1.618,
        scale=1.0,
        shade=True,
        **interval_kwargs
    )

    all_keys = set().union(*[set(result.keys()) for result in results_list])
    for attr_names in ATTR_GROUPS:
        if CPROFILE:
            pr = cProfile.Profile()
            pr.enable()

        print(f"\n-----------------------------------------------------\n{attr_names=}")

        if attr_names == ["per_iteration_decisions"]:
            plot_af_iterations = True
        else:
            plot_af_iterations = False
            attr_names = [a for a in attr_names if a in all_keys]
        
        post = copy.deepcopy(POST)
        attr_a = copy.deepcopy(ATTR_A)
        put_attr_a_into_line = plot_af_iterations
        if put_attr_a_into_line:
            post[0] += attr_a
            attr_a = []
        if len(attr_names) == 1:
            attrs_groups_list = [*PRE] + \
            ([] if len(attr_a) == 0 else [attr_a]) + \
            ([] if len(ATTR_B) == 0 else [ATTR_B]) + [*post]
        else:
            attrs_groups_list = [
                *PRE,
                [*attr_a, *ATTR_B],
                *post
            ]
        
        attrs_groups_list = [set(group) for group in attrs_groups_list]
        
        if plot_af_iterations:
            if PER_ITERATION_DECISIONS_SPLIT_INTO_FOLDERS:
                if ONE_FIGURE and not PLOT_ALL_SEEDS:
                    attrs_groups_list.insert(-2, {"attr_name"})
                else:
                    attrs_groups_list.insert(-1, {"attr_name"})
            else:
                attrs_groups_list.insert(-1, {"attr_name"})
            attr_names = list(range(args.n_iterations))
        else:
            if len(attrs_groups_list) >= 3:
                # Right before the one before "line" (-2) level
                attrs_groups_list.insert(-3, {"attr_name"})
            else:
                # Right before "line" (-2) level
                attrs_groups_list.insert(-2, {"attr_name"})
        
        this_reformatted_configs = [{**cfg, 'attr_name': name, 'index': i}
             for name in attr_names
             for i, cfg in enumerate(reformatted_configs)
             if not (plot_af_iterations and cfg.get('gp_af') == 'random search')
        ]
        if not plot_af_iterations:
            for cfg in this_reformatted_configs:
                if "nn_model_name" in cfg:
                    cfg.pop("nn_model_name")
        
        if plot_af_iterations and not PLOT_ALL_SEEDS and not ONE_FIGURE:
            last = attrs_groups_list[-1]
            attrs_groups_list = attrs_groups_list[:-1]
            if PLOT_ALL_SEEDS:
                idx = -1
            else:
                idx = -2
            attrs_groups_list.insert(idx, last)

        plot_config, new_attrs_groups_list = group_by_nested_attrs(
            this_reformatted_configs,
            attrs_groups_list,
            dict_to_str_func=plot_dict_to_str,
            add_extra_index=-2 # -2 is the "line" level
        )

        if plot_af_iterations and PLOT_ALL_SEEDS:
            new_attrs_groups_list.append(None)

        if plot_af_iterations and PER_ITERATION_DECISIONS_SPLIT_INTO_FOLDERS:
            if ONE_FIGURE:
                use_rows = True
                use_cols = False
            else:
                if not PLOT_ALL_SEEDS:
                    new_attrs_groups_list.insert(-1, None)
                use_rows = False
                use_cols = False
        else:
            use_rows = args.use_rows
            use_cols = args.use_cols

        n_groups = len(new_attrs_groups_list)
        if n_groups < 2:
            raise ValueError(
                "There are not enough levels of plot grouping (at least 2 are required). "
                f"{new_attrs_groups_list=}")
        
        # Make script config
        script_config = {**vars(args)}
        # script_config.pop("nn_base_config")
        # script_config.pop("nn_experiment_config")
        script_config["nn_train_and_bo_config"] = refined_config
        script_config["plots_config"] = [
            group if group is None else sorted(list(group))
            for group in new_attrs_groups_list]
        
        # SPECIAL
        script_config["gp_af_names"] = list(GP_AF_DICT)

        if plot_af_iterations:
            if PER_ITERATION_DECISIONS_SPLIT_INTO_FOLDERS:
                if ONE_FIGURE:
                    levels_to_add = ["line"]
                else:
                    if PLOT_ALL_SEEDS:
                        levels_to_add = ["line"]
                    else:
                        levels_to_add = ["random", "line"]
            else:
                levels_to_add = ["line"]
        else:
            levels_to_add = ["random", "line"]
        if use_cols:
            levels_to_add.append("col")
        if use_rows:
            levels_to_add.append("row")

        levels_reversed = levels_to_add[:n_groups]

        levels_reversed.append('fname')
        if len(levels_reversed) > n_groups:
            # need to add a file name level
            plot_config = {
                "results": {
                    "items": plot_config
                }
            }
            new_attrs_groups_list = [None] + new_attrs_groups_list
        elif len(levels_reversed) < n_groups:
            levels_reversed.extend(['folder'] * (n_groups - len(levels_reversed)))

        save_dir_this_attrs = os.path.join(save_dir, "-".join(map(str, attr_names)))
        print(f"Saving plots to {save_dir_this_attrs}")
        
        # Add folder level
        levels_reversed.append("folder")
        new_attrs_groups_list = [None] + new_attrs_groups_list
        plot_config = {
            save_dir_this_attrs: {
                "items": plot_config,
                "vals": script_config
            }
        }

        level_names = list(reversed(levels_reversed))

        print("Plotting configuration:")
        for level_name, attrs in zip(level_names, new_attrs_groups_list):
            # if attrs:
            print(f"  {level_name}: {attrs}")

        save_json(plot_config, "config/plot_config.json", indent=2)

        if plot_af_iterations:
            def get_result(index):
                cfg = this_reformatted_configs[index]
                idx = cfg['index']
                results = results_list[idx]
                attr_name = cfg['attr_name']

                c = existing_cfgs[idx]
                
                bo_policy_args = c['bo_policy_args']
                
                stuff = pre_run_bo(c['objective_args'], c['bo_policy_args'],
                                       c['gp_af_args'])
                ret = {
                    'results': results,
                    'attr_name': attr_name,
                    'index': idx,
                    'lamda': cfg.get('lamda'),
                    'lamda_min': cfg.get('nn.lamda_min'),
                    'lamda_max': cfg.get('nn.lamda_max'),
                    'objective_gp': stuff.get('objective_gp'),
                    'objective_octf': stuff.get('objective_octf'),
                    'objective': stuff['objective_fn']
                }
                
                if 'nn_model_name' in bo_policy_args:
                    # print(f"{c=}")
                    # print()
                    # print(f"{cfg=}")
                    # print()
                    # print(f"{stuff=}")
                    # exit()
                    nn = load_nn_acqf(bo_policy_args['nn_model_name'],
                                        return_model_path=False,
                                        load_weights=True,
                                        verbose=False)
                    ret['nn_model'] = nn
                    ret['method'] = bo_policy_args['nn.method'] # == cfg['nn.method']
                    # return None
                else:
                    # print(f"{c=}")
                    # print()
                    # print(f"{cfg=}")
                    # print()
                    # print(f"{stuff=}")
                    # exit()
                    af_options = stuff['af_options']
                    if not af_options: # Random Search
                        return None
                    ret['gp_model'] = af_options['optimizer_kwargs_per_function'][0]['model']
                    # ret['af_class'] = af_options['acquisition_function_class']
                    # ret['fit_params'] = af_options['fit_params']
                    ret['gp_af'] = cfg['gp_af'] # == c['gp_af_args']['gp_af'] e.g. "LogEI"
                    ret['gp_af_fit'] = cfg['gp_af.fit'] # == c['gp_af_args']['fit'] e.g. "exact"
                return ret

            plot_ax_func = get_plot_ax_af_iterations_func(get_result)
        else:
            def get_result(index):
                cfg = this_reformatted_configs[index]
                idx = cfg['index']
                results = results_list[idx]
                attr_name = cfg['attr_name']
                
                # bo_policy_args = existing_cfgs[idx]['bo_policy_args']
                
                if attr_name in results:
                    ret = {
                        attr_name: results[attr_name],
                        'attr_name': attr_name,
                        'index': idx
                    }
                    # if 'nn_model_name' in bo_policy_args:
                    #     ret['nn_model_name'] = bo_policy_args['nn_model_name']
                    return ret
                return None

            plot_ax_func = get_plot_ax_bo_stats_vs_iteration_func(get_result)

        
        this_script_plot_kwargs = {**script_plot_kwargs}
        if plot_af_iterations and PER_ITERATION_DECISIONS_SPLIT_INTO_FOLDERS \
                and ONE_FIGURE:
            this_script_plot_kwargs["aspect"] = 3.0
            # this_script_plot_kwargs["scale"] = 0.87

        save_figures_from_nested_structure(
            plot_config,
            plot_ax_func,
            new_attrs_groups_list,
            level_names,
            attr_name_to_title=ATTR_NAME_TO_TITLE,
            base_folder=save_dir_this_attrs,
            print_pbar=True,
            all_seeds=not plot_af_iterations or PLOT_ALL_SEEDS,
            **this_script_plot_kwargs
        )

        if CPROFILE:
            pr.disable()
            output_path = os.path.join(save_dir_this_attrs, 'cprofile_stats.txt')
            with open(output_path, 'w') as s:
                ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
                ps.print_stats()

if __name__ == "__main__":
    main()
