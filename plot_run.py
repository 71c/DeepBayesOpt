import copy
import os
import cProfile, pstats
import numpy as np

from experiments.registry import get_registry
from nn_af.acquisition_function_net_save_utils import load_module, MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN
from utils.plot_sorting import plot_dict_to_str
from utils.plot_utils import (
    create_plot_directory, get_plot_ax_af_iterations_func, get_plot_ax_bo_stats_vs_iteration_func, group_by_nested_attrs, save_figures_from_nested_structure)
from utils_general.utils import group_by
from utils_general.io_utils import save_json
from utils_general.experiments.experiment_config_utils import CONFIG_DIR

from submit import get_bo_experiments_parser, generate_gp_bo_job_specs
from single_run import GP_AF_DICT, pre_run_bo
from utils_general.plot_utils import add_plot_args
from utils_general.utils import dict_to_str, get_arg_names


CPROFILE = True

PER_ITERATION_DECISIONS_SPLIT_INTO_FOLDERS = True
ONE_FIGURE = False
PLOT_ALL_SEEDS = True

INCLUDE_TIMES = False

# "optimize_process_time"
ATTR_GROUPS = [
    # ["per_iteration_decisions"],
    # ["best_y"],
    ["normalized_regret"],
    # ["model_fitting_errors"],
    # ["regret"],
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
    "regret": "Regret",
    "normalized_regret": "Normalized Regret",
    "model_fitting_errors": "Cumulative Model Fitting Errors",
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
              'When --assume_normal is not set, you may choose "mean" or "median".')
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


def compute_auto_max_iterations(
        results_list,
        attr_name='normalized_regret',
        threshold=1e-6,
        buffer_fraction=0.25,
        min_buffer=5):
    """
    Automatically determine the optimal number of iterations to plot.

    This function analyzes all BO methods to find which ones achieve regret below
    the threshold, determines when they first cross that threshold, and returns
    a sensible max iteration count with a buffer for aesthetic plotting.

    Args:
        results_list: List of result dictionaries from BO runs
        attr_name: The attribute to analyze (default: 'normalized_regret')
        threshold: The regret threshold to consider as "converged" (default: 1e-6)
        buffer_fraction: Fraction of extra iterations to add as buffer (default: 0.25)
        min_buffer: Minimum number of iterations to add as buffer (default: 5)

    Returns:
        int or None: The suggested max iterations to plot, or None if auto-detection
                     should not be applied
    """
    if attr_name not in {'normalized_regret', 'regret'}:
        # Auto-detection only works for regret metrics
        return None

    convergence_iterations = []
    max_available_iterations = 0

    for result in results_list:
        if attr_name not in result:
            continue

        data = result[attr_name]
        if len(data.shape) != 2:
            continue

        # data has shape (n_seeds, n_iterations+1)
        # We look at each seed separately
        n_seeds, n_iters_plus_one = data.shape
        n_iters = n_iters_plus_one - 1
        max_available_iterations = max(max_available_iterations, n_iters)

        for seed_idx in range(n_seeds):
            seed_data = data[seed_idx, :]

            # Find first iteration where regret goes below threshold
            below_threshold = seed_data < threshold
            if np.any(below_threshold):
                # Find the first index where it goes below threshold
                first_below_idx = np.argmax(below_threshold)
                convergence_iterations.append(first_below_idx)

    if len(convergence_iterations) == 0:
        # No method converged below threshold, don't auto-limit
        return None

    # Take the maximum convergence iteration across all methods/seeds that converged
    max_convergence_iter = max(convergence_iterations)

    # Add buffer
    buffer_iters = max(
        int(max_convergence_iter * buffer_fraction),
        min_buffer
    )
    suggested_max = max_convergence_iter + buffer_iters

    # Cap at available iterations
    suggested_max = min(suggested_max, max_available_iterations)

    # Make sure we show at least some iterations (minimum 10)
    suggested_max = max(suggested_max, 10)

    return suggested_max


def add_plot_formatting_args(parser):
    plot_formatting_group = parser.add_argument_group("Plot formatting")
    plot_formatting_group.add_argument(
        '--add_grid',
        action='store_true',
        help='If set, add a grid to the plots'
    )
    plot_formatting_group.add_argument(
        '--add_markers',
        action='store_true',
        help='If set, add markers to the lines in the plots at each iteration'
    )
    plot_formatting_group.add_argument(
        '--min_regret_for_plot',
        type=float,
        default=1e-6,
        help='Minimum regret value to display in log-scale plots. Values below this '
             'will be clipped to this value to prevent extremely small regrets from '
             'compressing the y-axis range. Default: 1e-6'
    )
    
    return plot_formatting_group


def add_plot_iterations_args(parser):
    plot_iterations_group = parser.add_argument_group("Plot iterations args")
    plot_iterations_group.add_argument(
        '--n_iterations',
        type=int,
        default=40,
        help='Number of iterations to plot for the acquisition function animation'
    )
    plot_iterations_group.add_argument(
        '--max_iterations_to_plot',
        type=int,
        default=None,
        help='Maximum number of iterations to display in the BO regret plots (must be <= n_iter ran). '
             'If not specified, all iterations will be plotted.'
    )
    plot_iterations_group.add_argument(
        '--auto_max_iterations_buffer',
        type=float,
        default=0.25,
        help='When auto-detecting max_iterations_to_plot, add this fraction as a buffer '
             'beyond the convergence point (default: 0.25 = 25%% extra iterations)'
    )
    plot_iterations_group.add_argument(
        '--auto_max_iterations_min_buffer',
        type=int,
        default=5,
        help='Minimum number of iterations to add as buffer when auto-detecting '
             'max_iterations_to_plot (default: 5)'
    )
    return plot_iterations_group


def main():
    ############################### CREATE PARSER ######################################
    (parser, train_base_config_name, train_experiment_config_name, run_base_config_name,
     run_experiment_config_name) = get_bo_experiments_parser(train=False)
    add_plot_args(parser)
    interval_group = add_plot_interval_args(parser)
    plot_formatting_group = add_plot_formatting_args(parser)
    plot_iterations_group = add_plot_iterations_args(parser)
    
    ############################### PARSE ARGUMENTS ####################################
    args = parser.parse_args()

    ########################### SETUP PLOTTING CONFIG ##################################
    try:
        get_registry().setup_plotting_from_args(args, 'run_plot', globals())
        print("Successfully applied auto-plotting configuration")
    except Exception as e:
        print(f"Auto-plotting failed, using manual configuration: {e}")

    ############### GET JOB CONFIGS AND CORRESPONDING RESULTS ##########################
    jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config \
        = generate_gp_bo_job_specs(
            args,
            train_base_config=getattr(args, train_base_config_name),
            train_experiment_config=getattr(args, train_experiment_config_name),
            run_base_config=getattr(args, run_base_config_name),
            run_experiment_config=getattr(args, run_experiment_config_name)
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
    results_list = []
    for config, r in existing_cfgs_and_results:
        func_name, trials_dir, result_data = next(iter(r))
        item = {}
        for k, v in result_data.items():
            try:
                item[k] = v[0, :]
            except IndexError:
                print(f"DEBUG IndexError: {k}: shape={getattr(v, 'shape', 'no shape')}, ndim={getattr(v, 'ndim', 'no ndim')}, type={type(v)}")
                if hasattr(v, 'shape') and len(v.shape) > 0:
                    print(f"DEBUG IndexError: {k} first few values: {v.flat[:min(5, v.size)]}")
                    print(f"{func_name=}, {trials_dir=}")
                    print(f"{config=}")
                raise  # Re-raise the error after printing debug info
        results_list.append(item)

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
    save_dir = create_plot_directory(
        args.plots_name, args.plots_group_name, is_run_plot=True)
    
    plot_formatting_kwargs = {
        k: getattr(args, k) for k in get_arg_names(plot_formatting_group)}
    interval_kwargs = {k: getattr(args, k) for k in get_arg_names(interval_group)}
    if interval_kwargs['interval_of_center']:
        if interval_kwargs['center_stat'] is None:
            raise ValueError(
                "When --interval_of_center is set, --center_stat must be specified")
    elif interval_kwargs['center_stat'] is not None:
        raise ValueError(
            "When --interval_of_center is not set, --center_stat cannot be specified")

    script_plot_kwargs = dict(
        sharey=True,
        aspect=1.618,
        scale=1.0,
        shade=True,
        **plot_formatting_kwargs,
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

        # Auto-detect max_iterations_to_plot if not specified and plotting regret
        effective_max_iterations = args.max_iterations_to_plot
        if (not plot_af_iterations and
            args.max_iterations_to_plot is None and
            len(attr_names) == 1 and
            attr_names[0] in {'normalized_regret', 'regret'}):

            auto_max = compute_auto_max_iterations(
                results_list,
                attr_name=attr_names[0],
                threshold=args.min_regret_for_plot,
                buffer_fraction=args.auto_max_iterations_buffer,
                min_buffer=args.auto_max_iterations_min_buffer
            )
            if auto_max is not None:
                effective_max_iterations = auto_max
                print(f"Auto-detected max_iterations_to_plot: {effective_max_iterations} "
                      f"(based on threshold={args.min_regret_for_plot})")

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
                # attrs_groups_list.insert(-3, {"attr_name"})
                attrs_groups_list = [{"attr_name"}] + attrs_groups_list
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
        # script_config.pop("train_base_config")
        # script_config.pop("train_experiment_config")
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
                    nn = load_module(bo_policy_args['nn_model_name'],
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
                    # Slice the data if max_iterations_to_plot is specified (or auto-detected)
                    data_value = results[attr_name]
                    if effective_max_iterations is not None:
                        # +1 to include the initial value before doing BO
                        # (when we only have evaluated at the sobol points)
                        data_value = data_value[:effective_max_iterations+1]

                    ret = {
                        attr_name: data_value,
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
