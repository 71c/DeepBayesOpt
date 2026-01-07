"""
Combined BO and Training History Plotting Script

This script generates plots that combine information from both:
1. Bayesian Optimization results (regret, best_y, etc.)
2. Neural network training history (validation loss, training loss, etc.)

The main use case is to create scatter plots showing the relationship between
validation loss and final BO performance across different model configurations.
"""

import copy
import os
import time
from tqdm import tqdm

from experiments.registry import get_registry
from nn_af.acquisition_function_net_save_utils import MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN
from utils.basic_model_save_utils import BASIC_SAVING
from utils.plot_utils import (
    create_plot_directory, get_plot_ax_loss_vs_regret_func, group_by_nested_attrs, save_figures_from_nested_structure)
from utils_general.utils import group_by
from utils_general.io_utils import load_json, save_json
from utils_general.experiments.experiment_config_utils import CONFIG_DIR
from utils.constants import MODELS_DIR

from submit import get_bo_experiments_parser, generate_gp_bo_job_specs
from plot_run import add_plot_interval_args, add_plot_formatting_args, ATTR_NAME_TO_TITLE
from utils_general.plot_utils import add_plot_args
from utils_general.utils import dict_to_str


# Default configuration - can be overridden by auto-plotting
PRE = []
ATTR_A = []
ATTR_B = []
POST = [
    # We still need to group by hyperparameters, but plot function will aggregate
    ["lamda", "gp_af", "nn.method", "nn.learning_rate"],
]

# NEW: Flag to control whether to create one combined plot or separate plots
CREATE_SINGLE_COMBINED_PLOT = True

ONE_FIGURE = True
# Set to True because we're using list-based aggregation in the plotting function
# (The name is misleading - it controls the plot structure, not whether we aggregate)
PLOT_ALL_SEEDS = True

# Plot type to generate
ATTR_GROUPS = [
    ["loss_vs_regret"],
]


def main():
    ## Create parser
    (parser, train_base_config_name, train_experiment_config_name, run_base_config_name,
     run_experiment_config_name) = get_bo_experiments_parser(train=False)
    add_plot_args(parser)
    interval_group = add_plot_interval_args(parser)
    plot_formatting_group = add_plot_formatting_args(parser)

    parser.add_argument(
        '--max_iterations_to_plot',
        type=int,
        default=-1,
        help='Which BO iteration to use for regret measurement. Default is -1 (final iteration).'
    )
    
    parser.add_argument(
        '--plot-mode',
        type=str,
        choices=['scatter', 'density'],
        default='scatter',
        help='Visualization mode: scatter plot (default) or 2D density heatmap'
    )

    ## Parse arguments
    args = parser.parse_args()

    try:
        get_registry().setup_plotting_from_args(args, 'combined_plot', globals())
        print("Successfully applied auto-plotting configuration")
    except Exception as e:
        print(f"Auto-plotting failed, using manual configuration: {e}")

    interval_kwargs = {
        'alpha': args.alpha,
        'interval_of_center': args.interval_of_center,
        'center_stat': args.center_stat if args.center_stat else 'mean',
        'assume_normal': args.assume_normal
    }

    plot_formatting_kwargs = {
        'min_regret_for_plot': args.min_regret_for_plot,
        'add_grid': args.add_grid if hasattr(args, 'add_grid') else False,
    }

    ## Get the configurations and corresponding BO results
    print("\n[TIMING] Starting to generate BO job specs...")
    t_start = time.time()

    jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config \
        = generate_gp_bo_job_specs(
            args,
            train_base_config=getattr(args, train_base_config_name),
            train_experiment_config=getattr(args, train_experiment_config_name),
            run_base_config=getattr(args, run_base_config_name),
            run_experiment_config=getattr(args, run_experiment_config_name)
        )
    save_json(jobs_spec, os.path.join(CONFIG_DIR, "combined_dependencies.json"), indent=4)

    t_jobspec = time.time()
    print(f"[TIMING] Job spec generation took {t_jobspec - t_start:.2f}s")

    print(f"\nNumber of new configs: {len(new_cfgs)}")
    print(f"Number of existing configs: {len(existing_cfgs_and_results)}")

    if len(existing_cfgs_and_results) == 0:
        raise ValueError("There are no saved BO configs to plot.")

    existing_cfgs, existing_results = zip(*existing_cfgs_and_results)

    gr = group_by(existing_cfgs, dict_to_str)
    assert all(len(v) == 1 for v in gr.values())

    # Extract BO results and load training history
    print(f"\n[TIMING] Starting to extract BO results and load training histories...")
    t_extract_start = time.time()

    combined_results_list = []
    valid_configs = []

    # Cache for training histories (same model used across multiple seeds)
    training_history_cache = {}

    for config, r in tqdm(existing_cfgs_and_results, desc="Loading combined data"):
        func_name, trials_dir, result_data = next(iter(r))

        # Extract BO results
        bo_item = {}
        for k, v in result_data.items():
            try:
                bo_item[k] = v[0, :]
            except IndexError:
                print(f"DEBUG IndexError: {k}: shape={getattr(v, 'shape', 'no shape')}")
                raise

        # Get model name and load training history
        nn_model_name = config['bo_policy_args'].get('nn_model_name')
        if nn_model_name is None:
            # This is probably a GP or random search baseline, skip
            print(f"Skipping config without NN model: {config.get('gp_af_args', {}).get('gp_af', 'unknown')}")
            continue

        try:
            # Check cache first
            if nn_model_name in training_history_cache:
                training_history_data = training_history_cache[nn_model_name]
            else:
                # Get model path
                model_and_info_path = os.path.join(MODELS_DIR, nn_model_name)
                model_path = BASIC_SAVING.get_latest_model_path(model_and_info_path)

                # Load training history
                training_history_data = load_json(
                    os.path.join(model_path, 'training_history_data.json'))

                # Cache it
                training_history_cache[nn_model_name] = training_history_data

            # Combine BO results and training history
            combined_item = {
                **bo_item,
                'training_history_data': training_history_data
            }
            combined_results_list.append(combined_item)
            valid_configs.append(config)

        except FileNotFoundError as e:
            print(f"Warning: Could not load training history for {nn_model_name}: {e}")
            continue

    t_extract_end = time.time()
    print(f"[TIMING] Extracting/loading took {t_extract_end - t_extract_start:.2f}s")
    print(f"[TIMING] Unique NN models: {len(training_history_cache)}")
    print(f"[TIMING] Total configs processed: {len(combined_results_list)}")
    print(f"[TIMING] Cache hit rate: {100*(len(combined_results_list)-len(training_history_cache))/len(combined_results_list):.1f}%")

    if len(combined_results_list) == 0:
        raise ValueError("No valid combined results found (need both BO results and training history)")

    print(f"\nSuccessfully loaded {len(combined_results_list)} combined results")

    # Reformat configs with NN parameters
    reformatted_configs = []
    for item in valid_configs:
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

        reformatted_config = {
            **{k if k == 'dimension' else f'objective.{k}': v
               for k, v in item['objective_args'].items()},
            **{k: v for k, v in bo_policy_args.items() if k != 'random_search'},
            **{k if k == 'gp_af' else f'gp_af.{k}': v
               for k, v in item['gp_af_args'].items()}
        }

        # Remove seed-related attributes that we want to aggregate over
        # instead of creating separate groups for
        keys_to_remove = ['bo_seed', 'objective.id', 'objective.gp_seed', 'hpob_seed']
        for key in keys_to_remove:
            reformatted_config.pop(key, None)

        reformatted_configs.append(reformatted_config)

    # Folder name
    save_dir = create_plot_directory(
        args.plots_name, args.plots_group_name, is_run_plot=True)

    script_plot_kwargs = dict(
        sharey=True,
        aspect=1.618,
        scale=1.0,
        max_iterations_to_plot=args.max_iterations_to_plot,
        plot_mode=args.plot_mode,
        **plot_formatting_kwargs,
        **interval_kwargs
    )

    # Generate plots
    print(f"\n[TIMING] Starting plot generation...")
    t_plot_start = time.time()

    for attr_names in ATTR_GROUPS:
        print(f"\n-----------------------------------------------------\n{attr_names=}")

        if attr_names != ["loss_vs_regret"]:
            raise ValueError(f"Unknown attribute group: {attr_names}")

        # Build the attribute grouping structure
        attrs_groups_list = [*PRE] + \
            ([] if len(ATTR_A) == 0 else [ATTR_A]) + \
            ([] if len(ATTR_B) == 0 else [ATTR_B]) + [*POST]

        attrs_groups_list = [set(group) for group in attrs_groups_list]

        # Add a dummy "attr_name" level for the plotting structure
        if len(attrs_groups_list) >= 3:
            attrs_groups_list.insert(-3, {"attr_name"})
        else:
            attrs_groups_list.insert(-2, {"attr_name"})

        # Create configs with attr_name
        this_reformatted_configs = [{**cfg, 'attr_name': 'loss_vs_regret', 'index': i}
             for i, cfg in enumerate(reformatted_configs)]

        # Debug: Check if seed attributes were successfully removed
        if len(this_reformatted_configs) > 0:
            sample_config = this_reformatted_configs[0]
            print(f"DEBUG: Sample config keys: {sorted(sample_config.keys())}")
            seed_keys = ['bo_seed', 'objective.id', 'objective.gp_seed', 'hpob_seed']
            for key in seed_keys:
                if key in sample_config:
                    print(f"WARNING: {key} still in config with value: {sample_config[key]}")

        def get_result_func(i):
            return combined_results_list[i]

        # Group by attributes
        grouped_plot_config, final_attrs_groups_list = group_by_nested_attrs(
            this_reformatted_configs,
            attrs_groups_list)

        # Define level names for plotting
        # We expect 2 levels: attr_name and hyperparameters (seeds are aggregated)
        level_names = ['fname', 'line']

        final_attrs_groups_list = [{'attr_name'}] + final_attrs_groups_list

        if len(final_attrs_groups_list) > 2:
            level_names = ['folder'] * (len(final_attrs_groups_list) - 2) + level_names
        
        print(f"{attrs_groups_list=}")
        print(f"{final_attrs_groups_list=}")
        print(f"{level_names=}")
        print(f"DEBUG: grouped_plot_config type: {type(grouped_plot_config)}")
        if isinstance(grouped_plot_config, dict):
            first_key = list(grouped_plot_config.keys())[0]
            print(f"DEBUG: grouped_plot_config keys (first 5): {list(grouped_plot_config.keys())[:5]}")
            print(f"DEBUG: First item structure: {grouped_plot_config[first_key]}")

        # Create the plot
        plot_ax_func = get_plot_ax_loss_vs_regret_func(get_result_func)

        attr_name_to_title = copy.deepcopy(ATTR_NAME_TO_TITLE)
        attr_name_to_title['loss_vs_regret'] = 'Validation Loss vs. BO Regret'

        if CREATE_SINGLE_COMBINED_PLOT:
            # Create a single combined plot with all data points
            import matplotlib.pyplot as plt
            from utils.plot_utils import BLUE

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # Create a plot config where each configuration is separate (not aggregated)
            # This ensures each configuration creates its own point
            simple_plot_config = {}
            for i in range(len(combined_results_list)):
                # Each configuration gets a unique key and its own index (not as a list)
                simple_plot_config[f'config_{i}'] = {
                    'items': i  # Single index, not a list
                }

            # Call the plotting function
            plot_ax_func(
                plot_config=simple_plot_config,
                ax=ax,
                plot_name='Validation Loss vs. BO Regret',
                attr_name_to_title=attr_name_to_title,
                **script_plot_kwargs
            )

            # Save the figure
            os.makedirs(save_dir, exist_ok=True)
            fig_path = os.path.join(save_dir, 'combined_scatter.pdf')
            fig.savefig(fig_path, dpi=300, format='pdf', bbox_inches='tight')
            plt.close(fig)

            print(f"Saved single combined plot to {fig_path}")
        else:
            # Original behavior - save separate figures
            save_figures_from_nested_structure(
                grouped_plot_config,
                plot_ax_func,
                attrs_groups_list,
                level_names,
                base_folder=save_dir,
                attr_name_to_title=attr_name_to_title,
                print_pbar=True,
                all_seeds=PLOT_ALL_SEEDS,
                **script_plot_kwargs
            )

            print(f"Saved plots to {save_dir}")

    t_plot_end = time.time()
    t_total = t_plot_end - t_start
    print(f"\n[TIMING] Plot generation took {t_plot_end - t_plot_start:.2f}s")
    print(f"[TIMING] Total time: {t_total:.2f}s")
    print(f"[TIMING] Breakdown:")
    print(f"  - Job spec generation: {t_jobspec - t_start:.2f}s ({100*(t_jobspec - t_start)/t_total:.1f}%)")
    print(f"  - Data extraction/loading: {t_extract_end - t_extract_start:.2f}s ({100*(t_extract_end - t_extract_start)/t_total:.1f}%)")
    print(f"  - Plot generation: {t_plot_end - t_plot_start:.2f}s ({100*(t_plot_end - t_plot_start)/t_total:.1f}%)")


if __name__ == '__main__':
    main()
