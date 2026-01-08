import argparse
import cProfile
import os
import pstats
from typing import Optional

import torch

from experiments.registry import get_registry
from dataset_factory import create_train_test_acquisition_datasets_from_args
from utils.basic_model_save_utils import BASIC_SAVING
from utils.plot_sorting import plot_dict_to_str
from utils.utils import get_lamda_for_bo_of_nn
from utils_general.experiments.experiment_config_utils import get_config_options_list
from utils.plot_utils import N_CANDIDATES_PLOT, create_plot_directory, group_by_nested_attrs, plot_acquisition_function_net_training_history_ax, plot_nn_vs_gp_acquisition_function_1d, save_figures_from_nested_structure
from utils_general.utils import group_by
from utils_general.io_utils import load_json, save_json

from nn_af.acquisition_function_net_save_utils import load_module, cmd_opts_train_to_args_module_paths
from submit_train import add_train_acqf_args, get_cmd_options_train_acqf
from utils_general.plot_utils import add_plot_args
from utils_general.utils import DEVICE, dict_to_str

CPROFILE = False


# # For 1dim_pointnet-max_history_input_variation-pbgi
# PRE = []
# ATTR_A = ["train_samples_size", "samples_addition_amount"]
# ATTR_B = ["max_history_input"]
# POST = [
#     ["learning_rate"]
# ]


ATTR_GROUPS = [
    ["0_training_history_train_test"],

    # ["0_training_history_train_test", "1_training_history_test_log_regret", "2_af_plot"],
    # ["0_training_history_train_test", "2_af_plot"],
    # ["1_training_history_test_log_regret"],
    # ["2_af_plot"],

    # ["0_training_history_train_test", "1_training_history_test_log_regret"],
]

ATTR_NAME_TO_TITLE = {
    "0_training_history_train_test": "Training history (train and test loss)",
    "1_training_history_test_log_regret": "Training history (test log regret)",
    "2_af_plot": "Acquisition function plot"
}


N_HISTORY = 10


def get_plot_train_ax_func(get_result_func):
    def plot_train_ax(
            plot_config,
            ax,
            plot_name: Optional[str]=None,
            label='',
            color=None,
            alpha=1.0,
            **plot_kwargs):
        if not isinstance(plot_config, int):
            import matplotlib.pyplot as plt
            prop_cycle = plt.rcParams['axes.prop_cycle']
            c = prop_cycle()
            for label, value in plot_config.items():
                plot_train_ax(
                    value['items'], ax, plot_name=plot_name,
                    label=label, color=next(c)['color'], alpha=0.7*alpha, **plot_kwargs)
            return

        info = get_result_func(plot_config)
        attr_name = info['attr_name']

        results = info['results']
        training_history_data = results['training_history_data']
        nn_model = results['model']

        if attr_name == "0_training_history_train_test":
            plot_acquisition_function_net_training_history_ax(
                ax, training_history_data, plot_maxei=False, plot_name=plot_name,
                plot_log_regret=False, label=label, color=color, alpha=alpha)
        elif attr_name == "1_training_history_test_log_regret":
            plot_acquisition_function_net_training_history_ax(
                ax, training_history_data, plot_maxei=False, plot_name=plot_name,
                plot_log_regret=True, label=label, color=color, alpha=alpha)
        elif attr_name == "2_af_plot":
            aq_dataset = results['dataset_getter']()

            it = iter(aq_dataset)
            item = next(it)
            try:
                while item.x_hist.shape[0] != N_HISTORY:
                    item = next(it)
            except StopIteration:
                raise ValueError("No item with the right number of history points.")
            
            # Get the data to plot
            x_hist, y_hist, x_cand, vals_cand = item.tuple_no_model
            x_cand_original = x_cand
            dimension = x_hist.size(1)
            x_cand_varying_component = torch.linspace(0, 1, N_CANDIDATES_PLOT)
            varying_index = 0
            if dimension == 1:
                # N_CANDIDATES_PLOT x 1
                x_cand = x_cand_varying_component.unsqueeze(1)
            else:
                torch.manual_seed(0)
                random_x = torch.rand(dimension)
                # N_CANDIDATES_PLOT x dimension
                x_cand = random_x.repeat(N_CANDIDATES_PLOT, 1)
                x_cand[:, varying_index] = x_cand_varying_component

            # Get the GP model
            gp_model = item.model if item.has_model else None

            cfg = info['config']

            lamda = get_lamda_for_bo_of_nn(
                    cfg.get('lamda'), cfg.get('lamda_min'), cfg.get('lamda_max'))
            plot_nn_vs_gp_acquisition_function_1d(
                ax=ax, x_hist=x_hist, y_hist=y_hist, x_cand=x_cand,
                # x_cand_original=x_cand_original, vals_cand=vals_cand,
                lamda=lamda,
                gp_model=gp_model, nn_model=nn_model, method=cfg['method'],
                gp_fit_methods=['exact'],
                min_x=0.0, max_x=1.0,
                nn_device=DEVICE, group_standardization=None,
                varying_index=varying_index
            )
            ax.set_title("Acquisition function plot")
        else:
            raise ValueError(f"Unknown attribute name: {attr_name}")
    
    return plot_train_ax


def main():
    ## Create parser
    parser = argparse.ArgumentParser()
    train_base_config_name, train_experiment_config_name = add_train_acqf_args(parser,
                                                                         train=False)
    add_plot_args(parser)

    ## Parse arguments
    args = parser.parse_args()

    # Auto-configure plotting parameters based on experiment
    try:
        get_registry().setup_plotting_from_args(args, 'train_plot', globals())
        print("Successfully applied auto-plotting configuration")
    except Exception as e:
        print(f"Auto-plotting failed, using manual configuration: {e}")
    
    PLOTS_CONFIG_SINGLE = [
        *PRE,
        ATTR_A,
        ATTR_B,
        *POST
    ]

    PLOTS_CONFIG_MULTIPLE = [
        *PRE,
        [*ATTR_A, *ATTR_B],
        *POST
    ]
    
    # Get all the configs of all the NNs
    all_cfgs_list, refined_config = get_config_options_list(
        getattr(args, train_base_config_name), getattr(args, train_experiment_config_name))
    
    # Get all the configs for which we have results, and the corresponding results
    existing_cfgs = []
    results_list = []
    caches = [None for _ in range(len(all_cfgs_list))]
    for i, cfg in enumerate(all_cfgs_list):
        (cmd_dataset, cmd_opts_dataset,
         cmd_nn_train, cmd_opts_nn) = get_cmd_options_train_acqf(cfg)
        
        (args_nn, pre_model, model_and_info_name, models_path
        ) = cmd_opts_train_to_args_module_paths(cmd_opts_nn)

        # Get the model (with the weights)
        if BASIC_SAVING.model_is_trained(model_and_info_name):
            model, model_path = load_module(
                model_and_info_name,
                return_model_path=True, load_weights=True, verbose=False)
        else:
            # If the model is not found, skip this iteration
            continue
        
        # Get the dataset
        def create_dataset_func():
            cached = caches[i]
            if cached is not None:
                return cached
            (train_aq_dataset, test_aq_dataset, small_test_aq_dataset
            ) = create_train_test_acquisition_datasets_from_args(args_nn)
            caches[i] = test_aq_dataset
            return test_aq_dataset
        
        training_history_data = load_json(
            os.path.join(model_path, 'training_history_data.json'))

        existing_cfgs.append(cfg)
        results_list.append({
            'training_history_data': training_history_data,
            'model': model,
            'dataset_getter': create_dataset_func
        })
    
    # Remove all the prefixes
    existing_cfgs = [
        {k.split(".")[-1]: v for k, v in cfg.items()} for cfg in existing_cfgs
    ]
    
    n_options = len(all_cfgs_list)
    n_options_trained = len(results_list)
    n_options_not_trained = n_options - n_options_trained
    print(f"Number of not-trained-yet (unavailable) options: {n_options_not_trained}")
    print(f"Number of trained (available) options: {n_options_trained}")

    if n_options_trained == 0:
        raise ValueError("No trained options available for plotting.")
    
    gr = group_by(existing_cfgs, dict_to_str)
    assert all(len(v) == 1 for v in gr.values())

    # Folder name
    save_dir = create_plot_directory(
        args.plots_name, args.plots_group_name, is_run_plot=False)

    if CPROFILE:
        pr = cProfile.Profile()
        pr.enable()
    
    script_plot_kwargs = dict(
        sharey=True,
        aspect=1.618,
        scale=1.0
    )

    for attr_names in ATTR_GROUPS:
        print(f"\n-----------------------------------------------------\n{attr_names=}")

        if len(attr_names) == 1:
            attrs_groups_list = PLOTS_CONFIG_SINGLE
        else:
            attrs_groups_list = PLOTS_CONFIG_MULTIPLE
                
        attrs_groups_list = [set(group) for group in attrs_groups_list]
        if len(attrs_groups_list) + 1 >= 2:
            # Right before the one before "line" level
            attrs_groups_list.insert(-1, {"attr_name"})
        else:
            # Right before "line" level
            attrs_groups_list.append({"attr_name"})
                
        this_configs = [{**cfg, 'attr_name': name, 'index': i}
             for name in attr_names
             for i, cfg in enumerate(existing_cfgs)]

        plot_config, new_attrs_groups_list = group_by_nested_attrs(
            this_configs,
            attrs_groups_list,
            dict_to_str_func=plot_dict_to_str,
            add_extra_index=-1 # the level above "line" level
        )
        
        # Make script config
        script_config = {**vars(args)}
        script_config["nn_train_config"] = refined_config
        script_config["plots_config"] = [
            sorted(list(group)) for group in new_attrs_groups_list]
        
        # new_attrs_groups_list.append(None) # for the "line" level (just a dummy value)

        n_groups = len(new_attrs_groups_list)
        
        levels_to_add = ["line"]
        if args.use_cols:
            levels_to_add.append("col")
        if args.use_rows:
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

        save_dir_this_attrs = os.path.join(save_dir, "-".join(attr_names))
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
            print(f"  {level_name}: {attrs}")

        save_json(plot_config, "config/plot_config.json", indent=2)

        def get_result(index):
            cfg = this_configs[index]
            idx = cfg['index']
            results = results_list[idx]
            attr_name = cfg['attr_name']
            return {
                'attr_name': attr_name,
                'results': results,
                'config': cfg
            }

        plot_train_ax = get_plot_train_ax_func(get_result)
        save_figures_from_nested_structure(
            plot_config,
            plot_train_ax,
            new_attrs_groups_list,
            level_names,
            attr_name_to_title=ATTR_NAME_TO_TITLE,
            base_folder=save_dir_this_attrs,
            **script_plot_kwargs
        )

    if CPROFILE:
        pr.disable()
        with open('stats_output_plots.txt', 'w') as s:
            ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()


if __name__ == "__main__":
    main()
