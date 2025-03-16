import argparse
import cProfile
import os
from datetime import datetime
import pstats
from typing import Optional

from run_bo import GP_AF_DICT
from utils.constants import PLOTS_DIR
from utils.experiments.experiment_config_utils import get_config_options_list
from utils.plot_utils import add_plot_args, create_plot_directory, plot_dict_to_str, save_figures_from_nested_structure
from utils.utils import dict_to_str, group_by, group_by_nested_attrs, load_json, save_json

from nn_af.acquisition_function_net_save_utils import load_nn_acqf
from train_acqf import add_train_acqf_args, cmd_opts_nn_to_model_and_info_name, get_cmd_options_train_acqf


CPROFILE = False


PRE = [
    ["nn.layer_width", "nn.train_samples_size"]
]

ATTR_A = ["nn.batch_size"]
ATTR_B = ["nn.learning_rate"]

POST = [
    None
]

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

ATTR_GROUPS = [
    ["training_history", "af_plot"]
]


def get_plot_ax_train_acqf_func(get_result_func):
    def train_acqf_plot_ax(
            plot_config: dict,
            ax,
            plot_name: Optional[str]=None,
            **plot_kwargs):
        pass # TODO: Implement this function
    
    return train_acqf_plot_ax


def main():
    ## Create parser
    parser = argparse.ArgumentParser()
    nn_base_config_name, nn_experiment_config_name = add_train_acqf_args(parser,
                                                                         train=False)
    add_plot_args(parser)

    ## Parse arguments
    args = parser.parse_args()

    options_list, refined_config = get_config_options_list(
        getattr(args, nn_base_config_name), getattr(args, nn_experiment_config_name))
    
    existing_cfgs = []
    results_list = []
    
    for options in options_list:
        (cmd_dataset, cmd_opts_dataset,
         cmd_nn_train, cmd_opts_nn) = get_cmd_options_train_acqf(options)
        
        model_and_info_name = cmd_opts_nn_to_model_and_info_name(cmd_opts_nn)

        try:
            model, model_path = load_nn_acqf(
                model_and_info_name, return_model_path=True, load_weights=True)
        except FileNotFoundError:
            # If the model is not found, skip this iteration
            continue
        
        training_history_data = load_json(
            os.path.join(model_path, 'training_history_data.json'))

        existing_cfgs.append(options)
        results_list.append({
            'training_history_data': training_history_data,
            'model': model
        })
    
    n_options = len(options_list)
    n_options_trained = len(results_list)
    n_options_not_trained = n_options - n_options_trained
    print(f"Number of not-trained-yet (unavailable) options: {n_options_not_trained}")
    print(f"Number of trained (available) options: {n_options_trained}")

    if n_options_trained == 0:
        raise ValueError("No trained options available for plotting.")
    
    gr = group_by(existing_cfgs, dict_to_str)
    assert all(len(v) == 1 for v in gr.values())

    # Folder name
    save_dir = create_plot_directory(args.plots_name, args.plots_group_name)

    if CPROFILE:
        pr = cProfile.Profile()
        pr.enable()
    
    script_plot_kwargs = dict(
        sharey=True,
        aspect=1.618,
        scale=1.0
    )

    all_keys = set().union(*[set(result.keys()) for result in results_list])
    for attr_names in ATTR_GROUPS:
        attr_names = [a for a in attr_names if a in all_keys]
        print(f"\n-----------------------------------------------------\n{attr_names=}")

        if len(attr_names) == 1:
            attrs_groups_list = PLOTS_CONFIG_SINGLE
        else:
            attrs_groups_list = PLOTS_CONFIG_MULTIPLE
        
        attrs_groups_list = [set(group) for group in attrs_groups_list]
        if len(attrs_groups_list) >= 2:
            # Right before the one before "line" (-1) level
            attrs_groups_list.insert(-2, {"attr_name"})
        else:
            # Right before "line" (-1) level
            attrs_groups_list.insert(-1, {"attr_name"})
        
        this_configs = [{**cfg, 'attr_name': name, 'index': i}
             for name in attr_names
             for i, cfg in enumerate(existing_cfgs)]

        plot_config, new_attrs_groups_list = group_by_nested_attrs(
            this_configs,
            attrs_groups_list,
            dict_to_str_func=plot_dict_to_str,
            add_extra_index=-2 # -2 is the level above "line" level
        )

        n_groups = len(new_attrs_groups_list)
        if n_groups < 2:
            raise ValueError(
                "There are not enough levels of plot grouping (at least 2 are required). "
                f"{new_attrs_groups_list=}")
        
        # Make script config
        script_config = {**vars(args)}
        script_config["nn_train_config"] = refined_config
        script_config["plots_config"] = [
            sorted(list(group)) for group in new_attrs_groups_list]
        
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
            if attrs:
                print(f"  {level_name}: {attrs}")

        save_json(plot_config, "config/plot_config.json", indent=2)

        def get_result(index):
            cfg = this_configs[index]
            idx = cfg['index']
            results = results_list[idx]
            attr_name = cfg['attr_name']
                        
            if attr_name in results:
                ret = {
                    attr_name: results[attr_name],
                    'attr_name': attr_name,
                    'index': idx
                }
                return ret
            return None

        train_acqf_plot_ax = get_plot_ax_train_acqf_func(get_result)
        
        save_figures_from_nested_structure(
            plot_config,
            train_acqf_plot_ax,
            new_attrs_groups_list,
            level_names,
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
