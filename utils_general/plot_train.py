import argparse
import cProfile
import os
import pstats
from types import SimpleNamespace
from typing import Callable, Optional

from utils_general.experiments.experiment_config_utils import add_submit_train_args, get_config_options_list
from utils_general.experiments.registry import ExperimentRegistryBase
from utils_general.experiments.submit_train_utils import SubmitTrainUtils
from utils_general.io_utils import load_json, save_json
from utils_general.plot_utils import add_plot_args
from utils_general.utils import group_by, dict_to_str


def plot_train(
        registry: ExperimentRegistryBase,
        train_submit_utils: SubmitTrainUtils,
        plot_utils: SimpleNamespace,
        attr_name_to_plot_func: dict[str, Callable],
        attr_name_to_title: dict[str, str],
        attr_groups: list[list[str]],
        use_cprofile: bool = False
    ) -> None:
    def _get_plot_train_ax_func(get_result_func):
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
            if attr_name in attr_name_to_plot_func:
                attr_name_to_plot_func[attr_name](
                    ax, results['training_history_data'], results['dataset_getter'],
                    results['model'], info['config'],
                    plot_name=plot_name, label=label, color=color, alpha=alpha)
            else:
                raise ValueError(f"Unknown attribute name: {attr_name}")
        
        return plot_train_ax

    ## Create parser
    parser = argparse.ArgumentParser()
    train_base_config_name, train_experiment_config_name = add_submit_train_args(parser,
                                                                        train=False)
    add_plot_args(parser)

    ## Parse arguments
    args = parser.parse_args()

    # Configure plotting parameters based on experiment
    registry.setup_plotting_from_args(args, 'train_plot', globals())
    print("Successfully applied train plotting configuration for this experiment")
    
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
    
    torch_model_save_instance = train_submit_utils.torch_model_save_instance
    basic_save_utils = torch_model_save_instance.basic_save_utils
    
    # Get all the configs for which we have results, and the corresponding results
    existing_cfgs = []
    results_list = []
    caches = [None for _ in range(len(all_cfgs_list))]
    for i, cfg in enumerate(all_cfgs_list):
        (cmd_dataset, cmd_opts_dataset,
        cmd_nn_train, cmd_opts_nn) = train_submit_utils.get_dataset_and_train_cmd_options(cfg)
        
        (args_nn, pre_model, model_and_info_name, models_path
        ) = torch_model_save_instance.cmd_opts_train_to_args_module_paths(cmd_opts_nn)

        # Get the model (with the weights)
        if basic_save_utils.model_is_trained(model_and_info_name):
            model, model_path = torch_model_save_instance.load_module(
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
            result = train_submit_utils.create_datasets_func(args_nn)
            caches[i] = result
            return result
        
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
    save_dir = plot_utils.create_plot_directory(
        args.plots_name, args.plots_group_name, is_run_plot=False)

    if use_cprofile:
        pr = cProfile.Profile()
        pr.enable()
    
    script_plot_kwargs = dict(
        sharey=True,
        aspect=1.618,
        scale=1.0
    )

    for attr_names in attr_groups:
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

        plot_config, new_attrs_groups_list = plot_utils.group_by_nested_attrs(
            this_configs,
            attrs_groups_list,
            dict_to_str_func=plot_utils.plot_dict_to_str,
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

        plot_train_ax = _get_plot_train_ax_func(get_result)
        plot_utils.save_figures_from_nested_structure(
            plot_config,
            plot_train_ax,
            new_attrs_groups_list,
            level_names,
            attr_name_to_title=attr_name_to_title,
            base_folder=save_dir_this_attrs,
            **script_plot_kwargs
        )

    if use_cprofile:
        pr.disable()
        with open('stats_output_plots.txt', 'w') as s:
            ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
