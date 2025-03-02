import itertools
import os
import yaml
from datetime import datetime
from bo_experiments_gp import get_bo_experiments_parser, gp_bo_jobs_spec_cfgs_from_args
from plot_utils import save_figures_from_nested_structure
from run_bo import GP_AF_DICT
from submit_dependent_jobs import CONFIG_DIR
from tictoc import tic, tocl
from train_acqf import MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN
from utils import dict_to_str, group_by, group_by_nested_attrs, save_json


script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')

TEST = False


def plot_key_value_to_str(k, v):
    if k == "attr_name":
        return (2, v)
    if k != "nn.lamda" and k.startswith("nn."):
        k = k[3:]
    return (1, f"{k}={v}")


def plot_dict_to_str(d):
    for key_name, prefix, plot_name in [
        ("nn.method", "nn.", "NN, method="),
        ("gp_af", "gp_af.", "GP, ")
    ]:
        if key_name in d:
            d_method = {}
            d_non_method = {}
            for k, v in d.items():
                if k == key_name:
                    continue
                if k.startswith(prefix):
                    d_method[k[len(prefix):]] = v
                else:
                    d_non_method[k] = v
            method = d[key_name]
            ret = f"{plot_name}{method}"
            if d_method:
                s = dict_to_str(d_method, include_space=True)
                ret += f" ({s})"
            if d_non_method:
                s = dict_to_str(d_non_method, include_space=True)
                ret += f", {s}"
            return ret
    
    items = [
        plot_key_value_to_str(k, v)
        for k, v in d.items()
    ]
    items = sorted(items)
    return ", ".join([item[1] for item in items])


def main():
    parser, bo_loop_group = get_bo_experiments_parser(train=False)

    plot_group = parser.add_argument_group("Plotting")
    plot_group.add_argument(
        '--plots_config', 
        type=str,
        required=True,
        help='YAML file containing the configuration for organizing the plots'
    )
    plot_group.add_argument(
        '--use_cols', 
        action='store_true',
        help='Whether to use columns for subplots in the plots'
    )
    plot_group.add_argument(
        '--use_rows', 
        action='store_true',
        help='Whether to use rows for subplots in the plots'
    )
    plot_group.add_argument(
        '--plot_runtime',
        action='store_true',
        help='Plots the runtimes and stuff (as opposed to regret or highest function value)'
    )

    args = parser.parse_args()

    jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config \
          = gp_bo_jobs_spec_cfgs_from_args(args, bo_loop_group)
    
    save_json(jobs_spec, os.path.join(CONFIG_DIR, "dependencies.json"), indent=4)
    
    print(f"Number of new configs: {len(new_cfgs)}")
    print(f"Number of existing configs: {len(existing_cfgs_and_results)}")
    
    if TEST:
        # TEMPORARY TEST (NOT WHAT I WANT TO DO) to simulate having data:
        print("Simulating having data by pretending to have already computed everything")
        existing_cfgs_and_results += [(cfg, None) for cfg in new_cfgs]
    
    if len(existing_cfgs_and_results) == 0:
        raise ValueError("There are no saved BO configs to plot.")
    
    existing_cfgs, existing_results = zip(*existing_cfgs_and_results)
    
    gr = group_by(existing_cfgs, dict_to_str)
    assert all(len(v) == 1 for v in gr.values())

    reformatted_configs = []
    for item in existing_cfgs:
        nn_model_name = item['bo_policy_args'].get('nn_model_name')
        if nn_model_name is not None:
            item['bo_policy_args'].update(
                {"nn." + k: v
                 for k, v in MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN[nn_model_name].items()
                 }
            )
            item['bo_policy_args'].pop('nn_model_name')
        reformatted_configs.append({
            **{k if k == 'dimension' else f'objective.{k}': v
               for k, v in item['objective_args'].items()},
            **item['bo_policy_args'],
            **{k if k == 'gp_af' else f'gp_af.{k}': v
               for k, v in item['gp_af_args'].items()}
        })
    
    if args.plot_runtime:
        attr_names = ["process_time", "n_evals", "mean_eval_process_time", "optimize_process_time"]
    else:
        attr_names = ["best_y"]
    
    results_list = [
        {k: v[0, :] for k, v in next(iter(r))[1].items()}
        for r in existing_results
    ]
    
    all_keys = set().union(*[set(result.keys()) for result in results_list])
    attr_names = [a for a in attr_names if a in all_keys]
    print(f"{attr_names=}")
    
    reformatted_configs = [
        {**cfg, 'attr_name': name, 'index': i}
        for name in attr_names
        for i, cfg in enumerate(reformatted_configs)
    ]

    with open(args.plots_config, 'r') as f:
        attrs_groups_list = yaml.safe_load(f)
    attrs_groups_list = [set(group) for group in attrs_groups_list]

    if len(attrs_groups_list) >= 3:
        attrs_groups_list.insert(-3, {"attr_name"}) # Right before the one before "line"
    else:
        attrs_groups_list.insert(-2, {"attr_name"}) # right before "line"
    
    plot_config, new_attrs_groups_list = group_by_nested_attrs(
        reformatted_configs,
        attrs_groups_list,
        dict_to_str_func=plot_dict_to_str,
        add_extra_index=-2)

    n_groups = len(new_attrs_groups_list)
    if n_groups < 2:
        raise ValueError(
            "There are not enough levels of plot grouping (at least 2 are required). "
            f"{new_attrs_groups_list=}")
    
    # Make script config
    script_config = vars(args)
    script_config.pop("base_config")
    script_config.pop("experiment_config")
    script_config["nn_train_config"] = refined_config
    script_config["plots_config"] = [
        sorted(list(group)) for group in new_attrs_groups_list]
    
    # SPECIAL
    script_config["gp_af_names"] = list(GP_AF_DICT)

    levels_to_add = ["random", "line"]
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

    # Folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PLOTS_DIR, timestamp)
    
    # Add folder level
    levels_reversed.append("folder")
    new_attrs_groups_list = [None] + new_attrs_groups_list
    plot_config = {
        save_dir: {
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
        cfg = reformatted_configs[index]
        idx = cfg['index']
        results = results_list[idx]
        attr_name = cfg['attr_name']
        if attr_name in results:
            return {'attr_name': attr_name, attr_name: results[attr_name]}
        return None

    save_figures_from_nested_structure(
        plot_config,
        get_result,
        new_attrs_groups_list,
        level_names,
        base_folder=save_dir
    )

# e.g.,
# python bo_experiments_gp_plot.py --base_config config/train_acqf.yml --experiment_config config/train_acqf_experiment_test1.yml --n_gp_draws 16 --seed 8 --n_iter 100 --n_initial_samples 1 --plots_config config/plots_config_1.yml

if __name__ == "__main__":
    main()
