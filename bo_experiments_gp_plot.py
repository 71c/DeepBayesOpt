import itertools
import os
import yaml
from datetime import datetime
from bo_experiments_gp import get_bo_experiments_parser, generate_gp_bo_job_specs
from plot_utils import save_figures_from_nested_structure
from run_bo import GP_AF_DICT
from submit_dependent_jobs import CONFIG_DIR
from tictoc import tic, tocl
from train_acqf import MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN
from utils import dict_to_str, group_by, group_by_nested_attrs, save_json


script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')


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
            if method == "random search":
                ret = method
            else:
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

PRE = [
    ["nn.layer_width", "nn.train_samples_size", "gen_candidates"]
]

ATTR_A = ["nn.batch_size"]
ATTR_B = ["nn.learning_rate"]

POST = [
    ["lamda", "gp_af", "nn.method"],
    ["objective.gp_seed"]
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
    ["process_time", "n_evals",
     "mean_eval_process_time", "optimize_process_time"],
    ["best_y", "mean_eval_process_time", "process_time", "n_evals"],
    ["process_time"],
    ["mean_eval_process_time"],
    ["best_y"]
]

def main():
    (parser, nn_base_config_name, nn_experiment_config_name, bo_base_config_name,
     bo_experiment_config_name) = get_bo_experiments_parser(train=False)

    plot_group = parser.add_argument_group("Plotting")
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

    args = parser.parse_args()

    jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config \
        = generate_gp_bo_job_specs(
            args,
            nn_base_config=getattr(args, nn_base_config_name),
            nn_experiment_config=getattr(args, nn_experiment_config_name),
            bo_base_config=getattr(args, bo_base_config_name),
            bo_experiment_config=getattr(args, bo_experiment_config_name)
        )
    print(new_cfgs)
    save_json(jobs_spec, os.path.join(CONFIG_DIR, "dependencies.json"), indent=4)
    
    print(f"Number of new configs: {len(new_cfgs)}")
    print(f"Number of existing configs: {len(existing_cfgs_and_results)}")
    
    if len(existing_cfgs_and_results) == 0:
        raise ValueError("There are no saved BO configs to plot.")
    
    existing_cfgs, existing_results = zip(*existing_cfgs_and_results)
    
    gr = group_by(existing_cfgs, dict_to_str)
    assert all(len(v) == 1 for v in gr.values())

    results_list = [
        {k: v[0, :] for k, v in next(iter(r))[1].items()}
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
            # bo_policy_args.pop('nn_model_name')
        
        if 'random_search' in bo_policy_args:
            random_search = bo_policy_args.pop('random_search')
            if random_search:
                item['gp_af_args']['gp_af'] = 'random search'

        reformatted_configs.append({
            **{k if k == 'dimension' else f'objective.{k}': v
               for k, v in item['objective_args'].items()},
            **bo_policy_args,
            **{k if k == 'gp_af' else f'gp_af.{k}': v
               for k, v in item['gp_af_args'].items()}
        })

    # Folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PLOTS_DIR, timestamp)

    all_keys = set().union(*[set(result.keys()) for result in results_list])
    for attr_names in ATTR_GROUPS:
        attr_names = [a for a in attr_names if a in all_keys]
        print(f"\n-----------------------------------------------------\n{attr_names=}")

        if len(attr_names) == 1:
            attrs_groups_list = PLOTS_CONFIG_SINGLE
        else:
            attrs_groups_list = PLOTS_CONFIG_MULTIPLE
        
        attrs_groups_list = [set(group) for group in attrs_groups_list]
        if len(attrs_groups_list) >= 3:
            attrs_groups_list.insert(-3, {"attr_name"}) # Right before the one before "line"
        else:
            attrs_groups_list.insert(-2, {"attr_name"}) # right before "line"
        
        this_reformatted_configs = [{**cfg, 'attr_name': name, 'index': i}
             for name in attr_names
             for i, cfg in enumerate(reformatted_configs)]

        plot_config, new_attrs_groups_list = group_by_nested_attrs(
            this_reformatted_configs,
            attrs_groups_list,
            dict_to_str_func=plot_dict_to_str,
            add_extra_index=-2
        )

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
            cfg = this_reformatted_configs[index]
            idx = cfg['index']
            results = results_list[idx]
            attr_name = cfg['attr_name']
            
            cfg = existing_cfgs[idx]
            bo_policy_args = cfg['bo_policy_args']
            
            
            if attr_name in results:
                ret = {
                    'attr_name': attr_name,
                    attr_name: results[attr_name]
                }
                if 'nn_model_name' in bo_policy_args:
                    ret['nn_model_name'] = bo_policy_args['nn_model_name']
                return ret
            return None

        plot_kwargs=dict(
            alpha=0.05,
            sharey=True,
            aspect=1.618,
            scale=1.0)
        save_figures_from_nested_structure(
            plot_config,
            get_result,
            new_attrs_groups_list,
            level_names,
            base_folder=save_dir_this_attrs,
            **plot_kwargs
        )

if __name__ == "__main__":
    main()
