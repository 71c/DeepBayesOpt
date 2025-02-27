import os
import yaml
from datetime import datetime
from bo_experiments_gp import get_bo_experiments_parser, gp_bo_jobs_spec_cfgs_from_args
from plot_utils import save_figures_from_nested_structure
from run_bo import GP_AF_DICT
from train_acqf import MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN
from utils import dict_to_str, group_by, group_by_nested_attrs, save_json


script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')

TEST = False


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
        '--use_subplots', 
        action='store_true',
        help='Whether to use subplots in the plots'
    )

    args = parser.parse_args()

    jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config \
          = gp_bo_jobs_spec_cfgs_from_args(args, bo_loop_group)
    
    print(f"Number of new configs: {len(new_cfgs)}")
    print(f"Number of existing configs: {len(existing_cfgs_and_results)}")
        
    if TEST:
        # TEMPORARY TEST (NOT WHAT I WANT TO DO) to simulate having data:
        print("Simulating having data by swapping new and existing configs")
        existing_cfgs_and_results = [(cfg, None) for cfg in new_cfgs]
    
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

    with open(args.plots_config, 'r') as f:
        attrs_groups_list = yaml.safe_load(f)
    attrs_groups_list = [set(group) for group in attrs_groups_list]
    # print(f"{attrs_groups_list=}")
    
    plot_config, new_attrs_groups_list = group_by_nested_attrs(reformatted_configs,
                          attrs_groups_list,
                           add_extra_index=-2)
    # print(f"{new_attrs_groups_list=}")

    n_groups = len(new_attrs_groups_list)
    if n_groups < 2:
        raise ValueError(
            "There are not enough levels of plot grouping (at least 2 are required).")
    
    # Make script config
    script_config = vars(args)
    script_config["plots_config"] = [
        sorted(list(group)) for group in new_attrs_groups_list]
    script_config.pop("base_config")
    script_config.pop("experiment_config")
    script_config["nn_train_config"] = refined_config
    script_config["gp_af_names"] = list(GP_AF_DICT)
    
    level_names = ["line", "random"]
    if n_groups >= 3:
        if args.use_subplots:
            if n_groups >= 4:
                level_names = ['folder'] * (n_groups - 4) + ['fname', 'subplot'] + level_names
            else:
                level_names = ['subplot'] + level_names
        else:
            level_names = ['folder'] * (n_groups - 3) + ['fname'] + level_names
    
    if n_groups == 2 or (n_groups == 3 and args.use_subplots):
        # need to add a file name level
        plot_config = {
            "results": {
                "items": plot_config
            }
        }
        level_names = ["fname"] + level_names
        new_attrs_groups_list = [None] + new_attrs_groups_list

    # Folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PLOTS_DIR, timestamp)
    # Add folder level
    level_names = ["folder"] + level_names
    new_attrs_groups_list = [None] + new_attrs_groups_list
    plot_config = {
        save_dir: {
            "items": plot_config,
            "vals": script_config
        }
    }

    print("Plotting configuration:")
    for level_name, attrs in zip(level_names, new_attrs_groups_list):
        if attrs:
            print(f"  {level_name}: {attrs}")

    save_json(plot_config, "config/plot_config.json", indent=2)

    results_list = [
        {k: v[0, :] for k, v in next(iter(r))[1].items()}
        for r in existing_results
    ]

    save_figures_from_nested_structure(
        plot_config,
        results_list,
        new_attrs_groups_list,
        level_names,
        base_folder=save_dir
    )

# e.g.,
# python bo_experiments_gp_plot.py --base_config config/train_acqf.yml --experiment_config config/train_acqf_experiment_test1.yml --n_gp_draws 16 --seed 8 --n_iter 100 --n_initial_samples 1 --plots_config config/plots_config_1.yml

if __name__ == "__main__":
    main()
