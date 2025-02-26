from bo_experiments_gp import get_bo_experiments_parser, get_gp_bo_jobs_spec_and_configs_from_args
from train_acqf import MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN
from utils import dict_to_str, group_by, group_by_nested_attrs, save_json


TEST = True

def main():
    parser, bo_loop_group = get_bo_experiments_parser(train=False)

    plot_group = parser.add_argument_group("Plotting")
    plot_group.add_argument(
        '--plotting_config', 
        type=str,
        required=True,
        help='YAML file containing the configuration for organizing the plots'
    )

    args = parser.parse_args()

    jobs_spec, new_bo_configs, existing_bo_configs = get_gp_bo_jobs_spec_and_configs_from_args(
        args, bo_loop_group)
    
    print(f"Number of new configs: {len(new_bo_configs)}")
    print(f"Number of existing configs: {len(existing_bo_configs)}")
    
    # print(MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN)
    
    if TEST:
        # TEMPORARY TEST (NOT WHAT I WANT TO DO) to simulate having data:
        print("Simulating having data by swapping new and existing configs")
        existing_bo_configs = [(cfg, None) for cfg in new_bo_configs]
    
    if len(existing_bo_configs) == 0:
        raise ValueError("There are no saved BO configs to plot.")
    
    gr = group_by([cfg for cfg, result in existing_bo_configs], dict_to_str)
    assert all(len(v) == 1 for v in gr.values())

    reformatted_configs = []
    for item, result in existing_bo_configs:
        nn_model_name = item['bo_policy_args'].get('nn_model_name')
        if nn_model_name is not None:
            item['bo_policy_args'].update(
                {"nn." + k: v
                 for k, v in MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN[nn_model_name].items()
                 }
            )
            item['bo_policy_args'].pop('nn_model_name')
        reformatted_configs.append({
            **{f'objective.{k}': v for k, v in item['objective_args'].items()},
            **item['bo_policy_args'],
            **{k if k == 'gp_af' else f'gp_af.{k}': v for k, v in item['gp_af_args'].items()}
        })
    
    plot_config, attrs_groups_list = group_by_nested_attrs(reformatted_configs,
                          [{"nn.layer_width", "nn.train_samples_size"},
                           {"lamda", "gp_af", "nn.method"},
                           {"objective.gp_seed"}],
                           add_extra_index=-2)
    print(f"Plot {attrs_groups_list=}")
    save_json(plot_config, "config/plot_config.json", indent=2)


    
    # print(gr.values())
    
    # for u, v in gr.items():
    #     if len(v) > 1:
    #         print(u)
    #         print(v[0])
    #         print()
    
    

if __name__ == "__main__":
    main()
