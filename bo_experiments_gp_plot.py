from bo_experiments_gp import get_bo_experiments_parser, get_gp_bo_jobs_spec_and_configs_from_args
from train_acqf import MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN
from utils import dict_to_str, group_by, group_by_nested_attrs, save_json


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
    
    # print(MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN)

    # TEMPORARY TEST:
    existing_bo_configs = new_bo_configs
    
    n_existing = len(existing_bo_configs)
    print(f"Number of new configs: {len(new_bo_configs)}")
    print(f"Number of existing configs: {n_existing}")
    if n_existing == 0:
        raise ValueError("There are no saved BO configs to plot.")

    reformatted_configs = []
    for item in existing_bo_configs:
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
            **{f'gp_af.{k}': v for k, v in item['gp_af_args'].items()}
        })
    
    # save_json(reformatted_configs, "config/reformatted_configs.json", indent=2)

    y = group_by_nested_attrs(reformatted_configs,
                          [{"nn.layer_width", "nn.train_samples_size"},
                           {"lamda"},
                           {"objective.gp_seed"}],
                           add_extra_index=-2)
    save_json(y, "config/reformatted_configs.json", indent=2)

    # print(reformatted_configs[3708])
    # print(reformatted_configs[3724])

    # gr = group_by(existing_bo_configs, lambda x: dict_to_str(x))
    
    # gr = {u: len(v) for u, v in gr.items()}
    # print(gr.values())
    
    # for u, v in gr.items():
    #     if len(v) > 1:
    #         print(u)
    #         print(v[0])
    #         print()
    
    

if __name__ == "__main__":
    main()
