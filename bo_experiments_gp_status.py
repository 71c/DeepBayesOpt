from utils.experiments.submit_dependent_jobs import add_slurm_args
from bo_experiments_gp import generate_gp_bo_job_specs, get_bo_experiments_parser


def main():
    (parser, nn_base_config_name, nn_experiment_config_name, bo_base_config_name,
     bo_experiment_config_name) = get_bo_experiments_parser(train=False)

    args = parser.parse_args()

    jobs_spec, new_cfgs, existing_cfgs_and_results, refined_config \
        = generate_gp_bo_job_specs(
            args,
            nn_base_config=getattr(args, nn_base_config_name),
            nn_experiment_config=getattr(args, nn_experiment_config_name),
            bo_base_config=getattr(args, bo_base_config_name),
            bo_experiment_config=getattr(args, bo_experiment_config_name)
        )
    
    print(f"Number of new BO configs: {len(new_cfgs)}")
    print(f"Number of existing BO configs: {len(existing_cfgs_and_results)}")


if __name__ == "__main__":
    main()
