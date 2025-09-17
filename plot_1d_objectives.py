from bayesopt.bayesopt import outcome_transform_function
from gp_acquisition_dataset_manager import GPAcquisitionDatasetManager, get_gp_model_from_args_no_outcome_transform


command = "python gp_acquisition_dataset.py --dimension 1 --kernel Matern52 --lengthscale 0.05 --max_history 100 --min_history 100 --replacement --test_expansion_factor 1 --test_n_candidates 1 --test_samples_size 100 --train_acquisition_size 30000 --train_n_candidates 1 --train_samples_size 10000"

n_funcs = 3

base_af_params = dict(
    samples_size=n_funcs,
    acquisition_size=n_funcs,
    outcome_transform=None,
    standardize_outcomes=False,
    loguniform=False,
    pre_offset=None,
    min_history=1,
    max_history=80,
    samples_addition_amount=20,
    n_candidates=1,
    fix_n_samples=True,
    y_cand_indices="all",
    fix_function_samples=True,
    fix_acquisition_samples=False,
    lazy=False,
    cache=True,
    batch_size=16,
    name="testing_data",
    replacement=True,
    check_cached=False,
    load_dataset=True
)


gp_manager = GPAcquisitionDatasetManager()

gp_params_args = dict(
    dimension=1,
    kernel="Matern52",
    lengthscale=0.05,
    randomize_params=False,
)
models = [
    get_gp_model_from_args_no_outcome_transform(
        dimension=gp_params_args['dimension'],
        kernel=gp_params_args['kernel'],
        lengthscale=gp_params_args['lengthscale'],
        add_priors=gp_params_args['randomize_params'],
        device="cpu"
    )
]

gp_params = dict(
    dimension=gp_params_args['dimension'],
    randomize_params=gp_params_args['randomize_params'],
    xvalue_distribution="uniform",
    observation_noise=False,
    models=models,
    model_probabilities=None
)

dataset = gp_manager.create_acquisition_dataset(**base_af_params, **gp_params)
print(f"{dataset=}")
