import matplotlib.pyplot as plt
import numpy as np
from bayesopt.bayesopt import outcome_transform_function
from gp_acquisition_dataset_manager import GPAcquisitionDatasetManager, get_gp_model_from_args_no_outcome_transform
from lr_acquisition_dataset_manager import LogisticRegressionAcquisitionDatasetManager


command = "python gp_acquisition_dataset.py --dimension 1 --kernel Matern52 --lengthscale 0.05 --max_history 100 --min_history 100 --replacement --test_expansion_factor 1 --test_n_candidates 1 --test_samples_size 100 --train_acquisition_size 30000 --train_n_candidates 1 --train_samples_size 10000"

n_funcs = 6

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
        add_standardize=False,
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

function_samples_dataset_gp = GPAcquisitionDatasetManager().create_acquisition_dataset(
    **base_af_params, **gp_params).base_dataset

function_samples_dataset_lr = LogisticRegressionAcquisitionDatasetManager().create_acquisition_dataset(
    **base_af_params, log_lambda_range=(-8, 0)).base_dataset

datasets = [function_samples_dataset_gp, function_samples_dataset_lr]
dataset_names = ["GP", "Logistic Regression"]

fig, axes = plt.subplots(len(datasets), n_funcs, figsize=(5 * n_funcs, 4))
for k in range(len(datasets)):
    for i in range(n_funcs):
        ax = axes[k, i]
        item = datasets[k][i]
        x_values = item.x_values.cpu().numpy().flatten()
        y_values = item.y_values.cpu().numpy().flatten()
        # Sort for plotting
        sorted_indices = np.argsort(x_values)
        x_values = x_values[sorted_indices]
        y_values = y_values[sorted_indices]
        ax.plot(x_values, y_values, marker='o', linestyle='-')
        ax.set_title(f"{dataset_names[k]} Function Sample {i+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

plt.tight_layout()
plt.savefig("function_samples.png", dpi=300)
