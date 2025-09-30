"""
Utilities for loading and using the HPO-B benchmark datasets and surrogates.
Reference:
Arango, S. P., Jomaa, H. S., Wistuba, M., and Grabocka, J.
HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML.
Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks
2021.
https://arxiv.org/abs/2106.06257
The code here is adapted from the HPO-B repository:
https://github.com/machinelearningnuremberg/HPO-B
"""
from functools import cache
import os
from typing import Literal

import xgboost as xgb

import numpy as np
import torch
from datasets.function_samples_dataset import FunctionSamplesItem, ListMapFunctionSamplesDataset
from utils.constants import HPOB_DATA_DIR, HPOB_SAVED_SURROGATES_DIR
from utils.utils import load_json


_HPOB_PATHS = {
    'train': os.path.join(HPOB_DATA_DIR, "meta-train-dataset.json"),
    'validation': os.path.join(HPOB_DATA_DIR, "meta-validation-dataset.json"),
    'test': os.path.join(HPOB_DATA_DIR, "meta-test-dataset.json"),
}

_HPOB_JSON_CACHE = {}
def _get_hpob_dataset_json(
        search_space_id: str,
        dataset_type: Literal['train', 'validation', 'test']) -> dict:
    if dataset_type not in _HPOB_JSON_CACHE:
        data_path = _HPOB_PATHS[dataset_type]
        _HPOB_JSON_CACHE[dataset_type] = load_json(data_path)
    data = _HPOB_JSON_CACHE[dataset_type]
    try:
        return data[search_space_id]
    except KeyError:
        raise ValueError(f"Search space ID {search_space_id} not found in HPO-B "
                            f"{dataset_type} datasets. Available search space IDs: "
                            f"{sorted(data.keys())}")


def get_hpob_dataset_dimension(search_space_id: str) -> int:
    """Get the dimension of the search space for a given HPO-B search space ID."""
    search_space_data = _get_hpob_dataset_json(search_space_id, 'validation')
    dataset_id = list(search_space_data.keys())[0]
    X = search_space_data[dataset_id]['X']
    return len(X[0])


def get_hpob_dataset_ids(
        search_space_id: str,
        dataset_type: Literal['train', 'validation', 'test']) -> list[str]:
    """Get the list of dataset IDs for a given search space ID and dataset type."""
    search_space_data = _get_hpob_dataset_json(search_space_id, dataset_type)
    return sorted(search_space_data.keys())


def get_hpob_dataset(search_space_id: str,
                     dataset_type: Literal['train', 'validation', 'test'],
                     device: str = "cpu") -> ListMapFunctionSamplesDataset:
    """Get a function samples dataset from the HPO-B benchmark."""
    search_space_data = _get_hpob_dataset_json(search_space_id, dataset_type)
    list_of_datasets = []
    for dataset_id in sorted(search_space_data):
        Xy = search_space_data[dataset_id]
        X, y = Xy['X'], Xy['y']
        item = FunctionSamplesItem(
            torch.tensor(X, device=device), torch.tensor(y, device=device))
        list_of_datasets.append(item)
    return ListMapFunctionSamplesDataset(list_of_datasets)


@cache
def _get_hpob_initialization_helper():
    init_path = os.path.join(HPOB_DATA_DIR, "bo-initializations.json")
    if os.path.isfile(init_path):
        return load_json(init_path)
    raise FileNotFoundError(f"HPO-B initializations file not found: {init_path}")


def get_hpob_initialization(search_space_id: str,
                            dataset_id: str,
                            seed: Literal['test0', 'test1', 'test2', 'test3', 'test4']):
    """Returns HPO-B initialization for given search space ID, dataset ID, and seed"""
    available_dataset_ids = get_hpob_dataset_ids(search_space_id, 'test')
    if dataset_id not in available_dataset_ids:
        raise ValueError(
            f"Dataset ID {dataset_id} not found in HPO-B test datasets for search "
            f"space ID {search_space_id}. Available datasets: {available_dataset_ids}")
    init_ids = _get_hpob_initialization_helper()[search_space_id][dataset_id][seed]
    X = _get_hpob_dataset_json(search_space_id, 'test')[dataset_id]["X"]
    return torch.tensor([X[i] for i in init_ids])


SURROGATES_STATS_PATH = os.path.join(HPOB_SAVED_SURROGATES_DIR, "summary-stats.json")

@cache
def _load_hpob_surrogates_stats_helper():
    if os.path.isfile(SURROGATES_STATS_PATH):
        return load_json(SURROGATES_STATS_PATH)
    raise FileNotFoundError(f"Surrogates stats file not found: {SURROGATES_STATS_PATH}")


def _load_hpob_surrogates_stats(surrogate_name: str):
    """Load the HPO-B surrogates statistics from the JSON file."""
    return _load_hpob_surrogates_stats_helper()[surrogate_name]


## Note: The y_min and y_max values stored in the summary-stats.json file
## have been observed to sometimes incorrect, so instead we compute them from
## the actual dataset y values.
_USE_SURROGATE_STATS = False


_HPOB_FUNCTION_MIN_MAX_CACHE = {}
def get_hpob_function_min_max(
        search_space_id: str, dataset_id: str,
        use_surrogate_stats: bool = _USE_SURROGATE_STATS) -> tuple[float, float]:
    """Get the minimum and maximum function values for a given HPO-B search space ID
    and dataset ID.
    Args:
        search_space_id: The HPO-B search space ID.
        dataset_id: The HPO-B dataset ID.
        use_surrogate_stats: If True, use the y_min and y_max values from the
            surrogate stats file. If False, compute y_min and y_max from the dataset.
    Returns:
        A tuple (y_min, y_max) representing the minimum and maximum function values.
    """
    key = (search_space_id, dataset_id, use_surrogate_stats)
    if key not in _HPOB_FUNCTION_MIN_MAX_CACHE:
        dataset_ids = get_hpob_dataset_ids(search_space_id, 'test')
        if dataset_id not in dataset_ids:
            raise ValueError(
                f"Dataset ID {dataset_id} not found in HPO-B test datasets for search "
                f"space ID {search_space_id}. Available datasets: {dataset_ids}")
        if use_surrogate_stats:
            surrogate_name = 'surrogate-'+search_space_id+'-'+dataset_id
            surrogate_stats = _load_hpob_surrogates_stats(surrogate_name)
            y_min, y_max = surrogate_stats["y_min"], surrogate_stats["y_max"]
        else:
            y_vals = _get_hpob_dataset_json(search_space_id, 'test')[dataset_id]['y']
            y_vals = np.array(y_vals)
            y_min, y_max = y_vals.min(), y_vals.max()
        _HPOB_FUNCTION_MIN_MAX_CACHE[key] = (y_min, y_max)
    return _HPOB_FUNCTION_MIN_MAX_CACHE[key]


@cache
def _get_hpob_objective_function(
    search_space_id: str, dataset_id: str, scale_y: bool):
    dataset_ids = get_hpob_dataset_ids(search_space_id, 'test')
    if dataset_id not in dataset_ids:
        raise ValueError(
            f"Dataset ID {dataset_id} not found in HPO-B test datasets for search "
            f"space ID {search_space_id}. Available datasets: {dataset_ids}")
    surrogate_name = 'surrogate-'+search_space_id+'-'+dataset_id
    bst_surrogate = xgb.Booster()
    bst_surrogate.load_model(f"{HPOB_SAVED_SURROGATES_DIR}/{surrogate_name}.json")

    y_min, y_max = get_hpob_function_min_max(search_space_id, dataset_id)

    dim = get_hpob_dataset_dimension(search_space_id)

    def objective_function(x: torch.Tensor) -> torch.Tensor:
        """Evaluate the objective function at the given points.

        Args:
            x: A tensor of shape (n_points, dim) representing the input points.

        Returns:
            A tensor of shape (n_points,) representing the objective function values.
        """
        if x.dim() == 2:
            if x.size(1) != dim:
                raise ValueError(
                    f"Incorrect input {x.shape}: dimension does not match {dim}")
        else:
            raise ValueError(
                f"Incorrect input {x.shape}: should be of shape n x {dim}")
        x_np = x.cpu().numpy().reshape(-1, dim)
        x_q = xgb.DMatrix(x_np)
        new_y = bst_surrogate.predict(x_q)

        if scale_y:
            new_y = (new_y - y_min) / (y_max - y_min)
            new_y = np.clip(new_y, 0, 1)
        else:
            new_y = np.clip(new_y, y_min, y_max)

        new_y = torch.tensor(new_y, device=x.device, dtype=x.dtype)
        assert new_y.dim() == 1 and new_y.size(0) == x.size(0)
        return new_y
    
    return objective_function


def get_hpob_objective_function(
    search_space_id: str, dataset_id: str, scale_y: bool = False):
    """Get a function that evaluates the objective function for a given
    HPO-B search space ID and dataset ID.
    
    Args:
        search_space_id: The HPO-B search space ID.
        dataset_id: The HPO-B dataset ID.
        scale_y: If True, scale the output to [0, 1] using min-max scaling.
            If False, return the raw output clipped to [y_min, y_max].
            
    Returns:
        A function that takes a tensor of shape (n_points, dim) and returns
        a tensor of shape (n_points,) representing the objective function values.
    """
    return _get_hpob_objective_function(search_space_id, dataset_id, scale_y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    search_space_id = '5970' # ranger (5), 2 dimensions
    # search_space_id = '5965' # ranger (9), 10 dimensions
    # search_space_id = '5636'
    dataset_type = 'test'
    dataset = get_hpob_dataset(search_space_id, dataset_type)
    dataset_ids = get_hpob_dataset_ids(search_space_id, dataset_type)
    print(f"Number of datasets: {len(dataset)}")

    # Create 3 rows of subplots
    fig, axs = plt.subplots(1, len(dataset), figsize=(5*len(dataset), 5), squeeze=True)
    if len(dataset) == 1:
        axs = [axs]

    for i in range(len(dataset)):
        item = dataset[i]
        dataset_id = dataset_ids[i]

        y_values = item.y_values.flatten()

        # Split dataset into two random equal-sized parts
        n_samples = len(y_values)

        # Get original min/max for plotting
        min_y = y_values.min().item()
        max_y = y_values.max().item()
        min_x = item.x_values.min(axis=0).values.numpy()
        max_x = item.x_values.max(axis=0).values.numpy()
        min_y_surrogate, max_y_surrogate = get_hpob_function_min_max(
            search_space_id, dataset_id, use_surrogate_stats=True)
        print(
            f"Dataset {i}: X shape: {tuple(item.x_values.shape)}, "
            f"y shape: {tuple(y_values.shape)}, y min: {min_y}, y max: {max_y}, "
            f"y min (surrogate): {min_y_surrogate}, y max (surrogate): {max_y_surrogate}, "
            f"X min: {min_x}, X max: {max_x}")

        surrogate_function = get_hpob_objective_function(
            search_space_id, dataset_id, scale_y=False)

        y = y_values.cpu().numpy()
        y_pred = surrogate_function(item.x_values).cpu().numpy()

        axs[i].scatter(y, y_pred, alpha=0.5)
        axs[i].plot([min_y, max_y], [min_y, max_y], 'r--')
        axs[i].set_xlabel("True y")
        axs[i].set_ylabel("Predicted y")
        axs[i].set_title(f"Dataset {i} (ID {dataset_id})")
        axs[i].set_aspect('equal', 'box')
        axs[i].grid(True)

        ## Can observe in the plots that these min/max values are sometimes wrong
        axs[i].axhline(min_y_surrogate, color='g', linestyle='--', label='Surrogate min')
        axs[i].axhline(max_y_surrogate, color='b', linestyle='--', label='Surrogate max')
        axs[i].legend()

        # axs[i].axhline(min_y, color='g', linestyle='--', label='Surrogate min')
        # axs[i].axhline(max_y, color='b', linestyle='--', label='Surrogate max')
        # axs[i].legend()

    plt.tight_layout()
    plt.savefig("hpob_surrogate_check.pdf")
