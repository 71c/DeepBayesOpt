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
    X = np.array(_get_hpob_dataset_json(search_space_id, 'test')[dataset_id]["X"])
    return torch.tensor(X[init_ids])


SURROGATES_STATS_PATH = os.path.join(HPOB_SAVED_SURROGATES_DIR, "summary-stats.json")

@cache
def _load_hpob_surrogates_stats_helper():
    if os.path.isfile(SURROGATES_STATS_PATH):
        return load_json(SURROGATES_STATS_PATH)
    raise FileNotFoundError(f"Surrogates stats file not found: {SURROGATES_STATS_PATH}")


def _load_hpob_surrogates_stats(surrogate_name: str):
    """Load the HPO-B surrogates statistics from the JSON file."""
    return _load_hpob_surrogates_stats_helper()[surrogate_name]


@cache
def get_hpob_objective_function(search_space_id: str, dataset_id: str):
    """Get a function that evaluates the objective function for a given
    HPO-B search space ID and dataset ID."""
    dataset_ids = get_hpob_dataset_ids(search_space_id, 'test')
    if dataset_id not in dataset_ids:
        raise ValueError(
            f"Dataset ID {dataset_id} not found in HPO-B test datasets for search "
            f"space ID {search_space_id}. Available datasets: {dataset_ids}")
    surrogate_name = 'surrogate-'+search_space_id+'-'+dataset_id
    bst_surrogate = xgb.Booster()
    bst_surrogate.load_model(f"{HPOB_SAVED_SURROGATES_DIR}/{surrogate_name}.json")

    surrogate_stats = _load_hpob_surrogates_stats(surrogate_name)
    y_min = surrogate_stats["y_min"]
    y_max = surrogate_stats["y_max"]
    # print(f"{surrogate_name}: y_min={y_min}, y_max={y_max}")
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

        ## This is what HPO-B does, but I'm not sure whether it's a good idea.
        new_y = (new_y - y_min) / (y_max - y_min)
        new_y = np.clip(new_y, 0, 1)

        new_y = torch.tensor(new_y, device=x.device, dtype=x.dtype)
        assert new_y.dim() == 1 and new_y.size(0) == x.size(0)
        return new_y
    
    return objective_function


if __name__ == "__main__":
    # Example usage
    dataset = get_hpob_dataset('5970', 'train')
    print(f"Number of datasets: {len(dataset)}")
    for i in range(len(dataset)):
        item = dataset[i]
        min_y = item.y_values.min().item()
        max_y = item.y_values.max().item()
        min_x = item.x_values.min(axis=0).values.numpy()
        max_x = item.x_values.max(axis=0).values.numpy()
        print(
            f"Dataset {i}: X shape: {tuple(item.x_values.shape)}, "
            f"y shape: {tuple(item.y_values.shape)}, y min: {min_y}, y max: {max_y}, "
            f"X min: {min_x}, X max: {max_x}")
