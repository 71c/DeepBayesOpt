import os
from typing import Literal

import torch
from datasets.function_samples_dataset import FunctionSamplesItem, ListMapFunctionSamplesDataset
from utils.constants import HPOB_DATA_DIR
from utils.utils import load_json


_HPOB_PATHS = {
    'train': os.path.join(HPOB_DATA_DIR, "meta-train-dataset.json"),
    'validation': os.path.join(HPOB_DATA_DIR, "meta-validation-dataset.json"),
    'test': os.path.join(HPOB_DATA_DIR, "meta-test-dataset.json"),
}


_HPOB_JSON_CACHE = {}
def _get_hpob_dataset_json(search_space_id: str,
                           dataset_type: Literal['train', 'validation', 'test']):
    if dataset_type not in _HPOB_JSON_CACHE:
        data_path = _HPOB_PATHS[dataset_type]
        _HPOB_JSON_CACHE[dataset_type] = load_json(data_path)
    data = _HPOB_JSON_CACHE[dataset_type]
    try:
        return data[search_space_id]
    except KeyError:
        ValueError(f"Search space ID {search_space_id} not found in HPO-B dataset.")


def get_hpob_dataset_dimension(search_space_id: str) -> int:
    """Get the dimension of the search space for a given HPO-B search space ID."""
    search_space_data = _get_hpob_dataset_json(search_space_id, 'validation')
    dataset_id = list(search_space_data.keys())[0]
    X = search_space_data[dataset_id]['X']
    return len(X[0])


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
