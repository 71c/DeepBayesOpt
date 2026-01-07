import os
from datetime import datetime

from utils_general.io_utils import save_json


def get_new_timestamp_model_save_dir(models_path: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{timestamp}"
    return os.path.join(models_path, model_name), model_name


def mark_new_model_as_trained(models_path: str, model_name: str):
    latest_model_path = os.path.join(models_path, "latest_model.json")
    save_json({"latest_model": model_name}, latest_model_path)