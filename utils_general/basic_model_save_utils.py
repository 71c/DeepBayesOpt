import os
from datetime import datetime

from utils_general.io_utils import load_json, save_json


class BasicModelSaveUtils:
    """Basic utility functions for saving models with timestamped directories
    in a filesystem."""
    def __init__(self, models_path: str, models_subdir_name: str):
        self.models_path = models_path
        self.models_subdir_name = models_subdir_name

    def get_new_model_save_dir(self, models_path: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_{timestamp}"
        return os.path.join(models_path, model_name), model_name

    def mark_new_model_as_trained(self, models_path: str, model_name: str):
        latest_model_path = os.path.join(models_path, "latest_model.json")
        save_json({"latest_model": model_name}, latest_model_path)

    def get_latest_model_path(self, model_and_info_folder_name: str):
        model_and_info_path = os.path.join(self.models_path, model_and_info_folder_name)
        already_saved = os.path.isdir(model_and_info_path)
        if not already_saved:
            raise FileNotFoundError(f"Models path {model_and_info_path} does not exist")

        models_path = os.path.join(model_and_info_path, self.models_subdir_name)

        latest_model_path = os.path.join(models_path, "latest_model.json")
        try:
            latest_model_name = load_json(latest_model_path)["latest_model"]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Latest model path {latest_model_path} does not exist."
                " i.e., no models have been fully trained yet.")
        model_path = os.path.join(models_path, latest_model_name)

        # Handle a temporary bug from previous code (not really necessary)
        if not os.path.isdir(model_path):
            return os.path.join(model_and_info_path, latest_model_name)
        
        return model_path

    def model_is_trained(self, model_and_info_folder_name: str):
        try:
            self.get_latest_model_path(model_and_info_folder_name)
            return True
        except FileNotFoundError:
            return False
