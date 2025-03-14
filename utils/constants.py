import os

_script_dir = os.path.dirname(os.path.abspath(__file__)) # script directory
_ROOT_DIR = os.path.abspath(os.path.join(_script_dir, "..")) # root directory

_DATA_DIR = os.path.join(_ROOT_DIR, "data")
_UTILS_DIR = os.path.join(_ROOT_DIR, "utils")

JOB_ARRAY_SUB_PATH = os.path.join(_UTILS_DIR, "job_array.sub")

DATASETS_DIR = os.path.join(_DATA_DIR, "datasets")
RESULTS_DIR = os.path.join(_DATA_DIR, "bayesopt_results")
SWEEPS_DIR = os.path.join(_DATA_DIR, "sweeps")
PLOTS_DIR = os.path.join(_DATA_DIR, "plots")
MODELS_DIR = os.path.join(_DATA_DIR, "saved_models")
MODELS_VERSION = "v2"
