import os

_script_dir = os.path.dirname(os.path.abspath(__file__)) # script directory
_ROOT_DIR = os.path.abspath(os.path.join(_script_dir, "..")) # root directory

_DATA_DIR = os.path.join(_ROOT_DIR, "data")
_UTILS_GENERAL_DIR = os.path.join(_ROOT_DIR, "utils_general")

_EXPERIMENTS_DIR = os.path.join(_ROOT_DIR, "experiments")
REGISTRY_PATH = os.path.join(_EXPERIMENTS_DIR, "registry.yml")

JOB_ARRAY_SUB_PATH = os.path.join(_UTILS_GENERAL_DIR, "experiments", "job_array.sub")

DATASETS_DIR = os.path.join(_DATA_DIR, "datasets")
RESULTS_DIR = os.path.join(_DATA_DIR, "bayesopt_results")
SWEEPS_DIR = os.path.join(_DATA_DIR, "sweeps")
PLOTS_DIR = os.path.join(_DATA_DIR, "plots")
MODELS_DIR = os.path.join(_DATA_DIR, "saved_models")
HPOB_DATA_DIR = os.path.join(_DATA_DIR, "hpob-data")
HPOB_SAVED_SURROGATES_DIR = os.path.join(_DATA_DIR, "saved-surrogates")
MODELS_VERSION = "v3"
RUN_PLOTS_FOLDER = "bo_plots"
