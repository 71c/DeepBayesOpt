import os
from datetime import datetime
from utils.utils import dict_to_str, group_by, group_by_nested_attrs, save_json
from utils.plot_utils import save_figures_from_nested_structure
from utils.experiment_config_utils import CONFIG_DIR
from train_acqf import MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN


script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')


# TODO: Finish writing this script
# (can start by continuing to copy the code from bo_experiments_gp_plot.py)
