import os
from datetime import datetime
from plot_utils import save_figures_from_nested_structure
from submit_dependent_jobs import CONFIG_DIR
from train_acqf import MODEL_AND_INFO_NAME_TO_CMD_OPTS_NN
from utils import dict_to_str, group_by, group_by_nested_attrs, save_json


script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, 'plots')


def plot_key_value_to_str(k, v):
    if k == "attr_name":
        return (2, v)
    return (1, f"{k}={v}")


def plot_dict_to_str(d):
    for key_name, prefix, plot_name in [
        ("method", "", "method="),
    ]:
        if key_name in d:
            d_method = {}
            d_non_method = {}
            for k, v in d.items():
                if k == key_name:
                    continue
                if k.startswith(prefix):
                    d_method[k[len(prefix):]] = v
                else:
                    d_non_method[k] = v
            method = d[key_name]
            if d_method:
                s = dict_to_str(d_method, include_space=True)
                ret += f" ({s})"
            if d_non_method:
                s = dict_to_str(d_non_method, include_space=True)
                ret += f", {s}"
            return ret
    
    items = [
        plot_key_value_to_str(k, v)
        for k, v in d.items()
    ]
    items = sorted(items)
    return ", ".join([item[1] for item in items])

# TODO: Finish writing this script
# (can start by continuing to copy the code from bo_experiments_gp_plot.py)
