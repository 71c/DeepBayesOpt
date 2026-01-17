from typing import Optional

import torch

from utils_general.plot_train import plot_train

from utils_general.utils import DEVICE

# These three will be parameters
from experiments.registry import REGISTRY
from submit_train import AF_TRAIN_SUBMIT_UTILS
from utils.plot_utils import PROJECT_PLOT_UTILS


CPROFILE = False


# These are simply used for defining the project-specific plotting functions
N_HISTORY = 10
from utils.utils import get_lamda_for_bo_of_nn
from utils.plot_utils import (
    N_CANDIDATES_PLOT, plot_acquisition_function_net_training_history_ax,
    plot_nn_vs_gp_acquisition_function_1d)
 


def plot_0_training_history_train_test(
        ax,
        training_history_data,
        dataset_getter,
        nn_model,
        cfg,
        plot_name: Optional[str]=None,
        label='',
        color=None,
        alpha=1.0
    ):
    plot_acquisition_function_net_training_history_ax(
        ax, training_history_data, plot_maxei=False, plot_name=plot_name,
        plot_log_regret=False, label=label, color=color, alpha=alpha)


def plot_1_training_history_test_log_regret(
        ax,
        training_history_data,
        dataset_getter,
        nn_model,
        cfg,
        plot_name: Optional[str]=None,
        label='',
        color=None,
        alpha=1.0
    ):
    plot_acquisition_function_net_training_history_ax(
        ax, training_history_data, plot_maxei=False, plot_name=plot_name,
        plot_log_regret=True, label=label, color=color, alpha=alpha)


def plot_2_af_plot(
        ax,
        training_history_data,
        dataset_getter,
        nn_model,
        cfg,
        plot_name: Optional[str]=None,
        label='',
        color=None,
        alpha=1.0
    ):
    train_aq_dataset, test_aq_dataset, small_test_aq_dataset = dataset_getter()

    it = iter(test_aq_dataset)
    item = next(it)
    try:
        while item.x_hist.shape[0] != N_HISTORY:
            item = next(it)
    except StopIteration:
        raise ValueError("No item with the right number of history points.")
    
    # Get the data to plot
    x_hist, y_hist, x_cand, vals_cand = item.tuple_no_model
    x_cand_original = x_cand
    dimension = x_hist.size(1)
    x_cand_varying_component = torch.linspace(0, 1, N_CANDIDATES_PLOT)
    varying_index = 0
    if dimension == 1:
        # N_CANDIDATES_PLOT x 1
        x_cand = x_cand_varying_component.unsqueeze(1)
    else:
        torch.manual_seed(0)
        random_x = torch.rand(dimension)
        # N_CANDIDATES_PLOT x dimension
        x_cand = random_x.repeat(N_CANDIDATES_PLOT, 1)
        x_cand[:, varying_index] = x_cand_varying_component

    # Get the GP model
    gp_model = item.model if item.has_model else None

    lamda = get_lamda_for_bo_of_nn(
            cfg.get('lamda'), cfg.get('lamda_min'), cfg.get('lamda_max'))
    plot_nn_vs_gp_acquisition_function_1d(
        ax=ax, x_hist=x_hist, y_hist=y_hist, x_cand=x_cand,
        # x_cand_original=x_cand_original, vals_cand=vals_cand,
        lamda=lamda,
        gp_model=gp_model, nn_model=nn_model, method=cfg['method'],
        gp_fit_methods=['exact'],
        min_x=0.0, max_x=1.0,
        nn_device=DEVICE, group_standardization=None,
        varying_index=varying_index
    )
    ax.set_title("Acquisition function plot")


TRAIN_ATTR_NAME_TO_TITLE = {
    "0_training_history_train_test": "Training history (train and test loss)",
    "1_training_history_test_log_regret": "Training history (test log regret)",
    "2_af_plot": "Acquisition function plot"
}


TRAIN_ATTR_NAME_TO_PLOT_FUNC = {
    "0_training_history_train_test": plot_0_training_history_train_test,
    "1_training_history_test_log_regret": plot_1_training_history_test_log_regret,
    "2_af_plot": plot_2_af_plot
}


ATTR_GROUPS = [
    ["0_training_history_train_test"],

    # ["0_training_history_train_test", "1_training_history_test_log_regret", "2_af_plot"],
    # ["0_training_history_train_test", "2_af_plot"],
    # ["1_training_history_test_log_regret"],
    # ["2_af_plot"],

    # ["0_training_history_train_test", "1_training_history_test_log_regret"],
]


if __name__ == "__main__":
    plot_train(
        registry=REGISTRY,
        train_submit_utils=AF_TRAIN_SUBMIT_UTILS,
        plot_utils=PROJECT_PLOT_UTILS,
        attr_name_to_plot_func=TRAIN_ATTR_NAME_TO_PLOT_FUNC,
        attr_name_to_title=TRAIN_ATTR_NAME_TO_TITLE,
        attr_groups=ATTR_GROUPS,
        use_cprofile=CPROFILE
    )
