from datetime import datetime
import os
from typing import Literal, Optional, List
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import torch.distributions as dist
from utils.constants import BO_PLOTS_FOLDER
from datasets.acquisition_dataset import AcquisitionDataset
from nn_af.acquisition_function_net import AcquisitionFunctionNet, AcquisitionFunctionNetAcquisitionFunction
from utils.constants import PLOTS_DIR
from utils.exact_gp_computations import calculate_EI_GP
from utils.utils import dict_to_str, iterate_nested, save_json


def _calculate_center_and_interval(
        data,
        alpha,
        interval_of_center,
        assume_normal,
        center_stat:Optional[Literal['mean', 'median']] = None):
    """
    Calculate the center and an interval around it.

    Args:
        data:
            The data to calculate the center and interval of, shape (n_samples, n_stats)
            where each column is a different statistic with n_samples measurements.
        alpha:
            The confidence level, i.e., 1 - alpha is the proportion of the data
            that is within the confidence interval.
        interval_of_center:
            If True, calculate the confidence interval of the "center" (mean or median).
            If False, calculate the prediction interval of the data.
        assume_normal:
            If True, assume the data is normally distributed and use the t-distribution
            to calculate the confidence interval of the mean.
            If False, use bootstrapping to calculate the confidence interval of the
            center.
        center_stat: {'mean', 'median'}, optional
            The statistic used as the center. If None, defaults to 'mean' when
            assume_normal is True and to 'median' when assume_normal is False.
            If assume_normal is True and a value other than 'mean' is provided,
            a ValueError is raised.
    
    Returns:
        tuple (center, lo, hi)
    """
    if assume_normal:
        if center_stat is None:
            center_stat = 'mean'
        elif center_stat != 'mean':
            raise ValueError("Must use mean if using normal approximation")
    else:
        if center_stat is None:
            center_stat = "median"
        elif not (center_stat == 'mean' or center_stat == 'median'):
            raise ValueError("Must use mean or median center statistic")

    statistic = np.mean if center_stat == 'mean' else np.median
    center = statistic(data, axis=0)
    
    if assume_normal:
        n = data.shape[0]
        t = stats.t.ppf(1 - alpha / 2, n - 1)
        if interval_of_center:
            # https://stackoverflow.com/a/15034143
            se = stats.sem(data, axis=0, ddof=1)
            ci = se * t
        else:
            # https://en.wikipedia.org/wiki/Prediction_interval#Unknown_mean,_unknown_variance
            std = np.std(data, axis=0, ddof=1)
            ci = std * t * np.sqrt(1 + 1/n)
        lo, hi = center - ci, center + ci
    else:
        if interval_of_center:
            res = stats.bootstrap(
                (data,),
                statistic=statistic,
                axis=0,
                confidence_level=1 - alpha
            )
            lo = res.confidence_interval.low
            hi = res.confidence_interval.high
        else:
            # Calculate the interval of the data
            lo = np.quantile(data, alpha / 2, axis=0)
            hi = np.quantile(data, 1 - alpha / 2, axis=0)

    return center, lo, hi


def plot_error_bars(
        ax,
        center, lower, upper,
        label=None,
        shade=True):
    x = range(len(center))
    if shade:
        ax.plot(x, center, label=label)
        ax.fill_between(x, lower, upper, alpha=0.3)
    else:
        ax.errorbar(x, center, yerr=[center-lower, upper-center],
                    fmt='-o', capsize=4, markersize=5, capthick=None, label=label)


def plot_optimization_trajectories_error_bars(
        ax,
        data,
        label=None,
        shade=True,
        alpha=0.05,
        interval_of_center=True):
    """Plot the mean and confidence interval of the data
    alpha: confidence level, i.e., 1 - alpha is the proportion of the data
    that is within the confidence interval.
    """
    center, lower, upper = _calculate_center_and_interval(
        data=data,
        alpha=alpha, interval_of_center=interval_of_center,
        center_stat="median", assume_normal=False)
    
    plot_error_bars(ax, center, lower, upper, label=label, shade=shade)


def plot_gp_posterior(ax, posterior, test_x, train_x, train_y, color, name=None):
    if not hasattr(posterior, "mvn"):
        # This in general won't correspond to the actual probability distribution AT ALL
        # e.g. exponentiate then we get a lognormal distribution
        # but this is not that, the lower can go negative!
        # lower, upper = posterior.mean - posterior.variance.sqrt(), posterior.mean + posterior.variance.sqrt()
        return # Actually, lognormal distribution too extreme values so not comparable to plot, so just not plot it
    else:
        lower, upper = posterior.mvn.confidence_region()
    mean = posterior.mean.detach().squeeze().cpu().numpy()
    lower = lower.detach().squeeze().cpu().numpy()
    upper = upper.detach().squeeze().cpu().numpy()

    train_x = train_x.detach().squeeze().cpu().numpy()
    train_y = train_y.detach().squeeze().cpu().numpy()
    test_x = test_x.detach().squeeze().cpu().numpy()
    
    sorted_indices = np.argsort(test_x)
    test_x = test_x[sorted_indices]
    mean = mean[sorted_indices]
    lower = lower[sorted_indices]
    upper = upper[sorted_indices]

    extension = '' if name is None else f' {name}'

    # Plot posterior means as blue line
    ax.plot(test_x, mean, color, label=f'Mean{extension}')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x, lower, upper, color=color, alpha=0.5, label=f'Confidence{extension}')

# decided actually don't want to use this
normal = dist.Normal(0, 1)
def normalize_by_quantile(x, dim=-1):
    indices = torch.argsort(torch.argsort(x, dim=dim), dim=dim)
    max_index = x.size(dim) - 1
    quantiles = indices / max_index
    return normal.icdf(quantiles)


def get_means_and_stds(arrs: List[Tensor], correction=1, group=False):
    means = torch.tensor([arr.mean() for arr in arrs])
    variances = torch.tensor(
        [arr.var(dim=None, correction=correction, keepdim=False) for arr in arrs])
    if group:
        numels = torch.tensor([arr.numel() for arr in arrs], dtype=means.dtype)
        total_n = torch.sum(numels)
        group_mean = torch.sum(numels * means) / total_n
        group_variance = torch.sum(
            (numels - correction) * variances
            + numels * (means - group_mean)**2) / (total_n - correction)
        group_std = group_variance.sqrt()

        means = torch.full_like(means, group_mean)
        stds = torch.full_like(variances, group_std)
    else:
        stds = variances.sqrt()
    return means, stds


def standardize_arrs(arrs: List[Tensor], correction=1, group=False):
    means, stds = get_means_and_stds(arrs, correction, group)
    return [(arr - mean) / std for arr, mean, std in zip(arrs, means, stds)]


def plot_nn_vs_gp_acquisition_function_1d_grid(
        aq_dataset:AcquisitionDataset, nn_model:AcquisitionFunctionNet,
        policy_gradient:bool, plot_name:str,
        n_candidates:int, nrows:int, ncols:int, min_x:float=0., max_x:float=1.,
        plot_map:bool=True, nn_device:Optional[torch.device]=None,
        group_standardization:Optional[bool]=None):
    """Plot the acquisition function of a neural network and a GP on a grid of
      1D functions.

    Args:
        aq_dataset: The dataset of acquisition functions to plot.
        nn_model: The neural network model to use.
        policy_gradient: Whether the NN model uses policy gradient or not.
        plot_name: The name of the plot.
        n_candidates: The number of candidate points to plot.
        nrows: The number of rows of plots.
        ncols: The number of columns of plots.
        min_x: The minimum x value to plot.
        max_x: The maximum x value to plot.
        plot_map: Whether to plot the MAP GP acquisition function.
        nn_device: The device to use for the neural network.
        group_standardization: Whether to standardize the acquisition functions together.
        Default is to standardize them together if policy_gradient is False,
        and not standardize them together if policy_gradient is True.
        Setting this to True makes it so they can be compared on the same scale,
        which is applicable if predicting the improvement with MSE, but not
        if using policy gradient.
    """
    if group_standardization is None:
        group_standardization = not policy_gradient

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5*ncols, 2.5*nrows),
                            sharex=True, sharey=False)

    it = iter(aq_dataset)
    for row in range(nrows):
        for col in range(ncols):
            item = next(it)

            gp_model = item.model

            x_hist, y_hist, x_cand, vals_cand = item.tuple_no_model
            x_cand_original = x_cand
            x_cand = torch.linspace(0, 1, n_candidates).unsqueeze(1)
            item.x_cand = x_cand

            sorted_indices = np.argsort(x_cand.detach().numpy().flatten())
            sorted_x_cand = x_cand.detach().numpy().flatten()[sorted_indices]

            arrs_and_labels_to_plot = []

            # Compute GP EI acquisition function
            plot_gp = True
            try:
                gp_model.set_train_data_with_transforms(x_hist, y_hist, strict=False, train=False)
                posterior_true = gp_model.posterior(x_cand, observation_noise=False)

                ei_true = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, log=False)
                arrs_and_labels_to_plot.append((ei_true, "True GP"))

                if plot_map:
                    ei_map = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, fit_params=True, log=False)
                    arrs_and_labels_to_plot.append((ei_map, "MAP GP"))
            except NotImplementedError: # NotImplementedError: No mean transform provided.
                plot_gp = False
            
            # Compute NN acquisition function
            x_hist_nn, y_hist_nn, x_cand_nn, vals_cand_nn = item.to(nn_device).tuple_no_model
            aq_fn = AcquisitionFunctionNetAcquisitionFunction.from_net(
                nn_model, x_hist_nn, y_hist_nn, exponentiate=not policy_gradient, softmax=False)
            ei_nn = aq_fn(x_cand_nn.unsqueeze(1)).cpu()
            arrs_and_labels_to_plot.append((ei_nn, "NN"))

            ax = axs[row, col]

            arrs, labels = zip(*arrs_and_labels_to_plot)
            arrs = standardize_arrs(arrs, group=group_standardization)
            for arr, label in zip(arrs, labels):
                sorted_arr = arr.detach().numpy().flatten()[sorted_indices]
                ax.plot(sorted_x_cand, sorted_arr, label=label)
            
            # Plot training points as black stars
            ax.plot(x_hist, y_hist, 'b*', label=f'Observed Data')

            ax.plot(x_cand_original, vals_cand, 'ro', markersize=1, label=f'Candidate points')
            
            if plot_gp:
                plot_gp_posterior(
                    ax, posterior_true, x_cand, x_hist, y_hist, 'b', name='True')

            # ax.set_title(f"History: {x_hist.size(0)}")
            ax.set_xlim(min_x, max_x)
    
    # Add a single legend for all plots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    
    # axs[0, ncols - 1].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    axs[0, ncols - 1].legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1))
    fig.tight_layout(rect=[0.02, 0.02, 1, 1])
    # fig.suptitle(f'{name} vs x', fontsize=16)
    fig.supxlabel("x", fontsize=10)
    fig.supylabel(f'{plot_name}', fontsize=10)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    
    return fig, axs


BLUE = '#1f77b4'
ORANGE = '#ff7f0e'


def plot_acquisition_function_net_training_history_ax(
        ax, training_history_data, plot_maxei=False, plot_log_regret=False,
        plot_name=None):
    stats_epochs = training_history_data['stats_epochs']
    stat_name =  'maxei' if plot_maxei else training_history_data['stat_name']
    
    train_stat = np.array(
        [epoch['train']['after_training'][stat_name] for epoch in stats_epochs])
    test_stat = np.array([epoch['test'][stat_name] for epoch in stats_epochs])
    
    # Determine the GP test stat
    if stat_name == 'mse' or stat_name == 'maxei':
        gp_prefix = 'true_gp_ei_'
    elif stat_name == 'ei_softmax':
        gp_prefix = 'true_gp_ei_maxei_'
    elif stat_name.startswith('gittins_loss'):
        gp_prefix = f'true_gp_gi_'
    else:
        raise ValueError
    gp_stat_name = gp_prefix + stat_name
    try:
        gp_test_stat = [epoch['test'][gp_stat_name] for epoch in stats_epochs]
        # Validate that all the values in `gp_test_stat` are the same:
        assert all(x == gp_test_stat[0] for x in gp_test_stat)
        gp_test_stat = gp_test_stat[0]
    except KeyError:
        gp_test_stat = None
    
    if plot_log_regret:
        if gp_test_stat is None:
            raise ValueError("Need to have GP test stat to plot log regret")
        to_plot = {
            'lines': [
                {
                    'label': 'Regret (NN)',
                    'data': np.abs(test_stat - gp_test_stat),
                    'color': ORANGE
                }
            ],
            'title': f'Test {stat_name} vs Epochs (log regret)',
            'xlabel': 'Epochs',
            'ylabel': f'{stat_name} regret',
            'log_scale': True
        }
    else:
        to_plot = {
            'lines': [
                {
                    'label': 'Train (NN)',
                    'data': train_stat,
                    'color': BLUE
                },
                {
                    'label': 'Test (NN)',
                    'data': test_stat,
                    'color': ORANGE
                }
            ],
            'title': f'Train and Test {stat_name} vs Epochs',
            'xlabel': 'Epochs',
            'ylabel': stat_name,
            'log_scale': False
        }

        if gp_test_stat is not None:
            to_plot['consts'] = [
                {
                    'label': f'Test (true GP value)',
                    'data': gp_test_stat,
                    'color': 'k',
                    'linestyle': '--',
                }
            ]

    epochs = np.arange(1, len(train_stat) + 1)

    line_properties = ['label', 'marker', 'linestyle', 'color']

    lines = to_plot.get('lines')
    if lines is not None:
        for line in lines:
            kwargs = {p: line[p] for p in line_properties if p in line}
            ax.plot(epochs, line['data'], **kwargs)
    consts = to_plot.get('consts')
    if consts is not None:
        for line in consts:
            kwargs = {p: line[p] for p in line_properties if p in line}
            ax.axhline(line['data'], **kwargs)
    
    plot_desc = to_plot['title']
    if plot_name is not None:
        plot_name = f"{plot_name} ({plot_desc})"
    else:
        plot_name = plot_desc

    ax.set_title(plot_name)
    ax.set_xlabel(to_plot['xlabel'])
    ax.set_ylabel(to_plot['ylabel'])
    ax.legend()
    ax.grid(True)
    if to_plot['log_scale']:
        ax.set_yscale('log')


def plot_acquisition_function_net_training_history(
        training_history_data, plot_maxei=False, plot_log_regret=True,
        **plot_kwargs):
    scale = plot_kwargs.get("scale", 1.0)
    aspect = plot_kwargs.get("aspect", 1.5)

    area = 50 * scale**2
    height = np.sqrt(area / aspect)
    width = aspect * height

    plot_log_regrets = [False]
    if plot_log_regret:
        plot_log_regrets.append(True)

    n_rows, n_cols = 1, len(plot_log_regrets)
    fig, axs = plt.subplots(n_rows, n_cols,
                             figsize=(width * n_cols, height * n_rows),
                             sharex=False, sharey=False,
                             squeeze=True)
    for ax, plot_log_regret in zip(axs, plot_log_regrets):
        plot_acquisition_function_net_training_history_ax(
            ax=ax,
            training_history_data=training_history_data,
            plot_maxei=plot_maxei,
            plot_log_regret=plot_log_regret
        )

    return fig


def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=30,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


_ERROR_LINES_CACHE = {}
def _get_error_lines(
        ids: list,
        data: list[np.ndarray],
        alpha,
        interval_of_center,
        center_stat,
        assume_normal):
    ids.sort()
    info = dict(
        ids=ids,
        alpha=alpha,
        interval_of_center=interval_of_center,
        center_stat=center_stat,
        assume_normal=assume_normal,
    )
    info_str = dict_to_str(info, include_space=False)

    if info_str in _ERROR_LINES_CACHE:
        return _ERROR_LINES_CACHE[info_str]

    ret = _calculate_center_and_interval(
        data=np.array(data),
        alpha=alpha, interval_of_center=interval_of_center,
        center_stat=center_stat, assume_normal=assume_normal)
    _ERROR_LINES_CACHE[info_str] = ret
    return ret # center, lower, upper

def get_plot_ax_bo_stats_vs_iteration_func(get_result_func):
    def ret(plot_config: dict,
            ax,
            plot_name: Optional[str]=None,
            attr_name_to_title: dict[str, str] = {},
            **plot_kwargs):
        attr_names = set()
        
        for legend_name, data in plot_config.items():
            this_ids = []
            this_data = []
            for v in data["items"].values():
                data_index = v["items"]
                if isinstance(data_index, list):
                    print([get_result_func(i) for i in data_index])
                    raise ValueError

                info = get_result_func(data_index)
                if info is not None:
                    attr_name = info['attr_name']
                    attr_names.add(attr_name)
                    val = info[attr_name]
                    if attr_name == 'best_y':
                        assert len(val.shape) == 2 and val.shape[1] == 1
                        val = val[:, 0]
                    else:
                        assert len(val.shape) == 1
                    
                    val_id = f"{info['index']}_{attr_name}"
                    this_ids.append(val_id)
                    this_data.append(val)
            
            if len(this_ids) == 0:
                continue

            center, lower, upper = _get_error_lines(
                this_ids, this_data,
                alpha=plot_kwargs['alpha'],
                interval_of_center=plot_kwargs['interval_of_center'],
                center_stat=plot_kwargs['center_stat'],
                assume_normal=plot_kwargs['assume_normal']
            )

            plot_error_bars(ax, center, lower, upper,
                            label=legend_name, shade=plot_kwargs['shade'])
        
        if len(attr_names) > 1:
            raise ValueError(f"Expected just one attribute to plot but got {attr_names=}")

        if len(attr_names) == 0:
            raise ValueError("No data to plot")
        
        attr_name = attr_names.pop()

        ax.set_xlabel('Iteration')
        attr_name_title = attr_name_to_title.get(attr_name, attr_name)
        ax.set_ylabel(attr_name_title)

        plot_desc = f"{attr_name_title} vs iteration"
        if plot_name:
            plot_name = f"{plot_name} ({plot_desc})"
        else:
            plot_name = plot_desc

        ax.set_title(plot_name)
        ax.legend()
    
    return ret


def _get_figure_from_nested_structure(
        plot_config: dict,
        plot_ax_func,
        attr_name_to_title: dict[str, str],
        attrs_groups_list: list[Optional[set]],
        level_names: list[str],
        figure_name: str,
        **plot_kwargs):
    this_attrs_group = attrs_groups_list[0]
    next_attrs_groups = attrs_groups_list[1:]
    this_level_name = level_names[0]
    next_level_names = level_names[1:]

    scale = plot_kwargs.get("scale", 1.0)
    aspect = plot_kwargs.get("aspect", 1.5)
    sharey = plot_kwargs.get("sharey", False)

    area = 50 * scale**2
    height = np.sqrt(area / aspect)
    width = aspect * height

    row_and_col = False
    col_names = None
    if this_level_name == "line":
        n_rows = 1
        n_cols = 1
        sharey = False
    elif this_level_name == "row":
        n_rows = len(plot_config)
        if next_level_names[0] == "col":
            row_and_col = True
            assert next_level_names[1] == "line"
            key_to_data = {}
            for v in plot_config.values():
                for kk, vv in v["items"].items():
                    if kk not in key_to_data:
                        key_to_data[kk] = list(sorted(vv["vals"].items()))

            col_names = list(sorted(key_to_data.keys(),
                                    key=lambda u: key_to_data[u]))

            col_name_to_col_index = {}
            for i, col_name in enumerate(col_names):
                col_name_to_col_index[col_name] = i
            n_cols = len(col_names)
            if "attr_name" in this_attrs_group:
                # The plot attribute is varied with each row.
                sharey = "row" # each subplot row will share an x- or y-axis.
            elif "attr_name" in next_attrs_groups[0]:
                # The plot attribute is varied with each column.
                sharey = "col" # each subplot column will share an x- or y-axis.
            else:
                sharey = True
        else:
            n_cols = 1
            sharey = "attr_name" not in this_attrs_group
    elif this_level_name == "col":
        n_rows = 1
        n_cols = len(plot_config)
        sharey = "attr_name" not in this_attrs_group
    else:
        raise ValueError
    
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(width * n_cols, height * n_rows),
                             sharex=True, sharey=sharey,
                             squeeze=False
                             )
    
    if this_level_name == "line":
        plot_ax_func(plot_config=plot_config, ax=axes[0, 0], plot_name=figure_name,
                     attr_name_to_title=attr_name_to_title,
                     **plot_kwargs)
    else:
        fig.suptitle(figure_name, fontsize=16, fontweight='bold')
        
        sorted_plot_config_items = list(sorted(
            plot_config.items(),
            key=lambda x: list(sorted(x[1]["vals"].items()))
        ))
        
        if row_and_col:
            for row, (row_name, row_data) in enumerate(sorted_plot_config_items):
                row_items = row_data["items"]
                for subplot_name, subplot_data in row_items.items():
                    col = col_name_to_col_index[subplot_name]
                    plot_ax_func(plot_config=subplot_data["items"], ax=axes[row, col],
                                 plot_name=None,  attr_name_to_title=attr_name_to_title,
                                 **plot_kwargs)
            row_names = [x[0] for x in sorted_plot_config_items]
            add_headers(fig, row_headers=row_names, col_headers=col_names)
        elif this_level_name == "row" or this_level_name == "col":
            assert next_level_names[0] == "line"
            if this_level_name == "col":
                axs = axes[0, :]
            else:
                axs = axes[:, 0]

            for ax, (subplot_name, subplot_data) in zip(axs, sorted_plot_config_items):
                plot_ax_func(plot_config=subplot_data["items"], ax=ax,
                             plot_name=subplot_name,
                             attr_name_to_title=attr_name_to_title,
                             **plot_kwargs)
        else:
            raise ValueError

    fig.tight_layout()

    return fig


def save_figures_from_nested_structure(
        plot_config: dict,
        plot_ax_func,
        attrs_groups_list: list[Optional[set]],
        level_names: list[str],
        base_folder='',
        attr_name_to_title: dict[str, str] = {},
        **plot_kwargs):
    this_attrs_group = attrs_groups_list[0]
    next_attrs_groups = attrs_groups_list[1:]
    this_level_name = level_names[0]
    next_level_names = level_names[1:]

    if this_attrs_group:
        save_json({"attrs": list(this_attrs_group)},
                    os.path.join(base_folder, "attrs.json"), indent=2)

    if this_level_name == "folder":
        for folder_name, data in plot_config.items():
            items = data["items"]
            dirname = os.path.join(base_folder, folder_name)
            
            if "vals" in data:
                save_json(data["vals"], os.path.join(dirname, "vals.json"), indent=2)

            save_figures_from_nested_structure(
                items, plot_ax_func, next_attrs_groups, next_level_names,
                base_folder=dirname,
                attr_name_to_title=attr_name_to_title,
                **plot_kwargs
            )
    elif this_level_name == "fname":        
        info_dict = {}
        # print(f"{plot_config=}")
        # exit()
        for fname_desc, data in plot_config.items():
            items = data["items"]
            if "vals" in data:
                info_dict[fname_desc] = data["vals"]

            fig = _get_figure_from_nested_structure(
                items, plot_ax_func, attr_name_to_title, next_attrs_groups,
                next_level_names, fname_desc, **plot_kwargs)
            
            fname = f"{fname_desc}.pdf"
            fpath = os.path.join(base_folder, fname)
            fig.savefig(fpath, dpi=300, format='pdf', bbox_inches='tight')
            plt.close(fig)
        
        if info_dict:
            save_json(info_dict,
                      os.path.join(base_folder, "vals_per_figure.json"), indent=2)
    else:
        raise ValueError


def _plot_key_value_to_str(k, v):
    if k == "attr_name":
        return (2, v)
    if k != "nn.lamda" and k.startswith("nn."):
        k = k[3:]
    return (1, f"{k}={v}")


def plot_dict_to_str(d):
    for key_name, prefix, plot_name in [
        ("nn.method", "nn.", "NN, method="),
        ("method", "", "method="),
        ("gp_af", "gp_af.", "GP, ")
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
            if method == "random search":
                ret = method
            else:
                ret = f"{plot_name}{method}"
            if d_method:
                s = dict_to_str(d_method, include_space=True)
                ret += f" ({s})"
            if d_non_method:
                s = dict_to_str(d_non_method, include_space=True)
                ret += f", {s}"
            return ret
    
    items = [
        _plot_key_value_to_str(k, v)
        for k, v in d.items()
    ]
    items = sorted(items)
    return ", ".join([item[1] for item in items])


def add_plot_args(parser):
    plot_group = parser.add_argument_group("Plotting organization")
    plot_group.add_argument(
        '--use_cols', 
        action='store_true',
        help='Whether to use columns for subplots in the plots'
    )
    plot_group.add_argument(
        '--use_rows', 
        action='store_true',
        help='Whether to use rows for subplots in the plots'
    )
    plot_group.add_argument(
        '--plots_group_name',
        type=str,
        help='Name of group of plots',
    )
    plot_group.add_argument(
        '--plots_name',
        type=str,
        help='Name of these plots'
    )


def create_plot_directory(plots_name=None, plots_group_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [timestamp]
    if plots_name is not None:
        parts = [plots_name] + parts
    folder_name = "_".join(parts)
    pp = [PLOTS_DIR] + (
        [plots_group_name] if plots_group_name else []
    ) + [BO_PLOTS_FOLDER, folder_name]
    save_dir = os.path.join(*pp)
    print(f"Saving plots to {save_dir}")
    return save_dir
