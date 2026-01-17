import copy
from typing import Literal, Optional, List

from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy.stats as stats
from scipy.special import softplus
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, to_rgba_array

import torch
from torch import Tensor
import torch.distributions as dist
from botorch.models.gpytorch import GPyTorchModel

from utils.utils import get_lamda_for_bo_of_nn
from utils.constants import RUN_PLOTS_FOLDER
from datasets.acquisition_dataset import AcquisitionDataset
from utils_train.acquisition_function_net import AcquisitionFunctionNet, AcquisitionFunctionNetAcquisitionFunction, ExpectedImprovementAcquisitionFunctionNet
from utils.constants import PLOTS_DIR
from utils.exact_gp_computations import calculate_EI_GP, calculate_gi_gp
from utils.plot_sorting import plot_key_value_to_str, sort_key_for_grouped_items
from utils.utils import add_outcome_transform
from utils_general.plot_utils import get_plot_utils_namespace
from utils_general.utils import DEVICE, dict_to_str


BLUE = '#1f77b4'
ORANGE = '#ff7f0e'


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
        shade=True,
        add_grid=False,
        add_markers=False):
    min_x, max_x = 0, len(center) - 1
    x = list(range(min_x, max_x + 1))
    if shade:
        tmp = dict(marker='o', markersize=3.5) if add_markers else {}
        ax.plot(x, center, label=label, **tmp)
        ax.fill_between(x, lower, upper, alpha=0.3)
    else:
        ax.errorbar(x, center, yerr=[center-lower, upper-center],
                    fmt='-o', capsize=4, markersize=5, capthick=None, label=label)
    if add_grid:
        ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(
        integer=True, nbins='auto', steps=[1, 2, 5, 10], prune=None))
    
    ticks = ax.get_xticks()

    def has_val(val, arr, tol=0.0):
        return np.any(np.isclose(arr, val, rtol=0, atol=tol))

    new_ticks = list(ticks)
    if not has_val(min_x, ticks):
        new_ticks.append(min_x)
    if not has_val(max_x, ticks):
        new_ticks.append(max_x)
    new_ticks = np.array(new_ticks)

    # Make sure within limits. e.g. if min_x=0, max_x=23, then
    # ax.get_xticks() = [-5.,  0.,  5., 10., 15., 20., 25.]
    new_ticks = new_ticks[(new_ticks >= min_x) & (new_ticks <= max_x)]
    
    ax.set_xticks(new_ticks)


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
    ax.fill_between(test_x, lower, upper,
                    color=color, alpha=0.3, label=f'Confidence{extension}')

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
        aq_dataset:AcquisitionDataset, nn_model:AcquisitionFunctionNet, plot_name:str,
        n_candidates:int, nrows:int, ncols:int,
        method:str,
        gp_fit_methods: list[Literal['map', 'mle', 'exact']]=['exact'],
        min_x:float=0., max_x:float=1.,
        lamda:Optional[float]=None,  # for Gittins
        nn_device:Optional[torch.device]=None,
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
        gp_fit_methods: The GP fit methods to use.
        nn_device: The device to use for the neural network.
        group_standardization: Whether to standardize the acquisition functions together.
        Default is to not standardize them together (group_standardization=False)
        if method='policy_gradient' 
        standardize them together (preserve their relaive values)
        (group_standardization=True) otherwise.
        Setting this to True makes it so they can be compared on the same scale,
        which is applicable if predicting the improvement with MSE, but not
        if using policy gradient.
    """
    if lamda is None:
        lamda = 0.01

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5*ncols, 2.5*nrows),
                            sharex=True, sharey=False)

    it = iter(aq_dataset)
    for row in range(nrows):
        for col in range(ncols):
            item = next(it)
            # Get the data to plot
            x_hist, y_hist, x_cand, vals_cand = item.tuple_no_model
            x_cand_original = x_cand
            x_cand = torch.linspace(0, 1, n_candidates).unsqueeze(1)
            # Get the GP model
            gp_model = item.model if item.has_model else None
            plot_nn_vs_gp_acquisition_function_1d(
                ax=axs[row, col], x_hist=x_hist, y_hist=y_hist, x_cand=x_cand,
                x_cand_original=x_cand_original, vals_cand=vals_cand,
                lamda=lamda, gp_model=gp_model, nn_model=nn_model, method=method,
                gp_fit_methods=gp_fit_methods,
                min_x=min_x, max_x=max_x,
                nn_device=nn_device, group_standardization=group_standardization,
                give_legend=False
            )
    
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


_FIT_METHOD_TO_INFO = {
    'exact': {"label": "True GP", "color": "green"},
    'map': {"label": "MAP GP", "color": "red"},
    'mle': {"label": "MLE GP", "color": "blue"},
    'nn': {"label": "NN", "color": ORANGE},
}

def plot_nn_vs_gp_acquisition_function_1d(
        ax, x_hist, y_hist, x_cand,
        gp_model: Optional[GPyTorchModel],
        nn_model: Optional[AcquisitionFunctionNet],
        method:str,
        gp_fit_methods: list[Literal['map', 'mle', 'exact']]=['exact'],
        min_x:float=0., max_x:float=1.,
        x_cand_original=None, vals_cand=None,
        lamda:float=0.01,  # for Gittins
        nn_device:Optional[torch.device]=None,
        group_standardization:Optional[bool]=None,
        give_legend=True,
        varying_index=0,
        log_ei=True,
        constant_y_hist_val=0.0,
        objective=None):
    r"""Plot the acquisition function of a neural network and a GP.
    
    Args:
        ax: The axis to plot on.
        x_hist: The history of x values. shape: `n_hist x d`
        y_hist: The history of y values. shape: `n_hist x n_hist_out`
        x_cand: The candidate x values. shape: `n_cand_plot x d`
        gp_model: The GP model to use.
        nn_model: The neural network model to use.
        method: The method to use for the acquisition function.
        min_x: The minimum x value to plot.
        max_x: The maximum x value to plot.
        x_cand_original: The original candidate x values. shape: `n_cand_dataset x d`
        vals_cand: The original candidate y values. shape: `n_cand_dataset x 1` (usually)
        nn_device: The device to use for the neural network.
        group_standardization: Whether to standardize the acquisition functions together.
    """
    dimension = x_hist.size(1)

    plot_data_y = method == 'gittins' # could change this
    plot_data_x = True
    
    # if dimension != 1:
    #     plot_data_y = False
    
    if group_standardization is None:
        group_standardization = method != 'policy_gradient'

    arrs_and_fit_methods_to_plot = []

    # Compute GP EI acquisition function
    if gp_model is not None:
        try:
            use_ei = method == 'mse_ei' or method == 'policy_gradient'
            if use_ei:
                kwargs = dict(log=log_ei)
                fnc = calculate_EI_GP
            elif method == 'gittins':
                kwargs = dict(lambda_cand=torch.tensor(lamda))
                fnc = calculate_gi_gp
            else:
                raise NotImplementedError(f"Unknown method {method}")

            for j, fit_method in enumerate(gp_fit_methods):
                gp_model_ = copy.deepcopy(gp_model)
                gp_model_.set_train_data_with_transforms(
                    x_hist, y_hist, strict=False, train=False)
                if j == 0:
                    posterior = gp_model_.posterior(x_cand, observation_noise=False)

                if fit_method == 'exact':
                    kk = dict(fit_params=False, mle=False)
                elif fit_method == 'mle':
                    kk = dict(fit_params=True, mle=True)
                elif fit_method == 'map':
                    kk = dict(fit_params=True, mle=False)
                else:
                    raise ValueError(f"Unknown fit method {fit_method}")

                af = fnc(gp_model_, x_hist, y_hist, x_cand, **kwargs, **kk)

                if use_ei and log_ei:
                    af = torch.exp(af)
                
                af = af.detach().numpy()
            
                arrs_and_fit_methods_to_plot.append((af, fit_method))
            
            plot_gp = dimension == 1
        except NotImplementedError:
            # NotImplementedError can happen when:
            # -- Outcome transform computations are not implemented.
            # -- The GP AF value for the method is not implemented.
            plot_gp = False
    else:
        plot_gp = False
    
    if nn_model is not None:
        # Compute NN acquisition function
        x_hist_nn = x_hist.to(nn_device)
        y_hist_nn = y_hist.to(nn_device)
        x_cand_nn = x_cand.to(nn_device)
        if method == 'mse_ei' or method == 'policy_gradient':
            kwargs = dict(exponentiate=(method == 'mse_ei'), softmax=False)
        else:
            kwargs = {}
        aq_fn = AcquisitionFunctionNetAcquisitionFunction.from_net(
            nn_model, x_hist_nn, y_hist_nn, **kwargs)
        ei_nn = aq_fn(x_cand_nn.unsqueeze(1)).detach().cpu().numpy()
        arrs_and_fit_methods_to_plot.append((ei_nn, "nn"))
    
    # Sort the x_cand and arrs
    x_cand_plot_component = x_cand[:, varying_index].detach().numpy()
    sorted_indices = np.argsort(x_cand_plot_component)
    sorted_x_cand = x_cand.detach().numpy()[sorted_indices]
    sorted_x_cand_plot_component = x_cand_plot_component[sorted_indices]

    arrs, fit_methods = zip(*arrs_and_fit_methods_to_plot)
    if plot_data_y and method != 'gittins':
        arrs = standardize_arrs(arrs, group=group_standardization)
    for arr, fit_method in zip(arrs, fit_methods):
        label = _FIT_METHOD_TO_INFO[fit_method]
        sorted_arr = arr[sorted_indices]
        ax.plot(sorted_x_cand_plot_component, sorted_arr, **label)
    
    if objective is not None:
        objective_vals = objective(
            torch.from_numpy(sorted_x_cand)
        )
        print(f"{objective_vals.dim()=}")
        if objective_vals.dim() > 1: # = 2
            objective_vals = objective_vals[:, 0]
        objective_vals = objective_vals.detach().numpy()

    if plot_data_y or plot_data_x:
        x_hist_varying_index = x_hist[:, varying_index]
        if plot_data_y:
            ax.plot(x_hist_varying_index, y_hist[:, 0], 'b*',
                    markersize=10.0, label='History points')
            if objective is not None:
                # Plot the objective function
                ax.plot(sorted_x_cand_plot_component, objective_vals,
                        'k--', label='Objective function')
        if plot_data_x:
            # yvals = torch.full_like(x_hist_varying_index, constant_y_hist_val)
            # ax.plot(x_hist_varying_index, yvals, 'b*', label='History points')
            for i, xval in enumerate(x_hist_varying_index):
                kk = dict(ymin=0, ymax=1, color='blue', alpha=0.5)
                if i == 0:
                    kk['label'] = 'History points'
                ax.axvline(xval, **kk)
        
    if objective is not None:
        argmax_objective = np.argmax(objective_vals)
        ax.axvline(sorted_x_cand_plot_component[argmax_objective],
                   color='k', linestyle='-', label='Max objective')

    if x_cand_original is not None and vals_cand is not None:
        if dimension == 1:
            ax.plot(x_cand_original, vals_cand[:, 0],
                    'ro', markersize=1, label=f'Candidate points')
    elif not (x_cand_original is None and vals_cand is None):
        raise ValueError("Either both or neither of x_cand_original and vals_cand "
                         "should be provided")
    
    if plot_data_y and plot_gp:
        fit_method = gp_fit_methods[0]
        fit_method_info = _FIT_METHOD_TO_INFO[fit_method]
        plot_gp_posterior(
            ax, posterior, x_cand, x_hist, y_hist,
            color=fit_method_info['color'],
            name=f'GP posterior (fit={fit_method_info["label"]})')

    ax.set_xlim(min_x - 0.02 * (max_x - min_x), max_x + 0.02 * (max_x - min_x))

    if give_legend:
        ax.legend(loc='upper right')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Acquisition function value')
    ax.set_title(f'Acquisition function ({method})')
    ax.grid(True)

    return arrs, fit_methods


def _get_gp_stat_info(training_history_data, plot_maxei):
    stats_epochs = training_history_data['stats_epochs']
    stat_name = 'maxei' if plot_maxei else training_history_data['stat_name']
    
    # Determine the GP test stat
    if stat_name == 'mse' or stat_name == 'maxei':
        gp_stat_name = 'true_gp_ei_' + stat_name
        is_loss = stat_name == 'mse'
    elif stat_name == 'ei_softmax':
        gp_stat_name = 'true_gp_ei_maxei'
        is_loss = False
    elif stat_name.startswith('gittins_loss'):
        gp_stat_name = 'true_gp_gi_' + stat_name
        is_loss = True
    else:
        raise ValueError
    try:
        gp_test_stat = [epoch['test'][gp_stat_name] for epoch in stats_epochs]
        # Validate that all the values in `gp_test_stat` are the same:
        assert all(x == gp_test_stat[0] for x in gp_test_stat)
        gp_test_stat = gp_test_stat[0]
    except KeyError:
        gp_test_stat = None
    return stats_epochs, stat_name, gp_stat_name, is_loss, gp_test_stat


def plot_acquisition_function_net_training_history_ax(
        ax, training_history_data, plot_maxei=False, plot_log_regret=False,
        plot_name=None, label='', color=None, alpha=1.0):
    stats_epochs, stat_name, gp_stat_name, is_loss, gp_test_stat = _get_gp_stat_info(
        training_history_data, plot_maxei)
    
    if label:
        label = f' ({label})'
    test_stat = np.array([epoch['test'][stat_name] for epoch in stats_epochs])
    
    # array [R, G, B, A]
    test_color = to_rgba_array(
        _FIT_METHOD_TO_INFO['nn']['color'] if color is None else color, alpha=alpha)[0]
    if plot_log_regret:
        if gp_test_stat is None:
            s = stats_epochs[0]['test']
            raise ValueError(
                f"Need to have GP test stat '{gp_stat_name}' to plot regret. "
                f"Fist test stats: {s}")
        # regret_data = np.abs(test_stat - gp_test_stat)
        regret_data = test_stat - gp_test_stat
        regret_data = -regret_data if regret_data[0] < 0 else regret_data
        # regret_data = np.maximum(differences, 1e-15)
        to_plot = {
            'lines': [
                {
                    'label': 'Regret (NN)' + label,
                    'data': regret_data,
                    'color': test_color
                }
            ],
            'title': f'Test {stat_name} vs Epochs (regret)',
            'xlabel': 'Epochs',
            'ylabel': f'{stat_name} regret',
            'log_scale_x': True,
            'log_scale_y': True
        }
    else:
        train_stat = np.array(
            [epoch['train']['after_training'][stat_name] for epoch in stats_epochs])
        to_plot = {
            'lines': [
                {
                    'label': 'Train (NN)' + label,
                    'data': train_stat,
                    'color': to_rgba_array(BLUE if color is None else color, alpha=alpha)[0]
                },
                {
                    'label': 'Test (NN)' + label,
                    'data': test_stat,
                    'color': test_color
                }
            ],
            'title': f'Train and Test {stat_name} vs Epochs',
            'xlabel': 'Epochs',
            'ylabel': stat_name,
            'log_scale_x': True,
            'log_scale_y': is_loss
        }
        if color is not None:
            # Make the train line dashed to distinguish it
            to_plot['lines'][0]['linestyle'] = '--'
            # Also add some transparency
            to_plot['lines'][0]['color'][3] *= 0.7

        if gp_test_stat is not None:
            to_plot['consts'] = [
                {
                    'label': f'Test (true GP value)',
                    'data': gp_test_stat,
                    'color': 'k',
                    'linestyle': '--',
                }
            ]

    epochs = np.arange(1, len(test_stat) + 1)

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
    if to_plot['log_scale_x']:
        ax.set_xscale('log')
    if to_plot['log_scale_y']:
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
        (stats_epochs, stat_name, gp_stat_name,
         is_loss, gp_test_stat) = _get_gp_stat_info(training_history_data, plot_maxei)
        if gp_test_stat is not None:
            plot_log_regrets.append(True)

    n_rows, n_cols = 1, len(plot_log_regrets)
    fig, axs = plt.subplots(n_rows, n_cols,
                             figsize=(width * n_cols, height * n_rows),
                             sharex=False, sharey=False,
                             squeeze=False)
    axs = axs.flatten()

    for ax, plt_log_regret in zip(axs, plot_log_regrets):
        plot_acquisition_function_net_training_history_ax(
            ax=ax,
            training_history_data=training_history_data,
            plot_maxei=plot_maxei,
            plot_log_regret=plt_log_regret
        )

    return fig


def plot_bo_violin(ax, arr, color='#1f77b4', alpha=0.5, width=0.8):
    """
    Plots vertical violin plots over iterations.

    Parameters:
    - ax: matplotlib axis
    - arr: np.array of shape (n_seeds, n_iter)
    - color: color of the violins (hex string or color name)
    - alpha: transparency level (0 = transparent, 1 = opaque)
    - width: width of each violin
    """
    n_seeds, n_iter = arr.shape
    positions = np.arange(n_iter)

    parts = ax.violinplot(
        [arr[:, i] for i in range(n_iter)],
        positions=positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Set color and transparency
    for body in parts['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)


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
            for k, v in data["items"].items():
                data_index = v["items"]
                if isinstance(data_index, list):
                    print([get_result_func(i) for i in data_index])
                    raise ValueError

                info = get_result_func(data_index)
                if info is not None:
                    attr_name = info['attr_name']
                    attr_names.add(attr_name)
                    val = info[attr_name]
                    if attr_name in {
                        'best_y', 'regret', 'normalized_regret', 'y', 'x', 'best_x'}:
                        # For x vals, get the first component (if dim > 1)
                        if len(val.shape) != 2:
                            raise ValueError(f"Expected 2D array, but got {val.shape=}")
                        if attr_name in {'best_y', 'regret', 'normalized_regret', 'y'} \
                                        and val.shape[1] != 1:
                            raise ValueError(
                                f"Expected {attr_name} to have shape (n_seeds, 1), "
                                f"but got {val.shape=}")
                        val = val[:, 0]
                    else:
                        assert len(val.shape) == 1

                    # Clip regret values to prevent extremely small values from compressing the plot
                    if attr_name in {'regret', 'normalized_regret'}:
                        min_regret = plot_kwargs.get('min_regret_for_plot', 1e-6)
                        val = np.maximum(val, min_regret)

                    val_id = f"{info['index']}_{attr_name}"
                    this_ids.append(val_id)
                    this_data.append(val)

            if len(this_ids) == 0:
                continue

            if attr_name == 'x':
                # Only plot one seed
                x_vals = this_data[0]
                ax.scatter(range(len(x_vals)), x_vals,
                           label=legend_name, s=4, alpha=0.5)
            else:
                center, lower, upper = _get_error_lines(
                    this_ids, this_data,
                    alpha=plot_kwargs['alpha'],
                    interval_of_center=plot_kwargs['interval_of_center'],
                    center_stat=plot_kwargs['center_stat'],
                    assume_normal=plot_kwargs['assume_normal']
                )
                plot_error_bars(ax, center, lower, upper,
                                label=legend_name, shade=plot_kwargs['shade'],
                                add_grid=plot_kwargs.get('add_grid', False),
                                add_markers=plot_kwargs.get('add_markers', False))
                # ax.set_xscale('log')
                if attr_name in {'regret', 'normalized_regret'}:
                    ax.set_yscale('log')
                    
                # plot_bo_violin(ax, np.array(this_data))
        
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


def get_plot_ax_loss_vs_regret_func(get_result_func):
    """
    Create a scatter plot showing validation loss vs. mean BO regret.

    This function creates a plot where:
    - X axis: Log(mean regret after N iterations of BO), averaged across seeds
    - Y axis: Log(best validation loss from training)
    - Each point represents a different configuration
    """
    def ret(plot_config: dict,
            ax,
            plot_name: Optional[str]=None,
            attr_name_to_title: dict[str, str] = {},
            **plot_kwargs):

        # Extract parameters from plot_kwargs
        max_iterations_to_plot = plot_kwargs.get('max_iterations_to_plot', -1)  # -1 means final iteration
        center_stat = plot_kwargs.get('center_stat', 'mean')
        scale = plot_kwargs.get('scale', 1.0)
        s_default = scale * 80.0  # marker size
        plot_mode = plot_kwargs.get('plot_mode', 'scatter')  # 'scatter' or 'density'

        # Storage for plot data
        x_data = []  # regret values
        y_data = []  # validation loss values
        labels = []
        colors = []

        def process_data_index(data_index, legend_name, data):
            """Helper function to extract and process a single data index or aggregate multiple indices"""
            if isinstance(data_index, list):
                # Multiple indices - aggregate them into a single point
                # Collect all regret values at the specified iteration
                all_regret_values = []
                validation_loss = None

                for idx in data_index:
                    info = get_result_func(idx)
                    if info is None:
                        continue

                    # Extract BO regret data
                    if 'normalized_regret' in info:
                        regret_data = info['normalized_regret']
                        if len(regret_data.shape) != 2:
                            raise ValueError(f"Expected 2D regret array, got {regret_data.shape=}")

                        # regret_data has shape (n_seeds, n_iterations+1)
                        # Use the last available iteration if requested iteration is out of bounds
                        iter_to_use = max_iterations_to_plot if max_iterations_to_plot >= 0 else -1
                        if iter_to_use >= regret_data.shape[1]:
                            iter_to_use = -1

                        # Extract regret at the specified iteration for all seeds in this run
                        regret_at_iter = regret_data[:, iter_to_use]
                        all_regret_values.extend(regret_at_iter.flatten())

                    # Extract validation loss (same for all indices since they share the same model)
                    if validation_loss is None and 'training_history_data' in info:
                        training_history = info['training_history_data']
                        stats_epochs = training_history.get('stats_epochs', [])

                        if len(stats_epochs) > 0:
                            stat_name = training_history.get('stat_name', 'mse')
                            test_losses = [epoch['test'][stat_name] for epoch in stats_epochs]
                            validation_loss = np.min(test_losses)

                if len(all_regret_values) == 0:
                    print(f"Warning: No regret data found for {legend_name}")
                    return

                if validation_loss is None:
                    print(f"Warning: No validation loss found for {legend_name}")
                    return

                # Compute mean or median across all seeds and objectives
                if center_stat == 'mean':
                    aggregated_regret = np.mean(all_regret_values)
                elif center_stat == 'median':
                    aggregated_regret = np.median(all_regret_values)
                else:
                    raise ValueError(f"Unknown center_stat: {center_stat}")

                # Clip small values to avoid log(0)
                min_regret = plot_kwargs.get('min_regret_for_plot', 1e-6)
                min_loss = plot_kwargs.get('min_loss_for_plot', 1e-10)
                aggregated_regret = max(aggregated_regret, min_regret)
                validation_loss = max(validation_loss, min_loss)

                # Add aggregated point to plot data
                x_data.append(aggregated_regret)
                y_data.append(validation_loss)
                labels.append(legend_name)
                colors.append(data.get('color', BLUE))
                return

            info = get_result_func(data_index)
            if info is None:
                return

            # Extract BO regret data
            if 'normalized_regret' in info:
                regret_data = info['normalized_regret']
                if len(regret_data.shape) != 2:
                    raise ValueError(f"Expected 2D regret array, got {regret_data.shape=}")

                # regret_data has shape (n_seeds, n_iterations+1)
                # Extract regret at the specified iteration across all seeds
                n_iterations_available = regret_data.shape[1] - 1  # -1 because includes initial point

                # Use the last available iteration if requested iteration is out of bounds
                iter_to_use = max_iterations_to_plot if max_iterations_to_plot >= 0 else -1
                if iter_to_use >= regret_data.shape[1]:
                    # Out of bounds - use last iteration
                    iter_to_use = -1

                regret_at_iter = regret_data[:, iter_to_use]

                # Compute mean or median across seeds
                if center_stat == 'mean':
                    mean_regret = np.mean(regret_at_iter)
                elif center_stat == 'median':
                    mean_regret = np.median(regret_at_iter)
                else:
                    raise ValueError(f"Unknown center_stat: {center_stat}")

                # Clip small values to avoid log(0)
                min_regret = plot_kwargs.get('min_regret_for_plot', 1e-6)
                mean_regret = max(mean_regret, min_regret)
            else:
                print(f"Warning: No normalized_regret found for {legend_name}")
                return

            # Extract validation loss from training history
            if 'training_history_data' in info:
                training_history = info['training_history_data']
                stats_epochs = training_history.get('stats_epochs', [])

                if len(stats_epochs) == 0:
                    print(f"Warning: No stats_epochs found for {legend_name}")
                    return

                # Get the stat name (e.g., 'mse' or 'gittins_loss')
                stat_name = training_history.get('stat_name', 'mse')

                # Extract test losses across all epochs
                test_losses = [epoch['test'][stat_name] for epoch in stats_epochs]

                # Use the best (minimum) validation loss
                best_val_loss = np.min(test_losses)

                # Clip small values to avoid log(0)
                min_loss = plot_kwargs.get('min_loss_for_plot', 1e-10)
                best_val_loss = max(best_val_loss, min_loss)
            else:
                print(f"Warning: No training_history_data found for {legend_name}")
                return

            # Add to plot data
            x_data.append(mean_regret)
            y_data.append(best_val_loss)
            labels.append(legend_name)

            # Use line color if available
            line_color = data.get('color', BLUE)
            colors.append(line_color)

        for legend_name, data in plot_config.items():
            items = data["items"]

            # Handle both nested and flat structures
            # Nested: items = {k: {'items': data_index}}
            # Flat: items = data_index (int or list)
            if isinstance(items, dict) and not isinstance(items, (int, list)):
                # Nested structure - iterate over sub-items
                for k, v in items.items():
                    data_index = v["items"]
                    process_data_index(data_index, legend_name, data)
            else:
                # Flat structure - items is directly the data_index
                process_data_index(items, legend_name, data)

        if len(x_data) == 0:
            raise ValueError("No data to plot for loss vs regret scatter plot")

        # Convert to arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Create plot based on mode
        if plot_mode == 'scatter':
            # Scatter plot: one point per configuration
            # Plot all points with the same color and NO legend
            ax.scatter(x_data, y_data, s=s_default, alpha=0.7, color=BLUE)
            # No legend in scatter mode

        elif plot_mode == 'density':
            # 2D density heatmap/histogram
            # Take log of data for better visualization
            log_x = np.log10(x_data)
            log_y = np.log10(y_data)

            # Create 2D histogram
            bins = plot_kwargs.get('density_bins', 30)
            hist, xedges, yedges = np.histogram2d(log_x, log_y, bins=bins)

            # Plot as heatmap
            extent = [10**xedges[0], 10**xedges[-1], 10**yedges[0], 10**yedges[-1]]
            im = ax.imshow(hist.T, origin='lower', extent=extent, aspect='auto',
                          cmap='viridis', interpolation='nearest', norm=plt.matplotlib.colors.LogNorm())

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Count')

        else:
            raise ValueError(f"Unknown plot_mode: {plot_mode}")

        # Set log scales (for scatter mode, density mode handles internally)
        if plot_mode == 'scatter':
            ax.set_xscale('log')
            ax.set_yscale('log')

        # Set labels
        iter_label = f"iteration {max_iterations_to_plot}" if max_iterations_to_plot >= 0 else "final iteration"
        ax.set_xlabel(f'Mean Normalized Regret ({iter_label})')
        ax.set_ylabel('Best Validation Loss')

        # Set title
        mode_str = "Scatter" if plot_mode == 'scatter' else "Density"
        if plot_name:
            ax.set_title(f"{plot_name} (Loss vs Regret - {mode_str})")
        else:
            ax.set_title(f'Validation Loss vs. BO Regret ({mode_str})')

        # Add grid if requested
        if plot_kwargs.get('add_grid', False):
            ax.grid(True, alpha=0.3)

    return ret


N_CANDIDATES_PLOT = 2_000
LOGSCALE_EI_AF_ITERATIONS_PLOTS = True


def get_plot_ax_af_iterations_func(get_result_func):
    def ret(plot_config: dict,
            ax,
            plot_name: Optional[str]=None,
            attr_name_to_title: dict[str, str] = {},
            **plot_kwargs):

        scale = plot_kwargs.get("scale", 1.0)
        s_default = scale * 120.0
    
        if type(plot_config) is int:
            data_index = plot_config
        else:
            k, v = next(iter(plot_config.items()))
            data_index = v["items"]
        
        if isinstance(data_index, list):
            print([get_result_func(i) for i in data_index])
            raise ValueError

        info = get_result_func(data_index)
        if info is None:
            return False
        
        gp_af = info.get('gp_af')
        log_ei = True
        if 'nn_model' in info:
            nn_model = info['nn_model']
            method = info['method'] # like, 'policy_gradient', 'mse_ei', 'gittins'
            if info['objective_gp'] is not None:
                gp_model = info['objective_gp']
                if info['objective_octf'] is not None:
                    add_outcome_transform(gp_model, info['objective_octf'])
                gp_fit_methods = ['exact']
            else:
                gp_model = None
                gp_fit_methods = [] # doesn't matter
        else:
            nn_model = None
            gp_model = info['gp_model']
            
            # af_class = info['af_class']
            # fit_params = info['fit_params']
            
            # This is hacky...hopefully later code will be better
            if gp_af == 'LogEI':
                method = 'mse_ei'
                log_ei = True
            elif gp_af == 'EI':
                method = 'mse_ei'
                log_ei = False
            elif gp_af == 'gittins':
                method = 'gittins'
            else:
                raise ValueError(f"Unknown gp_af {gp_af}")
            gp_fit_methods = [info['gp_af_fit']]
        
        results = info['results']
        all_x_hist = torch.from_numpy(results['x']) # shape (1+n_iterations, dimension)
        all_y_hist = torch.from_numpy(results['y'])
        n_iterations = len(all_x_hist) - 1
        iteration_index = info['attr_name']
        if not (0 <= iteration_index <= n_iterations - 1):
            raise ValueError(f"Invalid iteration index {iteration_index} for "
                                f"{n_iterations=}")
        x_hist = all_x_hist[:iteration_index + 1]
        y_hist = all_y_hist[:iteration_index + 1]
        x_chosen = all_x_hist[iteration_index + 1]

        dimension = x_hist.size(1)

        x_cand_varying_component = torch.linspace(0, 1, N_CANDIDATES_PLOT)
        varying_index = 0
        if dimension == 1:
            # N_CANDIDATES_PLOT x 1
            x_cand = x_cand_varying_component.unsqueeze(1)
        else:
            x_fixed = x_chosen # Might also want to consider random x
            # N_CANDIDATES_PLOT x dimension
            x_cand = x_fixed.repeat(N_CANDIDATES_PLOT, 1)
            x_cand[:, varying_index] = x_cand_varying_component
        
        lamda = get_lamda_for_bo_of_nn(
                info.get('lamda'), info.get('lamda_min'), info.get('lamda_max'))

        # shape (n_iterations,)
        acqf_values = results.get('acqf_value_exponentiated')
        if acqf_values is None:
            acqf_values = results['acqf_value']
            # This is a hack to get the exponentiated value from old results
            # when using the expected improvement acquisition function.
            if nn_model is not None:
                if isinstance(nn_model, ExpectedImprovementAcquisitionFunctionNet):
                    # Assume that the NN uses softplus to make values positive.
                    acqf_values = softplus(acqf_values)
            elif gp_af == 'LogEI':
                acqf_values = np.exp(acqf_values)
        
        ymax = np.max(acqf_values)
        # ymin = 0.0 if method == 'mse_ei' else np.min(acqf_values)
        ymin = None
        if method == 'mse_ei' and LOGSCALE_EI_AF_ITERATIONS_PLOTS:
            ax.set_yscale('log')
            # ymin = 1e-9 if nn_model is not None else 1e-14
            if nn_model is not None:
                ymin = 1e-14
            else:
                # ymin = np.min(acqf_values[acqf_values > 0])
                ymin = 1e-40
            log_ymax = np.log10(ymax)
            log_ymin = np.log10(ymin)
            ymax = 10**(log_ymax + 0.1 * (log_ymax - log_ymin))
            ymin = 10**(log_ymin - 0.1 * (log_ymax - log_ymin))

            ymax_mid = 10**(log_ymax + 0.05 * (log_ymax - log_ymin))
            constant_y_hist_val = ymax_mid
        else:
            current_ymin, current_ymax = ax.get_ylim()
            ymax = ymax + 0.1 * (ymax - current_ymin)
            constant_y_hist_val = 0.0
        
        arrs, fit_methods = plot_nn_vs_gp_acquisition_function_1d(
            ax=ax, x_hist=x_hist, y_hist=y_hist, x_cand=x_cand,
            lamda=lamda,
            gp_model=gp_model, nn_model=nn_model, method=method,
            gp_fit_methods=gp_fit_methods,
            min_x=0.0, max_x=1.0,
            nn_device=DEVICE, group_standardization=None,
            varying_index=varying_index,
            log_ei=log_ei,
            give_legend=False,
            constant_y_hist_val=constant_y_hist_val,
            objective=info.get('objective')
        )
        fit_method_to_af_vals = dict(zip(fit_methods, arrs))

        fit_method = gp_fit_methods[0] if nn_model is None else 'nn'
        
        acqf_value_chosen = acqf_values[iteration_index]
        fit_method_info =  _FIT_METHOD_TO_INFO[fit_method]
        af_color = fit_method_info['color']
        if nn_model is not None:
            for fit_method_gp in gp_fit_methods:
                af_vals_gp = fit_method_to_af_vals[fit_method_gp]
                af_argmax_gp = np.argmax(af_vals_gp)
                acqf_value_chosen_gp = af_vals_gp[af_argmax_gp]
                x_chosen_gp = x_cand[af_argmax_gp]
                fit_method_gp_info = _FIT_METHOD_TO_INFO[fit_method_gp]
                ax.scatter(x_chosen_gp[varying_index], acqf_value_chosen_gp,
                            color=fit_method_gp_info['color'],
                            label=f'Max point ({fit_method_gp_info["label"]})',
                            marker='x', s=s_default)

        ax.scatter(
            x_chosen[varying_index], acqf_value_chosen,
            color=af_color, s=0.35 * s_default,
            label=f"Chosen point ({fit_method_info['label']})")
        
        af_vals_plot = fit_method_to_af_vals[fit_method]
        af_argmax = np.argmax(af_vals_plot)
        x_chosen_max = x_cand[af_argmax]
        acqf_value_chosen_max = af_vals_plot[af_argmax]
        ax.scatter(x_chosen_max[varying_index], acqf_value_chosen_max,
                    color=af_color, s=s_default, marker='x',
                    label=f"Max point ({fit_method_info['label']})")
        
        plot_title = f"{plot_name} -- {gp_af if nn_model is None else method} acquisition function (iteration {iteration_index+1})"
        ax.set_title(plot_title)

        ax.set_ylim(bottom=ymin, top=ymax)

        # ax.legend(loc='lower right') # loc='upper right', loc='best'
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return True

    return ret


PROJECT_PLOT_UTILS = get_plot_utils_namespace(
    special_names={"attr_name", "nn.method", "method", "gp_af"},
    kv_to_str_func=plot_key_value_to_str,
    sort_key_for_grouped_items_func=sort_key_for_grouped_items,
    plots_dir_path=PLOTS_DIR,
    run_plots_folder_name=RUN_PLOTS_FOLDER,
    constant_keys_to_remove={"nn.method", "gp_af"},
)
