
from typing import Optional, List
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import torch.distributions as dist
from acquisition_dataset import AcquisitionDataset
from acquisition_function_net import AcquisitionFunctionNet, LikelihoodFreeNetworkAcquisitionFunction
from predict_EI_simple import calculate_EI_GP


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
            aq_fn = LikelihoodFreeNetworkAcquisitionFunction.from_net(
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
