import copy
import math
import os
from typing import Any, Optional
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical
from acquisition_function_net import AcquisitionFunctionNet, ExpectedImprovementAcquisitionFunctionNet, GittinsAcquisitionFunctionNet
from dataset_with_models import RandomModelSampler
from exact_gp_computations import calculate_EI_GP_padded_batch, calculate_gi_gp_padded_batch
from tqdm import tqdm
from acquisition_dataset import AcquisitionDataset
from botorch.exceptions import UnsupportedError
from tictoc import tic, toc
from utils import SaveableObject, convert_to_json_serializable, dict_to_hash, int_linspace, calculate_batch_improvement, load_json, probability_y_greater_than_gi_normal, save_json
from datetime import datetime


METHODS = ['mse_ei', 'policy_gradient', 'gittins']
METHODS_STR = ', '.join(f"'{x}'" for x in METHODS)


def check_2d_or_3d_tensors(*tensors):
    tensors = [t.squeeze(2) if t is not None and t.dim() == 3 else t
               for t in tensors]
    shape = None
    for t in tensors:
        if t is not None:
            shape = t.shape
            break
    for t in tensors:
        if t is not None:
            if t.dim() != 2:
                raise ValueError("All tensors must be 2D or 3D tensors")
            if shape is not None and t.shape != shape:
                raise ValueError("All tensors must have the same shape")
    return tensors


def max_one_hot(values, mask=None):
    values, mask = check_2d_or_3d_tensors(values, mask)
    if mask is not None:
        neg_inf = torch.zeros_like(values)
        neg_inf[~mask] = float("-inf")
        values = values + neg_inf
    return F.one_hot(torch.argmax(values, dim=1),
                     num_classes=values.size(1)).double()


def get_average_normalized_entropy(probabilities, mask=None, reduction="mean"):
    """
    Calculates the average normalized entropy of a probability distribution,
    where a uniform distribution would give a value of 1.

    Args:
        probabilities (torch.Tensor), shape (batch_size, n_cand):
            A tensor representing the probability distribution.
            Entries corresponding to masked out values are assumed to be zero.
        mask (torch.Tensor, optional), shape (batch_size, n_cand):
            A tensor representing a mask to apply on the probabilities. 
            Defaults to None, meaning no mask is applied.

    Returns:
        torch.Tensor: The average normalized entropy.
    """
    probabilities, mask = check_2d_or_3d_tensors(probabilities, mask)
    entropy = Categorical(probs=probabilities).entropy()
    if mask is None:
        counts = torch.tensor(probabilities.size(1), dtype=torch.double)
    else:
        counts = mask.sum(dim=1).double()
    normalized_entropies = entropy / torch.log(counts)
    if reduction == "mean":
        return normalized_entropies.mean()
    elif reduction == "sum":
        return normalized_entropies.sum()
    else:
        raise ValueError("'reduction' must be either 'mean' or 'sum'")


def myopic_policy_gradient_ei(probabilities, improvements, reduction="mean"):
    """Calculate the policy gradient expected 1-step improvement.

    Args:
        probabilities (Tensor): The output tensor from the model, assumed to be
            softmaxed. Shape (batch_size, n_cand) or (batch_size, n_cand, 1)
       improvements (Tensor): The improvements tensor.
            Shape (batch_size, n_cand) or (batch_size, n_cand, 1)
        Both tensors are assumed to be padded with zeros.
        Note: A mask is not needed because the padded values are zero and the
        computation works out even if there is a mask.
    """
    probabilities, improvements = check_2d_or_3d_tensors(probabilities, improvements)
    expected_improvements_per_batch = torch.sum(probabilities * improvements, dim=1)
    if reduction == "mean":
        return expected_improvements_per_batch.mean()
    elif reduction == "sum":
        return expected_improvements_per_batch.sum()
    else:
        raise ValueError("'reduction' must be either 'mean' or 'sum'")


def mse_loss(pred_improvements, improvements, mask=None, reduction="mean"):
    """Calculate the MSE loss. Handle padding with mask for the case that
    there is padding. This works because the padded values are both zero
    so (0 - 0)^2 = 0. Equivalent to reduction="mean" if no padding.

    Args:
        pred_improvements (Tensor): The output tensor from the model,
            assumed to be exponentiated. Shape (batch_size, n_cand)
        improvements (Tensor): The improvements tensor.
            Shape (batch_size, n_cand)
        mask (Tensor or None): The mask tensor, shape (batch_size, n_cand)
        reduction (str): The reduction method. Either "mean" or "sum".
    """
    mask_shape = mask.shape if mask is not None else None
    pred_improvements, improvements, mask = check_2d_or_3d_tensors(
        pred_improvements, improvements, mask)

    if reduction not in {"mean", "sum"}:
        raise ValueError("'reduction' must be either 'mean' or 'sum'")

    if mask is None:
        if reduction == "mean":
            return F.mse_loss(pred_improvements, improvements, reduction="mean")
        # reduction == "sum"
        mse_per_batch = F.mse_loss(
            pred_improvements, improvements, reduction="none"
            ).mean(dim=1)
        return mse_per_batch.sum()
    
    counts = mask.sum(dim=1).double()
    mse_per_batch = F.mse_loss(
        pred_improvements, improvements, reduction="none"
        ).sum(dim=1) / counts

    if reduction == "mean":
        return mse_per_batch.mean()
    return mse_per_batch.sum()


def calculate_gittins_loss(pred_gi, y, lamdas, costs=None,
                           normalize=False, known_costs=True,
                           mask=None, reduction="mean"):
    """Calculate the Gittins index loss.

    Args:
        pred_gi (Tensor):
            The predicted Gittins indices. Shape (batch_size, n_cand)
        y (Tensor):
            The y values. Shape (batch_size, n_cand)
        lamdas (Tensor):
            The lambda values. Shape (batch_size, n_cand), or a scalar tensor
        costs (Tensor or None):
            The costs tensor, shape (batch_size, n_cand)
        normalize (bool):
            Whether to normalize the loss function as proposed
        known_costs (bool):
            Whether the costs `costs` are known
            (only applicable if normalize=True and costs != None)
        mask (Tensor or None):
            The mask tensor, shape (batch_size, n_cand)
        reduction (str):
            The reduction method. Either "mean" or "sum".
    """
    if lamdas.dim() == 0:
        pred_gi, y, costs, mask = check_2d_or_3d_tensors(pred_gi, y, costs, mask)
    else:
        pred_gi, y, lamdas, costs, mask = check_2d_or_3d_tensors(
            pred_gi, y, lamdas, costs, mask)
    
    if reduction not in {"mean", "sum"}:
        raise ValueError("'reduction' must be either 'mean' or 'sum'")

    c = lamdas if costs is None else lamdas * costs
    losses = 0.5 * c**2 + c * (pred_gi - y) + 0.5 * F.relu(y - pred_gi)**2

    if normalize:
        normalize_c = c if known_costs else lamdas
        normalization_consts = probability_y_greater_than_gi_normal(
            cbar=normalize_c, sigma=1.0)
        losses = losses / normalization_consts.to(losses)

    if mask is None:
        mean_error_per_batch = losses.mean(dim=1)
    else:
        losses *= mask
        mean_error_per_batch = losses.sum(dim=1) / mask.sum(dim=1).double()

    if reduction == "mean":
        return mean_error_per_batch.mean()
    return mean_error_per_batch.sum()


def compute_maxei(output, improvements, cand_mask, reduction="mean"):
    probs_max = max_one_hot(output, cand_mask)
    return myopic_policy_gradient_ei(probs_max, improvements, reduction).item()


def compute_acquisition_output_batch_stats(
        output, cand_mask, method:str,
        improvements=None,
        y_cand=None, lambdas=None, normalize=None,
        return_loss:bool=False, name:str="", reduction="mean"):
    if not isinstance(return_loss, bool):
        raise ValueError("return_loss should be a boolean")
    if not isinstance(name, str):
        raise ValueError("name should be a string")
    if name != "":
        name = name + "_"

    output_detached = output.detach()

    ret = {}
    if method == "policy_gradient":
        if improvements is None:
            raise ValueError(
                "improvements must be specified for method='policy_gradient'")
        ret[name+"avg_normalized_entropy"] = get_average_normalized_entropy(
            output_detached, mask=cand_mask, reduction=reduction).item()
        ei_softmax = myopic_policy_gradient_ei(
            output if return_loss else output_detached, improvements, reduction)
        ret[name+"ei_softmax"] = ei_softmax.item()
        if return_loss:
            ret[name+"loss"] = -ei_softmax  # Note the negative sign
    elif method == "mse_ei":
        if improvements is None:
            raise ValueError("improvements must be specified for method='mse_ei'")
        mse = mse_loss(output if return_loss else output_detached,
                       improvements, cand_mask, reduction)
        ret[name+"mse"] = mse.item()
        if return_loss:
            ret[name+"loss"] = mse
    elif method == "gittins":
        if y_cand is None:
            raise ValueError("y_cand must be specified for method='gittins'")
        if lambdas is None:
            raise ValueError("lambdas must be specified for method='gittins'")
        if normalize is None and return_loss:
            raise ValueError(
                "normalize must be specified for method='gittins' if return_loss=True")
        normalizes = [False, True] if normalize is None else [normalize]
        for nrmlz in normalizes:
            gittins_loss = calculate_gittins_loss(
                output if return_loss else output_detached, y_cand, lambdas,
                costs=None, normalize=nrmlz, mask=cand_mask, reduction=reduction)
            nam = name + "gittins_loss" + ("_normalized" if nrmlz else "")
            ret[nam] = gittins_loss.item()
            if return_loss:
                ret["loss"] = gittins_loss

    if improvements is not None:
        ret[name+"maxei"] = compute_maxei(output_detached, improvements,
                                            cand_mask, reduction)

    return ret


def print_things(rows, prefix=''):
    # Calculate the maximum width for each column
    col_widths = [
        max(len(row[col_idx]) for row in rows if len(row) > col_idx)
        for col_idx in range(max(map(len, rows)))]

    # Print each row with appropriate spacing
    row_strings = [
        prefix + " ".join(
            col_val.rjust(col_widths[col_idx])
            for col_idx, col_val in enumerate(row)
        ) for row in rows]
    print("\n".join(row_strings))


def print_stat_summary(
        stats, things_to_print, best_stat:Optional[str]=None,
        inverse_ratio=False, sqrt_ratio=False, ratio_name='Ratio'):
    best_val = None if best_stat is None else stats.get(best_stat)
    direct_things_to_print = []
    for stat_key, stat_print_name, print_ratio in things_to_print:
        if stat_key in stats:
            val = stats[stat_key]
            this_thing = [stat_print_name+':', f'{val:>8f}']
            if best_val is not None and print_ratio:
                ratio = best_val / val if inverse_ratio else val / best_val
                if sqrt_ratio:
                    ratio = math.sqrt(ratio)
                this_thing.extend([f'  {ratio_name}:', f'{ratio:>8f}'])
            direct_things_to_print.append(this_thing)
    print_things(direct_things_to_print, prefix="  ")


def print_stats(stats:dict, dataset_name, method, normalize_gi_loss=False):
    print(f'{dataset_name}:\nExpected 1-step improvement:')

    things_to_print = [
        ('ei_softmax', 'NN (softmax)', True),
        ('maxei', 'NN (max)', True),
        ('true_gp_ei_maxei', 'True GP EI', False),
        ('map_gp_ei_maxei', 'MAP GP EI', True),
        ('true_gp_gi_maxei', 'True GP GI', True),
        ('ei_random_search', 'Random search', True),
        ('ei_ideal', 'Ideal', True),
        ('avg_normalized_entropy', 'Avg norm. entropy', False)]
    print_stat_summary(
        stats, things_to_print, best_stat='true_gp_ei_maxei',
        inverse_ratio=False, sqrt_ratio=False)
    
    if method == 'mse_ei':
        print('Improvement MSE:')
        things_to_print = [
            ('mse', 'NN', True),
            ('true_gp_ei_mse', 'True GP EI', False),
            ('map_gp_ei_mse', 'MAP GP EI', True),
            ('mse_always_predict_0', 'Always predict 0', True)]
        print_stat_summary(
            stats, things_to_print, best_stat='true_gp_ei_mse',
            inverse_ratio=True, sqrt_ratio=True, ratio_name='RMSE Ratio')
    elif method == 'gittins':
        print('Gittins index loss:')
        tmp = '_normalized' if normalize_gi_loss else ''
        things_to_print = [
            ('gittins_loss' + tmp, 'NN', True),
            ('true_gp_gi_gittins_loss' + tmp, 'True GP GI', False)
        ]
        print_stat_summary(
            stats, things_to_print, best_stat='true_gp_gi',
            inverse_ratio=True, sqrt_ratio=False)


def print_train_batch_stats(nn_batch_stats, nn_model, method,
                            batch_index, n_batches,
                            reduction="mean", batch_size=None,
                            normalize_gi_loss=False):
    if reduction == "mean":
        assert batch_size is None
    elif reduction == "sum":
        assert batch_size is not None
        # convert sum reduction to mean reduction
        nn_batch_stats = {k: v / batch_size for k, v in nn_batch_stats.items()}
    else:
        raise ValueError("'reduction' must be either 'mean' or 'sum'")

    suffix = ""
    if method == 'policy_gradient':
        prefix = "Expected 1-step improvement"
        avg_normalized_entropy = nn_batch_stats["avg_normalized_entropy"]
        suffix += f", avg normalized entropy={avg_normalized_entropy:>7f}"
        loss_value = nn_batch_stats["ei_softmax"]
    elif method == 'mse_ei':
        prefix = "MSE"
        loss_value = nn_batch_stats["mse"]
    elif method == 'gittins':
        prefix = "Gittins index loss"
        loss_value = nn_batch_stats[
            "gittins_loss" + ("_normalized" if normalize_gi_loss else "")]
    else:
        raise UnsupportedError(f"method '{method}' is not supported")
    if isinstance(nn_model, ExpectedImprovementAcquisitionFunctionNet):
        if nn_model.includes_alpha:
            suffix += f", alpha={nn_model.get_alpha():>7f}"
        if method == 'mse_ei':
            beta = nn_model.get_beta()
            tau = 1 / beta
            suffix += f", tau={tau:>7f}"
            if nn_model.transform.softplus_batchnorm:
                const = nn_model.transform.batchnorm.weight.get_value().item()
                suffix += f", batchnorm constant={const:>7f}"

    print(f"{prefix}: {loss_value:>7f}{suffix}  [{batch_index+1:>4d}/{n_batches:>4d}]")


def get_average_stats(stats_list, batch_reduction:str, total_n_samples=None):
    if batch_reduction == "mean":
        assert total_n_samples is None
        divisor = len(stats_list)
    elif batch_reduction == "sum":
        divisor = total_n_samples
    else:
        raise ValueError("'batch_reduction' must be either 'mean' or 'sum'")
    return {key: sum(d[key] for d in stats_list) / divisor
            for key in stats_list[0]}


def shape_or_none(x):
    return x if x is None else list(x.shape)


def train_or_test_loop(dataloader: DataLoader,
                       nn_model: Optional[AcquisitionFunctionNet]=None,
                       train:Optional[bool]=None,
                       nn_device=None,
                       method:Optional[str]=None, # ONLY used when training NN
                       verbose:bool=True,
                       desc:Optional[str]=None,
                       
                       n_train_printouts:Optional[int]=None,
                       optimizer:Optional[torch.optim.Optimizer]=None,

                       get_true_gp_stats:Optional[bool]=None,
                       get_map_gp_stats:bool=False,
                       get_basic_stats:bool=True,

                       # Only used when method="mse_ei" or "policy_gradient"
                       # (only does anything if method="policy_gradient")
                       alpha_increment:Optional[float]=None,
                       
                       # Only used when method="gittins" and train=True
                       normalize_gi_loss:bool=False):
    if not isinstance(dataloader, DataLoader):
        raise ValueError("dataloader must be a torch DataLoader")
    
    if dataloader.drop_last:
        raise ValueError("dataloader must have drop_last=False")
    
    n_batches = len(dataloader)
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    has_models = dataset.has_models

    n_training_batches = n_batches
    if len(dataset) % batch_size != 0:
        n_training_batches -= 1
    
    if not isinstance(dataset, AcquisitionDataset):
        raise ValueError("The dataloader must contain an AcquisitionDataset")

    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")
    
    if get_true_gp_stats is None:
        get_true_gp_stats = has_models and dataset.data_is_fixed
    if not isinstance(get_true_gp_stats, bool):
        raise ValueError("get_true_gp_stats must be a boolean")
    if not isinstance(get_map_gp_stats, bool):
        raise ValueError("get_map_gp_stats must be a boolean")
    if not isinstance(get_basic_stats, bool):
        raise ValueError("get_basic_stats must be a boolean")
    
    if not has_models:
        if get_true_gp_stats:
            raise ValueError("get_true_gp_stats must be False if no models are present")
        if get_map_gp_stats:
            raise ValueError("get_map_gp_stats must be False if no models are present")
    
    if n_train_printouts == 0:
        n_train_printouts = None
    
    if nn_model is not None: # evaluating a NN model
        if not isinstance(nn_model, AcquisitionFunctionNet):
            raise ValueError("nn_model must be a AcquisitionFunctionNet instance")
        if not isinstance(train, bool):
            raise ValueError("'train' must be a boolean if evaluating a NN model")
        
        if method not in METHODS:
            raise ValueError(
                f"'method' must be one of {METHODS_STR} if evaluating a NN model; "
                f"it was {method}")
        if method == "gittins":
            if not isinstance(nn_model, GittinsAcquisitionFunctionNet):
                raise ValueError("nn_model must be a GittinsAcquisitionFunctionNet "
                                 "instance if method='gittins'")
            if nn_model.costs_in_history:
                raise UnsupportedError("nn_model.costs_in_history=True is currently not"
                                       " supported for method='gittins'")
            if nn_model.cost_is_input:
                raise UnsupportedError("nn_model.cost_is_input=True is currently not"
                                       " supported for method='gittins'")

            if not isinstance(normalize_gi_loss, bool):
                raise ValueError("normalize_gi_loss must be a boolean")
            nnei = False
        elif method == 'policy_gradient' or method == 'mse_ei':
            if not isinstance(nn_model, ExpectedImprovementAcquisitionFunctionNet):
                raise ValueError(
                    "nn_model must be a ExpectedImprovementAcquisitionFunctionNet "
                    "instance if method='policy_gradient' or method='mse_ei'")
            nnei = True
        else:
            raise UnsupportedError(f"method '{method}' is not supported")

        if train:
            if optimizer is None:
                raise ValueError("optimizer must be specified if training")
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise ValueError("optimizer must be a torch Optimizer instance")
            if verbose:
                if n_train_printouts is not None:
                    if not (isinstance(n_train_printouts, int) and n_train_printouts >= 0):
                        raise ValueError("n_train_printouts must be a non-negative integer")
            if alpha_increment is not None:
                if not (isinstance(alpha_increment, float) and alpha_increment >= 0):
                    raise ValueError("alpha_increment must be a positive float")
            nn_model.train()
        else:
            nn_model.eval()

    else: # just evaluating the dataset, no NN model
        if method is not None:
            raise ValueError("'method' must not be specified if not evaluating a NN model")
        if train is not None:
            raise ValueError("'train' must not be specified if not evaluating a NN model")
        if nn_device is not None:
            raise ValueError("'nn_device' must not be specified if not evaluating a NN model")

    # More of an annoyance than anything, so just ignore
    # if not (train and verbose):
    #     if n_train_printouts is not None:
    #         raise ValueError("n_train_printouts can't be be specified if train != True or verbose != True")
    
    if not train:
        if optimizer is not None:
            raise ValueError("optimizer must not be specified if train != True")
        if alpha_increment is not None:
            raise ValueError("alpha_increment must not be specified if train != True")
    
    if verbose:
        if not (desc is None or isinstance(desc, str)):
            raise ValueError("desc must be a string or None if verbose")
    # elif desc is not None:  # Just ignore desc in this case.
    #     raise ValueError("desc must be None if not verbose")
    
    has_true_gp_stats = hasattr(dataset, "_cached_true_gp_stats")
    has_map_gp_stats = hasattr(dataset, "_cached_map_gp_stats")
    has_basic_stats = hasattr(dataset, "_cached_basic_stats")
    if not dataset.data_is_fixed:
        assert not (has_true_gp_stats or has_map_gp_stats or has_basic_stats)
    
    compute_true_gp_stats = get_true_gp_stats and not has_true_gp_stats
    compute_map_gp_stats = get_map_gp_stats and not has_map_gp_stats
    compute_basic_stats = get_basic_stats and not has_basic_stats

    if compute_true_gp_stats:
        true_gp_stats_list = []
    if compute_map_gp_stats:
        map_gp_stats_list = []
    if compute_basic_stats:
        basic_stats_list = []
    if nn_model is not None:
        nn_batch_stats_list = []
    
    # If we are not computing any stats, then don't actually need to go through
    # the dataset. This probably won't happen in practice though because we
    # always will be evaluating the NN. Also make verbose=False in this case.
    if not (compute_true_gp_stats or compute_map_gp_stats or
            compute_basic_stats or nn_model is not None):
        it = []
        verbose = False

    it = dataloader
    if verbose:
        if train and n_train_printouts is not None:
            print(desc)
            print_indices = set(int_linspace(
                0, n_training_batches - 1,
                min(n_train_printouts, n_training_batches)))
        else:
            it = tqdm(it, desc=desc)
        tic(desc)
        
    dataset_length = 0
    for i, batch in enumerate(it):
        x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask = batch.tuple_no_model

        n_out_cand = vals_cand.size(-1)
        if not (n_out_cand == 1 or n_out_cand == 2):
            raise ValueError("Expected either 1 or 2 output values per candidate")

        vals_cand_0 = vals_cand if n_out_cand == 1 else vals_cand[..., 0].unsqueeze(-1)
        if batch.give_improvements:
            improvements = vals_cand_0
        else:
            y_cand = vals_cand_0
            improvements = calculate_batch_improvement(y_hist, y_cand, hist_mask, cand_mask)

        # print("(DEBUG) Mean Improvement:", improvements.mean())

        # print("x_hist shape:", shape_or_none(x_hist), "y_hist shape:", shape_or_none(y_hist), "x_cand shape:", shape_or_none(x_cand), "improvements shape:", shape_or_none(improvements), "hist_mask shape:", shape_or_none(hist_mask), "cand_mask shape:", shape_or_none(cand_mask))

        this_batch_size = improvements.size(0)
        assert this_batch_size <= batch_size
        is_degenerate_batch = this_batch_size < batch_size
        if i != n_batches - 1:
            assert not is_degenerate_batch
        
        dataset_length += this_batch_size

        if has_models:
            models = batch.model

        if nn_model is not None:
            (x_hist_nn, y_hist_nn, x_cand_nn, vals_cand_nn,
             hist_mask_nn, cand_mask_nn) = batch.to(nn_device).tuple_no_model
            
            # Only check this when we are training the NN
            if method == 'gittins':
                if n_out_cand != 2:
                    raise ValueError(f"Gittins index method requires 2 cand-vals (y, lambda), but got {n_out_cand} cand-vals")
            else:
                if n_out_cand != 1:
                    raise UnsupportedError(
                        "Expected 1 output value per candidate for training when method is not 'gittins'")
            
            if batch.give_improvements:
                if method == 'gittins':
                    raise RuntimeError(
                        "Has batch.give_improvements==True but we need the y values for Gittins index loss")
                improvements_nn = vals_cand_nn
            else:
                # y_cand_nn shape: batch x n_cand x 1
                if method == 'gittins':
                    y_cand_nn = vals_cand_nn[..., 0].unsqueeze(-1)
                else:
                    y_cand_nn = vals_cand_nn
                improvements_nn = calculate_batch_improvement(
                    y_hist_nn, y_cand_nn, hist_mask_nn, cand_mask_nn)
            
            if method == 'gittins':
                log_lambdas_nn = vals_cand_nn[..., 1].unsqueeze(-1)
                lambdas_nn = torch.exp(log_lambdas_nn)
            else:
                lambdas_nn = None

            with torch.set_grad_enabled(train and not is_degenerate_batch):
                if method == 'gittins':
                    if nn_model.variable_lambda:
                        lambda_cand_nn = log_lambdas_nn
                    else:
                        lambda_cand_nn = None
                    nn_output = nn_model(
                        x_hist_nn, y_hist_nn, x_cand_nn,
                        lambda_cand=lambda_cand_nn,
                        hist_mask=hist_mask_nn, cand_mask=cand_mask_nn,
                        is_log=True
                    )
                else: # method = 'mse_ei' or 'policy_gradient' (nnei=True)
                    nn_output = nn_model(
                        x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, cand_mask_nn,
                        exponentiate=(method == "mse_ei"),
                        softmax=(method == "policy_gradient"))

                nn_batch_stats = compute_acquisition_output_batch_stats(
                    nn_output, cand_mask_nn, method,
                    improvements=improvements_nn,
                    y_cand=y_cand_nn, lambdas=lambdas_nn, normalize=normalize_gi_loss,
                    return_loss=train, reduction="sum")
                # print("NN", nn_batch_stats)
                # print("(DEBUG) Mean EI NN:", nn_output.mean())
                
                if train and not is_degenerate_batch:
                    # convert sum to mean so that this is consistent across batch sizes
                    loss = nn_batch_stats.pop("loss") / this_batch_size # (== batch_size)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if nnei and nn_model.includes_alpha and alpha_increment is not None:
                        nn_model.set_alpha(nn_model.get_alpha() + alpha_increment)
                    
                    if verbose and n_train_printouts is not None and i in print_indices:
                        print_train_batch_stats(nn_batch_stats, nn_model,
                                                method, i,
                                                n_training_batches,
                                                reduction="sum",
                                                batch_size=this_batch_size,
                                                normalize_gi_loss=normalize_gi_loss)
                        
                        # # Debug code
                        # if method == 'mse_ei':
                        #     nn_output_no_transform = nn_model(
                        #         x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, cand_mask_nn,
                        #         exponentiate=False, softmax=False)
                        #     mean = nn_output_no_transform.mean().item()
                        #     std = nn_output_no_transform.std().item()
                        #     min_val = nn_output_no_transform.min().item()
                        #     max_val = nn_output_no_transform.max().item()
                        #     print(f"(DEBUG) nn_output_no_transform mean={mean:>4f}, "
                        #           f"std={std:>4f}, min={min_val:>4f}, max={max_val:>4f}")
                        #     min_tf = nn_output.min().item()
                        #     max_tf = nn_output.max().item()
                        #     print(f"(DEBUG) nn_output min={min_tf:>4f}, max={max_tf:>4f}")

                            # if has_models:
                            #     ei_values_true_model = calculate_EI_GP_padded_batch(
                            #         x_hist, y_hist, x_cand, hist_mask, cand_mask, models)
                            #     min_ei_gp = ei_values_true_model.min().item()
                            #     max_ei_gp = ei_values_true_model.max().item()
                            #     print(f"(DEBUG) ei_values_true_model min={min_ei_gp:>4f}, max={max_ei_gp:>4f}")

                nn_batch_stats_list.append(nn_batch_stats)

        with torch.no_grad():
            if compute_true_gp_stats:
                # Calculate true GP EI stats
                # print(models[0].training)
                ei_values_true_model = calculate_EI_GP_padded_batch(
                    x_hist, y_hist, x_cand, hist_mask, cand_mask, models)
                # print(f"{ei_values_true_model=}, {ei_values_true_model.shape=}")
                # print(f"{improvements=}, {improvements.shape=}")
                # print(models[0].training, models[0])

                # print(f"Here, {x_cand.shape=}, {vals_cand.shape=}, {cand_mask.shape=}, {improvements.shape=}")

                true_gp_batch_stats = compute_acquisition_output_batch_stats(
                    ei_values_true_model, cand_mask, method='mse_ei',
                    improvements=improvements,
                    return_loss=False, name="true_gp_ei", reduction="sum")

                if n_out_cand == 2: # Gittins index
                    log_lambdas = vals_cand[..., 1].unsqueeze(-1)
                    lambdas = torch.exp(log_lambdas)
                    gi_values_true_model = calculate_gi_gp_padded_batch(
                        models,
                        x_hist, y_hist, x_cand,
                        lambda_cand=lambdas,
                        hist_mask=hist_mask, cand_mask=cand_mask,
                        is_log=False
                    )
                    # normalize=None here means normalize both True and False
                    true_gp_batch_stats_gi = compute_acquisition_output_batch_stats(
                        gi_values_true_model, cand_mask, method='gittins',
                        improvements=improvements,
                        y_cand=y_cand, lambdas=lambdas, normalize=None,
                        return_loss=False, name="true_gp_gi", reduction="sum")
                    true_gp_batch_stats = {**true_gp_batch_stats, **true_gp_batch_stats_gi}

                # print(compute_myopic_acquisition_output_batch_stats(
                #     ei_values_true_model, cand_mask, method='mse_ei',
                #     improvements=improvements, return_loss=False,
                #     name="true_gp_ei", reduction="mean"))
                # print()
                # print("GP", true_gp_batch_stats)
                
                # model0 = models[0]
                # if hasattr(model0, "outcome_transform"):
                #     print("Outcome transform:",
                #           model0.outcome_transform._original_transform.means,
                #           model0.outcome_transform._original_transform.stdvs)
                
                # print("(DEBUG) Mean EI GP:", ei_values_true_model.mean())

                true_gp_stats_list.append(true_gp_batch_stats)
            
            if compute_basic_stats:
                # Calculate the E(I) of selecting a point at random,
                # the E(I) of selecting the point with the maximum I (cheating), and
                # the MSE loss of always predicting 0
                if cand_mask is None:
                    random_search_probs = torch.ones_like(vals_cand_0) / vals_cand_0.size(1)
                else:
                    random_search_probs = cand_mask.double() / cand_mask.sum(
                        dim=1, keepdim=True).double()
                basic_stats_list.append({
                    "ei_random_search": myopic_policy_gradient_ei(
                        random_search_probs, improvements, reduction="sum").item(),
                    "ei_ideal": compute_maxei(improvements, improvements,
                                               cand_mask, reduction="sum"),
                    "mse_always_predict_0": mse_loss(
                        torch.zeros_like(vals_cand_0), improvements, cand_mask,
                        reduction="sum").item()
                })
        # print()
        
        if compute_map_gp_stats: # I'm not updating this part anymore, I don't care
            # Calculate the MAP GP EI values
            ei_values_map = calculate_EI_GP_padded_batch(
                x_hist, y_hist, x_cand, hist_mask, cand_mask, models, fit_params=True)
            map_gp_batch_stats = compute_acquisition_output_batch_stats(
                    ei_values_map, cand_mask, method='mse_ei',
                    improvements=improvements, return_loss=False,
                    name="map_gp_ei", reduction="sum")
            map_gp_stats_list.append(map_gp_batch_stats)
    assert dataset_length == len(dataset)

    if verbose:
        toc(desc)
    
    ret = {}

    if get_true_gp_stats:
        if not has_true_gp_stats:
            true_gp_stats = get_average_stats(true_gp_stats_list, "sum", dataset_length)
            if dataset.data_is_fixed:
                dataset._cached_true_gp_stats = true_gp_stats
        else:
            true_gp_stats = dataset._cached_true_gp_stats
        ret.update(true_gp_stats)
    if get_map_gp_stats:
        if not has_map_gp_stats:
            map_gp_stats = get_average_stats(map_gp_stats_list, "sum", dataset_length)
            if dataset.data_is_fixed:
                dataset._cached_map_gp_stats = map_gp_stats
        else:
            map_gp_stats = dataset._cached_map_gp_stats
        ret.update(map_gp_stats)
    if get_basic_stats:
        if not has_basic_stats:
            basic_stats = get_average_stats(basic_stats_list, "sum", dataset_length)
            if dataset.data_is_fixed:
                dataset._cached_basic_stats = basic_stats
        else:
            basic_stats = dataset._cached_basic_stats
        ret.update(basic_stats)
    
    if nn_model is not None:
        ret.update(get_average_stats(nn_batch_stats_list, "sum", dataset_length))
    
    return ret


import logging

# Set to True to enable debug logging
DEBUG = False

# Based on EarlyStopping class from PyTorch Ignite
# https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping
class EarlyStopper:
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
    ):
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")
        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")
        
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            self.logger.debug("EarlyStopper: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self.logger.info("EarlyStopper: Stop training")
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


BASIC_STATS = {"ei_random_search", "ei_ideal", "mse_always_predict_0"}
def split_nn_stats(stats):
    nn_stats = stats.copy()
    non_nn_stats = {}
    for stat_name in stats:
        if stat_name in BASIC_STATS or stat_name.startswith("true_gp") or stat_name.startswith("map_gp"):
            non_nn_stats[stat_name] = nn_stats.pop(stat_name)
    return nn_stats, non_nn_stats


def train_acquisition_function_net(
        nn_model: AcquisitionFunctionNet,
        train_dataset: AcquisitionDataset,
        optimizer: torch.optim.Optimizer,
        method: str,
        n_epochs: int,
        batch_size: int,
        nn_device=None,
        verbose:bool=True,
        n_train_printouts_per_epoch:Optional[int]=None,

        # Only used when method="mse_ei" or "policy_gradient"
        # (only does anything if method="policy_gradient")
        alpha_increment:Optional[float]=None,
        
        # Only used when method="gittins" and train=True
        normalize_gi_loss:bool=False,
        
        test_dataset: Optional[AcquisitionDataset]=None,
        small_test_dataset:Optional[AcquisitionDataset]=None,
        test_during_training:Optional[bool]=None,

        get_train_stats_while_training:bool=True,
        get_train_stats_after_training:bool=True,
        
        get_train_true_gp_stats:Optional[bool]=None,
        get_train_map_gp_stats:bool=False,
        get_test_true_gp_stats:Optional[bool]=None,
        get_test_map_gp_stats:Optional[bool]=None,

        save_dir:Optional[str]=None,
        save_incremental_best_models:bool=True,
        
        early_stopping:bool=True,
        patience:int=10,
        min_delta:float=0.0,
        cumulative_delta:bool=False
    ):
    if not (isinstance(n_epochs, int) and n_epochs >= 1):
        raise ValueError("n_epochs must be a positive integer")
    if not (isinstance(batch_size, int) and batch_size >= 1):
        raise ValueError("batch_size must be a positive integer")
    if not (test_during_training is None or isinstance(test_during_training, bool)):
        raise ValueError("test_during_training must be a boolean or None")
    if not isinstance(verbose, bool):
        raise ValueError("verbose should be a boolean")

    if not isinstance(get_train_stats_while_training, bool):
        raise ValueError("get_train_stats_while_training should be a boolean")
    if not isinstance(get_train_stats_after_training, bool):
        raise ValueError("get_train_stats_after_training should be a boolean")
    if not (get_train_stats_while_training or get_train_stats_after_training):
        raise ValueError("You probably want to get some train stats...specify "
                         "either get_train_stats_while_training=True or "
                         "get_train_stats_after_training=True or both.")

    test_during_training, test_after_training = get_test_during_after_training(
        test_dataset, small_test_dataset, test_during_training)

    if test_during_training:
        if small_test_dataset is None:
            small_test_dataset = test_dataset
        small_test_dataloader = small_test_dataset.get_dataloader(
            batch_size=batch_size, drop_last=False)

    if test_during_training or test_after_training:
        if get_test_map_gp_stats is None:
            get_test_map_gp_stats = False # default
    elif get_test_true_gp_stats or get_test_map_gp_stats:
        raise ValueError("Can't get GP stats of test dataset because there is none specified")
    
    # If get_train_stats_after_training=True, then we are running through the
    # train dataset twice each epoch. If furthermore the training data is not
    # fixed, then with these two runs through it, the data will be different.
    # But we'd like to have the stats of during training vs after training
    # directly comparable, so we will freeze the data with each epoch.
    fix_train_dataset_each_epoch = get_train_stats_after_training and not train_dataset.data_is_fixed

    # Due to this, need to explicitly set the default value here because train_or_test_loop
    # won't get it right because we'll fix the data even though it isn't fixed
    if get_train_true_gp_stats is None:
        get_train_true_gp_stats = train_dataset.has_models and train_dataset.data_is_fixed

    if not fix_train_dataset_each_epoch:
        train_dataloader = train_dataset.get_dataloader(batch_size=batch_size, drop_last=False)
    
    if save_incremental_best_models and save_dir is None:
        raise ValueError("Need to specify save_dir if save_incremental_best_models=True")
    
    best_score = None
    best_epoch = None
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        training_history_path = os.path.join(save_dir, 'training_history_data.json')
    
    if early_stopping:
        early_stopper = EarlyStopper(patience, min_delta, cumulative_delta)

    training_history_data = {
        'stats_epochs': []
    }

    for t in range(n_epochs):
        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")
        
        if fix_train_dataset_each_epoch:
            # cache_pads=True: Training! took 85.434673 seconds
            # cache_pads=False: Training! took 80.610601 seconds
            train_dataloader = train_dataset.fix_samples(lazy=True) \
                .get_dataloader(batch_size=batch_size, drop_last=False, cache_pads=False)
        
        train_stats = {}
        
        train_stats_while_training = train_or_test_loop(
            train_dataloader, nn_model, train=True,
            nn_device=nn_device, method=method,
            verbose=verbose, desc=f"Epoch {t+1} train",
            n_train_printouts=n_train_printouts_per_epoch,
            optimizer=optimizer,
            alpha_increment=alpha_increment,
            normalize_gi_loss=normalize_gi_loss,
            get_true_gp_stats=get_train_true_gp_stats,
            get_map_gp_stats=get_train_map_gp_stats,
            get_basic_stats=True)

        (train_nn_stats_while_training,
         non_nn_train_stats) = split_nn_stats(train_stats_while_training)
        train_stats['non_nn_stats'] = non_nn_train_stats
        
        if get_train_stats_while_training:
            train_stats['while_training'] = train_nn_stats_while_training
            if verbose:
                print_stats({**train_stats['while_training'], **non_nn_train_stats},
                            "Train stats while training", method, normalize_gi_loss)

        if get_train_stats_after_training:
            train_stats['after_training'] = train_or_test_loop(
                train_dataloader, nn_model, train=False,
                nn_device=nn_device, method=method,
                verbose=verbose, desc=f"Epoch {t+1} compute train stats",
                normalize_gi_loss=normalize_gi_loss,
                # Don't need to compute non-NN stats because already computed them
                # while training, and we ensured that the train dataset is fixed for this epoch.
                get_true_gp_stats=False,
                get_map_gp_stats=False,
                get_basic_stats=False)
            if verbose:
                print_stats({**train_stats['after_training'], **non_nn_train_stats},
                            "Train stats after training", method, normalize_gi_loss)
        
        epoch_stats = {'train': train_stats}

        if test_during_training:
            test_stats = train_or_test_loop(
                small_test_dataloader, nn_model, train=False,
                nn_device=nn_device, method=method,
                verbose=verbose, desc=f"Epoch {t+1} compute test stats",
                normalize_gi_loss=normalize_gi_loss,
                get_true_gp_stats=get_test_true_gp_stats,
                get_map_gp_stats=get_test_map_gp_stats,
                get_basic_stats=True)
            epoch_stats['test'] = test_stats
            if verbose:
                print_stats(test_stats, "Test stats", method, normalize_gi_loss)
        
        training_history_data['stats_epochs'].append(epoch_stats)

        # Determine the maxei statistic. Decreasing order of preference.
        # We would usually only do these if test_during_training=True,
        # but why not cover all cases.
        if test_during_training:
            cur_score = test_stats["maxei"]
        elif get_train_stats_after_training:
            cur_score = train_stats["after_training"]["maxei"]
        else:
            cur_score = train_nn_stats_while_training["maxei"]
        
        # If the best score increased, then update that and maybe save
        if best_score is None or cur_score > best_score:
            prev_best_score = best_score
            best_score = cur_score
            best_epoch = t

            if verbose and prev_best_score is not None:
                msg = (f"Best score increased from {prev_best_score:>8f}"
                        f" to {best_score:>8f}.")

            if save_incremental_best_models:
                fname = f"model_{best_epoch}.pth"
                if verbose and prev_best_score is not None:
                    print(msg + f" Saving weights to {fname}.")
                torch.save(nn_model.state_dict(), os.path.join(save_dir, fname))
            else:
                if verbose and prev_best_score is not None:
                    print(msg)
                # If we don't save the best models during training, then
                # we still want to save the best state_dict so need to
                # keep a deepcopy of the best state_dict.
                best_state_dict = copy.deepcopy(nn_model.state_dict())
        
        if save_dir is not None:
            # Saving every epoch because why not
            save_json(training_history_data, training_history_path, indent=4)
        
        if early_stopping and early_stopper(cur_score):
            if verbose:
                print(
                    "Early stopping at epoch %i; counter is %i / %i" %
                    (t+1, early_stopper.counter, early_stopper.patience)
                )
            break
    
    best_model_fname = f"model_{best_epoch}.pth"
    
    # Load the best model weights to return
    if save_incremental_best_models:
        best_state_dict = torch.load(os.path.join(save_dir, best_model_fname))
    nn_model.load_state_dict(best_state_dict)

    if test_after_training:
        if test_during_training and (test_dataset is small_test_dataset):
            # If we already computed it then don't need to compute again
            final_test_stats = training_history_data['stats_epochs'][best_epoch]['test']
        else:
            # Consider if the test datasets are not fixed.
            # Then the test-after-training default is not good.
            if get_test_true_gp_stats is None and not test_dataset.data_is_fixed \
                and test_dataset.has_models:
                get_test_true_gp_stats_after_training = True
            else:
                get_test_true_gp_stats_after_training = get_test_true_gp_stats
            
            test_dataloader = test_dataset.get_dataloader(
                batch_size=batch_size, drop_last=False)
            final_test_stats = train_or_test_loop(
                test_dataloader, nn_model, train=False,
                nn_device=nn_device, method=method,
                verbose=verbose, desc=f"Compute final test stats",
                normalize_gi_loss=normalize_gi_loss,
                get_true_gp_stats=get_test_true_gp_stats_after_training,
                get_map_gp_stats=get_test_map_gp_stats,
                get_basic_stats=True)
        training_history_data['final_test_stats'] = final_test_stats
        if verbose:
            print_stats(final_test_stats, "Final test stats", method, normalize_gi_loss)
    
    if save_dir is not None:
        save_json(training_history_data, training_history_path, indent=4)

        save_json({"best_model_fname": best_model_fname},
                  os.path.join(save_dir, "best_model_fname.json"))
        if not save_incremental_best_models:
            # Save the best model weights if not already saved
            path = os.path.join(save_dir, best_model_fname)
            torch.save(best_state_dict, path)
    
    return training_history_data


def get_test_during_after_training(
        test_dataset, small_test_dataset, test_during_training):
    if test_dataset is not None:
        if small_test_dataset is not None: # Both test and small-test specified
            # Test during & after training
            if test_during_training == False:
                raise ValueError("Small and big test datasets specified but test_during_training == False")
            test_during_training = True # it can be either None or True
            test_after_training = True
        else: # Only test but not small-test specified
            # Whether to test during & after training is ambiguous
            if test_during_training is None:
                raise ValueError("test but not small-test dataset is specified but test_during_training is not specified")
            if test_during_training:
                # Then the during-train & after-train dataset are the same, and
                # we can test after training too (which is kind of redundant)
                pass # will set small_test_dataset = test_dataset
            test_after_training = True # want to test after training both cases
    else: # Test dataset not specified
        if small_test_dataset is not None: # Small-test but not test specified
            # Test during but not after training
            if test_during_training == False:
                raise ValueError("Small-test but not test specified, but test_during_training == False")
            test_during_training = True # it can be either None or True
            test_after_training = False
        else: # Neither are specified
            # Test neither during nor after training
            if test_during_training:
                raise ValueError("No test datasets were specified but got test_during_training == True")
            test_during_training = False
            test_after_training = False
    return test_during_training, test_after_training


def save_acquisition_function_net_configs(
        models_directory: str,
        model: AcquisitionFunctionNet,
        training_config: dict[str, Any],
        gp_realization_config: dict[str, Any],
        dataset_size_config: dict[str, Any],
        n_points_config: dict[str, Any],
        dataset_transform_config: dict[str, Any],
        model_sampler: RandomModelSampler):
    gp_realization_config_json = convert_to_json_serializable(gp_realization_config)
    dataset_transform_config_json = convert_to_json_serializable(dataset_transform_config)
    model_sampler_json = convert_to_json_serializable({
            '_models': model_sampler._models,
            '_initial_params_list': model_sampler._initial_params_list,
            'model_probabilities': model_sampler.model_probabilities,
            'randomize_params': model_sampler.randomize_params
        },
        include_priors=True, hash_gpytorch_modules=True,
        hash_include_str=False, hash_str=True)
    
    all_info_json = {
        'model': model.get_info_dict(),
        'training_config': training_config,
        'gp_realization_config': gp_realization_config_json,
        'dataset_size_config': dataset_size_config,
        'n_points_config': n_points_config,
        'dataset_transform_config': dataset_transform_config_json,
        'model_sampler': model_sampler_json
    }
    all_info_hash = dict_to_hash(all_info_json)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_and_info_folder_name = f"model_{timestamp}_{all_info_hash}"
    model_and_info_path = os.path.join(models_directory, model_and_info_folder_name)
    os.makedirs(model_and_info_path, exist_ok=True)

    print(f"Saving model and configs to {model_and_info_folder_name}")

    # Save model config
    model_path = os.path.join(model_and_info_path, "model")
    model.save_init(model_path)

    # Save training config
    save_json(training_config,
              os.path.join(model_and_info_path, "training_config.json"))

    # Save GP dataset config
    save_json(gp_realization_config_json,
              os.path.join(model_and_info_path, "gp_realization_config.json"))
    save_json(dataset_size_config,
              os.path.join(model_and_info_path, "dataset_size_config.json"))
    save_json(n_points_config,
              os.path.join(model_and_info_path, "n_points_config.json"))
    model_sampler.save(
        os.path.join(model_and_info_path, "model_sampler"))

    # Save dataset transform config
    save_json(dataset_transform_config_json,
             os.path.join(model_and_info_path, "dataset_transform_config.json"))
    outcome_transform = dataset_transform_config['outcome_transform']
    if outcome_transform is not None:
        torch.save(outcome_transform,
                   os.path.join(model_and_info_path, "outcome_transform.pt"))
    
    return model_and_info_folder_name, model_and_info_path, model_path


def load_model(model_and_info_path: str):
    model_path = os.path.join(model_and_info_path, "model")
    print(f"Loading model from {model_path}")
    
    # Load model (without weights)
    model = SaveableObject.load_init(model_path)
    
    # Load best weights
    best_model_fname_json_path = os.path.join(model_path, "best_model_fname.json")
    if os.path.exists(best_model_fname_json_path):
        best_model_fname = load_json(best_model_fname_json_path)['best_model_fname']
        best_model_path = os.path.join(model_path, best_model_fname)
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        raise ValueError("No best model found")
    
    return model


def load_configs(model_and_info_path: str):
    gp_realization_config = load_json(
        os.path.join(model_and_info_path, "gp_realization_config.json"))
    model_sampler = RandomModelSampler.load(
        os.path.join(model_and_info_path, "model_sampler"))
    if gp_realization_config['models'] is not None:
        gp_realization_config['models'] = model_sampler.initial_models
        gp_realization_config['model_probabilities'] = model_sampler.model_probabilities
        assert model_sampler.randomize_params == gp_realization_config['randomize_params']
    dataset_size_config = load_json(
        os.path.join(model_and_info_path, "dataset_size_config.json"))
    n_points_config = load_json(
        os.path.join(model_and_info_path, "n_points_config.json"))
    
    dataset_transform_config = load_json(
        os.path.join(model_and_info_path, "dataset_transform_config.json"))
    if dataset_transform_config['outcome_transform'] is not None:
        dataset_transform_config['outcome_transform'] = torch.load(
            os.path.join(model_and_info_path, "outcome_transform.pt"))
    
    return gp_realization_config, dataset_size_config, n_points_config, \
        dataset_transform_config, model_sampler
 