from typing import Optional
from tqdm import tqdm
from datasets.acquisition_dataset import AcquisitionDataset
import torch
from torch.utils.data import DataLoader
from botorch.exceptions import UnsupportedError

from utils_general.tictoc import tic, toc
from utils.utils import calculate_batch_improvement
from utils_general.train_utils import (
    get_average_stats, calculate_gittins_loss, GI_NORMALIZATIONS)
from utils_general.utils import int_linspace
from utils.exact_gp_computations import (
    calculate_EI_GP_padded_batch, calculate_gi_gp_padded_batch)
from utils_train.acquisition_function_net import (
    AcquisitionFunctionNet, ExpectedImprovementAcquisitionFunctionNet,
    GittinsAcquisitionFunctionNet)
from utils_train.acquisition_function_net_constants import METHODS
from utils_train.train_utils import compute_maxei, get_average_normalized_entropy, mse_loss, myopic_policy_gradient_ei


_METHODS_STR = ', '.join(f"'{x}'" for x in METHODS)


def _compute_acquisition_output_batch_stats(
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
        normalizes = [None] + GI_NORMALIZATIONS if normalize is None else [normalize]
        for nrmlz in normalizes:
            gittins_loss = calculate_gittins_loss(
                output if return_loss else output_detached, y_cand, lambdas,
                costs=None, normalize=nrmlz, mask=cand_mask, reduction=reduction)
            nam = name + "gittins_loss" + (f"_normalized_{nrmlz}" if nrmlz else "")
            ret[nam] = gittins_loss.item()
            if return_loss:
                ret["loss"] = gittins_loss

    if improvements is not None:
        ret[name+"maxei"] = compute_maxei(output_detached, improvements,
                                          cand_mask, reduction)

    return ret


def _print_train_batch_stats(nn_batch_stats, nn_model, method,
                            batch_index, n_batches,
                            reduction="mean", batch_size=None,
                            gi_loss_normalization=None):
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
            "gittins_loss" + (
                f'_normalized_{gi_loss_normalization}' \
                    if gi_loss_normalization is not None else '')]
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
                       gi_loss_normalization:Optional[str]=None,
                       
                       # Whether to return None if there is nothing to compute
                       return_none=False):
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
                f"'method' must be one of {_METHODS_STR} if evaluating a NN model; "
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
    
    if not train:
        if optimizer is not None:
            raise ValueError("optimizer must not be specified if train != True")
        if alpha_increment is not None:
            raise ValueError("alpha_increment must not be specified if train != True")
    
    if verbose:
        if not (desc is None or isinstance(desc, str)):
            raise ValueError("desc must be a string or None if verbose")
    
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
    
    if not (compute_true_gp_stats or compute_map_gp_stats or
            compute_basic_stats or nn_model is not None):
        if return_none:
            return None
        # If we are not computing any stats, then don't actually need to go through
        # the dataset. Also make verbose=False in this case.
        it = []
        verbose = False
        do_nothing = True
    else:
        it = dataloader
        do_nothing = False
    
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
            raise ValueError("Expected either 1 or 2 output values per candidate, "
                             f"but got {n_out_cand} output values")

        vals_cand_0 = vals_cand if n_out_cand == 1 else vals_cand[..., 0].unsqueeze(-1)
        if batch.give_improvements:
            improvements = vals_cand_0
        else:
            y_cand = vals_cand_0
            improvements = calculate_batch_improvement(y_hist, y_cand, hist_mask, cand_mask)

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

                nn_batch_stats = _compute_acquisition_output_batch_stats(
                    nn_output, cand_mask_nn, method,
                    improvements=improvements_nn,
                    y_cand=y_cand_nn, lambdas=lambdas_nn,
                    normalize=gi_loss_normalization,
                    return_loss=train, reduction="sum")
                
                if train and not is_degenerate_batch:
                    # convert sum to mean so that this is consistent across batch sizes
                    loss = nn_batch_stats.pop("loss") / this_batch_size # (== batch_size)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if nnei and nn_model.includes_alpha and alpha_increment is not None:
                        nn_model.set_alpha(nn_model.get_alpha() + alpha_increment)
                    
                    if verbose and n_train_printouts is not None and i in print_indices:
                        _print_train_batch_stats(nn_batch_stats, nn_model,
                                                method, i,
                                                n_training_batches,
                                                reduction="sum",
                                                batch_size=this_batch_size,
                                                gi_loss_normalization=gi_loss_normalization)

                nn_batch_stats_list.append(nn_batch_stats)

        with torch.no_grad():
            if compute_true_gp_stats:
                # Calculate true GP EI stats
                ei_values_true_model = calculate_EI_GP_padded_batch(
                    x_hist, y_hist, x_cand, hist_mask, cand_mask, models)

                true_gp_batch_stats = _compute_acquisition_output_batch_stats(
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
                    # normalize=None here means normalize all options
                    true_gp_batch_stats_gi = _compute_acquisition_output_batch_stats(
                        gi_values_true_model, cand_mask, method='gittins',
                        improvements=improvements,
                        y_cand=y_cand, lambdas=lambdas, normalize=None,
                        return_loss=False, name="true_gp_gi", reduction="sum")
                    true_gp_batch_stats = {**true_gp_batch_stats, **true_gp_batch_stats_gi}

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
        
        if compute_map_gp_stats: # I'm not updating this part anymore, I don't care
            # Calculate the MAP GP EI values
            ei_values_map = calculate_EI_GP_padded_batch(
                x_hist, y_hist, x_cand, hist_mask, cand_mask, models, fit_params=True)
            map_gp_batch_stats = _compute_acquisition_output_batch_stats(
                    ei_values_map, cand_mask, method='mse_ei',
                    improvements=improvements, return_loss=False,
                    name="map_gp_ei", reduction="sum")
            map_gp_stats_list.append(map_gp_batch_stats)
    if not do_nothing:
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
