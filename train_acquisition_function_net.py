import math
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical
from acquisition_function_net import AcquisitionFunctionNet
from predict_EI_simple import calculate_EI_GP_padded_batch
from tqdm import tqdm
from acquisition_dataset import AcquisitionDataset
from botorch.exceptions import UnsupportedError
from utils import to_device, unsupported_improvements
from tictoc import tic, toc


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def max_one_hot(values, mask=None):
    if mask is not None:
        neg_inf = torch.zeros_like(values)
        neg_inf[~mask] = float("-inf")
        values = values + neg_inf
    return F.one_hot(torch.argmax(values, dim=1),
                     num_classes=values.size(1)).double()


def get_average_normalized_entropy(probabilities, mask=None):
    """
    Calculates the average normalized entropy of a probability distribution,
    where a uniform distribution would give a value of 1.

    Args:
        probabilities (torch.Tensor):
            A tensor representing the probability distribution.
            Entries corresponding to masked out values are assumed to be zero.
        mask (torch.Tensor, optional):
            A tensor representing a mask to apply on the probabilities. 
            Defaults to None, meaning no mask is applied.

    Returns:
        torch.Tensor: The average normalized entropy.
    """
    entropy = Categorical(probs=probabilities).entropy()
    if mask is None:
        counts = torch.tensor(probabilities.size(1), dtype=torch.double)
    else:
        counts = mask.sum(dim=1).double()
    return (entropy / torch.log(counts)).mean()


def myopic_policy_gradient_ei(probabilities, improvements):
    """Calculate the policy gradient expected 1-step improvement.

    Args:
        probabilities (Tensor): The output tensor from the model, assumed to be
            softmaxed. Shape (batch_size, n_cand)
       improvements (Tensor): The improvements tensor.
            Shape (batch_size, n_cand)
        Both tensors are assumed to be padded with zeros.
        Note: A mask is not needed because the padded values are zero and the
        computation works out even if there is a mask.
    """
    expected_improvements_per_batch = torch.sum(probabilities * improvements, dim=1)
    return expected_improvements_per_batch.mean()


def mse_loss(pred_improvements, improvements, mask):
    """Calculate the MSE loss. Handle padding with mask for the case that
    there is padding. This works because the padded values are both zero
    so (0 - 0)^2 = 0. Equivalent to reduction="mean" if no padding.

    Args:
        pred_improvements (Tensor): The output tensor from the model,
            assumed to be exponentiated. Shape (batch_size, n_cand)
        improvements (Tensor): The improvements tensor.
            Shape (batch_size, n_cand)
        mask (Tensor): The mask tensor, shape (batch_size, n_cand)
    """
    if mask is None:
        return F.mse_loss(pred_improvements, improvements, reduction="mean")
    return F.mse_loss(pred_improvements, improvements, reduction="sum") / mask.sum()


def compute_ei_max(output, improvements, cand_mask):
    probs_max = max_one_hot(output, cand_mask)
    return myopic_policy_gradient_ei(probs_max, improvements).item()


def compute_myopic_acquisition_output_batch_stats(
        output, improvements, cand_mask, policy_gradient:bool,
        return_loss:bool=False, name:str=""):
    if not isinstance(policy_gradient, bool):
        raise ValueError("policy_gradient should be a boolean")
    if not isinstance(return_loss, bool):
        raise ValueError("return_loss should be a boolean")
    if not isinstance(name, str):
        raise ValueError("name should be a string")
    if name != "":
        name = name + "_"

    output_detached = output.detach()

    ret = {}
    if policy_gradient:
        ret[name+"avg_normalized_entropy"] = get_average_normalized_entropy(
            output_detached, mask=cand_mask).item()

        ei_softmax = myopic_policy_gradient_ei(
            output if return_loss else output_detached, improvements)
        ret[name+"ei_softmax"] = ei_softmax.item()
        if return_loss:
            ret[name+"loss"] = -ei_softmax  # Note the negative sign
    else:
        mse = mse_loss(output if return_loss else output_detached,
                       improvements, cand_mask)
        ret[name+"mse"] = mse.item()
        if return_loss:
            ret[name+"loss"] = mse

    ret[name+"ei_max"] = compute_ei_max(output_detached, improvements, cand_mask)

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


def print_stats(stats, dataset_name):
    print(f'{dataset_name}:\nExpected 1-step improvement:')
    
    has_true_gp = 'true_gp_ei_max' in stats
    if has_true_gp:
        true_gp_ei_max = stats['true_gp_ei_max']
        true_gp_mse = stats['true_gp_mse']
    
    things_to_print = [
        ('ei_softmax', 'NN (softmax)', True),
        ('ei_max', 'NN (max)', True),
        ('true_gp_ei_max', 'True GP', False),
        ('ei_random_search', 'Random search', True),
        ('ei_ideal', 'Ideal', True),
        ('avg_normalized_entropy', 'Avg norm. entropy', False)]

    direct_things_to_print = []
    for stat_key, stat_print_name, print_ratio in things_to_print:
        if stat_key in stats:
            val = stats[stat_key]
            this_thing = [stat_print_name+':', f'{val:>8f}']
            if has_true_gp and print_ratio:
                ratio = val / true_gp_ei_max
                this_thing.extend(['  Ratio:', f'{ratio:>8f}'])
            direct_things_to_print.append(this_thing)
    print_things(direct_things_to_print, prefix="  ")
    
    policy_gradient = 'ei_softmax' in stats
    if not policy_gradient:
        print('Improvement MSE:')

        things_to_print = [
            ('mse', 'NN', True),
            ('true_gp_mse', 'True GP', False),
            ('mse_always_predict_0', 'Always predict 0', True)]
        
        direct_things_to_print = []
        for stat_key, stat_print_name, print_ratio in things_to_print:
            if stat_key in stats:
                val = stats[stat_key]
                this_thing = [stat_print_name+':', f'{val:>8f}']
                if has_true_gp and print_ratio:
                    ratio = math.sqrt(true_gp_mse / val)
                    this_thing.extend(['  RMSE Ratio', f'{ratio:>8f}'])
                direct_things_to_print.append(this_thing)
        print_things(direct_things_to_print, prefix="  ")


def print_train_batch_stats(nn_batch_stats, nn_model, policy_gradient, batch_index, n_batches):
    prefix = "Expected 1-step improvement" if policy_gradient else "MSE"
    suffix = ""
    if policy_gradient:
        avg_normalized_entropy = nn_batch_stats["avg_normalized_entropy"]
        suffix += f", avg normalized entropy={avg_normalized_entropy:>7f}"
        loss_value = nn_batch_stats["ei_softmax"]
    else:
        loss_value = nn_batch_stats["mse"]
    if nn_model.includes_alpha:
        suffix += f", alpha={nn_model.get_alpha():>7f}"
    print(f"{prefix}: {loss_value:>7f}{suffix}  [{batch_index+1:>4d}/{n_batches:>4d}]")


def get_average_stats(stats_list):
    return {key: sum(d[key] for d in stats_list) / len(stats_list)
            for key in stats_list[0]}


def train_or_test_loop(dataloader: DataLoader,
                       nn_model: Optional[AcquisitionFunctionNet]=None,
                       train:Optional[bool]=None,
                       nn_device=None,
                       policy_gradient:Optional[bool]=None,
                       verbose:bool=True,
                       desc:Optional[str]=None,
                       
                       n_train_printouts:Optional[int]=None,
                       optimizer=None, 
                       alpha_increment:Optional[float]=None,

                       get_true_gp_stats:bool=True,
                       get_map_gp_stats:bool=False,
                       get_basic_stats:bool=True):
    if not isinstance(dataloader, DataLoader):
        raise ValueError("dataloader must be a torch DataLoader")
    
    n_batches = len(dataloader)
    dataset = dataloader.dataset
    has_models = dataset.has_models

    if not isinstance(dataset, AcquisitionDataset):
        raise ValueError("The dataloader must contain an AcquisitionDataset")

    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")
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
    
    if nn_model is not None: # evaluating a NN model
        if not isinstance(nn_model, AcquisitionFunctionNet):
            raise ValueError("nn_model must be a AcquisitionFunctionNet instance")
        if not isinstance(policy_gradient, bool):
            raise ValueError("'policy_gradient' must be a boolean if evaluating a NN model")
        if not isinstance(train, bool):
            raise ValueError("'train' must be a boolean if evaluating a NN model")
        
        if train:
            if optimizer is None:
                raise ValueError("optimizer must be specified if training")
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise ValueError("optimizer must be a torch Optimizer instance")
            if verbose:
                if n_train_printouts is None:
                    n_train_printouts = 10
                if not (isinstance(n_train_printouts, int) and n_train_printouts > 0):
                    raise ValueError("n_train_printouts must be a positive integer")
                every_n_batches = n_batches // n_train_printouts
            if alpha_increment is not None:
                if not (isinstance(alpha_increment, float) and alpha_increment >= 0):
                    raise ValueError("alpha_increment must be a positive float")
            nn_model.train()
        else:
            nn_model.eval()

    else: # just evaluating the dataset, no NN model
        if policy_gradient is not None:
            raise ValueError("'policy_gradient' must not be specified if not evaluating a NN model")
        if train is not None:
            raise ValueError("'train' must not be specified if not evaluating a NN model")

    if not (train and verbose):
        if n_train_printouts is not None:
            raise ValueError("n_train_printouts can't be be specified if train != True or verbose != True")
    
    if not train:
        if optimizer is not None:
            raise ValueError("optimizer must not be specified if train != True")
        if alpha_increment is not None:
            raise ValueError("alpha_increment must not be specified if train != True")
    
    if verbose:
        if not (desc is None or isinstance(desc, str)):
            raise ValueError("desc must be a string or None if verbose")
    elif desc is not None:
        raise ValueError("desc must be None if not verbose")
    
    has_true_gp_stats = hasattr(dataset, "_true_gp_stats")
    has_map_gp_stats = hasattr(dataset, "_map_gp_stats")
    has_basic_stats = hasattr(dataset, "_basic_stats")
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

    it = dataloader
    if verbose:
        if train:
            print(desc)
        else:
            it = tqdm(it, desc=desc)
        tic(desc)
    
    # If we are not computing any stats, then don't actually need to go through the dataset.
    # This probably won't happen in practice though because we always will be evaluating the NN.
    if not (compute_true_gp_stats or compute_map_gp_stats or compute_basic_stats or nn_model is not None):
        it = []
    
    for i, batch in enumerate(unsupported_improvements(it)):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask = batch.tuple_no_model

        if nn_model is not None:
            (x_hist_nn, y_hist_nn, x_cand_nn, improvements_nn,
             hist_mask_nn, cand_mask_nn) = batch.to(nn_device).tuple_no_model

            with torch.set_grad_enabled(train):
                nn_output = nn_model(
                    x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, cand_mask_nn,
                    exponentiate=not policy_gradient, softmax=policy_gradient)
                
                nn_batch_stats = compute_myopic_acquisition_output_batch_stats(
                    nn_output, improvements_nn, cand_mask_nn, policy_gradient,
                    return_loss=train)
                
                if train:
                    loss = nn_batch_stats.pop("loss")
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if nn_model.includes_alpha and alpha_increment is not None:
                        nn_model.set_alpha(nn_model.get_alpha() + alpha_increment)
                    
                    if verbose and i % every_n_batches == 0:
                        print_train_batch_stats(nn_batch_stats, nn_model,
                                                policy_gradient, i, n_batches)
                
                nn_batch_stats_list.append(nn_batch_stats)
        
        if has_models:
            models = batch.model
        
        with torch.no_grad():
            if compute_true_gp_stats:
                # Calculate true GP EI stats
                ei_values_true_model = calculate_EI_GP_padded_batch(
                    x_hist, y_hist, x_cand, hist_mask, cand_mask, models)
                true_gp_batch_stats = compute_myopic_acquisition_output_batch_stats(
                    ei_values_true_model, improvements, cand_mask,
                    policy_gradient=False, return_loss=False, name="true_gp")
                true_gp_stats_list.append(true_gp_batch_stats)
            
            if compute_basic_stats:
                # Calculate the E(I) of selecting a point at random,
                # the E(I) of selecting the point with the maximum I (cheating), and
                # the MSE loss of always predicting 0
                if cand_mask is None:
                    random_search_probs = torch.ones_like(improvements) / improvements.size(1)
                else:
                    random_search_probs = cand_mask.double() / cand_mask.sum(dim=1, keepdim=True).double()
                basic_stats_list.append({
                    "ei_random_search": myopic_policy_gradient_ei(
                        random_search_probs, improvements).item(),
                    "ei_ideal": compute_ei_max(improvements, improvements, cand_mask),
                    "mse_always_predict_0": mse_loss(torch.zeros_like(improvements),
                                                    improvements, cand_mask).item()
                })
        
        if compute_map_gp_stats:
            # Calculate the MAP GP EI values
            ei_values_map = calculate_EI_GP_padded_batch(
                x_hist, y_hist, x_cand, hist_mask, cand_mask, models, fit_params=True)
            map_gp_batch_stats = compute_myopic_acquisition_output_batch_stats(
                    ei_values_map, improvements, cand_mask,
                    policy_gradient=False, return_loss=False, name="map_gp")
            map_gp_stats_list.append(map_gp_batch_stats)
    
    if verbose:
        toc(desc)
    
    ret = {}

    if get_true_gp_stats:
        if not has_true_gp_stats:
            true_gp_stats = get_average_stats(true_gp_stats_list)
            if dataset.data_is_fixed:
                dataset._true_gp_stats = true_gp_stats
        else:
            true_gp_stats = dataset._true_gp_stats
        ret.update(true_gp_stats)
    if get_map_gp_stats:
        if not has_map_gp_stats:
            map_gp_stats = get_average_stats(map_gp_stats_list)
            if dataset.data_is_fixed:
                dataset._map_gp_stats = map_gp_stats
        else:
            map_gp_stats = dataset._map_gp_stats
        ret.update(map_gp_stats)
    if get_basic_stats:
        if not has_basic_stats:
            basic_stats = get_average_stats(basic_stats_list)
            if dataset.data_is_fixed:
                dataset._basic_stats = basic_stats
        else:
            basic_stats = dataset._basic_stats
    
    if nn_model is not None:
        ret.update(get_average_stats(nn_batch_stats_list))
    
    return ret
