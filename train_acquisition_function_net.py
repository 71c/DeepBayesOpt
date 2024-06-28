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
from utils import int_linspace


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
            softmaxed. Shape (batch_size, n_cand)
       improvements (Tensor): The improvements tensor.
            Shape (batch_size, n_cand)
        Both tensors are assumed to be padded with zeros.
        Note: A mask is not needed because the padded values are zero and the
        computation works out even if there is a mask.
    """
    expected_improvements_per_batch = torch.sum(probabilities * improvements, dim=1)
    if reduction == "mean":
        return expected_improvements_per_batch.mean()
    elif reduction == "sum":
        return expected_improvements_per_batch.sum()
    else:
        raise ValueError("'reduction' must be either 'mean' or 'sum'")


def mse_loss(pred_improvements, improvements, mask, reduction="mean"):
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


def compute_ei_max(output, improvements, cand_mask, reduction="mean"):
    probs_max = max_one_hot(output, cand_mask)
    return myopic_policy_gradient_ei(probs_max, improvements, reduction).item()


def compute_myopic_acquisition_output_batch_stats(
        output, improvements, cand_mask, policy_gradient:bool,
        return_loss:bool=False, name:str="", reduction="mean"):
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
            output_detached, mask=cand_mask, reduction=reduction).item()

        ei_softmax = myopic_policy_gradient_ei(
            output if return_loss else output_detached, improvements, reduction)
        ret[name+"ei_softmax"] = ei_softmax.item()
        if return_loss:
            ret[name+"loss"] = -ei_softmax  # Note the negative sign
    else:
        mse = mse_loss(output if return_loss else output_detached,
                       improvements, cand_mask, reduction)
        ret[name+"mse"] = mse.item()
        if return_loss:
            ret[name+"loss"] = mse

    ret[name+"ei_max"] = compute_ei_max(output_detached, improvements,
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
        ('map_gp_ei_max', 'MAP GP', True),
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
            ('map_gp_mse', 'MAP GP', True),
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


def print_train_batch_stats(nn_batch_stats, nn_model, policy_gradient,
                            batch_index, n_batches,
                            reduction="mean", batch_size=None):
    if reduction == "mean":
        assert batch_size is None
    elif reduction == "sum":
        assert batch_size is not None
        # convert sum reduction to mean reduction
        nn_batch_stats = {k: v / batch_size for k, v in nn_batch_stats.items()}
    else:
        raise ValueError("'reduction' must be either 'mean' or 'sum'")

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


def train_or_test_loop(dataloader: DataLoader,
                       nn_model: Optional[AcquisitionFunctionNet]=None,
                       train:Optional[bool]=None,
                       nn_device=None,
                       policy_gradient:Optional[bool]=None,
                       verbose:bool=True,
                       desc:Optional[str]=None,
                       
                       n_train_printouts:Optional[int]=None,
                       optimizer:Optional[torch.optim.Optimizer]=None, 
                       alpha_increment:Optional[float]=None,

                       get_true_gp_stats:Optional[bool]=None,
                       get_map_gp_stats:bool=False,
                       get_basic_stats:bool=True):
    if not isinstance(dataloader, DataLoader):
        raise ValueError("dataloader must be a torch DataLoader")
    
    if dataloader.drop_last:
        raise ValueError("dataloader must have drop_last=False")
    
    n_batches = len(dataloader)
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    has_models = dataset.has_models

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
        if policy_gradient is not None:
            raise ValueError("'policy_gradient' must not be specified if not evaluating a NN model")
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
                0, n_batches - 1, min(n_train_printouts, n_batches)
            ))
        else:
            it = tqdm(it, desc=desc)
        tic(desc)
    
    dataset_length = 0
    for i, batch in enumerate(unsupported_improvements(it)):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask = batch.tuple_no_model

        this_batch_size = improvements.size(0)
        assert this_batch_size <= batch_size
        is_degenerate_batch = this_batch_size < batch_size
        if i != n_batches - 1:
            assert not is_degenerate_batch
        
        dataset_length += this_batch_size

        if nn_model is not None:
            (x_hist_nn, y_hist_nn, x_cand_nn, improvements_nn,
             hist_mask_nn, cand_mask_nn) = batch.to(nn_device).tuple_no_model

            with torch.set_grad_enabled(train and not is_degenerate_batch):
                nn_output = nn_model(
                    x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, cand_mask_nn,
                    exponentiate=not policy_gradient, softmax=policy_gradient)
                
                nn_batch_stats = compute_myopic_acquisition_output_batch_stats(
                    nn_output, improvements_nn, cand_mask_nn, policy_gradient,
                    return_loss=train, reduction="sum")
                
                if train and not is_degenerate_batch:
                    # convert sum to mean so that this is consistent across batch sizes
                    loss = nn_batch_stats.pop("loss") / this_batch_size # (== batch_size)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if nn_model.includes_alpha and alpha_increment is not None:
                        nn_model.set_alpha(nn_model.get_alpha() + alpha_increment)
                    
                    if verbose and n_train_printouts is not None and i in print_indices:
                        print_train_batch_stats(nn_batch_stats, nn_model,
                                                policy_gradient, i, n_batches,
                                                reduction="sum",
                                                batch_size=this_batch_size)
                
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
                    policy_gradient=False, return_loss=False,
                    name="true_gp", reduction="sum")
                true_gp_stats_list.append(true_gp_batch_stats)
            
            if compute_basic_stats:
                # Calculate the E(I) of selecting a point at random,
                # the E(I) of selecting the point with the maximum I (cheating), and
                # the MSE loss of always predicting 0
                if cand_mask is None:
                    random_search_probs = torch.ones_like(improvements) / improvements.size(1)
                else:
                    random_search_probs = cand_mask.double() / cand_mask.sum(
                        dim=1, keepdim=True).double()
                basic_stats_list.append({
                    "ei_random_search": myopic_policy_gradient_ei(
                        random_search_probs, improvements, reduction="sum").item(),
                    "ei_ideal": compute_ei_max(improvements, improvements,
                                               cand_mask, reduction="sum"),
                    "mse_always_predict_0": mse_loss(
                        torch.zeros_like(improvements), improvements, cand_mask,
                        reduction="sum").item()
                })
        
        if compute_map_gp_stats:
            # Calculate the MAP GP EI values
            ei_values_map = calculate_EI_GP_padded_batch(
                x_hist, y_hist, x_cand, hist_mask, cand_mask, models, fit_params=True)
            map_gp_batch_stats = compute_myopic_acquisition_output_batch_stats(
                    ei_values_map, improvements, cand_mask,
                    policy_gradient=False, return_loss=False,
                    name="map_gp", reduction="sum")
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
        policy_gradient: bool,
        n_epochs: int,
        batch_size: int,
        nn_device=None,
        alpha_increment:Optional[float]=None,
        verbose:bool=True,
        n_train_printouts_per_epoch:Optional[int]=None,
        
        test_dataset: Optional[AcquisitionDataset]=None,
        small_test_dataset:Optional[AcquisitionDataset]=None,
        test_during_training:Optional[bool]=None,

        get_train_stats_while_training:bool=True,
        get_train_stats_after_training:bool=True,
        
        get_train_true_gp_stats:Optional[bool]=None,
        get_train_map_gp_stats:bool=False,
        get_test_true_gp_stats:Optional[bool]=None,
        get_test_map_gp_stats:Optional[bool]=None,
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

    if test_dataset is not None:
        test_dataloader = test_dataset.get_dataloader(batch_size=batch_size, drop_last=False)

        if small_test_dataset is not None: # Both test and small-test specified
            # Test during & after training
            if test_during_training == False:
                raise ValueError("Small and big test datasets specified but test_during_training == False")
            test_during_training = True # it can be either None or True
            test_after_training = True
            small_test_dataloader = small_test_dataset.get_dataloader(batch_size=batch_size, drop_last=False)
        else: # Only test but not small-test specified
            # Whether to test during & after training is ambiguous
            if test_during_training is None:
                raise ValueError("test but not small-test dataset is specified but test_during_training is not specified")
            if test_during_training:
                # Then the during-train & after-train dataset are the same, and
                # we can test after training too (which is kind of redundant)
                small_test_dataset = test_dataset
                small_test_dataloader = test_dataloader
            test_after_training = True # want to test after training both cases
    else: # Test dataset not specified
        if small_test_dataset is not None: # Small-test but not test specified
            # Test during but not after training
            if test_during_training == False:
                raise ValueError("Small-test but not test specified, but test_during_training == False")
            test_during_training = True # it can be either None or True
            test_after_training = False
            small_test_dataloader = small_test_dataset.get_dataloader(batch_size=batch_size, drop_last=False)
        else: # Neither are specified
            # Test neither during nor after training
            if test_during_training:
                raise ValueError("No test datasets were specified but got test_during_training == True")
            test_during_training = False
            test_after_training = False

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

    train_stats_list = []
    if test_during_training:
        test_stats_list = []

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
            nn_device=nn_device, policy_gradient=policy_gradient,
            verbose=verbose, desc=f"Epoch {t+1} train",
            n_train_printouts=n_train_printouts_per_epoch,
            optimizer=optimizer,
            alpha_increment=alpha_increment,
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
                            "Train stats while training")

        if get_train_stats_after_training:
            train_stats['after_training'] = train_or_test_loop(
                train_dataloader, nn_model, train=False,
                nn_device=nn_device, policy_gradient=policy_gradient,
                verbose=verbose, desc=f"Epoch {t+1} compute train stats",
                # Don't need to compute non-NN stats because already computed them
                # while training, and we ensured that the train dataset is fixed for this epoch.
                get_true_gp_stats=False,
                get_map_gp_stats=False,
                get_basic_stats=False)
            if verbose:
                print_stats({**train_stats['after_training'], **non_nn_train_stats},
                            "Train stats after training")
        
        train_stats_list.append(train_stats)

        if test_during_training:
            test_stats = train_or_test_loop(
                small_test_dataloader, nn_model, train=False,
                nn_device=nn_device, policy_gradient=policy_gradient,
                verbose=verbose, desc=f"Epoch {t+1} compute test stats",
                get_true_gp_stats=get_test_true_gp_stats,
                get_map_gp_stats=get_test_map_gp_stats,
                get_basic_stats=True)
            test_stats_list.append(test_stats)
            if verbose:
                print_stats(test_stats, "Test stats")
    
    ret = {
        'train_stats_epochs': train_stats_list
    }
    if test_during_training:
        ret['test_stats_epochs'] = test_stats_list
    
    if test_after_training:
        # Consider if the test datasets are not fixed.
        # Then the test-after-training default is not good.
        if get_test_true_gp_stats is None and not test_dataset.data_is_fixed and test_dataset.has_models:
            get_test_true_gp_stats_after_training = True
        else:
            get_test_true_gp_stats_after_training = get_test_true_gp_stats
        
        test_dataloader = test_dataset.get_dataloader(batch_size=batch_size, drop_last=False)
        final_test_stats = train_or_test_loop(
            test_dataloader, nn_model, train=False,
            nn_device=nn_device, policy_gradient=policy_gradient,
            verbose=verbose, desc=f"Compute final test stats",
            get_true_gp_stats=get_test_true_gp_stats_after_training,
            get_map_gp_stats=get_test_map_gp_stats,
            get_basic_stats=True)
        ret['final_test_stats'] = final_test_stats
        if verbose:
            print_stats(final_test_stats, "Final test stats")
    
    return ret
