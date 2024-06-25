import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from predict_EI_simple import calculate_EI_GP_padded_batch
from tqdm import tqdm
from acquisition_dataset import AcquisitionDataset
from botorch.exceptions import UnsupportedError


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


def myopic_policy_gradient_loss(probabilities, improvements):
    """Calculate the policy gradient loss.

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
    return -expected_improvements_per_batch.mean()


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


def _unsupported_improvements(dataloader):
    for batch in dataloader:
        if not batch.give_improvements:
            raise UnsupportedError(
                "The acquisition dataset must provide improvements; calculating " \
                "them from a batch would be possible but is currently unsupported.")
        yield batch


def train_loop(dataloader, model, optimizer, every_n_batches=10,
               policy_gradient=False, alpha_increment=None):
    """Trains the acquisition function network.

    Args:
        dataloader (torch.utils.data.DataLoader):
            The dataloader containing the training data.
        model (AcquisitionFunctionNet):
            The acquisition function network model.
        optimizer (torch.optim.Optimizer):
            The optimizer used for training.
        every_n_batches (int, optional, default=10):
            The frequency at which to print the training progress.
        policy_gradient (bool, default=False):
            Whether to use policy gradient loss
        alpha_increment (float, default=None):
            The amount to increase alpha by after each batch.
        train_with_ei:
            Whether to train to predict the EI rather than predict the I.
            Only used if policy_gradient is False.
    """
    if not isinstance(dataloader.dataset, AcquisitionDataset):
        raise ValueError("The dataloader must contain an AcquisitionDataset")
    
    model.train()

    n_batches = len(dataloader)
    # average_train_loss = 0.

    for i, batch in enumerate(_unsupported_improvements(dataloader)):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask = batch.tuple_no_model

        # if i % every_n_batches == 0:
        #     print("Test")
        # continue

        # print(type(models))
        # for gp_model in models:
        #     print(gp_model)
        #     for name, param in gp_model.named_parameters():
        #         print(name, param)
        #     print()
        #     print("lengthscale:", gp_model.covar_module.base_kernel.lengthscale)
        #     print("outputscale:", gp_model.covar_module.outputscale)
        #     print("mean:", gp_model.mean_module.constant)
        # print()
        # exit()

        if policy_gradient:
            output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=False, softmax=True)
            loss = myopic_policy_gradient_loss(output, improvements)
        else:
            output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=True, softmax=False)
            loss = mse_loss(output, improvements, cand_mask)
        
        # average_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model.includes_alpha and alpha_increment is not None:
            model.set_alpha(model.get_alpha() + alpha_increment)

        if i % every_n_batches == 0:
            prefix = "Expected 1-step improvement" if policy_gradient else "MSE"
            
            suffix = ""
            if policy_gradient:
                avg_normalized_entropy = get_average_normalized_entropy(output, mask=cand_mask).item()
                suffix += f", avg normalized entropy={avg_normalized_entropy:>7f}"
            if model.includes_alpha:
                suffix += f", alpha={model.get_alpha():>7f}"
            
            loss_value = -loss.item() if policy_gradient else loss.item()
            print(f"{prefix}: {loss_value:>7f}{suffix}  [{i+1:>4d}/{n_batches:>4d}]")

    # multiplier = -1 if policy_gradient else 1
    # average_train_loss /= multiplier * n_batches

    # print(f"Average train loss: {average_train_loss:>7f}")
    # return average_train_loss


def to_device(tensor, device):
    if tensor is None or device is None:
        return tensor
    return tensor.to(device)



def compute_stats_nn(dataloader, model, policy_gradient=False, nn_device=None):
    if not isinstance(dataloader.dataset, AcquisitionDataset):
        raise ValueError("The dataloader must contain an AcquisitionDataset")
    
    model.eval()

    test_loss = 0.
    test_ei_max = 0.
    if policy_gradient:
        avg_normalized_entropy = 0.
    
    for batch in _unsupported_improvements(tqdm(dataloader)):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask = batch.tuple_no_model
        
        # # For testing
        # test_ei_true_gp += 1. # to avoid division by zero
        # continue
        
        with torch.no_grad():
            x_hist_nn = to_device(x_hist, nn_device)
            y_hist_nn = to_device(y_hist, nn_device)
            x_cand_nn = to_device(x_cand, nn_device)
            improvements_nn = to_device(improvements, nn_device)
            hist_mask_nn = to_device(hist_mask, nn_device)
            cand_mask_nn = to_device(cand_mask, nn_device)

            if policy_gradient:
                # Calculate the softmax EI of the NN
                probabilities_nn = model(x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, cand_mask_nn, exponentiate=False, softmax=True)
                avg_normalized_entropy += get_average_normalized_entropy(probabilities_nn, mask=cand_mask).item()
                test_loss += myopic_policy_gradient_loss(probabilities_nn, improvements_nn).item()

                probabilities_nn_max = max_one_hot(probabilities_nn, cand_mask)
            else:
                # Calculate the MSE of the NN
                ei_values_nn = model(x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, cand_mask_nn, exponentiate=True)
                test_loss += mse_loss(ei_values_nn, improvements_nn, cand_mask_nn).item()

                probabilities_nn_max = max_one_hot(ei_values_nn, cand_mask_nn)
            
            test_ei_max += myopic_policy_gradient_loss(probabilities_nn_max, improvements_nn).item()

    n_batches = len(dataloader)
    
    ret = {}

    if policy_gradient:
        test_loss /= -n_batches
        ret["ei_softmax"] = test_loss
        
        avg_normalized_entropy /= n_batches
        ret["avg_normalized_entropy"] = avg_normalized_entropy
    else:
        test_loss /= n_batches
        ret["mse"] = test_loss

    test_ei_max /= -n_batches
    ret["ei_max"] = test_ei_max

    return ret


def compute_stats(dataloader, compute_gp_stats=True, fit_map_gp=False):
    dataset = dataloader.dataset

    if not isinstance(dataset, AcquisitionDataset):
        raise ValueError("The dataloader must contain an AcquisitionDataset")
    
    has_stats = hasattr(dataset, "_cached_stats")
    if dataset.data_is_fixed:
        if has_stats:
            return dataset._cached_stats
    else:
        assert not has_stats
    
    has_models = dataset.has_models
    
    mse_always_predict_0 = 0.
    ei_ideal = 0.
    ei_random_search = 0.

    if has_models and compute_gp_stats:
        mse_true_gp = 0.
        ei_true_gp = 0.
        if fit_map_gp:
            mse_map_gp = 0.
            ei_map_gp = 0.

    for batch in _unsupported_improvements(tqdm(dataloader)):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask = batch.tuple_no_model

        if has_models:
            models = batch.model
        
        with torch.no_grad():
            if has_models and compute_gp_stats:
                # Calculate true GP EI values
                ei_values_true_model = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models)

                # Calculate the MSE loss of the true GP model
                mse_true_gp += mse_loss(ei_values_true_model, improvements, cand_mask).item()

                # Calculate true GP actual E(I) of slecting the point with maximum EI
                probabilities_true_model = max_one_hot(ei_values_true_model, cand_mask)
                ei_true_gp += myopic_policy_gradient_loss(probabilities_true_model, improvements).item()

            # Calculate the E(I) of selecting a point at random
            if cand_mask is None:
                random_search_probabilities = torch.ones_like(improvements) / improvements.size(1)
            else:
                random_search_probabilities = cand_mask.double() / cand_mask.sum(dim=1, keepdim=True).double()
            ei_random_search += myopic_policy_gradient_loss(random_search_probabilities, improvements).item()
            
            # Calculate the MSE loss of always predicting 0
            mse_always_predict_0 += mse_loss(torch.zeros_like(improvements), improvements, cand_mask).item()
            
            # Calculate the E(I) of selecting the point with the maximum I (cheating)
            ideal_probabilities = max_one_hot(improvements, cand_mask)
            ei_ideal += myopic_policy_gradient_loss(ideal_probabilities, improvements).item()
        
        if has_models and fit_map_gp and compute_gp_stats:
            # Calculate the MAP GP EI values
            ei_values_map = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models, fit_params=True)

            # Calculate the MSE loss of the MAP GP model
            mse_map_gp += mse_loss(ei_values_map, improvements, cand_mask).item()
            
            # Calculate MAP GP actual E(I) of slecting the point with maximum EI
            probabilities_ei_map = max_one_hot(ei_values_map, cand_mask)
            ei_map_gp += myopic_policy_gradient_loss(probabilities_ei_map, improvements).item()

    n_batches = len(dataloader)
    
    
    mse_always_predict_0 /= n_batches
    ei_ideal /= -n_batches
    ei_random_search /= -n_batches

    ret = {
        "mse_always_predict_0": mse_always_predict_0,
        "ei_ideal": ei_ideal,
        "ei_random_search": ei_random_search
    }

    if has_models and compute_gp_stats:
        mse_true_gp /= n_batches
        ei_true_gp /= -n_batches
        ret.update({
            "mse_true_gp": mse_true_gp,
            "ei_true_gp": ei_true_gp})

        if fit_map_gp:
            mse_map_gp /= n_batches
            ei_map_gp /= -n_batches
            ret["mse_map_gp"] = mse_map_gp
            ret["ei_map_gp"] = ei_map_gp
    
    if dataloader.dataset.data_is_fixed:
        dataloader.dataset._cached_stats = ret
    
    return ret



