import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from predict_EI_simple import calculate_EI_GP_padded_batch
from tqdm import tqdm


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


def train_loop(dataloader, model, optimizer, every_n_batches=10,
               policy_gradient=False, alpha_increment=None, train_with_ei=False):
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
    """
    model.train()

    n_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask, models = batch

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

            if train_with_ei:
                ei_values_true_model = calculate_EI_GP_padded_batch(
                    x_hist, y_hist, x_cand, hist_mask, cand_mask, models)
                loss = mse_loss(output, ei_values_true_model, cand_mask)
            else:
                loss = mse_loss(output, improvements, cand_mask)

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


def test_loop(dataloader, model, policy_gradient=False, fit_map_gp=False):
    model.eval()

    test_loss = 0.
    test_loss_true_gp = 0.
    if fit_map_gp:
        test_loss_gp_map = 0.
    always_predict_0_loss = 0.
    test_loss_max = 0.
    if policy_gradient:
        ideal_loss = 0.
        avg_normalized_entropy_nn = 0.
    else:
        test_ei_true_gp = 0.
    
    for batch in tqdm(dataloader):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask, models = batch
        
        with torch.no_grad():
            ei_values_true_model = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models)
            probabilities_true_model = max_one_hot(ei_values_true_model, cand_mask)
            ei_true_gp = myopic_policy_gradient_loss(probabilities_true_model, improvements).item()

            if policy_gradient:
                probabilities_nn = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=False, softmax=True)
                avg_normalized_entropy_nn += get_average_normalized_entropy(probabilities_nn, mask=cand_mask).item()
                # print(probabilities_nn)
                test_loss += myopic_policy_gradient_loss(probabilities_nn, improvements).item()

                if cand_mask is None:
                    always_predict_0_probabilities = torch.ones_like(probabilities_nn) / probabilities_nn.size(1)
                else:
                    always_predict_0_probabilities = cand_mask.double() / cand_mask.sum(dim=1, keepdim=True).double()
                # print(always_predict_0_probabilities)
                always_predict_0_loss += myopic_policy_gradient_loss(always_predict_0_probabilities, improvements).item()

                test_loss_true_gp += ei_true_gp

                ideal_probabilities = max_one_hot(improvements, cand_mask)
                ideal_loss += myopic_policy_gradient_loss(ideal_probabilities, improvements).item()

                probabilities_nn_max = max_one_hot(probabilities_nn, cand_mask)
            else:
                ei_values_nn = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=True)
                test_loss += mse_loss(ei_values_nn, improvements, cand_mask).item()

                always_predict_0_loss += mse_loss(torch.zeros_like(ei_values_nn), improvements, cand_mask).item()

                test_loss_true_gp += mse_loss(ei_values_true_model, improvements, cand_mask).item()

                test_ei_true_gp += ei_true_gp

                probabilities_nn_max = max_one_hot(ei_values_nn, cand_mask)
            
            test_loss_max += myopic_policy_gradient_loss(probabilities_nn_max, improvements).item()

        if fit_map_gp:
            ei_values_map = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models, fit_params=True)
            
            if policy_gradient:
                probabilities_ei_map = max_one_hot(ei_values_map, cand_mask)
                # print(ei_values_map)
                # print(probabilities_ei_map)
                test_loss_gp_map += myopic_policy_gradient_loss(probabilities_ei_map, improvements).item()
            else:
                test_loss_gp_map += mse_loss(ei_values_map, improvements, cand_mask).item()


    n_batches = len(dataloader)
    multiplier = -1 if policy_gradient else 1
    
    test_loss /= multiplier * n_batches
    test_loss_true_gp /= multiplier * n_batches
    if fit_map_gp:
        test_loss_gp_map /= multiplier * n_batches
    always_predict_0_loss /= multiplier * n_batches
    if policy_gradient:
        ideal_loss /= multiplier * n_batches
        avg_normalized_entropy_nn /= n_batches
        test_loss_max /= multiplier * n_batches
    else:
        test_loss_max /= -n_batches
        test_ei_true_gp /= -n_batches

    mse_desc = "Expected 1-step improvement" if policy_gradient else"Improvement MSE"
    map_str = f" MAP GP: {test_loss_gp_map:>8f}\n" if fit_map_gp else ""
    naive_desc = "Random search" if policy_gradient else "Always predict 0"
    eval_str = f"Test {mse_desc}:\n NN ({'softmax' if policy_gradient else 'loss'}): {test_loss:>8f}\n"
    if policy_gradient:
        eval_str += f" NN (max): {test_loss_max:>8f}\n"
    eval_str += f" True GP: {test_loss_true_gp:>8f}\n"
    if policy_gradient:
        eval_str += f" Ratio: {test_loss_max/test_loss_true_gp:>8f}\n"
    eval_str += f"{map_str} {naive_desc}: {always_predict_0_loss:>8f}\n"
    if policy_gradient:
        eval_str += f" Ideal: {ideal_loss:>8f}\n NN avg normalized entropy: {avg_normalized_entropy_nn:>8f}\n"
    if not policy_gradient:
        eval_str += "Expected 1-step improvement\n"
        eval_str += f" NN (max): {test_loss_max:>8f}\n"
        eval_str += f" True GP: {test_ei_true_gp:>8f}\n"
        eval_str += f" Ratio: {test_loss_max/test_ei_true_gp:>8f}\n"
    print(eval_str)

