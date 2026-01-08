import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from utils_general.train_utils import check_2d_or_3d_tensors


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


def _max_one_hot(values, mask=None):
    values, mask = check_2d_or_3d_tensors(values, mask)
    if mask is not None:
        neg_inf = torch.zeros_like(values)
        neg_inf[~mask] = float("-inf")
        values = values + neg_inf
    return F.one_hot(torch.argmax(values, dim=1),
                     num_classes=values.size(1)).double()


def compute_maxei(output, improvements, cand_mask, reduction="mean"):
    probs_max = _max_one_hot(output, cand_mask)
    return myopic_policy_gradient_ei(probs_max, improvements, reduction).item()
