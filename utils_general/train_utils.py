import logging
from typing import Optional
import torch.nn.functional as F
from utils_general.math_utils import probability_y_greater_than_gi_normal


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


GI_NORMALIZATIONS = ["normal"]


def calculate_gittins_loss(pred_gi, y, lamdas, costs=None,
                           normalize=None, known_costs=True,
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
        normalize (str or None):
            How to normalize the loss function as proposed
        known_costs (bool):
            Whether the costs `costs` are known
            (only applicable if normalize != None and costs != None)
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

    if normalize == "normal":
        normalize_c = c if known_costs else lamdas
        normalization_consts = probability_y_greater_than_gi_normal(
            cbar=normalize_c, sigma=1.0)
        losses = losses / normalization_consts.to(losses)
    elif normalize is not None:
        raise ValueError(f"normalize must be one of {GI_NORMALIZATIONS} or None")

    if mask is None:
        mean_error_per_batch = losses.mean(dim=1)
    else:
        losses *= mask
        mean_error_per_batch = losses.sum(dim=1) / mask.sum(dim=1).double()

    if reduction == "mean":
        return mean_error_per_batch.mean()
    return mean_error_per_batch.sum()


# Set to True to enable debug logging
EARLY_STOPPER_DEBUG = False

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
        self.logger.setLevel(logging.DEBUG if EARLY_STOPPER_DEBUG else logging.WARNING)
    
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
