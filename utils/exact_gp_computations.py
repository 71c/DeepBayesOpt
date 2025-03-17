import math
from botorch.utils.probability.utils import get_constants_like, ndtr as Phi
import numpy as np
import scipy
from scipy.optimize import newton
import torch
from torch import Tensor
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior, Posterior, TransformedPosterior
from botorch.exceptions import UnsupportedError
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement, _log_ei_helper
from typing import Optional
from utils.utils import fit_model, pad_tensor, remove_priors, add_priors
from utils.nn_utils import check_xy_dims

torch.set_default_dtype(torch.double)


# Trying to debug a crazy problem....
DEBUG = False


def calculate_EI_GP(model: SingleTaskGP, X_hist: Tensor, y_hist: Tensor,
                    X: Tensor, y: Optional[Tensor]=None, log=False,
                    fit_params=False, mle=False):
    """Calculate the exact Expected Improvements at `n_eval` points,
    given `N` histories each of length `n_train`.

    Args:
        model: A SingleTaskGP model. Its batch shape must be compatible with the
            batch shape N (e.g. be 1 or N).
        X_hist: History x values, of shape `(N, n_train, d)`
        y_hist: History y values, of shape `(N, n_train)` or `(N, n_train, 1)`
        X: Evaluation x points, of shape `(N, n_eval, d)`
        y: True values of y corresponding to the X, of shape `(N, n_eval)`
            or `(N, n_eval, 1)`
        fit_params: whether to fit parameters before calculating EI
        mle: whether remove the model's prior when fitting parameters.
            If True, the model's prior will be removed before fitting and then
            re-added after fitting, so that the MLE estimate is done.
            If False, the model's prior will be used as the prior for fitting,
            making the estimate MAP.

    Returns:
        A `(N, n_eval)`-dim tensor of Expected Improvement values at the
        given design points `X`. Or, if `y` is not None, also return the true
        improvement values as a tuple, ei_values, improvement_values which both
        have shape `(N, n_eval)`.
    
    Note: for X_hist, y_hist, X, the batch dimension N can optionally be
        consistently omitted in all of them, which is equivalent to N=1.
        In this case, the return shape is `(n_eval,)`.
    """
    # If X_hist is only 2-dimensional, then it is assumed that N=1 and
    # that y_hist and X similarly omit this batch dimension.
    # So we will add this dimension to all of them, and the assertion statements
    # later will verify that the inputs were correctly given.
    batch_dim_not_given = False
    if X_hist.dim() == 2: # (n_train, d)
        batch_dim_not_given = True
        X_hist = X_hist.unsqueeze(0)
        y_hist = y_hist.unsqueeze(0)
        X = X.unsqueeze(0)
        if y is not None:
            y = y.unsqueeze(0)

    # Get y_hist into (N, n_train, 1) shape for BoTorch,
    # and same for y if y is given.
    # (actually not necessary but whatever I already wrote the code)
    if y_hist.dim() == 2: # (N, n_train)
        y_hist = y_hist.unsqueeze(-1)
        if y is not None:
            assert y.dim() == 2 # (N, n_eval)
            y = y.unsqueeze(-1)
    elif y_hist.dim() == 3: # (N, n_train, 1)
        assert y_hist.size(2) == 1
        if y is not None:
            assert y.dim() == 3
            assert y.size(2) == 1
    else:
        raise AssertionError("y_hist dimension is incorrect")

    assert X_hist.dim() == X.dim() == 3
    
    assert X_hist.size(0) == y_hist.size(0) == X.size(0) # N=N=N
    if y is not None:
        assert y.size(0) == X.size(0) # N=N

        assert X.size(1) == y.size(1) # n_eval=n_eval
    
    assert X_hist.size(1) == y_hist.size(1) # n_train=n_train
    
    assert X_hist.size(2) == X.size(2) # d=d

    has_outcome_transform = hasattr(model, "outcome_transform")

    if DEBUG and has_outcome_transform:
        means = model.outcome_transform._original_transform.means
        stdvs = model.outcome_transform._original_transform.stdvs
    
        # y_hist, _ = model.outcome_transform(y_hist)
        y_hist = means + stdvs * y_hist

        del model.outcome_transform

    if DEBUG and has_outcome_transform:
        model.set_train_data(X_hist, y_hist.squeeze(-1), strict=False)
    else:
        # reset the data in the model to be this data
        model.set_train_data_with_transforms(X_hist, y_hist, strict=False, train=fit_params)

    if fit_params:
        if mle: # remove priors for MLE
            named_priors_tuple_list = remove_priors(model)

        if hasattr(model, "initial_params"):
            # print("initial params:", model.initial_params) # This is ok
            model.initialize(**model.initial_params)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        if mle: # add back the priors
            add_priors(named_priors_tuple_list)
    
    # try:
    #     print("DEBUG means,stdvs:",
    #           model.outcome_transform._original_transform.means.item(),
    #           model.outcome_transform._original_transform.stdvs.item())
    #     print("DEBUG model.train_targets:",
    #         model.train_targets.squeeze(), model.train_targets.mean().item(),
    #         model.train_targets.std().item())
    #     print("DEBUG y_hist:", y_hist.squeeze(), y_hist.mean().item(), y_hist.std().item())
    #     print("DEBUG untransformed y_hist:", model.outcome_transform(y_hist.squeeze()))
    #     print()
    # except AttributeError:
    #     pass

    # best_f has shape (N,)
    best_f = y_hist.squeeze(2).amax(1) # unsqueezed so need to squeeze again
    if log:
        EI_object = LogExpectedImprovement(model, best_f=best_f, maximize=True)
    else:
        EI_object = ExpectedImprovement(model, best_f=best_f, maximize=True)

    # X currently has shape (N, n_eval, d)
    # Make it have shape (b_1, b_2, 1, d) where (b_1, b_2) = (N, n_eval)
    # The added "1" would be the "q" for "q-batch" in general
    X = X.unsqueeze(2)

    # But also need to swap the batch dimensions to align with what it says here
    # https://botorch.org/docs/batching#batched-models
    X = torch.transpose(X, 0, 1) # Now has shape (n_eval, N, 1, d)

    ei_values = EI_object(X) # shape (n_eval, N)

    if DEBUG and has_outcome_transform:
        ei_values = ei_values / stdvs

    # need to swap again to get to shape (N, n_eval)
    ei_values = torch.transpose(ei_values, 0, 1)

    if batch_dim_not_given: # remove first dimension N=1
        ei_values = ei_values.squeeze(0)

    if y is not None:
        # y has shape (N, n_eval, 1); best_f has shape (N,)
        y = y.squeeze(2) # Get into shape (N, n_eval)
        best_f = best_f.unsqueeze(1) # Get into shape (N, 1)
        improvement_values = torch.nn.functional.relu(y - best_f, inplace=True)
        if batch_dim_not_given: # remove first dimension N=1
            improvement_values = improvement_values.squeeze(0)
        
        return ei_values, improvement_values

    return ei_values


def calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask,
                                 models, fit_params=False, mle=False):
    r"""Calculate the EI with GP model on a batch
    
    Args:
        x_hist: shape batch_size x n_hist x dimension
        y_hist: shape batch_size x n_hist x 1
        x_cand: shape batch_size x n_cand x dimension
        hist_mask: shape batch_size x n_hist x 1
        cand_mask: shape batch_size x n_cand x 1
        fit_params: whether to fit parameters before calculating EI
        mle: whether to use the model's prior when fitting parameters.
    """
    n_hist = x_hist.size(1) # = y_hist.size(1)
    n_cand = x_cand.size(1)
    batch_size = x_hist.size(0) # = y_hist.size(0) = x_cand.size(0) = len(models)

    hist_lengths = None if hist_mask is None else hist_mask.sum(dim=1)
    cand_lengths = None if cand_mask is None else cand_mask.sum(dim=1)

    ei_values = []
    for i in range(batch_size):
        x_hist_i_padded = x_hist[i] # shape (n_hist, dimension)
        y_hist_i_padded = y_hist[i] # shape (n_hist, 1)
        x_cand_i_padded = x_cand[i] # shape (n_cand, dimension)
        n_hist_i = n_hist if hist_mask is None else hist_lengths[i].item()
        n_cand_i = n_cand if cand_mask is None else cand_lengths[i].item()

        x_hist_i = x_hist_i_padded[:n_hist_i]
        y_hist_i = y_hist_i_padded[:n_hist_i]
        x_cand_i = x_cand_i_padded[:n_cand_i]

        # shape (n_cand_i, 1)
        ei_value = calculate_EI_GP(models[i], x_hist_i, y_hist_i, x_cand_i,
                                   fit_params=fit_params, mle=mle)
        # shape (n_cand, 1)
        ei_value_padded = pad_tensor(ei_value, n_cand, 0, add_mask=False)
        ei_values.append(ei_value_padded)
    return torch.stack(ei_values) # shape (batch_size, n_cand)


def _safe_log(x):
    if x is None:
        return None
    return torch.where(x > 0, torch.log(x), torch.zeros_like(x))


def calculate_gi_gp_already_fit(model: GPyTorchModel,
                    x_cand:Tensor,
                    lambda_cand:Tensor,
                    cost_cand:Optional[Tensor]=None):
    r"""Calculate the exact Gittins index with GP model.

    Args:
        model: GPyTorchModel
            A fitted single-outcome model or a fitted two-outcome model, where the first
            output corresponds to the objective and the second one to the log-cost.
        x_cand (torch.Tensor):
            Candidate input tensor with shape `batch_shape x n_cand x d`.
        lambda_cand (torch.Tensor):
            A `batch_shape x n_cand` or `batch_shape x n_cand x 1` tensor of lambda
            values for the candidate points.
        cost_cand (torch.Tensor, optional):
            A `batch_shape x n_cand` or `batch_shape x n_cand x 1` tensor of costs for
            the candidate points.
    
    Returns:
        A `batch_shape x n_cand` tensor of Gittins index values
    """
    if len(model.batch_shape) != 0:
        raise UnsupportedError(
            f"model must be single-batch, but has batch shape {model.batch_shape}")

    # Make lambda_cand and cost_cand both have shape `batch_shape x n_cand`
    if lambda_cand.dim() != 0:
        lambda_cand = check_xy_dims(
            x_cand, lambda_cand, "x_cand", "lambda_cand", expected_y_dim=1).squeeze(-1)
    if cost_cand is not None:
        cost_cand = check_xy_dims(
            x_cand, cost_cand, "x_cand", "cost_cand", expected_y_dim=1).squeeze(-1)

    # Must be either GPyTorchPosterior or TransformedPosterior
    posterior = model.posterior(x_cand)

    if isinstance(posterior, TransformedPosterior):
        raise ValueError(
            "Exact Gittins index computation only supports normal distribution "
            "posteriors (GPyTorchPosterior), but got a TransformedPosterior")
    else:
        assert isinstance(posterior, GPyTorchPosterior)

    # mean and sigma are both batch_shape x n_cand x model.num_outputs
    mean = posterior.mean
    min_var = 1e-12 # from AnalyticAcquisitionFunction._mean_and_sigma default
    var = posterior.variance.clamp_min(min_var)
    assert mean.shape == var.shape
    assert mean.dim() == x_cand.dim()
    assert mean.size(-1) == model.num_outputs

    # mean and sigma have shape `batch_shape x n_cand`
    mean_y, sigma_y = mean[..., 0], var[..., 0].sqrt()
        
    if model.num_outputs == 2:
        if cost_cand is None:
            # mean of lognormal distribution
            cost_cand = torch.exp(mean[..., 1] + 0.5 * var[..., 1])
    elif model.num_outputs != 1:
        raise ValueError("model.num_outputs must be 1 or 2")
    
    lc = lambda_cand if cost_cand is None else lambda_cand * cost_cand

    return gi_normal(lc, mean_y, sigma_y)


def calculate_gi_gp(model, x_hist, y_hist, x_cand, lambda_cand,
             cost_cand=None, fit_params=False, mle=False):
    fit_model(model, x_hist, y_hist, fit_params, mle)
    return calculate_gi_gp_already_fit(model, x_cand, lambda_cand, cost_cand)


def calculate_gi_gp_padded_batch(
        models,
        x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
        lambda_cand:Tensor,
        cost_hist:Optional[Tensor]=None,
        cost_cand:Optional[Tensor]=None,
        hist_mask:Optional[Tensor]=None, cand_mask:Optional[Tensor]=None,
        is_log:bool=False, fit_params:bool=False, mle:bool=False):
    r"""Calculate the Gittins index with GP model on a batch
    
    Args:
        x_hist (torch.Tensor):
            A `batch_shape x n_hist x d` tensor of training features.
        y_hist (torch.Tensor):
            A `batch_shape x n_hist x 1` tensor of training observations.
        x_cand (torch.Tensor):
            Candidate input tensor with shape `batch_shape x n_cand x d`.
        lambda_cand (torch.Tensor):
            A `batch_shape x n_cand` or `batch_shape x n_cand x 1` tensor of lambda
            values for the candidate points.
        cost_hist (torch.Tensor, optional):
            A `batch_shape x n_hist` or `batch_shape x n_hist x 1` tensor of costs
            for the history points.
        cost_cand (torch.Tensor, optional):
            A `batch_shape x n_cand` or `batch_shape x n_cand x 1` tensor of costs
            for the candidate points.
        hist_mask (torch.Tensor, optional):
            Mask tensor for the history inputs with shape `batch_shape x n_hist`
            or `batch_shape x n_hist x 1`. If None, then mask is all ones.
        cand_mask (torch.Tensor, optional):
            Mask tensor for the candidate inputs with shape `batch_shape x n_cand`
            or `batch_shape x n_cand x 1`. If None, then mask is all ones.
        is_log (bool, default: False):
            Whether lambda_cand, cost_hist, and cost_cand (if applicable) are
            already log-transformed.
            If True, they will be unchanged. If False, their log will be taken
            before passing them to the acquisition function network.
        fit_params (bool, default: False):
            whether to fit parameters before calculating EI
        mle (bool, default: False):
            whether to use the model's prior when fitting parameters.
    """
    # make sure y_hist has `batch_shape x n_hist x 1`
    y_hist = check_xy_dims(x_hist, y_hist, "x_hist", "y_hist", expected_y_dim=1)

    if lambda_cand.dim() == 3:
        lambda_cand = lambda_cand.squeeze(-1)
    if is_log:
        lambda_cand = torch.exp(lambda_cand)
    
    if cost_cand is not None:
        if cost_cand.dim() == 3:
            cost_cand = cost_cand.squeeze(-1)
        if is_log:
            cost_cand = torch.exp(cost_cand)

    n_cand = x_cand.size(1)
    batch_size = x_hist.size(0) # = y_hist.size(0) = x_cand.size(0) = len(models)

    hist_lengths = None if hist_mask is None else hist_mask.sum(dim=1)
    cand_lengths = None if cand_mask is None else cand_mask.sum(dim=1)

    models_num_outputs = [model.num_outputs for model in models]
    if all(x == 1 for x in models_num_outputs):
        costs_in_history = False
    elif all(x == 2 for x in models_num_outputs):
        costs_in_history = True
    else:
        raise ValueError("All models should either be 1 or 2 output")

    if costs_in_history:
        if cost_hist is None:
            raise ValueError("cost_hist must be specified if 2-output model")
        cost_hist = check_xy_dims(x_hist, cost_hist, "x_hist", "cost_hist", expected_y_dim=1)
        if not is_log:
            cost_hist = _safe_log(cost_hist)
        y_hist = torch.cat((y_hist, cost_hist), dim=-1)
    elif cost_hist is not None:
        raise ValueError("cost_hist should not be specified if 1-output model")

    gi_values = []
    for i in range(batch_size):
        x_hist_i = x_hist[i] # shape (n_hist, dimension)
        y_hist_i = y_hist[i] # shape (n_hist, 1 or 2)
        if hist_mask is not None:
            n_hist_i = hist_lengths[i].item()
            x_hist_i = x_hist_i[:n_hist_i]
            y_hist_i = y_hist_i[:n_hist_i]
        
        x_cand_i = x_cand[i]
        lambda_cand_i = lambda_cand if lambda_cand.dim() == 0 else lambda_cand[i]
        cost_cand_i = None if cost_cand is None else cost_cand[i]
        if cand_mask is not None:
            n_cand_i = cand_lengths[i].item()
            x_cand_i = x_cand_i[:n_cand_i]
            if lambda_cand.dim() != 0:
                lambda_cand_i[:n_cand_i]
            cost_cand_i = None if cost_cand is None else cost_cand_i[:n_cand_i]

        model_i = models[i]
        
        gi_value = calculate_gi_gp(
            model_i, x_hist_i, y_hist_i, x_cand_i, lambda_cand_i, cost_cand_i,
            fit_params=fit_params, mle=mle)
        
        gi_value_padded = pad_tensor(gi_value, n_cand, 0, add_mask=False)
        gi_values.append(gi_value_padded)

    return torch.stack(gi_values) # shape (batch_size, n_cand)


_neg_inv_sqrt_2 = -(1 / math.sqrt(2))
_inv_sqrt_2pi = 1 / math.sqrt(math.tau)
# _constant_1 = math.log(0.5 / math.sqrt(math.tau))
_constant_1 = -0.5 * math.log(8.0 * math.pi)
_constant_2 = 0.5755

def _approx_ei_helper_inverse(v: Tensor) -> Tensor:
    const1, const2 = get_constants_like((_constant_1, _constant_2), v)
    tmp_log = const1 - torch.log(v)
    val_low = -torch.sqrt(2 * (tmp_log - torch.log(tmp_log)))
    val_med = const2 * torch.log(-1 + torch.exp(v / const2))
    return torch.where(
        v <= 0.05, val_low,
        torch.where(v <= 10.0, val_med, v)
    )

def _phi_numpy(x: Tensor) -> Tensor:
    r"""Standard normal PDF."""
    return _inv_sqrt_2pi * np.exp(-0.5 * x**2)

def _Phi_numpy(x):
    r"""Standard normal CDF."""
    return 0.5 * scipy.special.erfc(_neg_inv_sqrt_2 * x)

def _ei_helper_numpy(u):
    return _phi_numpy(u) + u * _Phi_numpy(u)

def _ei_helper_inverse(v: Tensor) -> Tensor:
    v = v.detach().cpu()
    if not torch.is_tensor(v):
        raise ValueError("v should be a torch tensor")
    log_v = torch.log(v).numpy()

    def f(x):
        x = torch.from_numpy(np.asarray(x))
        return _log_ei_helper(x).numpy() - log_v
    def fprime(x):
        return _Phi_numpy(x) / _ei_helper_numpy(x)
    def fprime2(x):
        return (
            _phi_numpy(x) * _ei_helper_numpy(x) - _Phi_numpy(x)**2) / _ei_helper_numpy(x)**2

    x0 = _approx_ei_helper_inverse(v).numpy()
    result = newton(f, x0, fprime=fprime, fprime2=fprime2, tol=1e-10, maxiter=50)
    return torch.tensor(result, dtype=v.dtype, device=v.device)


def gi_normal(cbar: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    u = _ei_helper_inverse(cbar / sigma)
    return mu - sigma * u


def probability_y_greater_than_gi_normal(cbar: Tensor, sigma: Tensor) -> Tensor:
    u = _ei_helper_inverse(cbar / sigma)
    return Phi(u)
