import torch
from torch import Tensor
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from typing import Optional
from utils import pad_tensor, remove_priors, add_priors

torch.set_default_dtype(torch.double)


def calculate_EI_GP(model: SingleTaskGP, X_hist: Tensor, y_hist: Tensor,
                    X: Tensor, y: Optional[Tensor]=None, fit_params=False,
                    use_priors=True, log=False):
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
        use_priors: whether to use the model's prior when fitting parameters.
            If False, the model's prior will be removed before fitting and then
            re-added after fitting, so that the MLE estimate is done.
            If True, the model's prior will be used as the prior for fitting,
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

    # reset the data in the model to be this data
    model.set_train_data_with_transforms(X_hist, y_hist, strict=False, train=fit_params)

    if fit_params:
        if not use_priors: # remove priors for MLE
            named_priors_tuple_list = remove_priors(model)

        if hasattr(model, "initial_params"):
            # print("initial params:", model.initial_params) # This is ok
            model.initialize(**model.initial_params)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        if not use_priors: # add back the priors
            add_priors(named_priors_tuple_list)

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
                                   fit_params=fit_params, use_priors=not mle)
        # shape (n_cand, 1)
        ei_value_padded = pad_tensor(ei_value, n_cand, 0, add_mask=False)
        ei_values.append(ei_value_padded)
    return torch.stack(ei_values) # shape (batch_size, n_cand)

