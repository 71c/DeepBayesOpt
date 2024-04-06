import torch
import botorch
import gpytorch
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.analytic import ExpectedImprovement

from torch.distributions import Uniform, Normal, Independent, Distribution
from torch import Tensor
from botorch.models.model import Model
from typing import List

import pyro


torch.set_default_dtype(torch.double)


def calculate_EI_GP(model: SingleTaskGP, X_hist: Tensor, y_hist: Tensor,
                    X: Tensor, y: Tensor=None, fit_params=False):
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
        fit_params: whether to fit parameters by maximizing the marginal log
            likelihood

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
    assert X_hist.size(0) == y_hist.size(0) == X.size(0) == y.size(0) # N=N=N=N
    assert X_hist.size(1) == y_hist.size(1) # n_train=n_train
    assert X.size(1) == y.size(1) # n_eval=n_eval
    assert X_hist.size(2) == X.size(2) # d=d

    # reset the data in the model to be this data
    # (As a hack, need to remove last dimension of y_values because
    # set_train_data isn't really supported in BoTorch)
    model.set_train_data(X_hist, y_hist.squeeze(-1), strict=False)

    if fit_params:
        # TODO: provide option to not use model's prior (remove it) (?)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

    # best_f has shape (N,)
    best_f = y_hist.squeeze(2).amax(1) # unsqueezed so need to squeeze again
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


# https://pytorch.org/docs/stable/data.html#map-style-datasets
# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
class GaussainProcessRandomDataset(torch.utils.data.IterableDataset):
    def __init__(self, dimension, n_datapoints:int=None,
                 n_datapoints_random_gen=None,
                 xvalue_distribution: Distribution=None,
                 models: List[SingleTaskGP]=None,
                 model_probabilities=None, observation_noise:bool=False,
                 set_random_model_train_data=False,
                 device=None):
        """Create a dataset that generates random Gaussian Process data.

        Args:
            dimension: The dimension of the feature space d
            n_datapoints: number of (x,y) pairs to generate with each sample;
                could be None
            n_datapoints_random_gen: a callable that returns a random natural
                number that is the number of datapoints.
                Note: exactly one of n_datapoints and n_datapoints_random_gen
                should be speicified (not be None).
            xvalue_distribution: a torch.distributions.Distribution object that
                represents the probability distribution for generating each iid
                value $x \in \mathbb{R}^{dimension}$. Default: iid Uniform(0,1)
            models: a list of SingleTaskGP models to choose from randomly,
                with their priors 
                defaults to a single SingleTaskGP model with the default BoTorch
                Matern 5/2 kernel and Gamma priors for the lengthscale,
                outputscale; and noise level also if observation_noise==True.
                It is assumed that each model provided is single-batch.
            model_probabilities: list of probability of choosing each model
            observation_noise: boolean specifying whether to generate the data
                to include the "observation noise" given by the model's
                likelihood (True), or not (False)
            set_random_model_train_data: whether to set the random model train
                data to the random data with each returned values
            device: torch.device, optional -- the desired device to use for
                computations.
        """
        self.dimension = dimension
        self.observation_noise = observation_noise

        # exacly one of them should be specified; verify this by xor
        assert (n_datapoints is None) ^ (n_datapoints_random_gen is None)
        self.n_datapoints = n_datapoints
        self.n_datapoints_random_gen = n_datapoints_random_gen
        self.set_random_model_train_data = set_random_model_train_data
        self.device = device
        
        if xvalue_distribution is None:
            # m = Normal(torch.zeros(dimension), torch.ones(dimension))
            m = Uniform(torch.zeros(dimension, device=device),
                        torch.ones(dimension, device=device))
            xvalue_distribution = Independent(m, 1)
        self.xvalue_distribution = xvalue_distribution

        if models is None: # models is None implies model_probabilities is None
            assert model_probabilities is None

            train_X = torch.zeros(0, dimension, device=device)
            train_Y = torch.zeros(0, 1, device=device)

            # Default: Matern 5/2 kernel with gamma priors on
            # lengthscale and outputscale, and noise level also if
            # observation_noise.
            # If no observation noise is generated, then make the likelihood
            # be fixed noise at almost zero to correspond to what is generated.
            likelihood = None if observation_noise else GaussianLikelihood(
                    noise_prior=None, batch_shape=torch.Size(),
                    noise_constraint=GreaterThan(
                        0.0, transform=None, initial_value=1e-6
                    )
                )
            models = [SingleTaskGP(train_X, train_Y, likelihood=likelihood)]
            model_probabilities = torch.tensor([1.0])
        else:
            if model_probabilities is None:
                model_probabilities = torch.full([len(models)], 1/len(models))
            else: # if both were specified, then,
                model_probabilities = torch.as_tensor(model_probabilities)
                assert model_probabilities.dim() == 1
                assert len(models) == len(model_probabilities)
        for model in models:
            t = len(model.batch_shape)
            assert t == 0 or t == 1 and model.batch_shape[0] == 1
        self.models = models
        self.model_probabilities = model_probabilities
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # pick the model
        # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146
        model_index = self.model_probabilities.multinomial(num_samples=1, 
                                                           replacement=True)[0]
        model = self.models[model_index]

        # Randomly set the parameters based on the priors of the model

        # random_model = model.pyro_sample_from_prior()
        
        # Hack to sample in-place rather than doing a deep copy, significantly
        # speeding it up
        random_model = _pyro_sample_from_prior(model, memo=None, prefix="")

        # pick the number of data points
        if self.n_datapoints is None: # then it's given by a distribution
            n_datapoints = self.n_datapoints_random_gen()
        else:
            n_datapoints = self.n_datapoints

        # generate the x-values
        x_values = self.xvalue_distribution.sample(torch.Size([n_datapoints]))
        assert x_values.dim() == 2 # should have shape (n_datapoints, dimension)
        
        # make x_values have 1 batch for Botorch
        x_values_botorch = x_values.unsqueeze(0) if len(model.batch_shape) == 1 else x_values

        with gpytorch.settings.prior_mode(True): # sample from prior
            prior = random_model.posterior(
                x_values_botorch, 
                observation_noise=self.observation_noise)

        # shape (batch_shape, n_datapoints, 1)
        y_values = prior.sample(torch.Size([]))

        if self.set_random_model_train_data:
            # As a hack, need to remove last dimension of y_values because
            # set_train_data isn't really supported in BoTorch
            random_model.set_train_data(
                x_values_botorch, y_values.squeeze(-1), strict=False)

        return x_values, y_values.squeeze(), random_model, model


# https://docs.gpytorch.ai/en/stable/_modules/gpytorch/module.html#Module.pyro_sample_from_prior
def _pyro_sample_from_prior(module, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_priors"):
        for prior_name, (prior, closure, setting_closure) in module._priors.items():
            if prior is not None and prior not in memo:
                if setting_closure is None:
                    raise RuntimeError(
                        "Cannot use Pyro for sampling without a setting_closure for each prior,"
                        f" but the following prior had none: {prior_name}, {prior}."
                    )
                memo.add(prior)
                prior = prior.expand(closure(module).shape)
                value = pyro.sample(prefix + ("." if prefix else "") + prior_name, prior)
                setting_closure(module, value)

    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        _pyro_sample_from_prior(module=module_, memo=memo, prefix=submodule_prefix)

    return module
