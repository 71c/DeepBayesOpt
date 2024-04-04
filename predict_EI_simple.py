import torch
import botorch
import gpytorch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.analytic import ExpectedImprovement

from torch.distributions import Uniform, Normal, Independent, Distribution
from torch import Tensor
from botorch.models.model import Model
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)


def calculate_EI_GP(X_hist: Tensor, y_hist: Tensor, X: Tensor,
                    model: SingleTaskGP, fit_params=False):
    """Calculate the exact Expected Improvements at `n_eval` points,
    given `N` histories each of length `n_train`.
    Assumes noise-free observations, so gives a fixed noise level of 1e-6

    Args:
        X_hist: History x values, of shape `(N, n_train, d)`
        y_hist: History y values, of shape `(N, n_train)` or `(N, n_train, 1)`
        X: Evaluation x points, of shape `(N, n_eval, d)`
        model: A SingleTaskGP model. Its batch shape must be compatible with the
            batch shape N (e.g. be 1 or N).
        fit_params: whether to fit parameters by maximizing the marginal log
            likelihood

    Returns:
        A `(N, n_eval)`-dim tensor of Expected Improvement values at the
        given design points `X`.
    
    Note: for X_hist, y_hist, X, the batch dimension N can optionally be
        consistently omitted in all of them, which is equivalent to N=1.
        In this case, the return shape is `(n_eval,)`.
    """
    # If X_hist is only 2-dimensional, then it is assumed that N=1 and
    # that y_hist and X similarly omit this batch dimension.
    # So we will add this dimension to all of them, and the assertion statements
    # later will verify that the inputs were correctly given.
    batch_dim_not_given = False
    if X_hist.dim() == 2:
        batch_dim_not_given = True
        X_hist = X_hist.unsqueeze(0)
        y_hist = y_hist.unsqueeze(0)
        X = X.unsqueeze(0)

    # Get y_hist into (N, n_train, 1) shape for BoTorch
    if y_hist.dim() == 2:
        y_hist = y_hist.unsqueeze(-1)
    elif y_hist.dim() == 3:
        assert y_hist.size(2) == 1
    else:
        raise AssertionError("y_hist dimension is incorrect")

    assert X_hist.dim() == X.dim() == 3
    assert X_hist.size(0) == y_hist.size(0) == X.size(0) # N=N=N
    assert X_hist.size(1) == y_hist.size(1) # n_train=n_train
    assert X_hist.size(2) == X.size(2) # d=d

    # reset the data in the model to be this data
    model.set_train_data(X_hist, y_hist, strict=False)

    if fit_params:
        # TODO: provide option to not use model's prior (remove it) (?)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

    # best_f has shape (N,)
    best_f = y_hist.squeeze().amax(1) # unsqueezed so need to squeeze again
    EI_object = ExpectedImprovement(model, best_f=best_f, maximize=True)

    # X currently has shape (N, n_eval, d)
    # Make it have shape (b_1, b_2, 1, d) where (b_1, b_2) = (N, n_eval)
    # The added "1" would be the "q" for "q-batch" in general
    X = X.unsqueeze(2)

    # But also need to swap the batch dimensions to align with what it says here
    # https://botorch.org/docs/batching#batched-models
    X = torch.transpose(X, 0, 1) # Now has shape (n_eval, N, 1, d)

    EI_values = EI_object(X) # shape (n_eval, N)

    # need to swap again to get to shape (N, n_eval)
    EI_values = torch.transpose(EI_values, 0, 1)

    if batch_dim_not_given: # remove first dimension N=1
        EI_values = EI_values.squeeze(0)

    return EI_values


class GaussainProcessRandomDataset(torch.utils.data.IterableDataset):
    def __init__(self, dimension, n_datapoints:int=None,
                 n_datapoints_random_gen=None,
                 xvalue_distribution: Distribution=None,
                 models: List[SingleTaskGP]=None,
                 model_probabilities=None, observation_noise:bool=False):
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
                outputscale, and noise level.
                It is assumed that each model provided is single-batch.
            model_probabilities: list of probability of choosing each model
            observation_noise: boolean specifying whether to generate the data
                to include the "observation noise" given by the model's
                likelihood (True), or not (False)
        """
        self.dimension = dimension
        self.observation_noise = observation_noise

        # exacly one of them should be specified; verify this by xor
        assert (n_datapoints is None) ^ (n_datapoints_random_gen is None)
        self.n_datapoints = n_datapoints
        self.n_datapoints_random_gen = n_datapoints_random_gen
        
        if xvalue_distribution is None:
            # m = Normal(torch.zeros(dimension), torch.ones(dimension))
            m = Uniform(torch.zeros(dimension), torch.ones(dimension))
            xvalue_distribution = Independent(m, 1)
        self.xvalue_distribution = xvalue_distribution

        if models is None: # models is None implies model_probabilities is None
            assert model_probabilities is None

            train_X = torch.empty(1, 0, dimension) # (nbatch, n_data, dimension)
            train_Y = torch.empty(1, 0, 1)         # (nbatch, n_data, n_out)
            # Default: Matern 5/2 kernel with gamma priors on
            # lengthscale, outputscale, and noise level
            models = [SingleTaskGP(train_X, train_Y)]
            model_probabilities = torch.tensor([1.0])
        elif model_probabilities is None:
            model_probabilities = torch.full([len(models)], 1/len(models))
        else: # if both were specified, then,
            model_probabilities = torch.as_tensor(model_probabilities)
            assert model_probabilities.dim() == 1
            assert len(models) == len(model_probabilities)
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
        random_model = model.pyro_sample_from_prior()

        # pick the number of data points
        if self.n_datapoints is None: # then it's given by a distribution
            n_datapoints = self.n_datapoints_random_gen()
        else:
            n_datapoints = self.n_datapoints

        # generate the x-values
        x_values = self.xvalue_distribution.sample(torch.Size([n_datapoints]))
        assert x_values.dim() == 2 # should have shape (n_datapoints, dimension)

        with gpytorch.settings.prior_mode(True): # sample from prior
            prior = random_model.posterior(
                x_values.unsqueeze(0), # make x_values have 1 batch for Botorch
                observation_noise=self.observation_noise)

        #        n_samples, n_batch, n_datapoints, n_out
        # shape (1,         1,       n_datapoints, 1)
        y_values = prior.sample(torch.Size([1]))
        y_values = y_values.squeeze() # shape (n_datapoints,)

        return x_values, y_values, random_model

