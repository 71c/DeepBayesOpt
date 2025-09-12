import copy
import math
from typing import Optional, Union
from collections.abc import Sequence

import torch
from torch.utils.data import IterableDataset
from torch.distributions import Normal, Uniform, Bernoulli
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from datasets.dataset_with_models import TupleWithModel
from datasets.function_samples_dataset import (
    FunctionSamplesDataset, FunctionSamplesItem)
from utils.utils import SizedInfiniteIterableMixin


# We'll reuse FunctionSamplesItem from the existing infrastructure
from datasets.function_samples_dataset import FunctionSamplesItem


class LogisticRegressionRandomDataset(
    FunctionSamplesDataset, IterableDataset, SizedInfiniteIterableMixin):
    """An IterableDataset that generates random logistic regression datasets for hyperparameter optimization.

    Each sample generates a synthetic binary classification dataset with known ground truth parameters,
    suitable for evaluating regularization strength in logistic regression.
    
    Usage example:
    ```
    dataset = LogisticRegressionRandomDataset(
        n_samples_range=(50, 500),
        n_features_range=(5, 50),
        dataset_size=100
    )
    for x_values, y_values, dataset_params in dataset:
        # x_values has shape (n_samples, n_features)  
        # y_values has shape (n_samples,) with binary labels
        # dataset_params contains generation parameters
    ```
    """
    
    def __init__(self, 
                 n_datapoints: Optional[int] = None,
                 n_datapoints_random_gen = None,
                 n_samples_range: tuple = (50, 2000),
                 n_features_range: tuple = (5, 100),
                 bias_range: tuple = (-2.0, 2.0),
                 coefficient_std: float = 1.0,
                 noise_range: tuple = (0.01, 1.0),
                 log_lambda_range: tuple = (-6, 2),
                 log_uniform_sampling: bool = True,
                 device = None,
                 dataset_size: Optional[int] = None):
        """Create a dataset that generates random logistic regression hyperparameter optimization problems.
        
        Args:
            n_datapoints: Number of hyperparameter evaluations per sample (fixed)
            n_datapoints_random_gen: Generator for random number of hyperparameter evaluations
            n_samples_range: Tuple (min, max) for number of samples per classification dataset
            n_features_range: Tuple (min, max) for number of features per classification dataset  
            bias_range: Tuple (min, max) for bias term b
            coefficient_std: Standard deviation for coefficient vector c ~ N(0, coefficient_std^2 * I)
            noise_range: Tuple (min, max) for noise standard deviation σ_y
            log_lambda_range: Tuple (min, max) for log(lambda) range mapping from [0,1]
            log_uniform_sampling: Whether to use log-uniform sampling for ranges
            device: torch.device for computations
            dataset_size: Number of hyperparameter optimization problems to generate (None for infinite)
        """
        # Exactly one of n_datapoints and n_datapoints_random_gen should be specified
        if not ((n_datapoints is None) ^ (n_datapoints_random_gen is None)):
            raise ValueError("Exactly one of n_datapoints and n_datapoints_random_gen "
                             "should be specified.")
        if n_datapoints is not None and (
            not isinstance(n_datapoints, int) or n_datapoints <= 0):
            raise ValueError("n_datapoints should be a positive integer.")
            
        self.n_datapoints = n_datapoints
        self.n_datapoints_random_gen = n_datapoints_random_gen
        
        # Validate ranges
        assert len(n_samples_range) == 2 and n_samples_range[0] > 0 and n_samples_range[1] >= n_samples_range[0]
        assert len(n_features_range) == 2 and n_features_range[0] > 0 and n_features_range[1] >= n_features_range[0]
        assert len(bias_range) == 2 and bias_range[1] >= bias_range[0]
        assert coefficient_std > 0
        assert len(noise_range) == 2 and noise_range[0] > 0 and noise_range[1] >= noise_range[0]
        assert len(log_lambda_range) == 2 and log_lambda_range[1] >= log_lambda_range[0]
        
        self.n_samples_range = n_samples_range
        self.n_features_range = n_features_range
        self.bias_range = bias_range
        self.coefficient_std = coefficient_std
        self.noise_range = noise_range
        self.log_lambda_range = log_lambda_range
        self.log_uniform_sampling = log_uniform_sampling
        self.device = device if device is not None else torch.device('cpu')
        
        # Required for compatibility with FunctionSamplesDataset infrastructure
        # Logistic regression doesn't use models, so set to None
        self._model_sampler = None
        
        if dataset_size is None:
            dataset_size = math.inf
        else:
            if not isinstance(dataset_size, int) or dataset_size <= 0:
                raise ValueError("dataset_size should be a positive integer or None.")
        self._size = dataset_size
    
    @property
    def data_is_fixed(self):
        return False
    
    def data_is_loaded(self):
        return False
    
    def _init_params(self):
        return tuple(), dict(
            n_datapoints=self.n_datapoints,
            n_datapoints_random_gen=self.n_datapoints_random_gen,
            n_samples_range=self.n_samples_range,
            n_features_range=self.n_features_range,
            bias_range=self.bias_range,
            coefficient_std=self.coefficient_std,
            noise_range=self.noise_range,
            log_lambda_range=self.log_lambda_range,
            log_uniform_sampling=self.log_uniform_sampling,
            device=self.device,
            dataset_size=self._size
        )

    def save(self, dir_name: str, verbose: bool = True):
        raise NotImplementedError(
            "LogisticRegressionRandomDataset does not support saving to a file")

    def copy_with_new_size(self, dataset_size: int):
        """Create a copy of the dataset with a new dataset size."""
        ret = self.__new__(self.__class__)
        ret.n_datapoints = self.n_datapoints
        ret.n_datapoints_random_gen = self.n_datapoints_random_gen
        ret.n_samples_range = self.n_samples_range
        ret.n_features_range = self.n_features_range
        ret.bias_range = self.bias_range
        ret.coefficient_std = self.coefficient_std
        ret.noise_range = self.noise_range
        ret.log_lambda_range = self.log_lambda_range
        ret.log_uniform_sampling = self.log_uniform_sampling
        ret.device = self.device
        ret._size = dataset_size
        return ret

    def random_split(self, lengths: Sequence[Union[int, float]]):
        # Delegate to the mixin implementation
        from utils.utils import SizedInfiniteIterableMixin
        return SizedInfiniteIterableMixin.random_split(self, lengths)

    def _sample_range(self, range_tuple: tuple, log_uniform: bool = False) -> float:
        """Sample a value from a range, optionally using log-uniform distribution."""
        min_val, max_val = range_tuple
        if log_uniform and min_val > 0:
            # Log-uniform sampling
            log_min = math.log(min_val)
            log_max = math.log(max_val)
            log_val = torch.rand(1, device=self.device).item() * (log_max - log_min) + log_min
            return math.exp(log_val)
        else:
            # Uniform sampling
            return torch.rand(1, device=self.device).item() * (max_val - min_val) + min_val

    def _next(self):
        """Generate a random logistic regression hyperparameter optimization problem.

        Returns:
            FunctionSamplesItem with:
            - x_values: hyperparameter values in [0, 1] space (n_datapoints, 1)
            - y_values: corresponding log-likelihood evaluations (n_datapoints, 1)
        """
        # First, generate a synthetic logistic regression dataset
        if self.log_uniform_sampling:
            n_samples = int(self._sample_range(self.n_samples_range, log_uniform=True))
            n_features = int(self._sample_range(self.n_features_range, log_uniform=True))
            sigma_y = self._sample_range(self.noise_range, log_uniform=True)
        else:
            n_samples = torch.randint(self.n_samples_range[0], self.n_samples_range[1] + 1, (1,), device=self.device).item()
            n_features = torch.randint(self.n_features_range[0], self.n_features_range[1] + 1, (1,), device=self.device).item()
            sigma_y = self._sample_range(self.noise_range, log_uniform=False)
        
        # Sample dataset parameters
        b = self._sample_range(self.bias_range, log_uniform=False)  # Bias term
        c = torch.randn(n_features, device=self.device) * self.coefficient_std  # Coefficient vector
        
        # Generate covariates X ~ N(0, I_d)
        X = torch.randn(n_samples, n_features, device=self.device)
        
        # Generate labels Y_i ~ Bernoulli(σ(b + c^T X_i + ε_i))
        linear_terms = b + torch.matmul(X, c)  # b + c^T X_i for each i
        noise = torch.randn(n_samples, device=self.device) * sigma_y  # ε_i ~ N(0, σ_y^2)
        logits = linear_terms + noise
        probs = torch.sigmoid(logits)  # σ(b + c^T X_i + ε_i)
        
        # Sample binary labels from Bernoulli distribution
        y_class = torch.bernoulli(probs).long()  # Convert to long for integer labels
        
        # Determine number of hyperparameter evaluations
        if self.n_datapoints is None:  # then it's given by a distribution
            n_hyperparameter_evals = self.n_datapoints_random_gen()
            if not isinstance(n_hyperparameter_evals, int) or n_hyperparameter_evals <= 0:
                raise ValueError(
                    "n_datapoints_random_gen should return a positive integer.")
        else:
            n_hyperparameter_evals = self.n_datapoints
            
        # Now evaluate hyperparameter optimization: sample hyperparameter points and evaluate log-likelihood
        x_hyperparams = torch.rand(n_hyperparameter_evals, 1, device=self.device)  # Hyperparameters in [0,1]
        y_log_likelihoods = torch.zeros(n_hyperparameter_evals, 1, device=self.device)
        
        # Convert to numpy for sklearn
        X_np = X.detach().cpu().numpy()
        y_np = y_class.detach().cpu().numpy()
        
        # Evaluate log-likelihood for each hyperparameter value
        for i, x_norm in enumerate(x_hyperparams):
            # Map x ∈ [0,1] to log-lambda range using self.log_lambda_range
            min_log_lambda, max_log_lambda = self.log_lambda_range
            log_lambda = min_log_lambda + x_norm.item() * (max_log_lambda - min_log_lambda)
            lambda_val = 10 ** log_lambda
            C = 1.0 / lambda_val  # sklearn uses C = 1/lambda
            
            # Evaluate with cross-validation
            lr = LogisticRegression(C=C, random_state=42, max_iter=1000, solver='lbfgs')
            try:
                # Use fewer CV folds for efficiency
                scores = cross_val_score(lr, X_np, y_np, cv=3, scoring='neg_log_loss')
                log_likelihood = -scores.mean()  # Convert neg_log_loss to log_likelihood
                y_log_likelihoods[i, 0] = log_likelihood
            except:
                # If optimization fails, use a penalty
                y_log_likelihoods[i, 0] = -10.0
        
        return FunctionSamplesItem(x_hyperparams, y_log_likelihoods)