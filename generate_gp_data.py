from abc import ABC, abstractmethod
from functools import partial
import torch
import gpytorch
import pyro
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from botorch.models.gp_regression import SingleTaskGP

from torch.distributions import Uniform, Normal, Independent, Distribution
from torch.utils.data import Dataset, IterableDataset, DataLoader, random_split, Subset

# https://pytorch.org/maskedtensor/main/index.html
# https://pytorch.org/docs/stable/masked.html
# https://pytorch.org/tutorials/prototype/maskedtensor_overview
from torch.masked import is_masked_tensor

from utils import resize_iterable, uniform_randint, get_uniform_randint_generator, max_pad_tensors_batch, get_lengths_from_proportions
from utils import SizedIterableMixin, SizedInfiniteIterableMixin, len_or_inf, iterable_is_finite

from typing import Iterable, Optional, List, Tuple, Union
from collections.abc import Sequence
from botorch.exceptions import UnsupportedError

import os
import warnings
import math
from tqdm import tqdm

torch.set_default_dtype(torch.double)


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


class RandomModelSampler:
    """A class that samples a random model with random parameters from its prior
    from a list of models."""
    def __init__(self, models: List[SingleTaskGP], model_probabilities=None, randomize_params=True):
        """Initializes the RandomModelSampler instance.

        Args:
            models (List[SingleTaskGP]): A list of SingleTaskGP models to choose
                from randomly, with their priors.
            model_probabilities (Tensor, optional): 1D Tensor of probabilities
                OF choosing each model. If None, then set to be uniform.
        """
        for i, model in enumerate(models):
            if not isinstance(model, SingleTaskGP):
                raise UnsupportedError(f"models[{i}] should be a SingleTaskGP instance.")

            # Set the model indices as attributes so we can access them for purpose
            # of saving data
            model.index = i

        self._models = models
        # need to copy the initial parameters with clone()
        for model in models:
            model.initial_params = {name: param.detach().clone() for name, param in model.named_parameters()}

        if model_probabilities is None:
            model_probabilities = torch.full([len(models)], 1/len(models))
        else:
            model_probabilities = torch.as_tensor(model_probabilities)
            assert model_probabilities.dim() == 1
            assert len(models) == len(model_probabilities)

        self.model_probabilities = model_probabilities
        self.randomize_params = randomize_params
    
    def sample(self):
        # pick the model
        # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146
        model_index = self.model_probabilities.multinomial(num_samples=1, 
                                                           replacement=True)[0]
        model = self._models[model_index]

        # Initialize the model with the initial parameters
        # because they could have been changed by maximizing mll.
        # in particular the parameters that don't have priors.
        model.initialize(**model.initial_params)

        # Randomly set the parameters based on the priors of the model.
        # Instead of doing  `random_model = model.pyro_sample_from_prior()`
        # which does a deep copy which takes long,
        # sample in-place, significantly speeding it up.
        # This also avoids the parameters disappearing.
        if self.randomize_params:
            _pyro_sample_from_prior(model, memo=None, prefix="")

        return model
    
    def get_model(self, index, model_params=None):
        model = self._models[index]
        if model_params is None:
            model_params = model.initial_params

        # Remove the data from the model. Basically equivalent to
        # model.set_train_data(inputs=None, targets=None, strict=False)
        # except that would just do nothing
        model.train_inputs = None
        model.train_targets = None
        model.prediction_strategy = None

        model.initialize(**model_params)
        return model
    
    @property
    def initial_models(self):
        return [self.get_model(i) for i in range(len(self._models))]


class TupleWithModel:
    def __init__(self, *items, model, model_params=None):
        self._items = items
        self._indices = list(range(len(items) + 1))
        self._model = model
        if model_params is None:
            # need to copy the data, otherwise everything will be the same
            model_params = {name: param.detach().clone()
                            for name, param in model.named_parameters()}
        self.model_params = model_params
        self.model_index = model.index if hasattr(model, "index") else None
    
    @property
    def model(self):
        # self._model.train_inputs = None
        # self._model.train_targets = None
        # self._model.prediction_strategy = None
        self._model.initialize(**self.model_params)
        return self._model

    @property
    def _tuple(self):
        return self._items + (self.model,)
    
    def __getitem__(self, index):
        # return self._tuple[index] # basically equivalent to the below

        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))

        index = self._indices[index]
        if index == len(self._items):
            return self.model
        return self._items[index]
    
    def __len__(self):
        return len(self._items) + 1

    def __str__(self):
        return str(self._tuple)

    def __repr__(self):
        return repr(self._tuple)
    
    def __eq__(self, other):
        return type(self) == type(other) and self._tuple == other._tuple


class GPDatasetItem(TupleWithModel):
    def __init__(self, x_values, y_values, model, model_params=None):
        super().__init__(x_values, y_values, model=model, model_params=model_params)
        self.x_values = x_values
        self.y_values = y_values


class AcquisitionDatasetModelItem(TupleWithModel):
    def __init__(self, x_hist, y_hist, x_cand, vals_cand, model, model_params=None):
        super().__init__(x_hist, y_hist, x_cand, vals_cand, model=model, model_params=model_params)
        self.x_hist = x_hist
        self.y_hist = y_hist
        self.x_cand = x_cand
        self.vals_cand = vals_cand


class FunctionSamplesDataset(Dataset, ABC):
    """
    A dataset class for function samples.

    It is expected that the yielded values are either
    (x_values, y_values, model) or (x_values, y_values), where
    - x_values: A tensor of shape (n_datapoints, dimension)
    - y_values: A tensor of shape (n_datapoints,)
    - model: A SingleTaskGP instance that was used to generate the data

    Subclasses include:
    - GaussianProcessRandomDataset, which represents an iterable-style
      dataset where the samples are generated from random Gaussian processes,
      and each iter() can be either finite or infinite in length
    - FunctionSamplesMapDataset, which represents a map-style dataset where
        the samples are stored in memory and can be accessed by index.
    
    Usage example:
    ```
    dataset = some FunctionSamplesDataset instance
    for x_values, y_values, model in dataset:
        # x_values has shape (n_datapoints, dimension)
        # y_values has shape (n_datapoints,)
        # do something with x_values, y_values, and model
    ```

    This class also has the instance method `_get_items_generator_and_size` that
    can be used in subclasses.
    """
    
    @property
    @abstractmethod
    def has_models(self):
        """Boolean variable that is whether the dataset includes model
        information (i.e., GP models and their parameters)."""
        pass  # pragma: no cover
    
    @property
    @abstractmethod
    def model_sampler(self):
        """The RandomModelSampler instance that is used to sample random GP
        models for this dataset.
        Should raise an error if has_models is False."""
        pass  # pragma: no cover
    
    @abstractmethod
    def random_split(self, lengths: Sequence[Union[int, float]]):
        """Randomly splits the dataset into multiple subsets.

        Args:
            lengths (Sequence[Union[int, float]]): A sequence of lengths
            specifying the size of each subset, or the proportion of the
            dataset to include in each subset.
        """
        pass  # pragma: no cover

    @staticmethod
    def _get_items_generator(dataset_resized, has_models, verbose, verbose_message=None):
        if verbose and verbose_message is not None:
            print(verbose_message)
        it = tqdm(dataset_resized) if verbose else dataset_resized
        for item in it:
            if has_models and not isinstance(item, GPDatasetItem):
                x_values, y_values, model = item
                # need to copy the data, otherwise everything will be same
                model_params = {name: param.detach().clone()
                                for name, param in model.named_parameters()}
                item = GPDatasetItem(x_values, y_values, model, model_params)
            yield item
    
    def _get_items_generator_and_size(self, n_realizations: Optional[int]=None,
                                      verbose: bool=True, verbose_message=None):
        """
        Args:
            n_realizations (int, positive):
                The number of function realizations (samples) to generate.
                Optional if the dataset has finite size, in which case the size
                of the dataset is used.
            verbose (bool):
                Whether to print progress bar or not.
            verbose_message (str):
                The message to print if verbose is True.
        
        Returns: a tuple (items_generator, size) where
            items_generator (generator): A generator that yields the items.
            size (int): The size of the resized dataset.
        """
        dataset = self

        if n_realizations is None:
            if not iterable_is_finite(dataset):
                raise ValueError(f"Can't store an infinite-sized {dataset.__class__.__name__} if n_realizations "\
                                 "is not specified. Either specify n_realizations or use a finite-sized dataset.")
            # The dataset is finite and we want to save all of it
            new_data_iterable = dataset
        else:
            if not isinstance(n_realizations, int) or n_realizations <= 0:
                raise ValueError("n_realizations should be a positive integer.")
            try:
                new_data_iterable = resize_iterable(dataset, n_realizations)
            except ValueError as e:
                raise ValueError(f"To store finite samples from this {dataset.__class__.__name__}, cannot make n_realizations > len(dataset) since it " \
                                 f"is not a SizedInfiniteIterableMixin. Got {n_realizations=} and len(dataset)={len(dataset)}") from e
        
        items_generator = self._get_items_generator(
            new_data_iterable, dataset.has_models, verbose, verbose_message)
        return items_generator, len(new_data_iterable)

    @abstractmethod
    def data_is_loaded(self) -> bool:
        """Returns whether the data is loaded in memory or not."""
        pass  # pragma: no cover

    def save(self, dir_name: str, n_realizations:Optional[int]=None,
             verbose:Optional[bool]=None):
        """Saves the dataset to a specified directory. If the dataset includes
        models, the models are saved as well. If the directory does not
        exist, it will be created.

        Args:
            dir_name (str):
                The directory where the realizations should be saved.
            n_realizations (int):
                The number of realizations to save.
                If unspecified, all the realizations are saved.
                If specified, the first n_realizations realizations are saved.
        """
        if verbose is None:
            verbose = not self.data_is_loaded()
        if self.data_is_loaded():
            message = f"Saving realizations from {self.__class__.__name__}"
        else:
            message = f"Generating and saving realizations from {self.__class__.__name__}"
        items_generator, length = self._get_items_generator_and_size(
            n_realizations, verbose=verbose, verbose_message=message)

        if os.path.exists(dir_name):
            if not os.path.isdir(dir_name):
                raise NotADirectoryError(f"Path {dir_name} is not a directory")
        else:
            os.mkdir(dir_name)
        
        has_models = self.has_models

        # Save the models if we have them
        if has_models:
            models = self.model_sampler.initial_models
            torch.save(models, os.path.join(dir_name, "models.pt"))
        
        list_of_dicts = []
        for item in items_generator:
            if has_models:
                list_of_dicts.append({
                    'x_values': item.x_values,
                    'y_values': item.y_values,
                    'model_index': item.model_index,
                    'model_params': item.model_params
                })
            else:
                x_values, y_values = item
                list_of_dicts.append({
                    'x_values': x_values,
                    'y_values': y_values
                })

        torch.save(list_of_dicts, os.path.join(dir_name, "data.pt"))
    
    def __getitem__(self, index):
        if isinstance(self, IterableDataset):
            raise TypeError(f"{self.__class__.__name__} is an IterableDataset and should not be accessed by index.")
        raise NotImplementedError("Subclass must implement __getitem__ for map-style datasets")


class GaussianProcessRandomDataset(
    FunctionSamplesDataset, IterableDataset, SizedInfiniteIterableMixin):
    """An IterableDataset that generates random Gaussian Process data.

     Usage example:
    ```
    dataset = GaussianProcessRandomDataset(n_datapoints=15, dimension=5, dataset_size=100)
    for x_values, y_values, model in dataset:
        # x_values has shape (n_datapoints, dimension)
        # y_values has shape (n_datapoints,)
        # do something with x_values, y_values, and model
    ```
    """
    def __init__(self, n_datapoints:Optional[int]=None,
                 n_datapoints_random_gen=None,
                 observation_noise:bool=False,
                 xvalue_distribution: Union[Distribution,str]="uniform",
                 models: Optional[List[SingleTaskGP]]=None,
                 model_probabilities=None,
                 dimension:Optional[int]=None, device=None,
                 set_random_model_train_data=False,
                 dataset_size:Optional[int]=None,
                 randomize_params=True):
        """Create a dataset that generates random Gaussian Process data.
        
        Args:
            n_datapoints: number of (x,y) pairs to generate with each sample;
                could be None
            n_datapoints_random_gen: a callable that returns a random natural
                number that is the number of datapoints.
                Note: exactly one of n_datapoints and n_datapoints_random_gen
                should be speicified (not be None).
            observation_noise: boolean specifying whether to generate the data
                to include the "observation noise" given by the model's
                likelihood (True), or not (False)
            xvalue_distribution: a torch.distributions.Distribution object that
                represents the probability distribution for generating each iid
                value $x \in \mathbb{R}^{dimension}$, or a string 'uniform' or
                'normal' to specify iid uniform(0,1) or normal(0,I) distribution
            models: a list of SingleTaskGP models to choose from randomly.
                Each model must have a GaussianLikelihood likelihood; no other
                likelihood is supported.
                Each model must also be single-batch.
                The priors on the models are used for randomly sampling
                parameters if randomize_params is True.
                Default: a single SingleTaskGP model with the default BoTorch
                Matern 5/2 kernel and Gamma priors for the lengthscale,
                outputscale; and noise level also if observation_noise==True.
            model_probabilities: list of probabilities of choosing each model
            dimension: int, optional -- The dimension d of the feature space.
                Is only used if xvalue_distribution is None or models is None;
                in this case, it is required. Otherwise, it is ignored.
            device: torch.device, optional -- the desired device to use for
                computations. Is only used if xvalue_distribution is None or
                models is None; otherwise, it is ignored.
            set_random_model_train_data (bool, default: False):
                Whether to set the random model train data to the random data
                with each returned values
            dataset_size: int, optional -- The size of the dataset to generate.
                If None, the dataset is infinite. If specified, the dataset
                generates that many samples with each iter() call.
            randomize_params: bool, default: True -- Whether to randomize the
                parameters of the model with each sample. If False, the model
                parameters are kept the same as the initial parameters.
        """
        # exacly one of them should be specified; verify this by xor
        if not ((n_datapoints is None) ^ (n_datapoints_random_gen is None)):
            raise ValueError("Exactly one of n_datapoints and n_datapoints_random_gen should be specified.")
        if n_datapoints is not None and (not isinstance(n_datapoints, int) or n_datapoints <= 0):
            raise ValueError("n_datapoints should be a positive integer.")
        self.n_datapoints = n_datapoints
        self.n_datapoints_random_gen = n_datapoints_random_gen
        
        if not isinstance(observation_noise, bool):
            raise TypeError("observation_noise must be a boolean value.")
        self.observation_noise = observation_noise

        if dimension is not None and (not isinstance(dimension, int) or dimension <= 0):
            raise ValueError("dimension should be a positive integer.")

        if not isinstance(set_random_model_train_data, bool):
            raise TypeError("set_random_model_train_data must be a boolean value.")
        self.set_random_model_train_data = set_random_model_train_data

        # Set xvalue_distribution
        if xvalue_distribution == "uniform" or xvalue_distribution == "normal":
            if dimension is None:
                raise ValueError("dimension should be specified if xvalue_distribution is 'uniform' or 'normal' string.")
            dist_cls = {"uniform": Uniform, "normal": Normal}[xvalue_distribution]
            m = dist_cls(torch.zeros(dimension, device=device),
                        torch.ones(dimension, device=device))
            xvalue_distribution = Independent(m, 1)
        elif not isinstance(xvalue_distribution, Distribution):
            raise ValueError(f"xvalue_distribution should be a Distribution object or 'uniform' or 'normal' string, but got {xvalue_distribution}")
        self.xvalue_distribution = xvalue_distribution

        if models is None: # models is None implies model_probabilities is None
            if model_probabilities is not None:
                raise ValueError("model_probabilities should be None if models is None")
            if dimension is None:
                raise ValueError("dimension should be specified if models is None")

            train_X = torch.zeros(0, dimension, device=device)
            train_Y = torch.zeros(0, 1, device=device)

            # Default: Matern 5/2 kernel with gamma priors on
            # lengthscale and outputscale, and noise level also if
            # observation_noise.
            # If no observation noise is generated, then make the likelihood
            # be fixed noise at zero to correspond to what is generated.
            likelihood = None if observation_noise else GaussianLikelihood(
                    noise_prior=None, batch_shape=torch.Size(),
                    noise_constraint=GreaterThan(
                        0.0, transform=None, initial_value=0.0
                    )
                )

            models = [SingleTaskGP(train_X, train_Y, likelihood=likelihood)]
            model_probabilities = torch.tensor([1.0])
        
        for i, model in enumerate(models):
            if not isinstance(model, SingleTaskGP):
                raise UnsupportedError(f"models[{i}] should be a SingleTaskGP instance.")

            if not observation_noise:
                # If no observation noise, then keep the noise
                # level fixed so it can't be changed by optimization.
                # Make it so nothing in likelihood can change by gradient-based optimization.
                # Since only GaussianLikelihood is supported, we could just do
                # model.likelihood.noise_covar.raw_noise.requires_grad_(False),
                # but this code is more general in case we want to support other likelihoods.
                for param in model.likelihood.parameters():
                    param.requires_grad_(False)
            
            if isinstance(model.likelihood, GaussianLikelihood):
                if not observation_noise:
                    # This is only important for estimating parameters with MLL
                    # and not for sampling. Set noise level to zero so that
                    # parameter estimates correspond to the generated data.

                    # model.likelihood.noise_covar is a HomoskedasticNoise instance
                    # Could also do model.initialize(**{'likelihood.noise_covar.raw_noise': 0.0})
                    # Equivalent because GaussianLikelihood has @raw_noise.setter
                    model.likelihood.raw_noise = 0.0
            elif isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
                raise UnsupportedError(
                    f"models[{i}] has FixedNoiseGaussianLikelihood which is not supported in GaussianProcessRandomDataset. Use GaussianLikelihood instead.")

                # ## Could alternatively do it like this but that's too
                # ## messy/buggy/not necessary:
                # # model.likelihood.noise_covar is a FixedGaussianNoise instance
                # # Also has a @noise.setter but not a raw_noise attribute
                # # Fills the tensor with zeros (I think)
                # model.noise = 0.0
                # if model.likelihood.second_noise_covar is not None:
                #     # there is also a @second_noise.setter
                #     model.likelihood.second_noise = 0.0
            else:
                raise UnsupportedError(
                    f"models[{i}] has likelihood {model.likelihood.__class__.__name__} which is not supported in GaussianProcessRandomDataset. Use GaussianLikelihood instead.")

            # Verify that the model is single-batch
            t = len(model.batch_shape)
            assert t == 0 or t == 1 and model.batch_shape[0] == 1
        
        self._model_sampler = RandomModelSampler(
            models, model_probabilities, randomize_params=randomize_params)

        if dataset_size is None:
            dataset_size = math.inf
        else:
            if not isinstance(dataset_size, int) or dataset_size <= 0:
                raise ValueError("dataset_size should be a positive integer.")
        self._size = dataset_size
    
    @property
    def has_models(self):
        return True
    
    @property
    def model_sampler(self):
        return self._model_sampler
    
    def random_split(self, lengths: Sequence[Union[int, float]]):
        # Same check that pytorch does in torch.utils.data.random_split
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
        lengths_is_proportions = math.isclose(sum(lengths), 1) and sum(lengths) <= 1

        dataset_size = self._size
        if dataset_size == math.inf:
            if lengths_is_proportions:
                raise ValueError(
                    "The GaussianProcessRandomDataset should not be infinite if lengths is a list of proportions")
            dataset_size = sum(lengths)
        else:
            if lengths_is_proportions:
                lengths = get_lengths_from_proportions(dataset_size, lengths)
            
            if sum(lengths) != dataset_size:
                raise ValueError(
                    "Sum of input lengths does not equal the dataset size!")
        return [self.copy_with_new_size(length) for length in lengths]

    def data_is_loaded(self):
        return False

    def copy_with_new_size(self, dataset_size:int):
        """Create a copy of the dataset with a new dataset size.

        Args:
            dataset_size (int): The new dataset size for the copied dataset.

        Returns:
            GaussianProcessRandomDataset: A new instance of the dataset with the
            specified dataset size.
        """
        return GaussianProcessRandomDataset(
            n_datapoints=self.n_datapoints,
            n_datapoints_random_gen=self.n_datapoints_random_gen,
            observation_noise=self.observation_noise,
            xvalue_distribution=self.xvalue_distribution,
            models=self.model_sampler.initial_models,
            model_probabilities=self.model_sampler.model_probabilities,
            set_random_model_train_data=self.set_random_model_train_data,
            dataset_size=dataset_size, randomize_params=self.model_sampler.randomize_params)

    def _next(self):
        """Generate a random Gaussian Process model and sample from it.

        Returns:
            Tuple `(x_values, y_values, model)` where `model` is a
            `SingleTaskGP` instance that was used to generate the data.
            `x_values` has shape `(n_datapoints, dimension)`, and
            `y_values` has shape `(n_datapoints,)`.
        """
        # Get a random model
        model = self.model_sampler.sample()

        # pick the number of data points
        if self.n_datapoints is None: # then it's given by a distribution
            n_datapoints = self.n_datapoints_random_gen()
            if not isinstance(n_datapoints, int) or n_datapoints <= 0:
                raise ValueError("n_datapoints_random_gen should return a positive integer.")
        else:
            n_datapoints = self.n_datapoints

        # generate the x-values
        x_values = self.xvalue_distribution.sample(torch.Size([n_datapoints]))
        assert x_values.dim() == 2 # should have shape (n_datapoints, dimension)
        
        # make x_values have 1 batch for Botorch
        x_values_botorch = x_values.unsqueeze(0) if len(model.batch_shape) == 1 else x_values

        with gpytorch.settings.prior_mode(True): # sample from prior
            prior = model.posterior(
                x_values_botorch, 
                observation_noise=self.observation_noise)

        # shape (batch_shape, n_datapoints, 1)
        y_values = prior.sample(torch.Size([]))

        if self.set_random_model_train_data:
            # As a hack, need to remove last dimension of y_values because
            # set_train_data isn't really supported in BoTorch
            model.set_train_data(
                x_values_botorch, y_values.squeeze(-1), strict=False)

        return GPDatasetItem(x_values, y_values.squeeze(), model)


class FunctionSamplesMapDatasetBase(FunctionSamplesDataset):
    """A base class for `FunctionSamplesDataset` datasets that hold function
    samples in a map style (i.e. implement `__getitem__` and `__len__`).
    
    All subclasses should implement `__getitem__` and `__len__`.
    `__getitem__` is partially implemented for slices where it returns a
    `FunctionSamplesMapSubset` instance, so subclasses should check for
    slices and use super() accordingly.
    
    Subclasses should also have the `_model_sampler` attribute, which is a
    `RandomModelSampler` instance if models are associated with the dataset
    and is `None` otherwise.
    
    The `has_models`, `model_sampler`, `random_split`, and `save` methods
    are implemented in this class.
    """

    @abstractmethod
    def __getitem__(self, index):
        """Retrieves a single sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: If the dataset includes models, returns a tuple
            (x_values, y_values, model), where 'model' is the GP model that
            generated the sample. Otherwise, returns a tuple(x_values, y_values)
        """
        if isinstance(index, slice):
            indices = list(range(*index.indices(len(self))))
            return FunctionSamplesMapSubset(self, indices)
        raise NotImplementedError("Subclass must implement __getitem__ for non-slice keys")
    
    @abstractmethod
    def __len__(self):
        pass  # pragma: no cover

    @property
    def has_models(self):
        return self._model_sampler is not None

    @property
    def model_sampler(self):
        if not self.has_models:
            raise ValueError(f"This {self.__class__.__name__} does not have models")
        return self._model_sampler
    
    def random_split(self, lengths: Sequence[Union[int, float]]):
        subsets = random_split(self, lengths)
        return [FunctionSamplesMapSubset.from_subset(subset) for subset in subsets]


class FunctionSamplesMapDataset(FunctionSamplesMapDatasetBase):
    """A dataset class that holds function samples, as well as optionally
    their associated GP models and parameters.

    Attributes:
        model_sampler (RandomModelSampler): The random model sampler, if models
            are associated with the dataset.

    Example:
        ```
        rand_dataset = GaussianProcessRandomDataset(n_datapoints=15, dimension=5)
        function_samples_dataset = FunctionSamplesMapDataset.from_iterable_dataset(rand_dataset, 100)
        function_samples_dataset.save('path/to/directory')
        loaded_dataset = FunctionSamplesMapDataset.load('path/to/directory')
    """
    def __init__(self, data: Union[List[GPDatasetItem], List[tuple]],
                 model_sampler: Optional[RandomModelSampler]=None):
        """Initializes an instance of the class with the given data.

        Args:
            data: A list of tuples or GPDatasetItem containing the data.
            model_sampler: Optional RandomModelSampler instance to associate
        """
        if not (model_sampler is None or isinstance(model_sampler, RandomModelSampler)):
            raise ValueError("model_sampler should be None or a RandomModelSampler instance")

        if all(isinstance(x, GPDatasetItem) for x in data):
            if model_sampler is None:
                raise ValueError("model_sampler should not be None if all items are GPDatasetItem")
            for item in data:
                if item.model_index is None:
                    raise ValueError("model_index should be specified for each GPDatasetItem")
                else:
                    assert model_sampler._models[item.model_index] is item._model
        elif all(isinstance(x, tuple) for x in data):
            if model_sampler is None:
                # x_values, y_values
                assert all(len(x) == 2 for x in data)
            else:
                # x_values, y_values, model
                assert all(len(x) == 3 for x in data)
        else:
            raise ValueError("All items in data should be either GPDatasetItem or tuple")

        self._data = data
        self._model_sampler = model_sampler

    def __getitem__(self, index):
        if isinstance(index, slice):
            return super().__getitem__(index)
        return self._data[index]
    
    def __len__(self):
        return len(self._data)
    
    def data_is_loaded(self):
        return True

    @classmethod
    def load(cls, dir_name: str):
        """Loads a dataset from a given directory. The directory must contain a
        saved instance of FunctionSamplesMapDataset, including the data and
        optionally the models.

        Args:
            dir_name (str): The path to the directory from which the dataset
            should be loaded.

        Returns:
            FunctionSamplesMapDataset: The loaded dataset instance.
        """
        if not os.path.exists(dir_name): # Error if path doesn't exist
            raise FileNotFoundError(f"Path {dir_name} does not exist")
        if not os.path.isdir(dir_name): # Error if path isn't directory
            raise NotADirectoryError(f"Path {dir_name} is not a directory")

        list_of_dicts = torch.load(os.path.join(dir_name, "data.pt"))

        models_path = os.path.join(dir_name, "models.pt")
        if os.path.exists(models_path):
            models = torch.load(models_path)
            model_sampler = RandomModelSampler(models)

            data = []
            for item in list_of_dicts:
                model_index = item['model_index']
                data.append(GPDatasetItem(
                    item['x_values'], item['y_values'],
                    model=model_sampler._models[model_index],
                    model_params=item['model_params']))
            return cls(data, model_sampler)
        else:
            data = []
            for item in list_of_dicts:
                if 'model_index' in item or 'model_params' in item:
                    raise ValueError("Model information should not be present in the data if models are not saved.")
                data.append((item['x_values'], item['y_values']))
            return cls(data)
    
    @classmethod
    def from_iterable_dataset(cls, dataset: FunctionSamplesDataset,
                              n_realizations:Optional[int] = None,
                              verbose: bool = True):
        """Creates an instance of FunctionSamplesMapDataset from a given
        iterable-style FunctionSamplesDataset by sampling a specified number of
        data points.

        Args:
            dataset (FunctionSamplesDataset and IterableDataset):
                The dataset from which to generate samples.
                Must be both a FunctionSamplesDataset and a IterableDataset.
            n_realizations (int, positive):
                The number of function realizations (samples) to generate.
                Optional if the dataset has finite size, in which case the size
                of the dataset is used.

        Returns:
            FunctionSamplesMapDataset:
            A new instance of FunctionSamplesMapDataset containing the sampled
            function realizations.

        Example:
            ```
            dataset = GaussianProcessRandomDataset(n_datapoints=15, dimension=5)
            samples_dataset = FunctionSamplesMapDataset.from_iterable_dataset(dataset, 100)
        """
        if not (isinstance(dataset, FunctionSamplesDataset) and
                isinstance(dataset, IterableDataset)):
            raise TypeError("dataset should be an instance of both FunctionSamplesDataset and IterableDataset")
        message = f"Saving realizations from {dataset.__class__.__name__} into {cls.__name__}"
        items_generator, size = dataset._get_items_generator_and_size(
            n_realizations, verbose, verbose_message=message)
        data = list(items_generator)
        return cls(
            data, dataset.model_sampler if dataset.has_models else None)


class LazyFunctionSamplesMapDataset(FunctionSamplesMapDatasetBase):
    """A dataset class that lazily generates function samples.

    This class extends the `FunctionSamplesMapDatasetBase` class and provides
    lazy loading of function samples. It generates function samples on-the-fly
    when accessed, rather than loading all samples into memory at once.
    """
    def __init__(self, dataset: FunctionSamplesDataset,
                 n_realizations:Optional[int] = None):
        """
        Args:
            dataset (FunctionSamplesDataset):
                The underlying function samples dataset.
            n_realizations (Optional[int]):
                The number of realizations to generate.
                If None, all realizations will be generated.
        """
        if not (isinstance(dataset, FunctionSamplesDataset) and
                isinstance(dataset, IterableDataset)):
            raise TypeError("dataset should be an instance of both FunctionSamplesDataset and IterableDataset")
        items_generator, size = dataset._get_items_generator_and_size(
            n_realizations, verbose=False)
        self._items_generator = items_generator
        self._size = size
        self._data = [None] * size
        self._model_sampler = dataset.model_sampler if dataset.has_models else None
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return super().__getitem__(index)
        if self._data[index] is None:
            self._data[index] = next(self._items_generator)
        return self._data[index]
    
    def __len__(self):
        return self._size
    
    def data_is_loaded(self):
        return all(x is not None for x in self._data)


class FunctionSamplesMapSubset(Subset, FunctionSamplesMapDatasetBase):
    """A subset of a FunctionSamplesMapDatasetBase dataset."""
    def __init__(self, dataset: FunctionSamplesMapDatasetBase, indices: Sequence[int]) -> None:
        """
        Args:
            dataset (FunctionSamplesMapDatasetBase): The original dataset.
            indices (Sequence[int]): The indices of the samples in the subset.
        """
        if not isinstance(dataset, FunctionSamplesMapDatasetBase):
            raise ValueError("dataset should be an instance of FunctionSamplesMapDatasetBase")
        
        # Equivalent to self.dataset = dataset; self.indices = indices
        Subset.__init__(self, dataset, indices)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return FunctionSamplesMapDatasetBase.__getitem__(self, idx)
        return Subset.__getitem__(self, idx)
    
    def __len__(self):
        return Subset.__len__(self)
    
    # Can just define _model_sampler and then has_models and model_sampler are
    # inherited from FunctionSamplesMapDatasetBase.
    # random_split is also inherited from FunctionSamplesMapDatasetBase.
    @property
    def _model_sampler(self):
        return self.dataset._model_sampler
    
    def data_is_loaded(self):
        return self.dataset.data_is_loaded()
    
    @classmethod
    def from_subset(cls, subset: Subset):
        if not isinstance(subset, Subset):
            raise ValueError("subset should be an instance of torch.utils.data.Subset")
        return cls(subset.dataset, subset.indices)


class RepeatedFunctionSamplesIterableDataset(
    FunctionSamplesDataset, IterableDataset, SizedIterableMixin):
    """An iterable-style dataset that repeats the samples from another
    iterable-style dataset a specified number of times.
    With each iter(), there are different random samples from the base dataset
    which are repeated in a random order."""
    def __init__(self, base_dataset: FunctionSamplesDataset, size_factor: int):
        """
        Args:
            base_dataset (IterableDataset and FunctionSamplesDataset):
                A finite-sized iterable FunctionSamplesDataset
                from which to generate samples.
            size_factor (int): The number of times to repeat the samples.
        """
        if not (isinstance(base_dataset, FunctionSamplesDataset) and
                isinstance(base_dataset, IterableDataset)):
            raise TypeError("base_dataset should be an instance of " \
                            "both FunctionSamplesDataset and IterableDataset.")
        if not iterable_is_finite(base_dataset):
            raise ValueError(
                "base_dataset for a RepeatedFunctionSamplesIterableDataset " \
                "should be a finite-sized IterableDataset")
        if not isinstance(size_factor, int) or size_factor <= 0:
            raise ValueError("size_factor should be a positive integer")
        
        self.base_dataset = base_dataset
        self.size_factor = size_factor
    
    @property
    def _size(self):
        # Required by SizedIterableMixin
        return len(self.base_dataset) * self.size_factor
    
    def __iter__(self):
        this_iter_base = LazyFunctionSamplesMapDataset(self.base_dataset)
        sampler = torch.utils.data.RandomSampler(
            this_iter_base, replacement=False, num_samples=len(self))
        this_iter = DataLoader(this_iter_base, batch_size=None, sampler=sampler)
        return iter(this_iter)
    
    def has_models(self):
        return self.base_dataset.has_models
    
    def model_sampler(self):
        return self.base_dataset.model_sampler
    
    def random_split(self, lengths: Sequence[Union[int, float]]):
        # Need to convert from lengths to proportions if absolute lengths were
        # given...
        lengths_is_proportions = math.isclose(sum(lengths), 1) and sum(lengths) <= 1
        if not lengths_is_proportions:
            if sum(lengths) == len(self):
                lengths = [length / len(self) for length in lengths]
            else:
                # Assume that sum(lengths) == len(self.base_dataset)
                pass

        return [
            type(self)(split_dataset, self.size_factor)
            for split_dataset in self.base_dataset.random_split(lengths)]

    def data_is_loaded(self):
        return False


class ModelsWithParamsList:
    """
    A class representing a list of models with their corresponding parameters.
    """

    def __init__(self, models_and_params: List[Tuple[SingleTaskGP, dict]]):
        """Initializes a ModelsWithParamsList object.

        Args:
            models_and_params (List[Tuple[SingleTaskGP, dict]]): A list of tuples where each tuple contains a SingleTaskGP model and its corresponding parameters.
        """
        self._models_and_params = models_and_params
    
    def __getitem__(self, index):
        """Returns the model at the specified index or a slice of models.

        Args:
            index: The index or slice to retrieve the model(s) from.

        Returns:
            SingleTaskGP or ModelsWithParamsList:
            If `index` is an integer, the model at the specified index with its
            parameters initialized.
            If `index` is a slice, a new ModelsWithParamsList object containing
            the models in the specified slice.
        """
        if isinstance(index, slice):
            # `return [self[i] for i in range(*index.indices(len(self)))]``
            # would not work because there might be a single or a few model
            # instances that are shared among the items.
            # Instead, here is what should be done:
            return ModelsWithParamsList(self._models_and_params[index])
        
        model, params = self._models_and_params[index]
        model.initialize(**params)
        return model

    def __len__(self):
        return len(self._models_and_params)
    
    def __repr__(self):
        return f"ModelsWithParamsList({repr(self._models_and_params)})"

    def __str__(self):
        return f"ModelsWithParamsList({str(self._models_and_params)})"

    def __eq__(self, other):
        return type(self) == type(other) and self._models_and_params == other._models_and_params


class AcquisitionDataset(Dataset, ABC):
    """Abstract class for acquisition function datasets.
    Designed for training a "likelihood-free" DNN acquisition function.
    Generates training data consisting of historical observations and candidate
    points for acquisition function training.
    
    Attributes:
        give_improvements (bool): If True, the dataset includes improvement
            values as targets instead of raw y-values of the candidate points.
            Improvement is calculated as the positive difference between the
            candidate point's y-value and the current best observation.

    Yields:
        Tuple containing historical x-values, historical y-values, candidate
        x-values, and either candidate y-values or improvement values, depending
        on the value of `give_improvements`. If `dataset` has model info, then
        each item also includes the associated GP model.

    Example:
        dataset = GaussianProcessRandomDataset(
            n_datapoints=15, dimension=5, dataset_size=100)
        
        # Creating the training dataset for acquisition functions
        training_dataset = FunctionSamplesAcquisitionDataset(
            dataset=dataset, n_candidate_points=5, give_improvements=True)
        
        # Iterating over the dataset to train an acquisition function
        for x_hist, y_hist, x_cand, improvements, model in training_dataset:
            # Use x_hist, y_hist, x_cand, and improvements for training
            # and model for evaluation of the approximated acquisition function
            # x_hist shape: (n_hist, dimension)
            # y_hist shape: (n_hist,)
            # x_cand shape: (n_cand, dimension)
            # improvements shape: (n_cand,)
    """

    give_improvements: bool
    
    @abstractmethod
    def random_split(self, lengths: Sequence[Union[int, float]]):
        """Randomly splits the dataset into multiple subsets.

        Args:
            lengths (Sequence[Union[int, float]]): A sequence of lengths
            specifying the size of each subset, or the proportion of the
            dataset to include in each subset.
        """
        pass  # pragma: no cover
    
    ## TODO: Implement save method
    # def save(self, dir_name: str, n_samples:Optional[int]=None):
    #     """Saves the dataset to a specified directory. If the dataset includes
    #     a base FunctionSamplesDataset which has models, the models are saved as
    #     well. If the directory does not exist, it will be created.

    #     Args:
    #         dir_name (str):
    #             The directory where the samples should be saved.
    #         n_samples (int):
    #             The number of samples to save.
    #             If unspecified, all the samples are saved.
    #             If specified, the first n_samples samples are saved.
    #     """
    #     pass
    
    @staticmethod
    def _collate_train_acquisition_function_samples(samples_list, device=None):
        if isinstance(samples_list[0], AcquisitionDatasetModelItem):
            unzipped_lists_first_4 = list(zip(*
                    [x[:4] for x in samples_list]))
            models_list = ModelsWithParamsList(
                [(x._model, x.model_params) for x in samples_list])
            unzipped_lists = unzipped_lists_first_4 + [models_list]
        else:
            unzipped_lists = list(zip(*samples_list))
        # Each of these are tuples of tensors
        x_hists, y_hists, x_cands, vals_cands = unzipped_lists[:4]

        # x_hist shape: (n_hist, dimension)
        # y_hist shape: (n_hist,)
        # x_cand shape: (n_cand, dimension)
        # vals_cand shape: (n_cand,)

        x_hist = max_pad_tensors_batch(x_hists, add_mask=False)
        y_hist, hist_mask = max_pad_tensors_batch(y_hists, add_mask=True)
        x_cand = max_pad_tensors_batch(x_cands, add_mask=False)
        vals_cand, cand_mask = max_pad_tensors_batch(vals_cands, add_mask=True)

        if device is not None:
            x_hist = x_hist.to(device)
            y_hist = y_hist.to(device)
            x_cand = x_cand.to(device)
            vals_cand = vals_cand.to(device)
            if hist_mask is not None:
                hist_mask = hist_mask.to(device)
            if cand_mask is not None:
                cand_mask = cand_mask.to(device)

        return [x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask] + unzipped_lists[4:]
    
    def get_dataloader(self, batch_size=32, device=None, shuffle=True, **kwargs):
        """Returns a DataLoader object for the dataset.

        Args:
            batch_size (int): The batch size for the DataLoader. Default is 32.
            device (torch.device): The device to move the tensors to.
            shuffle (bool): Whether to shuffle the data. Default is True.
            **kwargs: Additional keyword arguments to be passed to the DataLoader constructor.

        Raises:
            ValueError: If 'collate_fn' is specified in kwargs.

        Returns:
            DataLoader: A DataLoader object for the dataset, where
            each batch contains a list of tensors
            [x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask, models] or
            [x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask] if models
            are not associated with the dataset.
            
            x_hist has shape (batch_size, n_hist, dimension),
            y_hist and hist_mask have shape (batch_size, n_hist),
            x_cand has shape (batch_size, n_cand, dimension),
            vals_cand and cand_mask have shape (batch_size, n_cand),
            and models is a batch_size length tuple of GP models associated
            with the dataset.
            Everything is padded with zeros along with the corresponding masks.
        """
        if 'collate_fn' in kwargs:
            raise ValueError("collate_fn should not be specified in get_dataloader; we do it for you")
        collate_fn = self._collate_train_acquisition_function_samples
        if device is not None:
            collate_fn = partial(collate_fn, device=device)
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_fn,
                          **kwargs)


class FunctionSamplesAcquisitionDataset(
    AcquisitionDataset, IterableDataset, SizedIterableMixin):
    """An IterableDataset designed for training a "likelihood-free" DNN
    acquisition function.
    It processes a FunctionSamplesDataset instance to generate training data
    consisting of historical observations and candidate points for acquisition
    function evaluation. The data is generated randomly on-the-fly.

    Attributes:
        give_improvements (bool): If True, the dataset includes improvement
            values as targets instead of raw y-values of the candidate points.
            Improvement is calculated as the positive difference between the
            candidate point's y-value and the current best observation.
        n_candidate_points
        n_samples
        min_n_candidates
        dataset_size_factor
    """
    def __init__(self,
                 dataset: FunctionSamplesDataset,
                 n_candidate_points:Union[int,str,Sequence[int]]=1,
                 n_samples:str="all", give_improvements:bool=True,
                 min_n_candidates=2,
                 dataset_size_factor:Optional[int]=None):
        r"""
        Args:
            dataset (FunctionSamplesDataset):
                The base dataset from which to generate training data for
                acquisition functions.
            
            n_candidate_points (default: 1): The number of candidate points
                to generate for each training example.
                Can be:
                    - A positive integer, in which case the number of candidate
                        points is fixed to that value.
                    - A tuple of two positive integers (min, max), in which case
                        the number of candidate points is chosen uniformly at
                        random from min to max.
                    - A string "uniform", in which case the number of candidate
                        points is chosen uniformly at random from
                        `[min_n_candidates...n_samples-1]`.
                    - A string "binomial", in which case the number of candidate
                        points is chosen from a binomial distribution with
                        parameters n_samples and 0.5, conditioned on being
                        between `min_n_candidates` and n_samples-1.

            n_samples (str; default: "all"): The number of samples to use from
                the dataset each iteration.
                - If "all", all samples are used.
                - If "uniform", a uniform random number of samples is used each
                iteration. Specifically,
                    - If n_candidate_points is "uniform" or "binomial", then
                    n_samples is chosen uniformly at random in
                    [min_n_candidates+1...n], where n is the number of samples
                    in the dataset iteration, and then n_candidate_points is
                    chosen based on that.
                    - If n_candidate_points is an integer or a tuple of two
                    integers, then n_candidate_points is first chosen and then
                    n_samples is chosen uniformly in [n_candidate_points+1...n].
            
            give_improvements (bool): Whether to generate improvement values as
                targets instead of raw y-values of the candidate points.
            
            min_n_candidates (int): The minimum number of candidate points for
                every iteration. Only used if n_candidate_points is "uniform" or
                "binomial"; ignored otherwise.
            
            dataset_size_factor (Optional[int]): If the base dataset is a
                map-style dataset
                (i.e. a FunctionSamplesMapDataset or FunctionSamplesMapSubset),
                this parameter specifies the expansion factor for the dataset
                size. The dataset size is determined by the size of the base
                dataset multiplied by this factor. Default is 1.
                If the base dataset is an iterable-style dataset
                (i.e. GaussianProcessRandomDataset), then this
                parameter should not be specified.
        """
        # whether to generate `n_candidates` first or not
        self._gen_n_candidates_first = True
        if isinstance(n_candidate_points, str):
            if not (n_candidate_points == "uniform" or n_candidate_points == "binomial"):
                raise ValueError(f"Invalid value for n_candidate_points: {n_candidate_points}")
            self._gen_n_candidates_first = False
        elif isinstance(n_candidate_points, int):
            if n_candidate_points <= 0:
                raise ValueError(f"n_candidate_points should be positive, but got n_candidate_points={n_candidate_points}")
            self._gen_n_candidates = lambda: n_candidate_points
        else: # n_candidate_points is a tuple (or list) of two integers
            try:
                if not (len(n_candidate_points) == 2 and
                        isinstance(n_candidate_points[0], int) and
                        isinstance(n_candidate_points[1], int) and
                        1 <= n_candidate_points[0] <= n_candidate_points[1]):
                    raise ValueError(f"n_candidate_points should be a positive integer or a tuple of two integers, but got n_candidate_points={n_candidate_points}")
                self._gen_n_candidates = get_uniform_randint_generator(*n_candidate_points)
            except TypeError:
                raise ValueError(f"n_candidate_points should be a string, positive integer, tuple of two integers, but got n_candidate_points={n_candidate_points}")
        self.n_candidate_points = n_candidate_points

        if n_samples == "all":
            self.n_samples = "all"
        elif n_samples == "uniform":
            self.n_samples = "uniform"
        else:
            raise ValueError(f"Invalid value for n_samples: {n_samples}")
        
        if not isinstance(min_n_candidates, int) or min_n_candidates <= 0:
            raise ValueError(f"min_n_candidates should be a positive integer, but got min_n_candidates={min_n_candidates}")
        self.min_n_candidates = min_n_candidates
        
        if not isinstance(give_improvements, bool):
            raise TypeError(f"give_improvements should be a boolean value, but got give_improvements={give_improvements}")
        self.give_improvements = give_improvements

        if not isinstance(dataset, FunctionSamplesDataset):
            raise TypeError(f"dataset should be an instance of FunctionSamplesDataset, but got {dataset=}")

        # Need to save these so that we can copy in random_split
        self.base_dataset = dataset
        self.dataset_size_factor = dataset_size_factor

        if dataset_size_factor is None:
            dataset_size_factor = 1
        elif not isinstance(dataset_size_factor, int) or dataset_size_factor <= 0:
            raise ValueError(f"dataset_size_factor should be a positive integer, but got {dataset_size_factor=}")

        self._dataset_is_iterable_style = isinstance(dataset, IterableDataset)

        if self._dataset_is_iterable_style: # e.g. GaussianProcessRandomDataset
            # Then it has __iter__ because IterableDataset is a subclass of
            # Iterable, but it should not have or not implement __getitem__.
            # However, can't check that it doesn't have __getitem__ because it
            # could be that it does have it but it's not implemented.

            if dataset_size_factor == 1:
                self._data_iterable = dataset
            else:
                self._data_iterable = RepeatedFunctionSamplesIterableDataset(
                    dataset, dataset_size_factor)
            
            self._size = len_or_inf(self._data_iterable)

            if n_samples == "uniform" and isinstance(dataset, GaussianProcessRandomDataset):
                warnings.warn("n_samples='uniform' for GaussianProcessRandomDataset is supported but wasteful. Consider using n_samples='all' and setting n_datapoints_random_gen in the dataset instead.")
        else: # e.g. FunctionSamplesMapDataset
            # Then it should be a map-style dataset and have __getitem__.
            assert callable(getattr(dataset, "__getitem__", None))

            base_dataset_size = len(dataset)
            if not isinstance(base_dataset_size, int) or base_dataset_size <= 0:
                raise ValueError(f"len(dataset) should be a positive integer, but got len(dataset)={base_dataset_size}")
            if dataset_size_factor == 1:
                self._size = base_dataset_size
                self._data_iterable = DataLoader(
                    dataset, batch_size=None, shuffle=True)
            else:
                self._size = dataset_size_factor * base_dataset_size
                self._data_iterable = DataLoader(
                    dataset, batch_size=None,
                    sampler=torch.utils.data.RandomSampler(
                        dataset, replacement=False, num_samples=self._size))
    
    # __len__ is implemented by SizedIterableMixin
    
    def copy_with_expanded_size(self, size_factor: int) -> "FunctionSamplesAcquisitionDataset":
        """Creates a copy of the dataset with an expanded size.

        Args:
            size_factor (int):
                The factor by which to expand the size of the dataset.

        Returns:
            FunctionSamplesAcquisitionDataset:
                A new instance of the dataset with the expanded size.
        """
        return type(self)(
                self.base_dataset,
                self.n_candidate_points,
                self.n_samples, self.give_improvements, self.min_n_candidates,
                size_factor)
    
    def copy_with_new_size(self, size: Optional[int] = None) -> "FunctionSamplesAcquisitionDataset":
        """Creates a copy of the dataset with a new size.
        If the base dataset has the `copy_with_new_size` method then it is used
        to create a copy of the base dataset with the specified size.
        Otherwise, the dataset must be a map-style dataset in which case it is
        expanded by a factor such that the new size is at least the specified
        size.

        Args:
            size (int or None): the new size of the dataset. If None, the size
                is not changed.

        Returns:
            FunctionSamplesAcquisitionDataset:
                A new instance of the dataset with the specified size.
        """
        if self._dataset_is_iterable_style:
            return type(self)(
                resize_iterable(self.base_dataset, size),
                self.n_candidate_points,
                self.n_samples, self.give_improvements, self.min_n_candidates)
        else:
            if size is None:
                size = len(self.base_dataset)
            if not isinstance(size, int) or size <= 0:
                raise ValueError("size should be a positive integer or None")
            return self.copy_with_expanded_size(math.ceil(size / len(self.base_dataset)))

    def _pick_random_n_samples_and_n_candidates(self, n_datapoints_original):
        if self._gen_n_candidates_first:
            # generate n_candidates first; either fixed or random
            n_candidates = self._gen_n_candidates()

            # Need to have at least 1 history point
            if not (n_candidates+1 <= n_datapoints_original):
                raise ValueError(f"n_datapoints_original={n_datapoints_original} should be at least n_candidates+1={n_candidates+1}")

            # generate n_samples
            if self.n_samples == "all":
                n_samples = n_datapoints_original
            elif self.n_samples == "uniform":
                n_samples = uniform_randint(n_candidates+1, n_datapoints_original)
        else:
            # n_candidates is "uniform" or "binomial"

            min_n_candidates = self.min_n_candidates

            if not (min_n_candidates+1 <= n_datapoints_original):
                raise ValueError(f"n_datapoints_original={n_datapoints_original} should be at least min_n_candidates+1={min_n_candidates+1}")

            # generate n_samples first; either "all" or "uniform"
            if self.n_samples == "all":
                n_samples = n_datapoints_original
            elif self.n_samples == "uniform":
                n_samples = uniform_randint(min_n_candidates+1, n_datapoints_original)

            # generate n_candidates
            if self.n_candidate_points == "uniform":
                n_candidates = uniform_randint(min_n_candidates, n_samples-1)
            elif self.n_candidate_points == "binomial":
                n_candidates = int(torch.distributions.Binomial(n_samples, 0.5).sample())
                while not (min_n_candidates <= n_candidates <= n_samples-1):
                    n_candidates = int(torch.distributions.Binomial(n_samples, 0.5).sample())
        
        if torch.is_tensor(n_samples):
            n_samples = n_samples.item()
        return n_samples, n_candidates

    def __iter__(self):
        has_models = self.base_dataset.has_models
        
        # x_values has shape (n_datapoints, dimension)
        # y_values has shape (n_datapoints,)
        for item in self._data_iterable:
            if has_models:
                if isinstance(item, GPDatasetItem):
                    x_values, y_values = item.x_values, item.y_values
                    model = item._model
                    model_params = item.model_params
                else:
                    x_values, y_values, model = item
                    # need to copy the data, otherwise everything will be same
                    model_params = {name: param.detach().clone()
                                    for name, param in model.named_parameters()}
            else:
                x_values, y_values = item
            
            n_datapoints = x_values.shape[0]

            n_samples, n_candidate = self._pick_random_n_samples_and_n_candidates(n_datapoints)

            rand_idx = torch.randperm(n_datapoints)
            candidate_idx = rand_idx[:n_candidate]
            hist_idx = rand_idx[n_candidate:n_samples]

            x_hist, y_hist = x_values[hist_idx], y_values[hist_idx]
            x_cand = x_values[candidate_idx]
            y_cand = y_values[candidate_idx]

            if self.give_improvements:
                best_f = y_hist.amax(0, keepdim=False) # both T and F work
                improvement_values = torch.nn.functional.relu(
                    y_cand - best_f, inplace=True)
                vals_cand = improvement_values
            else:
                vals_cand = y_cand

            if has_models:
                yield AcquisitionDatasetModelItem(
                    x_hist, y_hist, x_cand, vals_cand, model, model_params)
            else:
                yield x_hist, y_hist, x_cand, vals_cand

    def random_split(self, lengths: Sequence[Union[int, float]]):
        # Need to convert from lengths to proportions if absolute lengths were
        # given...
        lengths_is_proportions = math.isclose(sum(lengths), 1) and sum(lengths) <= 1
        if not lengths_is_proportions:
            if sum(lengths) == len(self):
                lengths = [length / len(self) for length in lengths]
            else:
                # Assume that sum(lengths) == len(self.base_dataset)
                pass

        return [
            type(self)(split_dataset, self.n_candidate_points, self.n_samples,
                       self.give_improvements, self.min_n_candidates,
                       self.dataset_size_factor)
            for split_dataset in self.base_dataset.random_split(lengths)]

    def get_dataloader(self, batch_size=32, device=None, **kwargs):
        """Returns a DataLoader object for the dataset.

        Args:
            batch_size (int): The batch size for the DataLoader. Default is 32.
            device (torch.device): The device to move the tensors to.
            **kwargs: Additional keyword arguments to be passed to the DataLoader constructor.

        Raises:
            ValueError: If 'collate_fn' is specified in kwargs.
            ValueError: If 'shuffle' is specified as True in kwargs.

        Returns:
            DataLoader: A DataLoader object for the dataset, where
            each batch contains a list of tensors
            [x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask, models] or
            [x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask] if models
            are not associated with the dataset.
            
            x_hist has shape (batch_size, n_hist, dimension),
            y_hist and hist_mask have shape (batch_size, n_hist),
            x_cand has shape (batch_size, n_cand, dimension),
            vals_cand and cand_mask have shape (batch_size, n_cand),
            and models is a batch_size length tuple of GP models associated
            with the dataset.
            Everything is padded with zeros along with the corresponding masks.
        """
        shuffle = kwargs.pop("shuffle", False)
        if shuffle:
            # Can't do shuffle=True on a IterableDataset
            raise ValueError(f"shuffle should not be specified as True in {self.__class__.__name__}.get_dataloader; the dataset is already shuffled")
        return super().get_dataloader(batch_size=batch_size, device=device,
                                      shuffle=False, **kwargs)
