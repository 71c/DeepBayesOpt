from typing import Optional, List, Union
from collections.abc import Sequence
from dataset_with_models import TupleWithModel, create_classes, RandomModelSampler
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.distributions import Uniform, Normal, Independent, Distribution
from botorch.models.gp_regression import SingleTaskGP
from botorch.exceptions import UnsupportedError, BotorchTensorDimensionError
from botorch.models.transforms.outcome import OutcomeTransform
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from utils import SizedInfiniteIterableMixin, SizedIterableMixin, get_lengths_from_proportions, iterable_is_finite
import math


class FunctionSamplesItem(TupleWithModel):
    args_names = ['x_values', 'y_values']
    kwargs_names = []


(FunctionSamplesDataset,
 MapFunctionSamplesDataset,
 ListMapFunctionSamplesDataset,
 LazyMapFunctionSamplesDataset,
 MapFunctionSamplesSubset) = create_classes(
     dataset_base_name='FunctionSamplesDataset',
     dataset_base_docstring=r"""A dataset class for function samples.

    It is expected that the yielded values are either
    (x_values, y_values, model) or (x_values, y_values), where
    - x_values: A tensor of shape (n_datapoints, dimension)
    - y_values: A tensor of shape (n_datapoints,)
    - model: A SingleTaskGP instance that was used to generate the data

    Subclasses include:
    - GaussianProcessRandomDataset, which represents an iterable-style
      dataset where the samples are generated from random Gaussian processes,
      and each iter() can be either finite or infinite in length
    - ListMapFunctionSamplesDataset, which represents a map-style dataset where
        the samples are stored in memory and can be accessed by index.
    
    Usage example:
    ```
    dataset = some FunctionSamplesDataset instance
    for x_values, y_values, model in dataset:
        # x_values has shape (n_datapoints, dimension)
        # y_values has shape (n_datapoints,)
        # do something with x_values, y_values, and model
    ```
    """,

     map_dataset_base_name='MapFunctionSamplesDataset',

     list_dataset_name='ListMapFunctionSamplesDataset',
     list_dataset_docstring=r"""A class for `FunctionSamplesDataset` datasets that hold function
    samples in a list and can be accessed by index.
    
    Example:
        ```
        rand_dataset = GaussianProcessRandomDataset(n_datapoints=15, dimension=5)
        function_samples_dataset = ListMapFunctionSamplesDataset.from_iterable_dataset(rand_dataset, 100)
        function_samples_dataset.save('path/to/directory')
        loaded_dataset = ListMapFunctionSamplesDataset.load('path/to/directory')
    """,
     
     lazy_dataset_name='LazyMapFunctionSamplesDataset',
     lazy_dataset_docstring=r"""A dataset class that lazily generates function samples.

    This class extends the `MapFunctionSamplesDataset` class and provides
    lazy loading of function samples. It generates function samples on-the-fly
    when accessed, rather than loading all samples into memory at once.
    """,
     
     map_subset_name='MapFunctionSamplesSubset',
     
     tuple_class=FunctionSamplesItem)




###### Current implementation of outcome transform of datasets follows.
###### TODO if needed: Generalize these to also optionally have input transform.
###### Could alternatively just do away with the OutcomeTransform and
###### InputTransform and just use a function (x,y) --> (x,y) or
###### FunctionSamplesItem --> FunctionSamplesItem,
###### or (less generally) two functions x --> x and y --> y.

@staticmethod
def _get_outcome_transform(transform: OutcomeTransform):
    """Get a function that transforms the outcome of an item."""
    def transform(item: FunctionSamplesItem):
        X, Y = item.tuple_no_model

        # Doing this silly check and stuff just to make sure that
        # Y has one or more output dimensions, just in case the
        # OutcomeTransform checks for this e.g. Standardize.
        # Most don't, such as Power, though, but it's still good to do this.
        y_has_no_output_dim = False
        if X.dim() != Y.dim():
            if (X.dim() - Y.dim() == 1) and (X.shape[:-1] == Y.shape):
                y_has_no_output_dim = True
                Y = Y.unsqueeze(-1)
            else:
                message = (
                    "Expected X and Y to have the same number of dimensions"
                    f" (got X with dimension {X.dim()} and Y with dimension"
                    f" {Y.dim()}).")
                raise BotorchTensorDimensionError(message)
        
        Y_tf, Yvar = transform(Y, None)
        assert Y_tf.shape == Y.shape
        if y_has_no_output_dim:
            Y_tf = Y_tf.squeeze(-1)

        return FunctionSamplesItem(X, Y_tf)
    return transform

FunctionSamplesDataset._get_outcome_transform = _get_outcome_transform


# See here for OutcomeTransform documentation:
# https://github.com/pytorch/botorch/blob/main/botorch/models/transforms/outcome.py
class TransformedFunctionSamplesIterableDataset(
    FunctionSamplesDataset, IterableDataset):
    def __init__(self, base_dataset: FunctionSamplesDataset,
                 transform):
        if not isinstance(base_dataset, FunctionSamplesDataset):
            raise ValueError("base_dataset must be a FunctionSamplesDataset")
        if not isinstance(base_dataset, IterableDataset):
            raise ValueError("base_dataset must be a IterableDataset")
        self.base_dataset = base_dataset
        self.transform = transform
        # Currently unknown whether this dataset has models
        self._has_models = None
    
    def __iter__(self):
        for item in self.base_dataset:
            transformed_item = self.transform(item)
            if self._has_models is None:
                self._has_models = transformed_item.has_model
            else:
                assert self._has_models == transformed_item.has_model
            yield transformed_item
    
    @property
    def has_models(self):
        if self._has_models is None:
            raise ValueError(
                "It is currently unknown whether this "
                f"{self.__class__.__name__} has models or not -- get at least "
                "one value from an iter() first.")
        return self._has_models
    
    @property
    def model_sampler(self):
        if not self.has_models:
            raise ValueError(f"This {self.__class__.__name__} does not have models")
        return self.base_dataset.model_sampler
    
    # Must not forget this one!
    def __len__(self):
        return self.base_dataset.__len__()
    
    def random_split(self, lengths):
        return [self.__class__(split_dataset, self.transform)
                for split_dataset in self.base_dataset.random_split(lengths)]
    
    def data_is_loaded(self):
        return self.base_dataset.data_is_loaded()
    
    @property
    def data_is_fixed(self):
        return self.base_dataset.data_is_fixed
    
    def _init_params(self):
        return (self.base_dataset, self.transform), {}
    
    def save(self, dir_name: str, verbose:bool=True):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support saving to a file")
    
    def save_samples(self, dir_name: str, n_realizations:Optional[int]=None,
             verbose:bool=True):
        # Get whether has models
        if self._has_models is None:
            next(iter(self)) # It should set the attribute
        
        if self.has_models:
            raise NotImplementedError(
            f"{self.__class__.__name__} does not support saving samples to a "
            "file if it has models")
        super().save_samples(dir_name, n_realizations, verbose)



def _transform_default_iterable(self, transform):
    """Transform the items of the dataset.

    Args:
        transform: A function mapping FunctionSamplesItem to FunctionSamplesItem
        that transforms the items.
    
    Returns:
        A new dataset where its items are transformed by the given transform.
        If the dataset is iterable-style, then the new dataset is also
        iterable-style. If the dataset is map-style, then the new dataset is
        expected to be map-style.
    """
    if isinstance(self, IterableDataset):
        return TransformedFunctionSamplesIterableDataset(self, transform)
    raise NotImplementedError(
        f"{self.__class__.__name__} has not implemented transform")

FunctionSamplesDataset.transform = _transform_default_iterable


# FunctionSamplesDataset.transform_outcomes = ...


class TransformedListMapFunctionSamplesDataset(MapFunctionSamplesDataset):
    def __init__(self, base_dataset: ListMapFunctionSamplesDataset,
                 transform):
        if not isinstance(base_dataset, ListMapFunctionSamplesDataset):
            raise ValueError("base_dataset must be a ListMapFunctionSamplesDataset")
        self.dataset = base_dataset
        self.transform = transform
        self._data = [transform(item) for item in base_dataset._data]

## TODO: Finish revising all of these classes and functions!

def _transform_listmap(self, transform):
    """Transform the items of the dataset.

    Args:
        transform: A function mapping FunctionSamplesItem to FunctionSamplesItem
        that transforms the items.
    
    Returns:
        A new ListMapFunctionSamplesDataset where its items are transformed
        by the given transform.
"""
    transformed_data = [transform(item) for item in self._data]
    if all(item.has_model for item in transformed_data):

    return ListMapFunctionSamplesDataset(transformed_data)

ListMapFunctionSamplesDataset.transform_outcomes = _transform_outcomes_listmap


class TransformedLazyMapFunctionSamplesDataset(MapFunctionSamplesDataset):
    def __init__(self, base_dataset: LazyMapFunctionSamplesDataset,
                 transform):
        if not isinstance(base_dataset, LazyMapFunctionSamplesDataset):
            raise ValueError("base_dataset must be a LazyMapFunctionSamplesDataset")
        self.dataset = base_dataset
        self.transform = transform
        self._data = [
            transform(item) if item is not None
            else item for item in base_dataset._data]
    
    @property
    def _model_sampler(self):
        return self.dataset._model_sampler

    def _init_params(self):
        return (self.dataset, self.transform), {}
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._map_base_class.__getitem__(self, index)
        if self._data[index] is None:
            self._data[index] = self._transform_outcome_of_item(
                self.dataset[index], self.transform)
        return self._data[index]

    def __len__(self) -> int:
        # Could also compute it as len(self._data), either way works
        return self.dataset.__len__()
    
    def data_is_loaded(self) -> bool:
        # Even if some of self._data are None, then as long as all of the base
        # dataset's items are loaded, then, since we assume that the transform
        # is cheap to compute, then the data is basically loaded.
        return self.dataset.data_is_loaded()


def _transform_outcomes_lazymap(self, transform: OutcomeTransform):
    """Transform the outcomes of the dataset.

    Args:
        transform: An OutcomeTransform instance that transforms the outcomes.
    
    Returns:
        A OutcomeTransformedLazyMapFunctionSamplesDataset instance (which
        behaves exactly like a LazyMapFunctionSamplesDataset), where its
        outcomes are transformed by the given transform.
    """
    return OutcomeTransformedLazyMapFunctionSamplesDataset(self, transform)

LazyMapFunctionSamplesDataset.transform_outcomes = _transform_outcomes_lazymap
#######################



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
        r"""Create a dataset that generates random Gaussian Process data.
        
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
                Is only used if xvalue_distribution is "uniform" or "normal",
                or if models is None; in these cases, it is required.
                Otherwise, it is ignored.
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

            # SingleTaskGP doesn't support initializing with no data.
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

            model = SingleTaskGP(train_X, train_Y, likelihood=likelihood)
            # hack to make it as if we initialized without any data
            model.train_inputs = None
            model.train_targets = None

            models = [model]

        if device is not None:
            for model in models:
                model.to(device)
        
        for i, model in enumerate(models):
            if not isinstance(model, SingleTaskGP):
                raise UnsupportedError(f"models[{i}] should be a SingleTaskGP instance.")

            # Verify that the model is single-batch
            if len(model.batch_shape) != 0:
                raise UnsupportedError(f"All models must be single-batch, but models[{i}] has batch shape {model.batch_shape}")
            # Verify that the model is single-output
            if model.num_outputs != 1:
                raise UnsupportedError(f"All models must be single-output, but models[{i}] is {model.num_outputs}-output")

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
        
        self._model_sampler = RandomModelSampler(
            models, model_probabilities, randomize_params=randomize_params)

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
            observation_noise=self.observation_noise,
            xvalue_distribution=self.xvalue_distribution,
            models=self.model_sampler.initial_models,
            model_probabilities=self.model_sampler.model_probabilities,
            set_random_model_train_data=self.set_random_model_train_data,
            dataset_size=self._size,
            randomize_params=self.model_sampler.randomize_params
        )

    def save(self, dir_name: str, verbose:bool=True):
        raise NotImplementedError("GaussianProcessRandomDataset does not support saving to a file")

    def copy_with_new_size(self, dataset_size:int):
        """Create a copy of the dataset with a new dataset size.

        Args:
            dataset_size (int): The new dataset size for the copied dataset.

        Returns:
            GaussianProcessRandomDataset: A new instance of the dataset with the
            specified dataset size.
        """
        t, d = self._get_params_dict()
        d['dataset_size'] = dataset_size
        return GaussianProcessRandomDataset(*t, **d)
    
    def random_split(self, lengths: Sequence[Union[int, float]]):
        # Same check that pytorch does in torch.utils.data.random_split
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
        lengths_is_proportions = math.isclose(sum(lengths), 1) and sum(lengths) <= 1

        dataset_size = self._size
        if dataset_size == math.inf:
            if lengths_is_proportions:
                raise ValueError(
                    "The GaussianProcessRandomDataset should not be infinite if lengths is a list of proportions")
        else:
            if lengths_is_proportions:
                lengths = get_lengths_from_proportions(dataset_size, lengths)
            
            if sum(lengths) != dataset_size:
                raise ValueError(
                    "Sum of input lengths does not equal the dataset size!")
        return [self.copy_with_new_size(length) for length in lengths]

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
        
        with gpytorch.settings.prior_mode(True): # sample from prior
            prior = model.posterior(
                x_values, observation_noise=self.observation_noise)

        # shape (n_datapoints, 1)
        y_values = prior.sample(torch.Size([]))
        assert y_values.dim() == 2 and y_values.size(1) == 1
        y_values_1dim = y_values.squeeze(1)

        if self.set_random_model_train_data:
            # As a hack, need to remove last dimension of y_values because
            # set_train_data isn't really supported in BoTorch
            model.set_train_data(
                x_values, y_values_1dim, strict=False)

        return FunctionSamplesItem(x_values, y_values_1dim, model)


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
    
    def save(self, dir_name: str, verbose:bool=True):
        raise NotImplementedError(
            "RepeatedFunctionSamplesIterableDataset does not support saving to a file")
    
    @property
    def data_is_fixed(self):
        return self.base_dataset.data_is_fixed

    def data_is_loaded(self):
        return self.base_dataset.data_is_loaded()
    
    def _init_params(self):
        return (self.base_dataset, self.size_factor), dict()
    
    @property
    def _size(self):
        # Required by SizedIterableMixin
        return len(self.base_dataset) * self.size_factor
    
    def __iter__(self):
        this_iter_base = LazyMapFunctionSamplesDataset(self.base_dataset)
        sampler = torch.utils.data.RandomSampler(
            this_iter_base, replacement=False, num_samples=len(self))
        this_iter = DataLoader(this_iter_base, batch_size=None, sampler=sampler)
        return iter(this_iter)
    
    def has_models(self):
        return self.base_dataset.has_models
    
    @property
    def _model_sampler(self):
        return self.base_dataset._model_sampler
    
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

