from abc import ABC, abstractmethod
from typing import Optional, List, Union
from collections.abc import Sequence

import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, IterableDataset, random_split, Subset

from botorch.models.gp_regression import SingleTaskGP
from botorch.exceptions import UnsupportedError
import pyro

from utils import resize_iterable, iterable_is_finite


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
    def __init__(self, *items, model:Optional[SingleTaskGP]=None,
                 model_params:Optional[dict]=None, items_list=None):
        if items_list is not None:
            assert len(items) == 0
            items = tuple(items_list)
        self._items = items
        self._model = model
        self._indices = list(range(len(self)))
        if self.has_model:
            if not isinstance(model, SingleTaskGP):
                raise ValueError("model should be a SingleTaskGP instance.")
            if model_params is None:
                # need to copy the data, otherwise everything will be the same
                model_params = {name: param.detach().clone()
                                for name, param in model.named_parameters()}
            self.model_params = model_params
            self.model_index = model.index if hasattr(model, "index") else None
    
    @property
    def has_model(self):
        return self._model is not None
    
    @property
    def model(self):
        if not self.has_model:
            raise AttributeError(
                "This TupleWithModel instance does not have a model.")
        self._model.initialize(**self.model_params)
        return self._model

    @property
    def _tuple(self):
        if not self.has_model:
            return self._items
        return self._items + (self.model,)
    
    def __getitem__(self, index):
        # return self._tuple[index] # basically equivalent to the below

        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))

        # This is to handle the edge case of -1, -2, etc.
        index = self._indices[index]
        
        if self.has_model and index == len(self._items):
            return self.model
        return self._items[index]
    
    def __len__(self):
        return len(self._items) + (1 if self.has_model else 0)

    def __str__(self):
        return str(self._tuple)

    def __repr__(self):
        return repr(self._tuple)
    
    def __eq__(self, other):
        return type(self) == type(other) and self._tuple == other._tuple
    
    def to_dict(self):
        if not self.has_model:
            return {
                'items_list': list(self._items)
            }
        return {
            'items_list': list(self._items),
            'model_index': self.model_index,
            'model_params': self.model_params
        }


def add_indent(s):
    return '\n'.join(['  ' + line for line in s.split('\n')])


class DatasetWithModels(Dataset, ABC):
    """A base class for datasets that have models.

    Subclasses should implement the `random_split` and `data_is_loaded` methods.

    Subclasses should have the `_model_sampler` attribute, which is a
    `RandomModelSampler` instance if models are associated with the dataset
    and is `None` otherwise.
    
    This class also has the instance method `_get_items_generator_and_size` that
    can be used in subclasses."""

    # subclass of TupleWithModel that the dataset holds
    _tuple_class: type
    
    # whatever concrete class corresponds to DatasetWithModels,
    # to be used in subclasses of that class
    _base_class: type

    @abstractmethod
    def random_split(self, lengths: Sequence[Union[int, float]]) -> List['DatasetWithModels']:
        """Randomly splits the dataset into multiple subsets.

        Args:
            lengths (Sequence[Union[int, float]]): A sequence of lengths
            specifying the size of each subset, or the proportion of the
            dataset to include in each subset.
        """
        pass  # pragma: no cover

    @abstractmethod
    def data_is_loaded(self) -> bool:
        """Returns whether the data is loaded in memory or not."""
        pass  # pragma: no cover

    @abstractmethod
    def _init_params(self) -> tuple[tuple, dict]:
        """Returns a tuple of the arguments and keyword arguments that are
        passed to the constructor of the class."""
        pass  # pragma: no cover

    @classmethod
    def _str_helper(cls, args, kwargs, is_str, include_class=True):
        if is_str:
            def f(x):
                if isinstance(x, str):
                    return repr(x)
                return str(x)
        else:
            f = repr
        
        args_str = ',\n'.join(f(arg) for arg in args)
        kwargs_str = ',\n'.join(f"{key}={f(value)}" for key, value in kwargs.items())

        things = []
        if args_str != '':
            things.append(add_indent(args_str))
        if kwargs_str != '':
            things.append(add_indent(kwargs_str))

        parts = ',\n'.join(things)

        if not include_class:
            return parts

        return f"{cls.__name__}(\n{parts}\n)"
    
    def __repr__(self):
        args, kwargs = self._init_params()
        return self._str_helper(args, kwargs, is_str=False)
    
    def __str__(self):
        args, kwargs = self._init_params()
        return self._str_helper(args, kwargs, is_str=True)

    def __new__(cls, *args, **kwargs):
        # Ensure the required class attribute is set correctly
        if not (hasattr(cls, '_tuple_class') and
                issubclass(cls._tuple_class, TupleWithModel)):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_tuple_class' that is a subclass of TupleWithModel.")
        if not (hasattr(cls, '_base_class')):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_base_class'.")
        return Dataset.__new__(cls)

    @property
    def has_models(self):
        if not (hasattr(self, "_model_sampler") and
                isinstance(self._model_sampler, (type(None), RandomModelSampler))):
            raise AttributeError(
                f"{self.__class__.__name__}, a subclass of {self._base_class.__name__}, "\
                    "must have '_model_sampler' attribute that is a RandomModelSampler instance or None.")
        return self._model_sampler is not None

    @property
    def model_sampler(self):
        if not self.has_models:
            raise ValueError(f"This {self.__class__.__name__} does not have models")
        return self._model_sampler

    @classmethod
    def _get_items_generator(cls, dataset_resized, verbose, verbose_message=None):
        if verbose and verbose_message is not None:
            print(verbose_message)
        it = tqdm(dataset_resized) if verbose else dataset_resized
        for item in it:
            if not isinstance(item, cls._tuple_class):
                raise RuntimeError(f"Item should be an instance of {cls._tuple_class.__name__}")
            yield item

    def _get_items_generator_and_size(self, n_samples: Optional[int]=None,
                                      verbose: bool=True, verbose_message=None):
        """
        Args:
            n_samples (int, positive):
                The number of samples to generate.
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

        if n_samples is None:
            if not iterable_is_finite(dataset):
                raise ValueError(f"Can't store an infinite-sized {dataset.__class__.__name__} if n_samples "\
                                 "is not specified. Either specify n_samples or use a finite-sized dataset.")
            # The dataset is finite and we want to save all of it
            new_data_iterable = dataset
        else:
            if not isinstance(n_samples, int) or n_samples <= 0:
                raise ValueError("n_samples should be a positive integer.")
            try:
                new_data_iterable = resize_iterable(dataset, n_samples)
            except ValueError as e:
                raise ValueError(f"To store finite samples from this {dataset.__class__.__name__}, cannot make n_samples > len(dataset) since it " \
                                 f"is not a SizedInfiniteIterableMixin. Got {n_samples=} and len(dataset)={len(dataset)}") from e
        
        items_generator = self._get_items_generator(
            new_data_iterable, verbose, verbose_message)
        return items_generator, len(new_data_iterable)
    
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
            list_of_dicts.append(item.to_dict())

        torch.save(list_of_dicts, os.path.join(dir_name, "data.pt"))
    
    def __getitem__(self, index):
        """Retrieves a single sample from the dataset at the specified index.
        Should only be implemented in subclasses that are map-style datasets.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            TupleWithModel: The sample at the specified index.
        """
        if isinstance(self, IterableDataset):
            raise TypeError(f"{self.__class__.__name__} is an IterableDataset and should not be accessed by index.")
        raise NotImplementedError("Subclass must implement __getitem__ for map-style datasets")


class MapDatasetWithModels(DatasetWithModels):
    """A base class for `DatasetWithModels` datasets that hold items
    in a map style (i.e. implement `__getitem__` and `__len__`).
    
    All subclasses should implement `__getitem__`, `__len__`, and `data_is_loaded`.
    
    `__getitem__` is partially implemented for slices where it returns a
    `MapDatasetWithModelsSubset` instance, so subclasses should check for
    slices and use super() accordingly."""

    # direct subclass of MapDatasetWithModels that the dataset is a subclass of
    _map_base_class: type
    # subclass of MapDatasetWithModelsSubset
    _map_subset_class: type
    
    def __new__(cls, *args, **kwargs):
        # Ensure the required class attribute is set correctly
        if not (hasattr(cls, '_map_base_class') and
                issubclass(cls._map_base_class, cls._base_class)):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_map_base_class' that is a subclass of {cls._base_class.__name__}.")
        if not (hasattr(cls, '_map_subset_class') and
                issubclass(cls._map_subset_class, cls._map_base_class)):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_map_subset_class' that is a subclass of {cls._map_base_class.__name__}.")
        return cls._base_class.__new__(cls)

    @abstractmethod
    def __getitem__(self, index):
        """Retrieves a single item from the dataset at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            TupleWithModel: The item at the specified index.
        """
        if isinstance(index, slice):
            indices = list(range(*index.indices(len(self))))
            return self._map_subset_class(self, indices)
        raise NotImplementedError("Subclass must implement __getitem__ for non-slice keys")
    
    @abstractmethod
    def __len__(self):
        pass  # pragma: no cover

    # Subclasses should implement data_is_loaded

    def random_split(self, lengths: Sequence[Union[int, float]]):
        return [self._map_subset_class.from_subset(subset)
                for subset in random_split(self, lengths)]


class ListMapDatasetWithModels(MapDatasetWithModels):
    """A base class for `DatasetWithModels` datasets that hold items in a list
    and can be accessed by index."""
    def __init__(self, data: List[TupleWithModel],
                 model_sampler: Optional[RandomModelSampler]=None):
        """Initializes an instance of the class with the given data.

        Args:
            data: A list of TupleWithModel containing the data.
            model_sampler: Optional RandomModelSampler instance to associate
        """
        if not (model_sampler is None or isinstance(model_sampler, RandomModelSampler)):
            raise ValueError("model_sampler should be None or a RandomModelSampler instance")

        if all(isinstance(x, self._tuple_class) for x in data):
            if all(x.has_model for x in data):
                if model_sampler is None:
                    raise ValueError(f"model_sampler should not be None if all items are {self._tuple_class.__name__} with models.")
            elif all(not x.has_model for x in data):
                pass
            else:
                raise ValueError("All items in data should either have models or not have models")

            for item in data:
                if item.has_model:
                    if item.model_index is None:
                        raise ValueError(f"model_index should be specified for each {self._tuple_class.__name__}")
                    else:
                        assert model_sampler._models[item.model_index] is item._model
        elif all(isinstance(x, tuple) for x in data):
            return type(self).__init__(self, [self._tuple_class(*x) for x in data], model_sampler)
        else:
            raise ValueError(f"All items in data should be of type {self._tuple_class.__name__}")

        self._data = data
        self._model_sampler = model_sampler
    
    def _init_params(self):
        kwargs = {'model_sampler': self._model_sampler} if self.has_models else {}
        return (self._data,), kwargs
    
    def __str__(self):
        args, kwargs = self._init_params()
        if len(self._data) <= 2:
            short_list = self._data
        else:
            short_list = self._data[:1] + ["..."] + self._data[-1:]
        tmp1 = self._str_helper(short_list, {},
                                is_str=True, include_class=False)
        tmp = self._str_helper(
            ("[\n" + tmp1 + "\n]",),
            kwargs, is_str=True)
        return f"{tmp} of length {len(self)}"

    def __getitem__(self, index):
        if isinstance(index, slice):
            # basically super().__getitem__(index)
            return self._map_base_class.__getitem__(self, index)
        return self._data[index]
    
    def __len__(self):
        return len(self._data)
    
    def data_is_loaded(self):
        return True

    @classmethod
    def load(cls, dir_name: str):
        """Loads a dataset from a given directory. The directory must contain a
        saved instance of ListMapDatasetWithModels, including the data and
        optionally the models.

        Args:
            dir_name (str): The path to the directory from which the dataset
            should be loaded.

        Returns:
            ListMapDatasetWithModels: The loaded dataset instance.
        """
        if not os.path.exists(dir_name): # Error if path doesn't exist
            raise FileNotFoundError(f"Path {dir_name} does not exist")
        if not os.path.isdir(dir_name): # Error if path isn't directory
            raise NotADirectoryError(f"Path {dir_name} is not a directory")

        list_of_dicts = torch.load(os.path.join(dir_name, "data.pt"))

        models_path = os.path.join(dir_name, "models.pt")
        has_models = os.path.exists(models_path)
        if has_models:
            model_sampler = RandomModelSampler(torch.load(models_path))
        else:
            model_sampler = None
        
        data = []
        for item in list_of_dicts:
            if has_models:
                model_index = item.pop('model_index')
                model = model_sampler._models[model_index]
                model_params = item.pop('model_params')
            else:
                if 'model_index' in item or 'model_params' in item:
                    raise ValueError("Model information should not be present in the data if models are not saved.")
                model = None
                model_params = None
            data.append(cls._tuple_class(
                **item,
                model=model,
                model_params=model_params))
        return cls(data, model_sampler)
    
    @classmethod
    def from_iterable_dataset(cls, dataset: DatasetWithModels,
                              n_realizations:Optional[int] = None,
                              verbose: bool = True):
        """Creates an instance of ListMapDatasetWithModels from a given
        iterable-style DatasetWithModels by sampling a specified number of
        data points.

        Args:
            dataset (DatasetWithModels and IterableDataset):
                The dataset from which to generate samples.
                Must be both a DatasetWithModels and a IterableDataset.
            n_realizations (int, positive):
                The number of function realizations (samples) to generate.
                Optional if the dataset has finite size, in which case the size
                of the dataset is used.

        Returns:
            ListMapDatasetWithModels:
            A new instance of ListMapDatasetWithModels containing the sampled
            function realizations.
        """
        if not (isinstance(dataset, cls._base_class) and
                isinstance(dataset, IterableDataset)):
            raise TypeError(f"dataset should be an instance of both {cls._base_class.__name__} and IterableDataset")
        message = f"Saving realizations from {dataset.__class__.__name__} into {cls.__name__}"
        items_generator, size = dataset._get_items_generator_and_size(
            n_realizations, verbose, verbose_message=message)
        data = list(items_generator)
        return cls(data, dataset.model_sampler if dataset.has_models else None)


class LazyMapDatasetWithModels(MapDatasetWithModels):
    """A dataset class that lazily generates function samples.

    This class extends the `MapDatasetWithModels` class and provides
    lazy loading of items. It generates items on-the-fly
    when accessed, rather than loading all samples into memory at once.
    """
    def __init__(self, dataset: DatasetWithModels,
                 n_realizations:Optional[int] = None):
        """
        Args:
            dataset (MapDatasetWithModels):
                The underlying items dataset.
            n_realizations (Optional[int]):
                The number of realizations to generate.
                If None, all realizations will be generated.
        """
        # DatasetWithModels
        if not (isinstance(dataset, self._base_class) and
                isinstance(dataset, IterableDataset)):
            raise TypeError(f"dataset should be an instance of both {self._base_class.__name__} and IterableDataset")
        
        # For __repr__
        self._dataset = dataset
        self._n_realizations = n_realizations
        
        items_generator, size = dataset._get_items_generator_and_size(
            n_realizations, verbose=False)
        self._items_generator = items_generator
        self._size = size
        self._data = [None] * size
        self._model_sampler = dataset.model_sampler if dataset.has_models else None
    
    def _init_params(self):
        kwargs = {'n_realizations': self._n_realizations} if self._n_realizations is not None else {}
        return (self._dataset,), kwargs
    
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


class MapDatasetWithModelsSubset(Subset, MapDatasetWithModels):
    """A subset of a MapDatasetWithModels dataset."""
    def __init__(self, dataset: MapDatasetWithModels, indices: Sequence[int]) -> None:
        """
        Args:
            dataset (MapDatasetWithModels): The original dataset.
            indices (Sequence[int]): The indices of the samples in the subset.
        """
        # MapDatasetWithModels
        if not isinstance(dataset, self._map_base_class):
            raise ValueError(f"dataset should be an instance of {self._map_base_class.__name__}")
        
        # Equivalent to self.dataset = dataset; self.indices = indices
        Subset.__init__(self, dataset, indices)
    
    def _init_params(self):
        return (self.dataset, self.indices), {}

    def __str__(self):
        tmp = self._str_helper((self.dataset,), {}, is_str=True)
        return f"{tmp} of length {len(self)}/{len(self.dataset)}"

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._map_base_class.__getitem__(self, idx)
        return Subset.__getitem__(self, idx)
    
    def __len__(self):
        return Subset.__len__(self)
    
    # Can just define _model_sampler and then has_models and model_sampler are
    # inherited from MapDatasetWithModels.
    # random_split is also inherited from MapDatasetWithModels.
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


def create_classes(dataset_base_name="DatasetWithModels",
                   map_dataset_base_name="MapDatasetWithModels",
                   list_dataset_name="ListMapDatasetWithModels",
                   lazy_dataset_name="LazyMapDatasetWithModels",
                   map_subset_name="MapDatasetWithModelsSubset",
                   dataset_base_docstring=None,
                   map_dataset_base_docstring=None,
                   list_dataset_docstring=None,
                   lazy_dataset_docstring=None,
                   map_subset_docstring=None,
                   tuple_class=None):
    """Creates the classes that inherit from the base classes."""
    if tuple_class is None:
        raise ValueError("tuple_class should be specified in create_classes.")
    if not issubclass(tuple_class, TupleWithModel):
        raise ValueError("create_classes: tuple_class should be a subclass of TupleWithModel.")
    
    def _docstring_replace(docstring):
        return docstring.replace("DatasetWithModels", dataset_base_name) \
                        .replace("MapDatasetWithModels", map_dataset_base_name) \
                        .replace("ListMapDatasetWithModels", list_dataset_name) \
                        .replace("LazyMapDatasetWithModels", lazy_dataset_name) \
                        .replace("MapDatasetWithModelsSubset", map_subset_name) \
                        .replace("TupleWithModel", tuple_class.__name__)

    def replace_docstrings(x, new_docstring=None):
        if new_docstring is not None:
            x.__doc__ = new_docstring
        elif x.__doc__ is not None:
            x.__doc__ = _docstring_replace(x.__doc__)
        
        for name, obj in x.__dict__.items():
            if obj.__doc__ is not None:
                try:
                    obj.__doc__ = _docstring_replace(obj.__doc__)
                except AttributeError:
                    pass

    BaseDataset = type(dataset_base_name,
                       DatasetWithModels.__bases__,
                       dict(DatasetWithModels.__dict__))
    BaseDataset._tuple_class = tuple_class
    BaseDataset._base_class = BaseDataset
    
    MapBaseDataset = type(map_dataset_base_name,
                          (BaseDataset,),
                          dict(MapDatasetWithModels.__dict__))
    MapBaseDataset._map_base_class = MapBaseDataset
    
    ListDataset = type(list_dataset_name,
                       (MapBaseDataset,),
                       dict(ListMapDatasetWithModels.__dict__))
    
    LazyDataset = type(lazy_dataset_name,
                       (MapBaseDataset,),
                       dict(LazyMapDatasetWithModels.__dict__))
    
    MapSubset = type(map_subset_name,
                     (Subset, MapBaseDataset),
                     dict(MapDatasetWithModelsSubset.__dict__))

    MapBaseDataset._map_subset_class = MapSubset
    
    replace_docstrings(BaseDataset, dataset_base_docstring)
    replace_docstrings(MapBaseDataset, map_dataset_base_docstring)
    replace_docstrings(ListDataset, list_dataset_docstring)
    replace_docstrings(LazyDataset, lazy_dataset_docstring)
    replace_docstrings(MapSubset, map_subset_docstring)

    return BaseDataset, MapBaseDataset, ListDataset, LazyDataset, MapSubset


if __name__ == "__main__":
    BaseDataset, MapBaseDataset, ListDataset, LazyDataset, MapSubset = create_classes(
        "BaseDataset", "MapBaseDataset", "ListDataset",
        "LazyDataset", "MapSubset",
        tuple_class=TupleWithModel)

    boo = ListDataset([
        TupleWithModel(1, 2),
        TupleWithModel(4, 5),
        TupleWithModel(7, 8)
    ])
    print(boo)
    print(boo[1])
    print(boo[:2])
    boo.save("boo")
    print(list(boo))
    boo2 = ListDataset.load("boo")#[1:]
    print(list(boo2))

    print(isinstance(boo2, MapBaseDataset), isinstance(boo2, ListDataset), isinstance(boo2, MapSubset))

    print(MapSubset.__doc__)
