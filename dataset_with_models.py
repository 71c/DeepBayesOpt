from abc import ABC, abstractmethod
import math
from typing import Optional, List, Tuple, Union
from collections.abc import Sequence

import os
import warnings
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, IterableDataset, random_split, Subset

from botorch.models.gp_regression import SingleTaskGP
from botorch.exceptions import UnsupportedError
import pyro

from utils import get_lengths_from_proportions, resize_iterable, iterable_is_finite, get_lengths_from_proportions_or_lengths


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


class TupleWithModel:
    def __init__(self, *items, model:Optional[SingleTaskGP]=None,
                 model_params:Optional[dict]=None, items_list=None,
                 **kwargs):
        if items_list is not None:
            assert len(items) == 0
            items = tuple(items_list)
        
        if hasattr(self, "args_names"):
            if len(items) != len(self.args_names):
                if len(items) == 0:
                    if set(self.args_names) <= set(kwargs.keys()):
                        items = []
                        for name in self.args_names:
                            items.append(kwargs.pop(name))
                        items = tuple(items)
                    else:
                        missing_keys = set(self.args_names) - set(kwargs.keys())
                        missing_keys_str = ", ".join(map(repr, missing_keys))
                        raise ValueError(f"{self.__class__.__name__}.__init__: Missing key{'' if len(missing_keys) == 1 else 's'} {missing_keys_str} in {self.__class__.__name__}")
                elif len(items) == len(self.args_names) + 1 and model is None:
                    model = items[-1]
                    items = items[:-1]
                elif len(items) == len(self.args_names) + 2 and model is None and model_params is None:
                    model_params = items[-1]
                    model = items[-2]
                    items = items[:-2]
                else:
                    raise ValueError(f"{self.__class__.__name__}.__init__: Number of items in should be {len(self.args_names)} but got {len(items)}")
            
            for name, item in zip(self.args_names, items):
                setattr(self, name, item)
        
            for key in kwargs:
                if key in self.args_names:
                    raise ValueError(
                        f"{self.__class__.__name__}.__init__: Keyword argument {key} should not be in the keyword arguments " \
                        f"because it is already the name of an element in the tuple of {self.__class__.__name__}.")
        
        self._items = items

        if hasattr(self, "kwargs_names"):
            given_kwargs_keys = set(kwargs.keys())
            expected_kwargs_keys = set(self.kwargs_names)
            if given_kwargs_keys < expected_kwargs_keys:
                missing_keys = expected_kwargs_keys - given_kwargs_keys
                missing_keys_str = ", ".join(map(repr, missing_keys))
                raise ValueError(f"{self.__class__.__name__}.__init__: Keyword arguments for {self.__class__.__name__} " \
                    f"should be {self.kwargs_names}; "\
                    f"missing key{'' if len(missing_keys) == 1 else 's'} {missing_keys_str}")
            elif given_kwargs_keys > expected_kwargs_keys:
                extra_keys = given_kwargs_keys - expected_kwargs_keys
                extra_keys_str = ", ".join(map(repr, extra_keys))
                raise ValueError(f"{self.__class__.__name__}.__init__: Keyword arguments for {self.__class__.__name__} " \
                    f"should be {self.kwargs_names} but got extra key" \
                    f"{'' if len(extra_keys) == 1 else 's'} {extra_keys_str}")
            elif given_kwargs_keys != expected_kwargs_keys:
                raise ValueError(f"{self.__class__.__name__}.__init__: Keyword arguments for {self.__class__.__name__} " \
                    f"should be {self.kwargs_names} but got {given_kwargs_keys}")

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._kwargs = kwargs
        
        self._model = model
        self._indices = list(range(len(self)))
        if self.has_model:
            if isinstance(model, ModelsWithParamsList):
                if model_params is not None:
                    raise ValueError(f"{self.__class__.__name__}.__init__: model_params should not be specified if model is a ModelsWithParamsList instance.")
            elif isinstance(model, SingleTaskGP):
                if model_params is None:
                    # need to copy the data, otherwise everything will be the same
                    model_params = {name: param.detach().clone()
                                    for name, param in model.named_parameters()}
                self.model_params = model_params
                self.model_index = model.index if hasattr(model, "index") else None
            else:
                raise ValueError(f"{self.__class__.__name__}.__init__: model should be a SingleTaskGP or ModelsWithParamsList instance.")
        elif model_params is not None:
            raise ValueError(
                f"{self.__class__.__name__}.__init__: model_params should not be specified if model is not specified.")
    
    @property
    def has_model(self):
        return self._model is not None
    
    @property
    def model(self):
        if not self.has_model:
            raise AttributeError(
                "This TupleWithModel instance does not have a model.")
        if isinstance(self._model, ModelsWithParamsList):
            return self._model
        # otherwise, it is a SingleTaskGP instance:
        self._model.initialize(**self.model_params)
        return self._model

    @property
    def _tuple(self):
        if not self.has_model:
            return self._items
        return self._items + (self.model,)

    @property
    def tuple_no_model(self):
        return self._items
    
    def to(self, *args, **kwargs):
        """Does the torch Tensor.to() operation on all of the tuple's items,
        but NOT on the kwargs or on the model(s). If any of the tensors
        changed by doing to() then returns a new object which is a copy of the
        current object but with the tensors moved. Otherwise, returns self."""
        new_items = tuple(
            item.to(*args, **kwargs) if torch.is_tensor(item) else item
            for item in self._items)
        if new_items == self._items:
            return self
        return self.__class__(*new_items,
                              model=self._model,
                              model_params=getattr(self, "model_params", None),
                              **self._kwargs)
    
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

    def __repr__(self):
        ret = repr(self._tuple)[1:-1]
        if self._kwargs:
            u = ", ".join(f"{key}={value!r}" for key, value in self._kwargs.items())
            ret += ", " + u
        return f"{self.__class__.__name__}({ret})"
    
    def __eq__(self, other):
        return type(self) == type(other) and self._tuple == other._tuple and self._kwargs == other._kwargs
    
    def to_dict(self):
        if hasattr(self, "args_names"):
            x = {name: item for name, item in zip(self.args_names, self._items)}
        else:
            x = {'items_list': list(self._items)}
        if self.has_model:
            if isinstance(self.model, ModelsWithParamsList):
                # Honestly I don't care about this case ...
                # It would probably be possible to make this work with
                # ModelsWithParamsList better but we don't need this functionality.
                x.update({
                    'model': self.model
                })
            else: # SingleTaskGP
                x.update({
                    'model_index': self.model_index,
                    'model_params': self.model_params
                })
        x.update(self._kwargs)
        return x

    @classmethod
    def from_dict(cls, x, model_sampler:Optional[RandomModelSampler]=None):
        if 'model' in x:
            # assume that model is a ModelsWithParamsList instance
            assert isinstance(x['model'], ModelsWithParamsList)
            model = x.pop('model')
            model_params = None
        else: # then assume that model is a SingleTaskGP instance
            if 'model_index' in x:
                if model_sampler is None:
                    raise ValueError("model_sampler should be specified if model information is present.")
                model_index = x.pop('model_index')
                model = model_sampler._models[model_index]
                model_params = x.pop('model_params', None)
            else:
                assert 'model_params' not in x
                if model_sampler is not None:
                    raise ValueError("model_sampler should not be specified if model information is not present.")
                model, model_params = None, None
        return cls(**x, model=model, model_params=model_params)


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
    
    # Corresponds to DatasetWithModels
    _base_class: type

    # Corresponds to MapDatasetWithModels
    _map_base_class: type

    # Corresponds to ListMapDatasetWithModels
    _list_map_class: type

    # Corresponds to LazyMapDatasetWithModels
    _lazy_map_class: type

    def __new__(cls, *args, **kwargs):
        # Ensure the required class attribute is set correctly
        if not (hasattr(cls, '_tuple_class') and
                issubclass(cls._tuple_class, TupleWithModel)):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_tuple_class' that is a subclass of TupleWithModel.")
        
        if not (hasattr(cls, '_base_class')):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_base_class'.")
        
        if not (hasattr(cls, '_map_base_class') and
                issubclass(cls._map_base_class, cls._base_class)):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_map_base_class' that is a subclass of {cls._base_class.__name__}.")
        
        if not (hasattr(cls, '_list_map_class') and
                issubclass(cls._list_map_class, cls._map_base_class)):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_list_map_class' that is a subclass of {cls._map_base_class.__name__}.")
        
        if not (hasattr(cls, '_lazy_map_class') and
                issubclass(cls._lazy_map_class, cls._map_base_class)):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_lazy_map_class' that is a subclass of {cls._map_base_class.__name__}.")
                
        return Dataset.__new__(cls)

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
    
    @property
    @abstractmethod
    def data_is_fixed(self) -> bool:
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

    def fix_samples(self, n_realizations:Optional[int]=None, lazy=True):
        if self.data_is_fixed:
            raise ValueError(f"{self.__class__.__name__} is already fixed so don't need to fix.")
        if lazy:
            return self._lazy_map_class(self, n_realizations)
        return self._list_map_class.from_iterable_dataset(self, n_realizations)

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

    # Corresponds to MapDatasetWithModelsSubset
    _map_subset_class: type
    
    def __new__(cls, *args, **kwargs):
        # Ensure the required class attribute is set correctly
        if not (hasattr(cls, '_map_subset_class') and
                issubclass(cls._map_subset_class, cls._map_base_class)):
            raise AttributeError(
                f"{cls.__name__} must have attribute '_map_subset_class' that is a subclass of {cls._map_base_class.__name__}.")
        return cls._base_class.__new__(cls)

    @property
    def data_is_fixed(self):
        return True

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
        # Check if any of the lengths are the length of the entire dataset.
        # If so, make that one be self.
        # This will have slightly different behavior because then it won't be
        # shuffled, but it doesn't matter.
        return [self if len(subset) == len(self) else self._map_subset_class.from_subset(subset)
                for subset in random_split(self, lengths)]


class _Dummy:
    def __init__(self, x):
        self.x = x
    
    def __repr__(self):
        return str(self.x)


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
            short_list = self._data[:1] + [_Dummy("...")] + self._data[-1:]
        tmp1 = self._str_helper(short_list, {},
                                is_str=True, include_class=False)
        tmp = self._str_helper(
            (_Dummy("[\n" + tmp1 + "\n]"),),
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
                if 'model_index' not in item:
                    raise ValueError("Model information should be present in the data if models are saved.")
            else:
                if 'model_index' in item or 'model_params' in item:
                    raise ValueError("Model information should not be present in the data if models are not saved.")
            data.append(cls._tuple_class.from_dict(item, model_sampler))
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
            return self._map_base_class.__getitem__(self, index)
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
    
    MapBaseDataset = type(map_dataset_base_name,
                          (BaseDataset,),
                          dict(MapDatasetWithModels.__dict__))
    
    ListDataset = type(list_dataset_name,
                       (MapBaseDataset,),
                       dict(ListMapDatasetWithModels.__dict__))
    
    LazyDataset = type(lazy_dataset_name,
                       (MapBaseDataset,),
                       dict(LazyMapDatasetWithModels.__dict__))
    
    MapSubset = type(map_subset_name,
                     (Subset, MapBaseDataset),
                     dict(MapDatasetWithModelsSubset.__dict__))
    
    BaseDataset._tuple_class = tuple_class
    BaseDataset._base_class = BaseDataset
    BaseDataset._map_base_class = MapBaseDataset
    BaseDataset._list_map_class = ListDataset
    BaseDataset._lazy_map_class = LazyDataset

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
