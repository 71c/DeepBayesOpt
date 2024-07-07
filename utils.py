from abc import ABC, abstractmethod
import math
import os
from typing import List, Optional, Sequence, Union, Iterable, Tuple
import warnings
from functools import partial, lru_cache
import json
import numpy as np
from scipy.optimize import root_scalar
import torch
from torch import Tensor

from botorch.exceptions.errors import (
    BotorchTensorDimensionError,
    InputDataError,
    UnsupportedError,
)
from botorch.exceptions.warnings import (
    _get_single_precision_warning,
    BotorchTensorDimensionWarning,
    InputDataWarning,
)
from botorch.posteriors import Posterior
from botorch.models.transforms.outcome import OutcomeTransform, Standardize

from botorch.models.model import Model
from botorch.models.gpytorch import GPyTorchModel, BatchedMultiOutputGPyTorchModel


class InverseOutcomeTransform(OutcomeTransform):
    def __init__(self, transform: OutcomeTransform):
        super().__init__()
        if not isinstance(transform, OutcomeTransform):
            raise ValueError("transform must be a OutcomeTransform instance")
        self._original_transform = transform
    
    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self._original_transform.untransform(Y, Yvar)

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        return self.__class__(self._original_transform.subset_output(idcs))

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self._original_transform.forward(Y, Yvar)
    
    @property
    def _is_linear(self) -> bool:
        return self._original_transform._is_linear

class Unstandardize(InverseOutcomeTransform):
    def __init__(self, standardizer: Standardize):
        super().__init__(standardizer)

        if not isinstance(standardizer, Standardize):
            raise ValueError("standardizer must be a Standardize instance")
        if not standardizer._is_trained:
            raise RuntimeError(
            "Can only invert a Standardize if it has been called on some outcome data")
        assert not standardizer.training

        new_tf = standardizer.__class__(
            m=standardizer._m,
            outputs=standardizer._outputs,
            batch_shape=standardizer._batch_shape,
            min_stdv=standardizer._min_stdv)

        new_stdvs = 1 / standardizer.stdvs
        new_tf.stdvs = new_stdvs
        new_tf._stdvs_sq = new_stdvs.pow(2)
        
        new_tf.means = -new_stdvs * standardizer.means
        
        new_tf._is_trained = standardizer._is_trained
        new_tf.eval()
        self.eval()

        self._inverse_transform = new_tf

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        return self._inverse_transform.untransform_posterior(posterior)



def _transform_Y(self, Y: Optional[Tensor]=None):
    if Y is not None and hasattr(self, "outcome_transform"):
        outcome_transform = self.outcome_transform
        is_training = outcome_transform.training
        # Put it into eval mode because we don't want to re-learn e.g.
        # mean and standard deviation for Standardize.
        # We're assuming that Standardize has been called at least once
        # (i.e. from being called on __init__ in SingleTaskGP) so that its
        # _is_trained attribute is True, so even if in training mode,
        # it has been trained at least once.
        if is_training:
            outcome_transform.eval()
        Y, _ = outcome_transform(Y)
        if is_training: # Set it back to train mode
            outcome_transform.train()
    return Y
Model._transform_Y = _transform_Y


def _set_train_data_with_X_transform(self,
                                     X: Optional[Tensor]=None,
                                     Y: Optional[Tensor]=None,
                                     strict=True):
    # If only one is given, then it does make sense to make sure it has the
    # right shape...
    strict = strict or ((X is None) ^ (Y is None))
    self.set_train_data(inputs=X, targets=Y, strict=strict)

    if X is not None and hasattr(self, "input_transform"):
        assert self.training == (not self._has_transformed_inputs)
        if not self.training:
            # Then self._has_transformed_inputs == True,
            # and self._original_train_inputs is wrong.
            # So we can set _has_transformed_inputs = False
            # to make it so that _set_transformed_inputs sets the
            # _original_train_inputs, transforms the inputs, and sets
            # self._has_transformed_inputs = True again.
            self._has_transformed_inputs = False
            self._set_transformed_inputs()
        else:
            # If in training mode, then train() doesn't do anything since
            # _has_transformed_inputs == False, and eval() makes
            # _original_train_inputs set correct. So don't need to do anything.
            # However, it doesn't hurt to just set it anyway for consistency:
            if self._original_train_inputs is not None:
                self._original_train_inputs = self.train_inputs[0]
Model._set_train_data_with_X_transform = _set_train_data_with_X_transform


def _set_train_data_with_transforms_Model(self,
                                          X: Optional[Tensor]=None,
                                          Y: Optional[Tensor]=None,
                                          strict=True):
    Y = self._transform_Y(Y)
    self._set_train_data_with_X_transform(X, Y, strict)
Model.set_train_data_with_transforms = _set_train_data_with_transforms_Model


def _set_train_data_with_transforms_GPyTorchModel(self,
                                                  X: Optional[Tensor]=None,
                                                  Y: Optional[Tensor]=None,
                                                  strict=True):
    Y = self._transform_Y(Y)
    
    if not (X is None and Y is None):
        proposed_X = X if X is not None else self.train_inputs[0]

        if Y is None:
            proposed_Y = self.train_targets
            if (proposed_X.dim() - proposed_Y.dim() == 1) and \
                (proposed_X.shape[:-1] == proposed_Y.shape):
                # Then the targets doesn't have an explicit output dimension
                proposed_Y = proposed_Y.unsqueeze(-1)
        else:
            proposed_Y = Y
        
        self._validate_tensor_args(X=proposed_X, Y=proposed_Y)

    self._set_train_data_with_X_transform(X, Y, strict)
GPyTorchModel.set_train_data_with_transforms = _set_train_data_with_transforms_GPyTorchModel


def _set_train_data_with_transforms_BatchedMultiOutputGPyTorchModel(self,
                                                  X: Optional[Tensor]=None,
                                                  Y: Optional[Tensor]=None,
                                                  strict=True):
    Y = self._transform_Y(Y)
    
    both_are_given = X is not None and Y is not None
    neither_is_given = X is None and Y is None
    
    if both_are_given:
        self._validate_tensor_args(X=X, Y=Y)
        self._set_dimensions(train_X=X, train_Y=Y)
        X, Y, _ = self._transform_tensor_args(X=X, Y=Y)
    elif not neither_is_given:
        if X is not None:
            # X is given but not Y

            # Try to do something akin to _validate_tensor_args
            if X.dim() < 2:
                raise BotorchTensorDimensionError(
                    f"X has shape {X.shape} but needs to have at least 2 dimensions"
                )
            given_input_batch_shape = X.shape[:-2]
            if given_input_batch_shape != self.batch_shape:
                raise BotorchTensorDimensionError(
                    f"Expected X to have batch shape {self.batch_shape} but has"
                    f" batch shape {given_input_batch_shape}"
                )
            
            # Akin to _transform_tensor_args
            if self.num_outputs > 1:
                X = X.unsqueeze(-3).expand(
                    X.shape[:-2] + torch.Size([self.num_outputs]) + X.shape[-2:]
                )
        elif Y is not None:
            # Y is given but not X

            # Try to do something akin to _validate_tensor_args
            if Y.dim() < 2:
                raise BotorchTensorDimensionError(
                    f"Y has shape {Y.shape} but needs to have at least 2 dimensions"
                )
            given_input_batch_shape = Y.shape[:-2]
            given_num_outputs = Y.shape[-1]
            if given_input_batch_shape != self.batch_shape:
                raise BotorchTensorDimensionError(
                    f"Expected Y to have batch shape {self.batch_shape} but has"
                    f" batch shape {given_input_batch_shape}"
                )
            if given_num_outputs != self.num_outputs:
                raise BotorchTensorDimensionError(
                    f"Expected Y to have number of outputs {self.num_outputs} but has"
                    f" number of outputs {given_num_outputs}"
                )
            
            # Akin to _transform_tensor_args
            if self.num_outputs > 1:
                Y = Y.transpose(-1, -2)
            else:
                Y = Y.squeeze(-1)
    
    self._set_train_data_with_X_transform(X, Y, strict)

    if not both_are_given and not neither_is_given:
        # In the case that only one of X or Y was provided, check the dtypes
        # akin to the last part of _validate_tensor_args
        new_X, new_Y = self.train_inputs[0], self.train_targets
        if new_X.dtype != new_Y.dtype:
            raise InputDataError(
                "Expected all inputs to share the same dtype. Got "
                f"{new_X.dtype} for X, {new_Y.dtype} for Y."
            )
        if new_X.dtype != torch.float64:
            warnings.warn(
                _get_single_precision_warning(str(new_X.dtype)),
                InputDataWarning,
                stacklevel=3,  # Warn at model constructor call.
            )
BatchedMultiOutputGPyTorchModel.set_train_data_with_transforms = _set_train_data_with_transforms_BatchedMultiOutputGPyTorchModel


def uniform_randint(min_val, max_val):
    return torch.randint(min_val, max_val+1, (1,), dtype=torch.int32).item()


def get_uniform_randint_generator(min_val, max_val):
    return partial(uniform_randint, min_val, max_val)


def loguniform_randint(min_val, max_val, size=1, pre_offset=0.0, offset=0):
    if not (isinstance(min_val, int) and isinstance(max_val, int) and isinstance(offset, int)):
        raise ValueError("min_val, max_val, and offset must be integers")
    if not (1 <= min_val <= max_val):
        raise ValueError("min_val must be between 1 and max_val")
    if not (pre_offset >= 0):
        raise ValueError("pre_offset must be non-negative")

    min_log = torch.log(torch.tensor(min_val + pre_offset))
    max_log = torch.log(torch.tensor(max_val+1 + pre_offset))
    random_log = torch.rand(size) * (max_log - min_log) + min_log
    ret = (torch.exp(random_log) - pre_offset).to(dtype=torch.int32) + offset
    if torch.numel(ret) == 1:
        return ret.item()
    return ret


def get_loguniform_randint_generator(min_val, max_val, pre_offset=0.0, offset=0):
    return partial(loguniform_randint, min_val, max_val, pre_offset=pre_offset, offset=offset)


def _int_linspace_naive(start, stop, num):
    return np.unique(np.round(np.linspace(start, stop, num)).astype(int))


@lru_cache(maxsize=128) # Not necessary but why not.
def int_linspace(start, stop, num):
    if not (isinstance(start, int) and isinstance(stop, int)):
        raise ValueError("start and stop should be integers")

    if num > stop - start + 1:
        raise ValueError('num must be less than or equal to stop - start + 1')
    ret = _int_linspace_naive(start, stop, num)
    length = len(ret)
    
    if length < num:
        sol = root_scalar(
            lambda x: len(_int_linspace_naive(start, stop, int(x))) - num,
            method='secant', x0=num, x1=2*num, xtol=1e-12, rtol=1e-12)
        
        k = int(sol.root)
        ret = _int_linspace_naive(start, stop, k)

        if len(ret) != num:
            if len(ret) > num:
                while len(ret) > num:
                    k -= 1
                    ret = _int_linspace_naive(start, stop, k)
            else:
                while len(ret) < num:
                    k += 1
                    ret = _int_linspace_naive(start, stop, k)

    return ret


def pad_tensor(vec, length, dim, add_mask=False):
    """Pads a tensor 'vec' to a size 'length' in dimension 'dim' with zeros.
    args:
        vec - tensor to pad
        length - the size to pad to in dimension 'dim'
        dim - dimension to pad
        add_mask - whether to return the mask as well

    returns:
        If add_mask=True, return a tuple (padded, mask).
        Otherwise, return the padded tensor only.
    """
    pad_size = length - vec.size(dim)
    if pad_size < 0:
        raise ValueError("Tensor cannot be padded to length less than it already is")
    
    pad_shape = list(vec.shape)
    pad_shape[dim] = pad_size
    if pad_size == 0: # Could pad with nothing but that's unnecessary
        padded = vec
    else:
        padding = torch.zeros(*pad_shape, dtype=vec.dtype, device=vec.device)
        padded = torch.cat([vec, padding], dim=dim)

    if add_mask:
        mask_true = torch.ones(vec.shape, dtype=torch.bool, device=vec.device)
        mask_false = torch.zeros(*pad_shape, dtype=torch.bool, device=vec.device)
        mask = torch.cat([mask_true, mask_false], dim=dim)
        return padded, mask

    return padded


# Training took 5.717105 seconds
# Training took 4.495734 seconds
# Training took 4.565008 seconds
# Training took 4.320932 seconds
# def max_pad_tensors_batch(tensors, dim=0, add_mask=False):
#     """Pads a batch of tensors with zeros along a dimension to match the maximum
#     length.

#     Args:
#         tensors (List[torch.Tensor]): A list of tensors to be padded.
#         dim (int, default: 0): The dimension along which to pad the tensors.
#         add_mask (bool, optional, default: False):
#             Whether to also return a mask tensor

#     Returns:
#         If add_mask=True, return a tuple (padded, mask).
#         If all tensors have the same length, mask is None.
#         Otherwise, returns the padded tensor only.
#     """
#     lengths = [x.shape[dim] for x in tensors]
#     max_length = max(lengths)
#     if all(length == max_length for length in lengths):
#         stacked = torch.stack(tensors) # Don't pad if we don't need to
#         if add_mask:
#             mask = None
#     else:
#         if add_mask:
#             padded_tensors, masks = zip(*[
#                 pad_tensor(x, max_length, dim=dim, add_mask=True)
#                 for x in tensors])
#             mask = torch.stack(masks)
#         else:
#             padded_tensors = [
#                 pad_tensor(x, max_length, dim=dim, add_mask=False)
#                 for x in tensors]
#         stacked = torch.stack(padded_tensors)
    
#     if add_mask:
#         return stacked, mask
#     return stacked


def equals(a, b):
    try:
        iter(a)
        is_iterable = True
    except TypeError:
        is_iterable = False
    if is_iterable:
        if len(a) != len(b):
            return False
        return type(b) is type(a) and all(equals(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    return a == b


# Training took 4.215692 seconds
# Training took 3.932536 seconds
# Training took 4.084472 seconds
# Training took 4.148941 seconds
# Training took 4.345846 seconds
# Training took 4.416536 seconds
## I don't know maybe this is a bit faster
def max_pad_tensors_batch(tensors, add_mask=False):
    """Pads a batch of tensors with zeros along a dimension to match the maximum
    length.

    Args:
        tensors (List[torch.Tensor]): A list of tensors to be padded.
        add_mask (bool, optional, default: False):
            Whether to also return a mask tensor

    Returns:
        If add_mask=True, return a tuple (padded, mask).
        If all tensors have the same length, mask is None.
        Otherwise, returns the padded tensor only.
    """
    lengths = [x.shape[0] for x in tensors]
    max_length = max(lengths)
    if all(length == max_length for length in lengths):
        padded = torch.stack(tensors) # Don't pad if we don't need to
        if add_mask:
            mask = None
    else:
        if add_mask:
            mask = torch.zeros(len(tensors), max_length, *tensors[0].shape[1:],
                               dtype=torch.bool, device=tensors[0].device)
            for i, x in enumerate(tensors):
                mask[i, :x.shape[0], ...] = True
        
        padded = torch.zeros(len(tensors), max_length, *tensors[0].shape[1:],
                             dtype=tensors[0].dtype, device=tensors[0].device)
        for i, x in enumerate(tensors):
            padded[i, :x.shape[0], ...] = x
    
    if add_mask:
        ret = padded, mask
    else:
        ret = padded

    # Testing
    # assert equals(ret, max_pad_tensors_batch_old(tensors, dim=0, add_mask=add_mask))

    return ret


# Taken from
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
def get_lengths_from_proportions(total_length: int, proportions: Sequence[float]):
    subset_lengths: List[int] = []
    for i, frac in enumerate(proportions):
        if frac < 0 or frac > 1:
            raise ValueError(f"Fraction at index {i} is not between 0 and 1")
        n_items_in_split = int(math.floor(total_length * frac))
        subset_lengths.append(n_items_in_split)
    remainder = total_length - sum(subset_lengths)
    # add 1 to all the lengths in round-robin fashion until the remainder is 0
    for i in range(remainder):
        idx_to_add_at = i % len(subset_lengths)
        subset_lengths[idx_to_add_at] += 1
    lengths = subset_lengths
    for i, length in enumerate(lengths):
        if length == 0:
            warnings.warn(
                f"Length of split at index {i} is 0. "
                f"This might result in an empty dataset."
            )
    return lengths


def get_lengths_from_proportions_or_lengths(
        total_length: int, lengths: Sequence[Union[int, float]]):
    lengths_is_proportions = math.isclose(sum(lengths), 1) and sum(lengths) <= 1
    if lengths_is_proportions:
        return get_lengths_from_proportions(total_length, lengths)
    return lengths


class SizedIterableMixin(Iterable):
    """A mixin class that provides functionality 'len()' for iterable objects.
    All subclasses should implement __iter__ because this class inherits from
    Iterable.

    Attributes:
        _size (int or inf): The size of the iterable object.
            math.inf if the size is infinite.
            Subclasses must use this attribute to hold the length of the object.
    """
    def _len_or_inf(self):
        if not hasattr(self, "_size"):
            raise AttributeError(
                f"{self.__class__.__name__}, a subclass of SizedIterableMixin, "\
                    "must have attribute '_size' to hold the length.")
        size = self._size
        if size != math.inf and (not isinstance(size, int) or size < 0):
            raise ValueError(
                f"self._size should inf or a non-negative integer but got {size}")
        return size
    
    def __len__(self):
        size = self._len_or_inf()
        if size == math.inf:
            raise TypeError(f"Length of the {self.__class__.__name__} is infinite")
        return size


class SizedInfiniteIterableMixin(SizedIterableMixin):
    """A mixin class that provides functionality for creating iterable objects
    with a specified size. If the size is inf, the object is considered to be
    infinite and so calling iter() then you can call next() indefinitely wihout
    any StopIteration exception.
    If the size is not inf, then the object is considered to be finite and
    calling iter() will return a generator that will yield the next element
    until the size is reached.

    Attributes:
        _size (Optional[int]): The size of the iterable object.
            inf if the size is infinite.
    """

    @abstractmethod
    def copy_with_new_size(self, size:int) -> "SizedInfiniteIterableMixin":
        """Creates a copy of the object with a new size.
        Should set the _size attribute of the new object to the specified size.

        Args:
            size (int): The new size for the object.

        Returns:
            A new instance of the object with the specified size.
        """
        pass  # pragma: no cover
    
    @abstractmethod
    def _next(self):
        """Returns the next element in the iterable."""
        pass  # pragma: no cover

    def __iter__(self):
        if self._len_or_inf() == math.inf:
            return self
        # Must separate this in a different function because otherwise,
        # iter will always return a generator, even if self._size == math.inf
        return self._finite_iterator()
    
    def _finite_iterator(self):
        for _ in range(len(self)):
            yield self._next()

    def __next__(self):
        if self._len_or_inf() == math.inf:
            return self._next()
        raise TypeError(f"Cannot call __next__ on a finitely sized {type(self)}. Use iter() first.")


class FirstNIterable(Iterable):
    """
    Creates an iterable for the first 'n' elements of a given iterable.

    Takes any iterable and an integer 'n', and provides an iterator
    that yields the first 'n' elements of the given iterable. If the original
    iterable contains fewer than 'n' elements, the iterator will yield only the
    available  elements without raising an error.

    Args:
        iterable (iterable): The iterable to wrap.
        n (int): The number of elements to yield from the iterable.

    Example:
        >>> numbers = range(10)  # A range object is an iterable
        >>> first_five = _FirstNIterable(numbers, 5)
        >>> list(first_five)
        [0, 1, 2, 3, 4]

        >>> words = ["apple", "banana", "cherry", "date"]
        >>> first_two = _FirstNIterable(words, 2)
        >>> list(first_two)
        ['apple', 'banana']
    """
    def __init__(self, iterable, n):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n should be a positive integer.")
        self.iterable = iterable
        self.n = n
    
    def __iter__(self):
        iterator = iter(self.iterable)
        for _ in range(self.n):
            try:
                yield next(iterator)
            except StopIteration:
                break
    
    def __len__(self):
        return min(len_or_inf(self.iterable), self.n)


def len_or_inf(x):
    try:
        l = len(x)
        if l != math.inf and (not isinstance(l, int) or l < 0):
            raise ValueError(
                f"len(x) should be inf or a non-negative integer but got {l}")
        return l
    except TypeError:
        # Then it has no length, so we can only say it's infinite if
        # it is an iterable.
        try: # Check if it is an iterable
            iter(x)
            return math.inf
        except TypeError:
            raise TypeError(
                f"Object of type {type(x)} is not iterable so it has no length")


def iterable_is_finite(x):
    return len_or_inf(x) != math.inf


def resize_iterable(it, new_length: Optional[int] = None):
    original_length = len_or_inf(it)

    if new_length is not None:
        if not isinstance(new_length, int) or new_length <= 0:
            raise ValueError("new_length should be a positive integer")
        if new_length != original_length:
            # Weaker condition than `if isinstance(it, SizedInfiniteIterableMixin):`
            if callable(getattr(it, "copy_with_new_size", None)):
                it = it.copy_with_new_size(new_length)
            else:
                if new_length > original_length:
                    raise ValueError(f"{new_length=} should be <= len(it)={original_length} if it is not a SizedInfiniteIterableMixin")
                it = FirstNIterable(it, new_length)

    return it


def to_device(tensor, device):
    if tensor is None or device is None:
        return tensor
    return tensor.to(device)


def unsupported_improvements(dataloader):
    for batch in dataloader:
        if not batch.give_improvements:
            raise UnsupportedError(
                "The acquisition dataset must provide improvements; calculating " \
                "them from a batch would be possible but is currently unsupported.")
        yield batch


def save_json(data, fname, **kwargs):
    with open(fname, 'w') as json_file:
        json.dump(data, json_file, **kwargs)


def load_json(fname, **kwargs):
    with open(fname, 'r') as json_file:
        return json.load(json_file)


# Based on
# https://docs.gpytorch.ai/en/stable/_modules/gpytorch/module.html#Module.initialize
def get_param_value(module, name):
    if "." in name:
        submodule, name = module._get_module_and_name(name)
        if isinstance(submodule, torch.nn.ModuleList):
            idx, name = name.split(".", 1)
            return get_param_value(submodule[int(idx)], name)
        else:
            return get_param_value(submodule, name)
    elif not hasattr(module, name):
        raise AttributeError("Unknown parameter {p} for {c}".format(p=name, c=module.__class__.__name__))
    elif name not in module._parameters and name not in module._buffers:
        return getattr(module, name)
    else:
        return module.__getattr__(name)


# Print out all parameters of a random model:
# random_model = model.pyro_sample_from_prior()
# for name, param in model.named_parameters(): 
#     print(name)
#     print(get_param_value(random_model, name))
#     print()
## OR,
# random_model_params_dict = {
#     name: get_param_value(random_model, name)
#     for name, param in model.named_parameters()
# }

