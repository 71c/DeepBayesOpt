from functools import partial
import os
from re import L
import torch
from botorch.models.gp_regression import SingleTaskGP

from torch.utils.data import IterableDataset, DataLoader

from dataset_with_models import ModelsWithParamsList, TupleWithModel, create_classes
from function_samples_dataset import FunctionSamplesDataset, GaussianProcessRandomDataset, RepeatedFunctionSamplesIterableDataset, FunctionSamplesItem

from utils import resize_iterable, uniform_randint, get_uniform_randint_generator, max_pad_tensors_batch, pad_tensor
from utils import SizedIterableMixin, len_or_inf, save_json, load_json

from typing import Optional, List, Tuple, Union
from collections.abc import Sequence

import warnings
import math
from tqdm import tqdm
from tictoc import tic, tocl


torch.set_default_dtype(torch.double)


# TODO: Could make it so that acqf_params are in here too
# so that GI lambda values can be stored and costant for the test dataset,
# but it's not absolutely necessary
class AcquisitionDatasetModelItem(TupleWithModel):
    """
    x_hist shape: `n_hist x d`
    y_hist shape: `n_hist x n_hist_out`
    x_cand shape: `n_cand x d`
    vals_cand shape: `n_cand x (k+z)`, 1 <= k <= n_hist_out, z >= 0
    """
    args_names = ['x_hist', 'y_hist', 'x_cand', 'vals_cand']
    kwargs_names = ['give_improvements']

    def validate_data(self):
        SingleTaskGP._validate_tensor_args(self.x_hist, self.y_hist)
        SingleTaskGP._validate_tensor_args(self.x_cand, self.vals_cand)
        assert type(self.give_improvements) is bool


(AcquisitionDataset,
 MapAcquisitionDataset,
 ListMapAcquisitionDataset,
 LazyMapAcquisitionDataset,
 MapAcquisitionSubset) = create_classes(
     dataset_base_name='AcquisitionDataset',
     dataset_base_docstring=r"""Abstract class for acquisition function datasets.
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
            # y_hist shape: (n_hist, 1)
            # x_cand shape: (n_cand, dimension)
            # improvements shape: (n_cand, 1)
    """,

     map_dataset_base_name='MapAcquisitionDataset',
     map_dataset_base_docstring=None,

     list_dataset_name='ListMapAcquisitionDataset',
     list_dataset_docstring=None,
     
     lazy_dataset_name='LazyMapAcquisitionDataset',
     lazy_dataset_docstring=None,
     
     map_subset_name='MapAcquisitionSubset',
     map_subset_docstring=None,
     
     tuple_class=AcquisitionDatasetModelItem)


class AcquisitionDatasetBatch(TupleWithModel):
    args_names = ['x_hist', 'y_hist', 'x_cand', 'vals_cand', 'hist_mask', 'cand_mask']
    kwargs_names = ['give_improvements']


@staticmethod
def _collate_train_acquisition_function_samples(samples_list, has_models, cached_pads=False):
    give_improvements = samples_list[0].give_improvements
    for x in samples_list:
        if not isinstance(x, AcquisitionDatasetModelItem):
            raise TypeError("All items in samples_list should be AcquisitionDatasetModelItem")
        if x.give_improvements != give_improvements:
            raise ValueError(
                "All items in samples_list should have the same value for give_improvements")
        if x.has_model != has_models:
            raise ValueError(
                "All items in samples_list should have the same value for has_models " \
                "and should be consistent with the dataset's has_models attribute")

    if has_models:
        models_list = ModelsWithParamsList(
            [(x._model, x.model_params) for x in samples_list])
    else:
        models_list = None
    
    if cached_pads:
        unzipped_lists = list(zip(*
            [x._padded_tensors for x in samples_list]))
        (x_hist_paddeds, y_hist_paddeds, x_cand_paddeds,
            vals_cand_paddeds, hist_masks, cand_masks) = unzipped_lists
        x_hist = torch.stack(x_hist_paddeds)
        y_hist = torch.stack(y_hist_paddeds)
        hist_mask = torch.stack(hist_masks)
        x_cand = torch.stack(x_cand_paddeds)
        vals_cand = torch.stack(vals_cand_paddeds)
        cand_mask = torch.stack(cand_masks)
    else:
        if has_models:
            unzipped_lists = list(zip(*
                    [x[:4] for x in samples_list]))
        else:
            unzipped_lists = list(zip(*samples_list))

        # Each of these are tuples of tensors
        x_hists, y_hists, x_cands, vals_cands = unzipped_lists

        # x_hist shape: (n_hist, dimension)
        # y_hist shape: (n_hist, n_hist_out)
        # x_cand shape: (n_cand, dimension)
        # vals_cand shape: (n_cand, k+z), 1 <= k <= n_hist_out, z >= 0

        x_hist = max_pad_tensors_batch(x_hists, add_mask=False)
        y_hist, hist_mask = max_pad_tensors_batch(y_hists, add_mask=True)
        x_cand = max_pad_tensors_batch(x_cands, add_mask=False)
        vals_cand, cand_mask = max_pad_tensors_batch(vals_cands, add_mask=True)

        # print(f"{x_hist.shape=}, {y_hist.shape=}, {x_cand.shape=}, {vals_cand.shape=}, hist_mask.shape={hist_mask.shape if hist_mask is not None else None}, cand_mask.shape={cand_mask.shape if cand_mask is not None else None}")

    return AcquisitionDatasetBatch(
        x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask,
        model=models_list, give_improvements=give_improvements)


def _make_mask(mask_shape, mask_index, n_ones, device):
    ones_shape = list(mask_shape)
    ones_shape[mask_index] = n_ones

    n_zeros = mask_shape[mask_index] - n_ones

    zeros_shape = list(mask_shape)
    zeros_shape[mask_index] = n_zeros

    mask_true = torch.ones(*ones_shape, dtype=torch.bool, device=device)
    mask_false = torch.zeros(*zeros_shape, dtype=torch.bool, device=device)

    return torch.cat((mask_true, mask_false), dim=mask_index)


# get_dataloader is a method of AcquisitionDataset
def _get_dataloader(self, batch_size=32, shuffle=None,
                   cache_pads:Optional[bool]=None, **kwargs):
    """Returns a DataLoader object for the dataset.

    Args:
        batch_size (int):
            The batch size for the DataLoader. Default is 32.
        shuffle (bool):
            Whether to shuffle the data. If self is an IterableDataset, then the
            data is already shuffled and this parameter is ignored, and shuffle
            is set to False in the DataLoader. Otherwise, self is assumed to be
            a map-style dataset, and shuffle can be either True or False and
            defaults to True.
        **kwargs:
            Additional keyword arguments to be passed to the DataLoader.

    Raises:
        ValueError: If 'collate_fn' is specified in kwargs.

    Returns:
        DataLoader: A DataLoader object for the dataset, where
        each batch contains a list of tensors
        [x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask, models] or
        [x_hist, y_hist, x_cand, vals_cand, hist_mask, cand_mask] if models
        are not associated with the dataset.
        
        x_hist has shape (batch_size, n_hist, dimension),
        y_hist and hist_mask have shape (batch_size, n_hist, n_hist_out),
        x_cand has shape (batch_size, n_cand, dimension),
        vals_cand and cand_mask have shape (batch_size, n_cand, k+z),
        and models is a batch_size length tuple of GP models associated
        with the dataset.
        Everything is padded with zeros along with the corresponding masks.
    """
    if isinstance(self, IterableDataset):
        shuffle = False # Can't do shuffle=True on a IterableDataset
    elif shuffle is None:
        shuffle = True  # Default to shuffle=True for map-style datasets

    if 'collate_fn' in kwargs:
        raise ValueError("collate_fn should not be specified in get_dataloader; we do it for you")

    self._has_cached_pads = getattr(self, '_has_cached_pads', False)
    if self._has_cached_pads or cache_pads:
        assert self.data_is_fixed
    if self.data_is_fixed:
        if cache_pads is None:
            cache_pads = True
        if cache_pads and not self._has_cached_pads:
            max_hist = max(item.x_hist.size(0) for item in self)
            max_cand = max(item.x_cand.size(0) for item in self)
            print(f"max_hist={max_hist}")
            tic("Padding")
            for item in tqdm(self, desc="Padding"):
                x_hist, y_hist, x_cand, vals_cand = item.tuple_no_model
                x_hist_padded = pad_tensor(x_hist, max_hist, dim=0, add_mask=False)
                x_cand_padded = pad_tensor(x_cand, max_cand, dim=0, add_mask=False)

                # y_hist is a tensor of shape (n_hist, n_hist_out)
                # vals_cand is a tensor of shape (n_cand, something)
                
                y_hist_padded = pad_tensor(y_hist, max_hist, dim=0, add_mask=False)
                vals_cand_padded = pad_tensor(vals_cand, max_cand, dim=0, add_mask=False)
                hist_mask = _make_mask((max_hist, 1), 0, y_hist.size(0), y_hist.device)
                cand_mask = _make_mask((max_cand, 1), 0, vals_cand.size(0), vals_cand.device)
                
                item._padded_tensors = (x_hist_padded, y_hist_padded, x_cand_padded,
                                        vals_cand_padded, hist_mask, cand_mask)
            tocl()
            self._has_cached_pads = True

    collate_fn = partial(self._collate_train_acquisition_function_samples,
                        has_models=self.has_models,
                        cached_pads=self._has_cached_pads)

    return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=collate_fn, **kwargs)


AcquisitionDataset._collate_train_acquisition_function_samples = _collate_train_acquisition_function_samples
AcquisitionDataset.get_dataloader = _get_dataloader


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
                 min_history=1, max_history=None,
                 dataset_size_factor:Optional[int]=None,
                 y_cand_indices:Union[str,Sequence[int]]="all"):
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
                    [min_n_candidates+min_history...n], where n is the number of
                    samples in the dataset iteration, and then n_candidate_points
                    is chosen based on that.
                    - If n_candidate_points is an integer or a tuple of two
                    integers, then n_candidate_points is first chosen and then
                    n_samples is chosen uniformly in
                    [n_candidate_points+min_history...n].
            
            give_improvements (bool): Whether to generate improvement values as
                targets instead of raw y-values of the candidate points.
            
            min_n_candidates (int): The minimum number of candidate points for
                every iteration. Only used if n_candidate_points is "uniform" or
                "binomial"; ignored otherwise.
            
            min_history (int): The minimum number of history points (default: 1).

            max_history (int): The maximum number of history points (default: None).
            
            dataset_size_factor (Optional[int]): If the base dataset is a
                map-style dataset
                (i.e. a ListMapFunctionSamplesDataset or MapFunctionSamplesSubset),
                this parameter specifies the expansion factor for the dataset
                size. The dataset size is determined by the size of the base
                dataset multiplied by this factor. Default is 1.
                If the base dataset is an iterable-style dataset
                (i.e. GaussianProcessRandomDataset), then this
                parameter should not be specified.
            
            y_cand_indices (Union[str, Sequence[int]], default: "all"):
                Which indices of the y values to include from the dataset as the
                candidate y values. "all" means all of them, and a list of indices means
                use those indices.
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

        if not isinstance(min_history, int) or min_history <= 0:
            raise ValueError(f"min_history should be a positive integer, but got min_history={min_history}")
        self.min_history = min_history

        if not ((isinstance(max_history, int) and max_history > 0) or max_history is None):
            raise ValueError(f"max_history should be a positive integer or None, but got max_history={max_history}")
        self.max_history = max_history
        
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
        
        if y_cand_indices == "all":
            self._y_cand_indices = "all"
        elif isinstance(y_cand_indices, Sequence):
            if not all(isinstance(i, int) for i in y_cand_indices):
                raise ValueError("y_cand_indices should be 'all' or a list of "
                                 f"integers, but got {y_cand_indices=}")
            self._y_cand_indices = list(y_cand_indices)
        else:
            raise ValueError(
                "y_cand_indices should be 'all' or a list of integers, "
                f"but got {y_cand_indices=}")

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
        else: # e.g. ListMapFunctionSamplesDataset
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
    
    def _init_params(self):
        return (self.base_dataset,), dict(
            n_candidate_points=self.n_candidate_points,
            n_samples=self.n_samples,
            give_improvements=self.give_improvements,
            min_n_candidates=self.min_n_candidates,
            min_history=self.min_history,
            max_history=self.max_history,
            dataset_size_factor=self.dataset_size_factor,
            y_cand_indices=self._y_cand_indices
        )

    def save(self, dir_name: str, verbose:bool=True):
        super().save(dir_name, verbose)

        base_dataset_path = os.path.join(dir_name, "base_dataset")
        kwargs_path = os.path.join(dir_name, "kwargs")

        (base_dataset,), kwargs = self._init_params()
        base_dataset.save(base_dataset_path, verbose)
        save_json(kwargs, kwargs_path)
    
    @classmethod
    def load(cls, dir_name: str, verbose=True):
        base_dataset_path = os.path.join(dir_name, "base_dataset")
        kwargs_path = os.path.join(dir_name, "kwargs")

        base_dataset = FunctionSamplesDataset.load(base_dataset_path)
        kwargs = load_json(kwargs_path)
        return cls(base_dataset, **kwargs)

    @property
    def data_is_fixed(self):
        return False
    
    def data_is_loaded(self):
        return self.base_dataset.data_is_loaded()

    @property
    def _model_sampler(self):
        if not hasattr(self.base_dataset, "_model_sampler"):
            raise RuntimeError(f"The base dataset of this {self.__class__.__name__} is a {self.base_dataset.__class__.__name__} "
                               " which does not have the _model_sampler attribute")
        return self.base_dataset._model_sampler

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
        args, kwargs = self._init_params()
        kwargs['dataset_size_factor'] = size_factor
        return type(self)(*args, **kwargs)

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
            size_factor = math.ceil(size / len(self.base_dataset))
            return self.copy_with_expanded_size(size_factor)

    def _pick_random_n_samples_and_n_candidates(self, n_datapoints_original):
        min_history = self.min_history
        max_history = math.inf if self.max_history is None else self.max_history

        if self._gen_n_candidates_first:
            # generate n_candidates first; either fixed or random
            n_candidates = self._gen_n_candidates()

            # Need to have at least `min_history` history points
            if not (n_candidates + min_history <= n_datapoints_original):
                raise ValueError(f"n_datapoints_original={n_datapoints_original} should be at least n_candidates+min_history={n_candidates}+{min_history}")

            # n_history = n_samples - n_candidates <= max_history
            # n_samples <= max_history + n_candidates

            # generate n_samples
            max_n_samples = min(n_datapoints_original, max_history + n_candidates)
            if self.n_samples == "all":
                n_samples = max_n_samples
            elif self.n_samples == "uniform":
                n_samples = uniform_randint(n_candidates+min_history, max_n_samples)
        else:
            # n_candidates is "uniform" or "binomial"

            min_n_candidates = self.min_n_candidates

            if not (min_n_candidates + min_history <= n_datapoints_original):
                raise ValueError(f"n_datapoints_original={n_datapoints_original} should be at least min_n_candidates+min_history={min_n_candidates}+{min_history}")

            # generate n_samples first; either "all" or "uniform"
            if self.n_samples == "all":
                n_samples = n_datapoints_original
            elif self.n_samples == "uniform":
                n_samples = uniform_randint(min_n_candidates+min_history, n_datapoints_original)

            # generate n_candidates
            # n_history = n_samples - n_candidates >= min_history
            # n_samples - min_history >= n_candidates
            # n_history = n_samples - n_candidates <= max_history
            # n_samples - max_history <= n_candidates
            min_cands = max(min_n_candidates, n_samples - max_history)
            if self.n_candidate_points == "uniform":
                n_candidates = uniform_randint(min_cands, n_samples-min_history)
            elif self.n_candidate_points == "binomial":
                n_candidates = int(torch.distributions.Binomial(n_samples, 0.5).sample())
                while not (min_cands <= n_candidates <= n_samples-min_history):
                    n_candidates = int(torch.distributions.Binomial(n_samples, 0.5).sample())
        
        if torch.is_tensor(n_samples):
            n_samples = n_samples.item()
        return n_samples, n_candidates

    def __iter__(self):
        # x_values has shape (n_datapoints, dimension)
        # y_values has shape (n_datapoints, n_out)
        for item in self._data_iterable:
            if not isinstance(item, FunctionSamplesItem):
                raise TypeError(f"item should be an instance of FunctionSamplesItem, but got {item=}")
            x_values, y_values = item.x_values, item.y_values
            if item.has_model:
                model, model_params = item._model, item.model_params
            else:
                model, model_params = None, None
            
            n_datapoints = x_values.shape[0]

            n_samples, n_candidate = self._pick_random_n_samples_and_n_candidates(n_datapoints)

            rand_idx = torch.randperm(n_datapoints)
            candidate_idx = rand_idx[:n_candidate]
            hist_idx = rand_idx[n_candidate:n_samples]

            x_hist, y_hist = x_values[hist_idx], y_values[hist_idx]
            x_cand = x_values[candidate_idx]
            y_cand = y_values[candidate_idx]

            if self.give_improvements:
                best_f = y_hist.amax(0, keepdim=True)
                improvement_values = torch.nn.functional.relu(
                    y_cand - best_f, inplace=True)
                vals_cand = improvement_values
            else:
                vals_cand = y_cand
            
            if self._y_cand_indices != "all":
                vals_cand = vals_cand[:, self._y_cand_indices]

            yield AcquisitionDatasetModelItem(
                x_hist, y_hist, x_cand, vals_cand, model, model_params,
                give_improvements=self.give_improvements)

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


class CostAwareAcquisitionDataset(
    AcquisitionDataset, IterableDataset, SizedIterableMixin):
    def __init__(self,
                 base_dataset: AcquisitionDataset,
                 lambda_min:float,
                 lambda_max:Optional[float]=None):
        if not isinstance(base_dataset, AcquisitionDataset):
            raise ValueError("CostAwareAcquisitionDataset: base_dataset must be a AcquisitionDataset")
        self.base_dataset = base_dataset

        self._size = len_or_inf(base_dataset) # for SizedIterableMixin

        if not (isinstance(lambda_min, float) and lambda_min > 0):
            raise ValueError(f"lambda_min must be a positive float; was {lambda_min}")
        self._log_lambda_min = math.log(lambda_min)
        
        if lambda_max == lambda_min:
            lambda_max = None
        elif lambda_max is not None:
            if not (isinstance(lambda_max, float) and lambda_max > 0):
                raise ValueError(f"lambda_max must be a positive float; was {lambda_max}")
            if not (lambda_min <= lambda_max):
                raise ValueError("lambda_min must be <= lambda_max")
            self._log_lambda_diff = math.log(lambda_max) - self._log_lambda_min
    
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
    
    def _init_params(self):
        return (self.base_dataset,), dict(
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max
        )

    def save(self, dir_name: str, verbose:bool=True):
        raise NotImplementedError
    
    @classmethod
    def load(cls, dir_name: str, verbose=True):
        raise NotImplementedError

    @property
    def data_is_fixed(self):
        return self.base_dataset.data_is_fixed and self.lambda_max is None
    
    def data_is_loaded(self):
        return self.base_dataset.data_is_loaded()

    @property
    def _model_sampler(self):
        if not hasattr(self.base_dataset, "_model_sampler"):
            raise RuntimeError(f"The base dataset of this {self.__class__.__name__} is a {self.base_dataset.__class__.__name__} "
                               " which does not have the _model_sampler attribute")
        return self.base_dataset._model_sampler
    
    def copy_with_expanded_size(self, size_factor: int) -> "CostAwareAcquisitionDataset":
        return type(self)(
            self.base_dataset.copy_with_expanded_size(size_factor),
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max
        )
    
    def copy_with_new_size(self, size: Optional[int] = None) -> "CostAwareAcquisitionDataset":
        return type(self)(
            self.base_dataset.copy_with_new_size(size),
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max
        )

    def random_split(self, lengths: Sequence[Union[int, float]]) -> List['CostAwareAcquisitionDataset']:
        return [
            type(self)(ds, lambda_min=self.lambda_min, lambda_max=self.lambda_max)
            for ds in self.base_dataset.random_split(lengths)
        ]

    def __iter__(self):
        for item in self.base_dataset:
            vals_cand = item.vals_cand # shape n_cand x (some number)
            n_cand = vals_cand.size(0)

            if self.lambda_max is None:
                log_lambdas = torch.full((n_cand, 1), self._log_lambda_min,
                                         dtype=vals_cand.dtype, device=vals_cand.device)
            else:
                log_lambdas = torch.rand(
                    n_cand, 1, dtype=vals_cand.dtype, device=vals_cand.device) \
                    * self._log_lambda_diff + self._log_lambda_min
            
            vals_cand = torch.cat((vals_cand, log_lambdas), dim=1)

            if item.has_model:
                yield AcquisitionDatasetModelItem(
                    item.x_hist, item.y_hist, item.x_cand, vals_cand,
                    item.model, item.model_params,
                    give_improvements=item.give_improvements)
            else:
                yield AcquisitionDatasetModelItem(
                    item.x_hist, item.y_hist, item.x_cand, vals_cand,
                    give_improvements=item.give_improvements)
