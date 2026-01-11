from typing import ClassVar, Optional, Type
from types import SimpleNamespace
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from utils_general.tictoc import tic, toc
from utils_general.utils import check_subclass, int_linspace


class TrainOrTestLoop(ABC):
    @classmethod
    def train_or_test_loop(
        cls,
        dataloader: DataLoader,
        nn_model: Optional[nn.Module]=None,
        train:Optional[bool]=None,
        nn_device=None,
        method:Optional[str]=None, # ONLY used when training NN
        verbose:bool=True,
        desc:Optional[str]=None,
        
        n_train_printouts:Optional[int]=None,
        optimizer:Optional[torch.optim.Optimizer]=None,

        # Whether to return None if there is nothing to compute
        return_none=False,

        **specific_kwargs
    ) -> Optional[dict]:
        loop_state = cls(
            dataloader=dataloader,
            nn_model=nn_model,
            train=train,
            nn_device=nn_device,
            method=method,
            verbose=verbose,
            desc=desc,
            n_train_printouts=n_train_printouts,
            optimizer=optimizer,
            **specific_kwargs
        )

        if loop_state.do_nothing:
            if return_none:
                return None
            # If we are not computing any stats, then don't actually need to go through
            # the dataset. Also make verbose=False in this case.
            it = []
            verbose = False
        else:
            it = dataloader
        
        if verbose:
            if train and n_train_printouts is not None:
                print(desc)
                print_indices = set(int_linspace(
                    0, loop_state.n_training_batches - 1,
                    min(n_train_printouts, loop_state.n_training_batches)))
            else:
                it = tqdm(it, desc=desc)
            tic(desc)
            
        dataset_length = 0
        for batch_index, batch in enumerate(it):
            loop_state.batch_index = batch_index

            batch_data = loop_state.get_data_from_batch(batch)
            this_batch_size = batch_data.batch_size
            
            assert this_batch_size <= loop_state.batch_size
            is_degenerate_batch = this_batch_size < loop_state.batch_size
            if batch_index != loop_state.n_batches - 1:
                assert not is_degenerate_batch
            dataset_length += this_batch_size

            if nn_model is not None:
                batch_data_nn = loop_state.get_data_from_batch_for_nn(batch)

                with torch.set_grad_enabled(train and not is_degenerate_batch):
                    nn_output = loop_state.evaluate_nn_on_batch(batch_data_nn)

                    nn_batch_stats = loop_state.compute_nn_batch_stats(nn_output, batch_data_nn)
                    
                    if train and not is_degenerate_batch:
                        # convert sum to mean so that this is consistent across batch sizes
                        loss = nn_batch_stats.pop("loss") / this_batch_size # (== batch_size)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        loop_state.perform_post_optimizer_step_updates()

                        if verbose and n_train_printouts is not None and batch_index in print_indices:
                            loop_state.print_train_batch_stats(nn_batch_stats, this_batch_size)
                            
                    loop_state.nn_batch_stats_list.append(nn_batch_stats)

            loop_state.compute_additional_batch_stats(batch_data)

        if not loop_state.do_nothing:
            assert dataset_length == len(loop_state.dataset)
        if verbose:
            toc(desc)
        return loop_state.aggregate_stats()

    methods: ClassVar[list[str]]
    module_class: ClassVar[Type[nn.Module]]

    def __init_subclass__(cls, **kwargs):
        # Validate that subclasses define 'methods' correctly
        super().__init_subclass__(**kwargs)
        
        error_message = f"Subclasses of {cls.__name__} must define a valid " \
            f"'methods' class variable"
        if not hasattr(cls, "methods"):
            raise TypeError(error_message)
        if not isinstance(cls.methods, list) or not all(
            isinstance(m, str) for m in cls.methods):
            raise TypeError(error_message)
        
        error_message = f"Subclasses of {cls.__name__} must define a valid " \
            f"'module_class' class variable"
        if not hasattr(cls, "module_class"):
            raise TypeError(error_message)
        try:
            check_subclass(cls.module_class, "module_class", nn.Module)
        except ValueError as e:
            raise TypeError(f"{error_message}: {e}") from e

    def __init__(
            self,
            dataloader: DataLoader,
            nn_model: Optional[nn.Module]=None,
            train:Optional[bool]=None,
            nn_device=None,
            method:Optional[str]=None, # ONLY used when training NN
            verbose:bool=True,
            desc:Optional[str]=None,
            
            n_train_printouts:Optional[int]=None,
            optimizer:Optional[torch.optim.Optimizer]=None,

            **specific_kwargs):
        """Initialize the TrainOrTestLoopState.
        Must set an attribute `do_nothing` to self, which is a boolean indicating
        whether there is anything to compute
        (True if there is nothing to compute, False otherwise)."""
        
        self.do_nothing = nn_model is None

        self.nn_model = nn_model
        self.train = train
        self.nn_device = nn_device
        self.method = method

        if nn_model is not None:
            self.nn_batch_stats_list = []
        
        self.n_batches = len(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size

        self.n_training_batches = self.n_batches
        if len(self.dataset) % self.batch_size != 0:
            self.n_training_batches -= 1

        if not isinstance(dataloader, DataLoader):
            raise ValueError("dataloader must be a torch DataLoader")
        
        if dataloader.drop_last:
            raise ValueError("dataloader must have drop_last=False")
        
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")
        
        if n_train_printouts == 0:
            n_train_printouts = None

        if nn_model is not None: # evaluating a NN model
            if not isinstance(nn_model, self.module_class):
                raise ValueError(
                    f"nn_model must be a {self.module_class.__name__} instance")
            if not isinstance(train, bool):
                raise ValueError("'train' must be a boolean if evaluating a NN model")
            
            if method not in self.methods:
                methods_str = ', '.join(f"'{x}'" for x in self.methods)
                raise ValueError(
                    f"'method' must be one of {methods_str} if evaluating a NN model; "
                    f"it was {method}")
            
            if train:
                if optimizer is None:
                    raise ValueError("optimizer must be specified if training")
                if not isinstance(optimizer, torch.optim.Optimizer):
                    raise ValueError("optimizer must be a torch Optimizer instance")
                if verbose:
                    if n_train_printouts is not None:
                        if not (isinstance(n_train_printouts, int) and n_train_printouts >= 0):
                            raise ValueError("n_train_printouts must be a non-negative integer")
                nn_model.train()
            else:
                nn_model.eval()
        else: # just evaluating the dataset, no NN model
            if method is not None:
                raise ValueError("'method' must not be specified if not evaluating a NN model")
            if train is not None:
                raise ValueError("'train' must not be specified if not evaluating a NN model")
            if nn_device is not None:
                raise ValueError("'nn_device' must not be specified if not evaluating a NN model")
        
        if not train:
            if optimizer is not None:
                raise ValueError("optimizer must not be specified if train != True")
        
        if verbose:
            if not (desc is None or isinstance(desc, str)):
                raise ValueError("desc must be a string or None if verbose")
    
    @abstractmethod
    def get_data_from_batch(self, batch) -> SimpleNamespace:
        pass

    @abstractmethod
    def get_data_from_batch_for_nn(self, batch) -> SimpleNamespace:
        pass

    @abstractmethod
    def evaluate_nn_on_batch(self, batch_data: SimpleNamespace) -> Tensor:
        """Evaluate self.nn_model on the given batch_data and return the output Tensor.
        """
        pass

    @abstractmethod
    def compute_nn_batch_stats(
            self, nn_output, batch_data: SimpleNamespace) -> dict:
        pass
    
    def perform_post_optimizer_step_updates(self):
        """Perform any updates needed after the optimizer step during training.
        This is a no-op by default, but can be overridden in subclasses."""
        pass

    @abstractmethod
    def print_train_batch_stats(self, nn_batch_stats: dict, batch_size: int):
        pass

    def compute_additional_batch_stats(self, batch_data: SimpleNamespace):
        """Compute any additional batch statistics that do not involve the NN model.
        This is a no-op by default, but can be overridden in subclasses."""
        pass

    def aggregate_stats(self) -> dict:
        """Aggregate the statistics collected during the loop.
        By default, only aggregates the NN batch statistics if a NN model was used.
        Subclasses can override this to add more statistics."""
        ret = {}
        if self.nn_model is not None:
            ret.update(self.get_average_stats(
                self.nn_batch_stats_list, "sum", self.dataset_length))
        return ret
    
    @classmethod
    def get_average_stats(cls, stats_list, batch_reduction:str, total_n_samples=None):
        if batch_reduction == "mean":
            assert total_n_samples is None
            divisor = len(stats_list)
        elif batch_reduction == "sum":
            divisor = total_n_samples
        else:
            raise ValueError("'batch_reduction' must be either 'mean' or 'sum'")
        return {key: sum(d[key] for d in stats_list) / divisor
                for key in stats_list[0]}
    
    @property
    def dataset_length(self):
        """Utility property to get the length of the dataset"""
        return 0 if self.do_nothing else len(self.dataset)
