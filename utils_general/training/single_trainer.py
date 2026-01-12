from abc import ABC, abstractmethod
import copy
import os
from typing import ClassVar, Optional, Sequence, Type
import cProfile, pstats
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils_general.nn_utils import count_parameters, count_trainable_parameters
from utils_general.tictoc import tic, tocl
from utils_general.torch_module_save_utils import TorchModuleSaveUtils
from utils_general.training.train_or_test_loop import TrainOrTestLoop
from utils_general.training.train_utils import EarlyStopper
from utils_general.io_utils import load_json, save_json
from utils_general.utils import DEVICE, check_subclass


class SingleTrainer(ABC):
    def __init__(self, torch_model_save_instance: TorchModuleSaveUtils):
        """Initialize SingleTrainer with model save/load utilities.

        Args:
            torch_model_save_instance: Instance for saving and loading PyTorch modules.
        """
        if not isinstance(torch_model_save_instance, TorchModuleSaveUtils):
            raise ValueError(
                "torch_model_save_instance must be an instance of TorchModuleSaveUtils")
        self.torch_model_save_instance = torch_model_save_instance

    def single_train(
            self,
            cmd_args: Optional[Sequence[str]]=None,
            save_incremental_best_models: bool=False,
            use_cprofile: bool=False,
            measure_time: bool=False,
            verbose: bool=True) -> None:
        """Train a neural network model from command-line arguments or defaults.

        Args:
            cmd_args: Command-line arguments for training configuration.
            save_incremental_best_models: Whether to save each new best model during training.
            use_cprofile: Whether to profile the training run.
            measure_time: Whether to measure and report training time.
            verbose: Whether to print training progress and statistics.
        """
        torch_model_save_instance = self.torch_model_save_instance
        basic_save_utils = torch_model_save_instance.basic_save_utils

        (args, model, model_and_info_folder_name, models_path
        ) = torch_model_save_instance.get_args_module_paths_from_cmd_args(cmd_args)

        root_models_path = basic_save_utils.models_path
        model_and_info_path = os.path.join(root_models_path, model_and_info_folder_name)

        if args.load_saved_model:
            model, model_path = torch_model_save_instance.load_module(
                model_and_info_folder_name, return_model_path=True)
        else:
            model_path = None

        model = model.to(DEVICE)

        print(model)
        print("Number of trainable parameters:", count_trainable_parameters(model))
        print("Number of parameters:", count_parameters(model))
        print(f"\nSaving model and configs to {model_and_info_path}\n")

        ####################### Make the train and test datasets #######################
        (train_dataset, test_dataset,
        small_test_dataset) = self.create_train_test_datasets_from_args(args)

        extra_kwargs = self.get_extra_kwargs_namespace(
                args, train_dataset, test_dataset, small_test_dataset)
        
        ######################## Train the model #######################################
        if args.train:
            if args.save_model:
                model_path, model_name = basic_save_utils.get_new_model_save_dir(models_path)
            else:
                model_path = None

            if use_cprofile:
                pr = cProfile.Profile()
                pr.enable()
            
            if measure_time:
                tic("Training")
            
            print(f"learning rate: {args.learning_rate}, batch size: {args.batch_size}")
            weight_decay = 0.0 if args.weight_decay is None else args.weight_decay
            c = torch.optim.AdamW if weight_decay > 0 else torch.optim.Adam
            optimizer = c(model.parameters(), lr=args.learning_rate,
                            weight_decay=weight_decay)
            # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
            # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

            training_history_data = self._train_neural_net(
                model, train_dataset, optimizer, args.epochs, args.batch_size,
                args.method,
                
                nn_device=DEVICE, verbose=verbose, n_train_printouts_per_epoch=10,
                test_dataset=test_dataset, small_test_dataset=small_test_dataset,
                get_train_stats_while_training=True,
                get_train_stats_after_training=True,
                
                save_dir=model_path,
                save_incremental_best_models=save_incremental_best_models and args.save_model,
                # early stopping
                early_stopping=args.early_stopping,
                patience=args.patience,
                min_delta=args.min_delta,
                cumulative_delta=args.cumulative_delta,
                # learning rate scheduler
                lr_scheduler=args.lr_scheduler,
                lr_scheduler_patience=args.lr_scheduler_patience,
                lr_scheduler_factor=args.lr_scheduler_factor,
                lr_scheduler_min_lr=args.lr_scheduler_min_lr,
                lr_scheduler_cooldown=args.lr_scheduler_cooldown,
                lr_scheduler_power=args.lr_scheduler_power,
                lr_scheduler_burnin=args.lr_scheduler_burnin,

                extra_kwargs=extra_kwargs
            )

            if args.save_model:
                basic_save_utils.mark_new_model_as_trained(models_path, model_name)
                print(f"Saved best weights to {model_and_info_path}")

            if measure_time:
                tocl()

            if use_cprofile:
                pr.disable()

                with open('stats_output.txt', 'w') as s:
                    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
                    ps.print_stats()

            if not measure_time:
                print("Done training")
        else:
            training_history_data = None

        ######################## Evaluate and plot model performance #######################
        if not args.train and model_path is not None:
            training_history_path = os.path.join(
                model_path, 'training_history_data.json')
            training_history_data = load_json(training_history_path)
            final_test_stats_original = training_history_data['final_test_stats']
            self.print_stats(final_test_stats_original,
                        "Final test stats on the original test dataset",
                        args.method, extra_kwargs=extra_kwargs, is_test=True)

            test_dataloader = self.get_test_dataloader(test_dataset, args.batch_size)
            
            final_test_stats = self.train_or_test_loop_class.train_or_test_loop(
                        test_dataloader, model, train=False,
                        nn_device=DEVICE, method=args.method,
                        verbose=False, desc=f"Compute final test stats",
                        **self.get_final_test_stats_kwargs(extra_kwargs))
            self.print_stats(final_test_stats,
                        "Final test stats on this test dataset (should be same as above)",
                        args.method, extra_kwargs=extra_kwargs, is_test=True)

    train_or_test_loop_class: ClassVar[Type[TrainOrTestLoop]]

    @classmethod
    @abstractmethod
    def get_test_dataloader(cls, test_ds: Dataset, batch_size: int) -> DataLoader:
        """Create a DataLoader for the test dataset.

        Args:
            test_ds: Test dataset.
            batch_size: Batch size for the DataLoader.

        Returns:
            DataLoader for the test dataset.
        """
        pass

    @classmethod
    @abstractmethod
    def get_train_dataloaders(
        cls,
        get_train_stats_after_training: bool,
        train_dataset: Dataset,
        batch_size: int,
        test_during_training: bool,
        small_test_dataset: Optional[Dataset]
    ) -> tuple[Optional[DataLoader], Optional[callable], DataLoader]:
        """Returns a tuple
        train_dataloader, per_epoch_get_train_dataloader, train_dataset_eval_dataloader
        """
        pass

    @classmethod
    @abstractmethod
    def print_stats(cls,
                    stats: dict,
                    dataset_name: str,
                    method: str,
                    extra_kwargs: SimpleNamespace,
                    is_test: bool=False) -> None:
        """Print training or test statistics in a formatted way.

        Args:
            stats: Dictionary of statistics to print.
            dataset_name: Name of the dataset (for display).
            method: Training method name.
            extra_kwargs: Additional configuration parameters.
            is_test: Whether these are test (vs training) statistics.
        """
        pass

    @classmethod
    @abstractmethod
    def get_stat_name_and_minimize_flag(cls, method: str, extra_kwargs: SimpleNamespace) -> tuple[str, bool]:
        """Determine the statistic name and whether to minimize it for a given method.

        Args:
            method: Training method name.
            extra_kwargs: Additional configuration parameters.

        Returns:
            Tuple of (stat_name, minimize_stat) where minimize_stat is True if lower is better.
        """
        pass

    @classmethod
    @abstractmethod
    def get_extra_kwargs_namespace(
        cls, args, train_dataset: Dataset, test_dataset: Optional[Dataset],
        small_test_dataset: Optional[Dataset]) -> SimpleNamespace:
        """Create a namespace of extra keyword arguments for training.

        Args:
            args: Parsed command-line arguments.
            train_dataset: Training dataset.
            test_dataset: Test dataset.
            small_test_dataset: Small test dataset for evaluation during training.

        Returns:
            SimpleNamespace containing additional configuration parameters.
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_train_stats_while_training_kwargs(cls, extra_kwargs: SimpleNamespace) -> dict:
        """Get kwargs for computing statistics during training (forward+backward pass).

        Args:
            extra_kwargs: Additional configuration parameters.

        Returns:
            Dictionary of keyword arguments for train_or_test_loop.
        """
        pass

    @classmethod
    @abstractmethod
    def get_train_stats_after_training_kwargs(cls, extra_kwargs: SimpleNamespace) -> dict:
        """Get kwargs for computing statistics after training (eval mode).

        Args:
            extra_kwargs: Additional configuration parameters.

        Returns:
            Dictionary of keyword arguments for train_or_test_loop.
        """
        pass

    @classmethod
    @abstractmethod
    def get_test_during_training_kwargs(cls, extra_kwargs: SimpleNamespace) -> dict:
        """Get kwargs for computing test statistics during training epochs.

        Args:
            extra_kwargs: Additional configuration parameters.

        Returns:
            Dictionary of keyword arguments for train_or_test_loop.
        """
        pass

    @classmethod
    @abstractmethod
    def get_test_after_training_kwargs(
        cls, extra_kwargs: SimpleNamespace, test_dataset: Dataset) -> dict:
        """Get kwargs for computing final test statistics after training completes.

        Args:
            extra_kwargs: Additional configuration parameters.
            test_dataset: Test dataset.

        Returns:
            Dictionary of keyword arguments for train_or_test_loop.
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_final_test_stats_kwargs(cls, extra_kwargs: SimpleNamespace) -> dict:
        """Get kwargs for computing final test statistics (used when loading saved models).

        Args:
            extra_kwargs: Additional configuration parameters.

        Returns:
            Dictionary of keyword arguments for train_or_test_loop.
        """
        pass

    @classmethod
    @abstractmethod
    def create_train_test_datasets_from_args(
        cls, args) -> tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """Returns (train_dataset, test_dataset, small_test_dataset)"""

    @classmethod
    def is_non_nn_stat_name(cls, stat_name: str) -> bool:
        """Returns True if the stat_name is a non-neural-net statistic.
        By default, always returns False. Override in subclasses if needed."""
        return False

    @classmethod
    def extra_checks(
        cls, extra_kwargs: SimpleNamespace, test_during_training: bool,
        test_after_training: bool) -> None:
        """Hook for any extra checks needed in specific implementations.
        Default does nothing."""
        pass

    @classmethod
    def _train_neural_net(
            cls,
            nn_model: nn.Module,
            train_dataset: Dataset,
            optimizer: torch.optim.Optimizer,
            n_epochs: int,
            batch_size: int,
            method: str,

            nn_device: Optional[torch.device] = None,
            verbose: bool = True,
            n_train_printouts_per_epoch: Optional[int] = None,

            test_dataset: Optional[Dataset] = None,
            small_test_dataset: Optional[Dataset] = None,
            test_during_training: Optional[bool] = None,

            get_train_stats_while_training: bool = True,
            get_train_stats_after_training: bool = True,

            save_dir: Optional[str] = None,
            save_incremental_best_models: bool = True,

            early_stopping: bool = True,
            patience: int = 10,
            min_delta: float = 0.0,
            cumulative_delta: bool = False,

            # learning rate scheduler
            lr_scheduler: Optional[str] = None,
            lr_scheduler_patience: int = 10,
            lr_scheduler_factor: float = 0.1,
            lr_scheduler_min_lr: float = 1e-6,
            lr_scheduler_cooldown: int = 0,
            lr_scheduler_power: float = 0.6,
            lr_scheduler_burnin: int = 1,

            extra_kwargs: SimpleNamespace = SimpleNamespace()
        ) -> dict:
        stat_name, minimize_stat = cls.get_stat_name_and_minimize_flag(method, extra_kwargs)
        
        if not (isinstance(n_epochs, int) and n_epochs >= 1):
            raise ValueError("n_epochs must be a positive integer")
        if not (isinstance(batch_size, int) and batch_size >= 1):
            raise ValueError("batch_size must be a positive integer")
        if not (test_during_training is None or isinstance(test_during_training, bool)):
            raise ValueError("test_during_training must be a boolean or None")
        if not isinstance(verbose, bool):
            raise ValueError("verbose should be a boolean")

        if not isinstance(get_train_stats_while_training, bool):
            raise ValueError("get_train_stats_while_training should be a boolean")
        if not isinstance(get_train_stats_after_training, bool):
            raise ValueError("get_train_stats_after_training should be a boolean")
        if not (get_train_stats_while_training or get_train_stats_after_training):
            raise ValueError("You probably want to get some train stats...specify "
                            "either get_train_stats_while_training=True or "
                            "get_train_stats_after_training=True or both.")
        # if not isinstance(min_delta, float):
        #     raise ValueError("min_delta should be a float")
        # if not isinstance(cumulative_delta, bool):
        #     raise ValueError("cumulative_delta should be a boolean")
        # if not isinstance(early_stopping, bool):
        #     raise ValueError("early_stopping should be a boolean")
        if not isinstance(save_incremental_best_models, bool):
            raise ValueError("save_incremental_best_models should be a boolean")
        
        # if not isinstance(lr_scheduler_patience, int):
        #     raise ValueError("lr_scheduler_patience should be an integer")
        # if not isinstance(lr_scheduler_factor, float):
        #     raise ValueError("lr_scheduler_factor should be a float")
        # if not isinstance(lr_scheduler_min_lr, float):
        #     raise ValueError("lr_scheduler_min_lr should be a float")
        # if not isinstance(lr_scheduler_cooldown, int):
        #     raise ValueError("lr_scheduler_cooldown should be an integer")

        test_during_training, test_after_training = cls._get_test_during_after_training(
            test_dataset, small_test_dataset, test_during_training)

        if test_during_training:
            if small_test_dataset is None:
                small_test_dataset = test_dataset
            small_test_dataloader = cls.get_test_dataloader(small_test_dataset, batch_size)

        cls.extra_checks(extra_kwargs, test_during_training, test_after_training)
        
        (train_dataloader, per_epoch_get_train_dataloader,
        train_dataset_eval_dataloader) = cls.get_train_dataloaders(
            get_train_stats_after_training,
            train_dataset,
            batch_size,
            test_during_training,
            small_test_dataset
        )
        
        if save_incremental_best_models and save_dir is None:
            raise ValueError("Need to specify save_dir if save_incremental_best_models=True")
        
        best_score = None
        best_epoch = None
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            training_history_path = os.path.join(save_dir, 'training_history_data.json')
        
        if early_stopping:
            early_stopper = EarlyStopper(patience, min_delta, cumulative_delta)

        if lr_scheduler is None:
            scheduler = None
        elif lr_scheduler == "ReduceLROnPlateau":
            mode = "min" if minimize_stat else "max"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, patience=lr_scheduler_patience,
                factor=lr_scheduler_factor, min_lr=lr_scheduler_min_lr,
                cooldown=lr_scheduler_cooldown)
        elif lr_scheduler == "power":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (
                    1 if epoch <= lr_scheduler_burnin
                    else (epoch + 1 - lr_scheduler_burnin) ** -lr_scheduler_power
                )
            )
        else:
            raise ValueError(f"Unknown lr_scheduler '{lr_scheduler}'")

        training_history_data = {
            'stats_epochs': [],
            'stat_name': stat_name
        }

        for t in range(n_epochs):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            
            this_epoch_train_dataloader = train_dataloader if train_dataloader is not None \
                else per_epoch_get_train_dataloader()
            
            train_stats = {}
            
            train_stats_while_training = cls.train_or_test_loop_class.train_or_test_loop(
                this_epoch_train_dataloader, nn_model, train=True,
                nn_device=nn_device, method=method,
                verbose=verbose, desc=f"Epoch {t+1} train",
                n_train_printouts=n_train_printouts_per_epoch,
                optimizer=optimizer,
                **cls.get_train_stats_while_training_kwargs(extra_kwargs)
            )

            (train_nn_stats_while_training,
            non_nn_train_stats) = cls._split_nn_stats(train_stats_while_training)
            train_stats['non_nn_stats'] = non_nn_train_stats
            
            if get_train_stats_while_training:
                train_stats['while_training'] = train_nn_stats_while_training
                if verbose:
                    cls.print_stats({**train_stats['while_training'], **non_nn_train_stats},
                                "Train stats while training", method, extra_kwargs=extra_kwargs,
                                is_test=False)

            if get_train_stats_after_training:
                train_stats['after_training'] = cls.train_or_test_loop_class.train_or_test_loop(
                    train_dataset_eval_dataloader, nn_model, train=False,
                    nn_device=nn_device, method=method,
                    verbose=verbose, desc=f"Epoch {t+1} compute train stats",
                    **cls.get_train_stats_after_training_kwargs(extra_kwargs)
                )
                if verbose:
                    cls.print_stats({**train_stats['after_training'], **non_nn_train_stats},
                                "Train stats after training", method, extra_kwargs=extra_kwargs,
                                is_test=False)
            
            epoch_stats = {'train': train_stats}

            if test_during_training:
                test_stats = cls.train_or_test_loop_class.train_or_test_loop(
                    small_test_dataloader, nn_model, train=False,
                    nn_device=nn_device, method=method,
                    verbose=verbose, desc=f"Epoch {t+1} compute test stats",
                    **cls.get_test_during_training_kwargs(extra_kwargs)
                )
                epoch_stats['test'] = test_stats
                if verbose:
                    cls.print_stats(test_stats, "Test stats", method, extra_kwargs=extra_kwargs,
                                is_test=True)
            
            training_history_data['stats_epochs'].append(epoch_stats)

            # Determine the maxei statistic. Decreasing order of preference.
            # We would usually only do these if test_during_training=True,
            # but why not cover all cases.
            if test_during_training:
                cur_score = test_stats[stat_name]
            elif get_train_stats_after_training:
                cur_score = train_stats["after_training"][stat_name]
            else:
                cur_score = train_nn_stats_while_training[stat_name]
            cur_score_maximize = -cur_score if minimize_stat else cur_score
            
            # If the best score increased, then update that and maybe save
            if best_score is None or cur_score_maximize > best_score:
                prev_best_score = best_score
                best_score = cur_score_maximize
                best_epoch = t

                if verbose and prev_best_score is not None:
                    if minimize_stat:
                        msg = (f"Best score decreased from {-prev_best_score:>8f}"
                            f" to {-best_score:>8f}.")
                    else:
                        msg = (f"Best score increased from {prev_best_score:>8f}"
                                f" to {best_score:>8f}.")

                if save_incremental_best_models:
                    fname = f"model_{best_epoch}.pth"
                    if verbose and prev_best_score is not None:
                        print(msg + f" Saving weights to {fname}.")
                    torch.save(nn_model.state_dict(), os.path.join(save_dir, fname))
                else:
                    if verbose and prev_best_score is not None:
                        print(msg)
                    # If we don't save the best models during training, then
                    # we still want to save the best state_dict so need to
                    # keep a deepcopy of the best state_dict.
                    best_state_dict = copy.deepcopy(nn_model.state_dict())
            
            if save_dir is not None:
                # Saving every epoch because why not
                save_json(training_history_data, training_history_path, indent=4)
            
            # Early stopping
            if early_stopping and early_stopper(cur_score_maximize):
                if verbose:
                    print(
                        "Early stopping at epoch %i; counter is %i / %i" %
                        (t+1, early_stopper.counter, early_stopper.patience)
                    )
                break

            # Learning rate scheduler
            if verbose and scheduler is not None:
                if lr_scheduler == "ReduceLROnPlateau":
                    scheduler.step(cur_score)
                elif lr_scheduler == "power":
                    scheduler.step()
                _lr = scheduler.get_last_lr()
                assert len(_lr) == 1
                print(f"Learning rate: {_lr[0]}")
        
        best_model_fname = f"model_{best_epoch}.pth"
        best_state_dict_path = os.path.join(save_dir, best_model_fname)
        
        # Load the best model weights to return
        if save_incremental_best_models:
            best_state_dict = torch.load(best_state_dict_path)
        nn_model.load_state_dict(best_state_dict)

        if test_after_training:
            if test_during_training and (test_dataset is small_test_dataset):
                # If we already computed it then don't need to compute again
                final_test_stats = training_history_data['stats_epochs'][best_epoch]['test']
            else:
                test_dataloader = cls.get_test_dataloader(test_dataset, batch_size)
                final_test_stats = cls.train_or_test_loop_class.train_or_test_loop(
                    test_dataloader, nn_model, train=False,
                    nn_device=nn_device, method=method,
                    verbose=verbose, desc=f"Compute final test stats",
                    **cls.get_test_after_training_kwargs(extra_kwargs, test_dataset))
            training_history_data['final_test_stats'] = final_test_stats
            if verbose:
                cls.print_stats(final_test_stats, "Final test stats",
                            method, extra_kwargs=extra_kwargs,
                            is_test=True)
        
        if save_dir is not None:
            save_json(training_history_data, training_history_path, indent=4)

            save_json({"best_model_fname": best_model_fname},
                    os.path.join(save_dir, "best_model_fname.json"))
            if not save_incremental_best_models:
                # Save the best model weights if not already saved
                torch.save(best_state_dict, best_state_dict_path)
        
        return training_history_data
    
    @classmethod
    def _get_test_during_after_training(
            cls, test_dataset: Optional[Dataset], small_test_dataset: Optional[Dataset],
            test_during_training: Optional[bool]) -> tuple[bool, bool]:
        if test_dataset is not None:
            if small_test_dataset is not None: # Both test and small-test specified
                # Test during & after training
                if test_during_training == False:
                    raise ValueError("Small and big test datasets specified but test_during_training == False")
                test_during_training = True # it can be either None or True
                test_after_training = True
            else: # Only test but not small-test specified
                # Whether to test during & after training is ambiguous
                if test_during_training is None:
                    raise ValueError("test but not small-test dataset is specified but test_during_training is not specified")
                if test_during_training:
                    # Then the during-train & after-train dataset are the same, and
                    # we can test after training too (which is kind of redundant)
                    pass # will set small_test_dataset = test_dataset
                test_after_training = True # want to test after training both cases
        else: # Test dataset not specified
            if small_test_dataset is not None: # Small-test but not test specified
                # Test during but not after training
                if test_during_training == False:
                    raise ValueError("Small-test but not test specified, but test_during_training == False")
                test_during_training = True # it can be either None or True
                test_after_training = False
            else: # Neither are specified
                # Test neither during nor after training
                if test_during_training:
                    raise ValueError("No test datasets were specified but got test_during_training == True")
                test_during_training = False
                test_after_training = False
        return test_during_training, test_after_training
    
    @classmethod
    def _split_nn_stats(cls, stats: dict) -> tuple[dict, dict]:
        nn_stats = stats.copy()
        non_nn_stats = {}
        for stat_name in stats:
            if cls.is_non_nn_stat_name(stat_name):
                non_nn_stats[stat_name] = nn_stats.pop(stat_name)
        return nn_stats, non_nn_stats
    
    def __init_subclass__(cls, **kwargs):
        # Validate that subclasses define train_or_test_loop_class correctly
        super().__init_subclass__(**kwargs)
        error_message = f"Subclasses of {cls.__name__} must define a valid " \
            f"'train_or_test_loop_class' class variable"
        if not hasattr(cls, "train_or_test_loop_class"):
            raise TypeError(error_message)
        try:
            check_subclass(cls.train_or_test_loop_class, "train_or_test_loop_class",
                           TrainOrTestLoop)
        except ValueError as e:
            raise TypeError(f"{error_message}: {e}") from e
