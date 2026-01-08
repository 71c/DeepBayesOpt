import copy
import math
import os
from typing import Optional
import torch

from utils_general.train_utils import EarlyStopper
from utils_general.io_utils import save_json
from utils_train.acquisition_function_net import AcquisitionFunctionNet
from datasets.acquisition_dataset import AcquisitionDataset
from utils_train.train_or_test_loop import train_or_test_loop


## Only used in print_stats
def _print_stat_summary(
        stats, things_to_print, best_stat:Optional[str]=None,
        inverse_ratio=False, sqrt_ratio=False, ratio_name='Ratio'):
    ## Prepare the rows to print
    best_val = None if best_stat is None else stats.get(best_stat)
    rows = []
    for stat_key, stat_print_name, print_ratio in things_to_print:
        if stat_key in stats:
            val = stats[stat_key]
            this_thing = [stat_print_name+':', f'{val:>8f}']
            if best_val is not None and print_ratio:
                ratio = best_val / val if inverse_ratio else val / best_val
                if sqrt_ratio:
                    ratio = math.sqrt(ratio)
                this_thing.extend([f'  {ratio_name}:', f'{ratio:>8f}'])
            rows.append(this_thing)
    
    ## Print the rows
    prefix = "  "
    # Calculate the maximum width for each column
    col_widths = [
        max(len(row[col_idx]) for row in rows if len(row) > col_idx)
        for col_idx in range(max(map(len, rows)))]

    # Print each row with appropriate spacing
    row_strings = [
        prefix + " ".join(
            col_val.rjust(col_widths[col_idx])
            for col_idx, col_val in enumerate(row)
        ) for row in rows]
    print("\n".join(row_strings))


def print_stats(stats:dict,
                dataset_name, method,
                gi_loss_normalization=None,
                print_dataset_ei=True):
    print(f'{dataset_name}:')
    if print_dataset_ei:
        print('Expected 1-step improvement:')
        things_to_print = [
            ('ei_softmax', 'NN (softmax)', True),
            ('maxei', 'NN (max)', True),
            ('true_gp_ei_maxei', 'True GP EI', False),
            ('map_gp_ei_maxei', 'MAP GP EI', True),
            ('true_gp_gi_maxei', 'True GP GI', True),
            ('ei_random_search', 'Random search', True),
            ('ei_ideal', 'Ideal', True),
            ('avg_normalized_entropy', 'Avg norm. entropy', False)]
        _print_stat_summary(
            stats, things_to_print, best_stat='true_gp_ei_maxei',
            inverse_ratio=False, sqrt_ratio=False)
    
    if method == 'mse_ei':
        print('Improvement MSE:')
        things_to_print = [
            ('mse', 'NN', True),
            ('true_gp_ei_mse', 'True GP EI', False),
            ('map_gp_ei_mse', 'MAP GP EI', True),
            ('mse_always_predict_0', 'Always predict 0', True)]
        _print_stat_summary(
            stats, things_to_print, best_stat='true_gp_ei_mse',
            inverse_ratio=True, sqrt_ratio=True, ratio_name='RMSE Ratio')
    elif method == 'gittins':
        print('Gittins index loss:')
        tmp = f'_normalized_{gi_loss_normalization}' if gi_loss_normalization is not None else ''
        things_to_print = [
            ('gittins_loss' + tmp, 'NN', True),
            ('true_gp_gi_gittins_loss' + tmp, 'True GP GI', False)
        ]
        _print_stat_summary(
            stats, things_to_print, best_stat='true_gp_gi',
            inverse_ratio=True, sqrt_ratio=False)


_BASIC_STATS = {"ei_random_search", "ei_ideal", "mse_always_predict_0"}
def _split_nn_stats(stats):
    nn_stats = stats.copy()
    non_nn_stats = {}
    for stat_name in stats:
        if stat_name in _BASIC_STATS or stat_name.startswith("true_gp") or stat_name.startswith("map_gp"):
            non_nn_stats[stat_name] = nn_stats.pop(stat_name)
    return nn_stats, non_nn_stats
_FIX_TRAIN_DATA_EACH_EPOCH = False


# Only used in train_acquisition_function_net [user-specified]
def _get_train_dataloaders(
    get_train_stats_after_training,
    train_dataset,
    batch_size,
    test_during_training,
    small_test_dataset
):
    # If get_train_stats_after_training=True, then we are running through the
    # train dataset twice each epoch. If furthermore the training data is not
    # fixed, then with these two runs through it, the data will be different.
    # But we'd like to have the stats of during training vs after training
    # directly comparable, so we will freeze the data with each epoch.
    need_fix_train_data = get_train_stats_after_training and not train_dataset.data_is_fixed
    fix_train_dataset_each_epoch = need_fix_train_data and _FIX_TRAIN_DATA_EACH_EPOCH

    if fix_train_dataset_each_epoch:
        train_dataloader = None
        def per_epoch_get_train_dataloader():
            return train_dataset.fix_samples(lazy=True) \
                .get_dataloader(
                    batch_size=batch_size, drop_last=False, cache_pads=False)
    else:
        train_dataloader = train_dataset.get_dataloader(
            batch_size=batch_size, drop_last=False)
        per_epoch_get_train_dataloader = None
    
    if need_fix_train_data and not _FIX_TRAIN_DATA_EACH_EPOCH:
        if test_during_training:
            num = len(small_test_dataset)
        else:
            num = None
        train_dataset_eval_dataloader = train_dataset \
            .fix_samples(n_realizations=num, lazy=False) \
            .get_dataloader(batch_size=batch_size, drop_last=False, cache_pads=True)
    else:
        train_dataset_eval_dataloader = train_dataloader
    
    return (
        train_dataloader, per_epoch_get_train_dataloader, train_dataset_eval_dataloader)


# Only used in train_acquisition_function_net [user-specified]
def _get_test_dataloader(test_ds, batch_size):
    return test_ds.get_dataloader(batch_size=batch_size, drop_last=False)


def train_acquisition_function_net(
        nn_model: AcquisitionFunctionNet,
        train_dataset: AcquisitionDataset,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        batch_size: int,
        nn_device=None,
        verbose:bool=True,
        n_train_printouts_per_epoch:Optional[int]=None,
        
        test_dataset: Optional[AcquisitionDataset]=None,
        small_test_dataset:Optional[AcquisitionDataset]=None,
        test_during_training:Optional[bool]=None,

        get_train_stats_while_training:bool=True,
        get_train_stats_after_training:bool=True,

        save_dir:Optional[str]=None,
        save_incremental_best_models:bool=True,
        
        early_stopping:bool=True,
        patience:int=10,
        min_delta:float=0.0,
        cumulative_delta:bool=False,

        # learning rate scheduler
        lr_scheduler:Optional[str]=None,
        lr_scheduler_patience:int=10,
        lr_scheduler_factor:float=0.1,
        lr_scheduler_min_lr:float=1e-6,
        lr_scheduler_cooldown:int=0,
        lr_scheduler_power:float=0.6,
        lr_scheduler_burnin:int=1,

        #### SPECIFIC
        method: str = "gittins",
        # Only used when method="mse_ei" or "policy_gradient"
        # (only does anything if method="policy_gradient")
        alpha_increment:Optional[float]=None,
        # Only used when method="gittins" and train=True
        gi_loss_normalization:Optional[str]=None,
        # get stats
        get_train_true_gp_stats:Optional[bool]=None,
        get_train_map_gp_stats:bool=False,
        get_test_true_gp_stats:Optional[bool]=None,
        get_test_map_gp_stats:Optional[bool]=None,
        # evaluation metric
        use_maxei=False
    ):
    ## SPECIFIC
    ## PURPOSE: give stat_name and minimize_stat to (generalised) train_acquisition_function_net
    if not isinstance(use_maxei, bool):
        raise ValueError("use_maxei should be a boolean")
    if use_maxei:
        stat_name = "maxei"
        minimize_stat = False
    elif method == "policy_gradient":
        stat_name = "ei_softmax"
        minimize_stat = False
    elif method == "mse_ei":
        stat_name = "mse"
        minimize_stat = True
    elif method == "gittins":
        stat_name = "gittins_loss" + \
            (f"_normalized_{gi_loss_normalization}" \
             if gi_loss_normalization is not None else "")
        minimize_stat = True
    
    ## SPECIFIC
    ## PURPOSE: give train_n_cand and test_n_cand to print_stats
    train_n_cand = next(iter(train_dataset)).x_cand.size(0)
    test_n_cand = next(iter(test_dataset)).x_cand.size(0)

    ## SPECIFIC
    ## PURPOSE: give get_train_true_gp_stats to train_or_test_loop
    # Due to this, need to explicitly set the default value here because train_or_test_loop
    # won't get it right because we'll fix the data even though it isn't fixed
    if get_train_true_gp_stats is None:
        get_train_true_gp_stats = train_dataset.has_models and train_dataset.data_is_fixed

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

    test_during_training, test_after_training = _get_test_during_after_training(
        test_dataset, small_test_dataset, test_during_training)

    if test_during_training:
        if small_test_dataset is None:
            small_test_dataset = test_dataset
        small_test_dataloader = _get_test_dataloader(small_test_dataset, batch_size)

    #### SPECIFIC
    #### TODO: figure out how to handle this
    if test_during_training or test_after_training:
        if get_test_map_gp_stats is None:
            get_test_map_gp_stats = False # default
    elif get_test_true_gp_stats or get_test_map_gp_stats:
        raise ValueError("Can't get GP stats of test dataset because there is none specified")
    
    #### SPECIFIC
    (train_dataloader, per_epoch_get_train_dataloader,
     train_dataset_eval_dataloader) = _get_train_dataloaders(
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
        
        train_stats_while_training = train_or_test_loop(
            this_epoch_train_dataloader, nn_model, train=True,
            nn_device=nn_device, method=method,
            verbose=verbose, desc=f"Epoch {t+1} train",
            n_train_printouts=n_train_printouts_per_epoch,
            optimizer=optimizer,
            alpha_increment=alpha_increment,
            gi_loss_normalization=gi_loss_normalization,
            get_true_gp_stats=get_train_true_gp_stats,
            get_map_gp_stats=get_train_map_gp_stats,
            get_basic_stats=True)

        (train_nn_stats_while_training,
         non_nn_train_stats) = _split_nn_stats(train_stats_while_training)
        train_stats['non_nn_stats'] = non_nn_train_stats
        
        if get_train_stats_while_training:
            train_stats['while_training'] = train_nn_stats_while_training
            if verbose:
                print_stats({**train_stats['while_training'], **non_nn_train_stats},
                            "Train stats while training", method, gi_loss_normalization,
                            print_dataset_ei=(train_n_cand > 1))

        if get_train_stats_after_training:
            train_stats['after_training'] = train_or_test_loop(
                train_dataset_eval_dataloader, nn_model, train=False,
                nn_device=nn_device, method=method,
                verbose=verbose, desc=f"Epoch {t+1} compute train stats",
                gi_loss_normalization=gi_loss_normalization,
                # Don't need to compute non-NN stats because already computed them
                # while training, and we ensured that the train dataset is fixed for this epoch.
                get_true_gp_stats=False,
                get_map_gp_stats=False,
                get_basic_stats=False)
            if verbose:
                print_stats({**train_stats['after_training'], **non_nn_train_stats},
                            "Train stats after training", method, gi_loss_normalization,
                            print_dataset_ei=(train_n_cand > 1))
        
        epoch_stats = {'train': train_stats}

        if test_during_training:
            test_stats = train_or_test_loop(
                small_test_dataloader, nn_model, train=False,
                nn_device=nn_device, method=method,
                verbose=verbose, desc=f"Epoch {t+1} compute test stats",
                gi_loss_normalization=gi_loss_normalization,
                get_true_gp_stats=get_test_true_gp_stats,
                get_map_gp_stats=get_test_map_gp_stats,
                get_basic_stats=True)
            epoch_stats['test'] = test_stats
            if verbose:
                print_stats(test_stats, "Test stats", method, gi_loss_normalization,
                            print_dataset_ei=(test_n_cand > 1))
        
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
            #### SPECIFIC
            # Consider if the test datasets are not fixed.
            # Then the test-after-training default is not good.
            if get_test_true_gp_stats is None and not test_dataset.data_is_fixed \
                and test_dataset.has_models:
                get_test_true_gp_stats_after_training = True
            else:
                get_test_true_gp_stats_after_training = get_test_true_gp_stats
            
            test_dataloader = _get_test_dataloader(test_dataset, batch_size)
            final_test_stats = train_or_test_loop(
                test_dataloader, nn_model, train=False,
                nn_device=nn_device, method=method,
                verbose=verbose, desc=f"Compute final test stats",
                gi_loss_normalization=gi_loss_normalization,
                get_true_gp_stats=get_test_true_gp_stats_after_training,
                get_map_gp_stats=get_test_map_gp_stats,
                get_basic_stats=True)
        training_history_data['final_test_stats'] = final_test_stats
        if verbose:
            print_stats(final_test_stats, "Final test stats",
                        method, gi_loss_normalization,
                        print_dataset_ei=(test_n_cand > 1))
    
    if save_dir is not None:
        save_json(training_history_data, training_history_path, indent=4)

        save_json({"best_model_fname": best_model_fname},
                  os.path.join(save_dir, "best_model_fname.json"))
        if not save_incremental_best_models:
            # Save the best model weights if not already saved
            torch.save(best_state_dict, best_state_dict_path)
    
    return training_history_data


## only used in train_acquisition_function_net
def _get_test_during_after_training(
        test_dataset, small_test_dataset, test_during_training):
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
