"""NOTE: This script has some arguments that are not actually needed for the transfer
BO baselines (specifically, those to do with acquisition dataset sampling), but for
ease of integration with the existing codebase, they are included for compatibility with
the way the existing code loads datasets."""
import argparse
import os
from typing import Optional, Sequence

from dataset_factory import (add_unified_acquisition_dataset_args,
                             create_train_test_acquisition_datasets_from_args,
                             validate_args_for_dataset_type)
from datasets.utils import get_cmd_options_sample_dataset
from nn_af.acquisition_function_net_save_utils import MODELS_SUBDIR, get_new_timestamp_model_save_dir, mark_new_model_as_trained, nn_acqf_is_trained
from transfer_bo_baselines.fsbo.fsbo_modules import FSBO
from utils.constants import MODELS_DIR
from utils.utils import dict_to_hash, save_json


TRANSFER_BO_BASELINE_NAMES = ['FSBO']


def get_dataset_hash_for_transfer_bo_baselines(
        function_samples_and_acquisition_dataset_args: dict,
        return_dict: bool=False):
    function_samples_dataset_info = get_cmd_options_sample_dataset(
        function_samples_and_acquisition_dataset_args)
    if function_samples_dataset_info.get('train_samples_size', None) is not None:
        try:
            n_candidates = function_samples_and_acquisition_dataset_args['n_candidates']
        except KeyError:
            # In our current setup, we always have
            # train_n_candidates == test_n_candidates == n_candidates
            n_candidates = function_samples_and_acquisition_dataset_args['train_n_candidates']
        function_samples_dataset_info['evals_per_function'] = \
            function_samples_and_acquisition_dataset_args['max_history'] + \
            function_samples_and_acquisition_dataset_args['samples_addition_amount'] + \
            n_candidates
    dataset_hash = dict_to_hash(function_samples_dataset_info)
    if return_dict:
        return dataset_hash, function_samples_dataset_info
    return dataset_hash


def get_relative_checkpoints_path_for_transfer_bo_baseline(
        transfer_bo_method: str, dataset_hash: str) -> str:
    return os.path.join("transfer_bo_baselines", transfer_bo_method, dataset_hash)


def get_checkpoints_path_for_transfer_bo_baseline(
        transfer_bo_method: str, dataset_hash: str) -> str:
    return os.path.join(MODELS_DIR, get_relative_checkpoints_path_for_transfer_bo_baseline(
        transfer_bo_method, dataset_hash))


def transfer_bo_baseline_is_trained(transfer_bo_method: str, dataset_hash: str) -> bool:
    relative_checkpoints_path = get_relative_checkpoints_path_for_transfer_bo_baseline(
        transfer_bo_method, dataset_hash)
    return nn_acqf_is_trained(relative_checkpoints_path)


_EPOCHS_REFERENCE = 6508
_N_REFERENCE = 320 + 1000
_TIME_REFERENCE = 14 + 34 / 60  # 14 hours 34 minutes, in hours
_FSBO_CONSTANT = _EPOCHS_REFERENCE * _N_REFERENCE / _TIME_REFERENCE
def _calculate_fsbo_epochs_for_time_budget(time_budget_hours: float, N: int) -> int:
    return int(_FSBO_CONSTANT * time_budget_hours / N)


def run_train_transfer_bo_baseline(cmd_args: Optional[Sequence[str]]=None):
    parser = argparse.ArgumentParser()
    dataset_group = parser.add_argument_group("Dataset options")
    groups_arg_names = add_unified_acquisition_dataset_args(
        parser, dataset_group, add_lamda_args_flag=False)
    
    parser.add_argument(
        '--transfer_bo_method',
        type=str,
        choices=TRANSFER_BO_BASELINE_NAMES,
        required=True,
        help='Transfer BO baseline method to use.'
    )

    fsbo_group = parser.add_argument_group("FSBO specific options")
    # 100K epochs is the default number of epochs in the original FSBO script, which
    # by a rough estimate from an early experiment would take around 9.32 days.
    # OTOH, Maraval et al only run FSBO for 10K epochs in their experiments, so we
    # plan to use only 10K epochs as well.
    fsbo_group.add_argument(
        '--fsbo_time_budget_hours',
        help='Approximate time budget for FSBO, in hours',
        type=float,
        default=14.0
    )

    args = parser.parse_args(args=cmd_args)
    validate_args_for_dataset_type(args, groups_arg_names)

    args.batch_size = 16 # need to set something arbitrary to avoid an error somewhere

    # Although we don't need to get the "acquisition datasets" and only need the
    # "function samples datasets" for transfer BO baselines, we still load the
    # acquisition datasets here due to the way that the codebase is already structured.
    (train_aq_dataset, test_aq_dataset,
     small_test_aq_dataset) = create_train_test_acquisition_datasets_from_args(
         args, fix_test_acquisition_dataset=False)
    
    train_data = train_aq_dataset.base_dataset
    valid_data = test_aq_dataset.base_dataset

    dataset_hash, dataset_info = get_dataset_hash_for_transfer_bo_baselines(
        vars(args), return_dict=True)
    
    # Verify that we calculated evals_per_function correctly.
    evals_per_function = dataset_info.get('evals_per_function', None)
    if evals_per_function is not None:
        true_train_evals_per_function = train_data[0].x_values.shape[0]
        if true_train_evals_per_function != evals_per_function:
            raise RuntimeError(
                f"Calculated evals_per_function {evals_per_function} does not match "
                f"true value {true_train_evals_per_function} in train data."
            )
        true_valid_evals_per_function = valid_data[0].x_values.shape[0]
        if true_valid_evals_per_function != evals_per_function:
            raise RuntimeError(
                f"Calculated evals_per_function {evals_per_function} does not match "
                f"true value {true_valid_evals_per_function} in validation data."
            )

    checkpoints_path = get_checkpoints_path_for_transfer_bo_baseline(
        args.transfer_bo_method, dataset_hash)
    
    # Save dataset info. Just as with our own method, it is not strictly necessary to
    # save it, but it may be useful for debugging or analysis later.
    os.makedirs(checkpoints_path, exist_ok=True)
    save_json(dataset_info, os.path.join(checkpoints_path, 'dataset_info.json'))

    model_path, model_name = get_new_timestamp_model_save_dir(checkpoints_path)

    if args.transfer_bo_method == 'FSBO':
        fsbo_model = FSBO(train_data=train_data, valid_data=valid_data,
                          checkpoint_path=model_path)
        fsbo_epochs = _calculate_fsbo_epochs_for_time_budget(
            args.fsbo_time_budget_hours, N=len(train_data) + len(valid_data))
        print(f"Training FSBO for {fsbo_epochs} epochs to fit in time budget of "
              f"{args.fsbo_time_budget_hours} hours")
        fsbo_model.meta_train(epochs=fsbo_epochs)
    
    models_path = os.path.join(checkpoints_path, MODELS_SUBDIR)
    mark_new_model_as_trained(models_path, model_name)
    print(f"Completed training, saved to {model_path}")


if __name__ == "__main__":
    run_train_transfer_bo_baseline()
