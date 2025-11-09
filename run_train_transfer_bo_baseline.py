"""NOTE: This script has some arguments that are not actually needed for the transfer
BO baselines, but are included for compatibility with the existing code structure."""
import argparse
from typing import Optional, Sequence

from dataset_factory import (add_unified_acquisition_dataset_args,
                             create_train_test_acquisition_datasets_from_args,
                             validate_args_for_dataset_type)
from datasets.hpob_dataset import get_hpob_dataset_dimension


TRANSFER_BO_BASELINE_NAMES = ['FSBO']


def run_train(cmd_args: Optional[Sequence[str]]=None):
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

    args = parser.parse_args(args=cmd_args)
    validate_args_for_dataset_type(args, groups_arg_names)

    dataset_type = getattr(args, 'dataset_type', 'gp')
    # manager = get_dataset_manager(dataset_type, device="cpu")
    # gp_af_dataset_configs = manager.get_dataset_configs(args, device=GP_GEN_DEVICE)
    if dataset_type in {'gp', 'cancer_dosage'}:
        dimension = None # already in args.dimension
    elif dataset_type == 'logistic_regression':
        dimension = 1
    elif dataset_type == 'hpob':
        # Make sure to set the dimension for non-GP datasets so that it is available
        # for model creation
        dimension = get_hpob_dataset_dimension(args.hpob_search_space_id)
    
    args.batch_size = 16 # need to set something arbitrary to avoid an error somewhere

    # Although we don't need to get the "acquisition datasets" and only need the
    # "function samples datasets" for transfer BO baselines, we still load the
    # acquisition datasets here due to the way that the codebase is already structured.
    (train_aq_dataset, test_aq_dataset,
     small_test_aq_dataset) = create_train_test_acquisition_datasets_from_args(
         args, fix_test_acquisition_dataset=False)
    
    print("Train function samples dataset:")
    print(train_aq_dataset.base_dataset)

    print("\nValidation function samples dataset:")
    print(test_aq_dataset.base_dataset)


if __name__ == "__main__":
    run_train()
