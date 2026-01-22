from typing import Any
import argparse
from functools import cache

from utils_general.utils import get_arg_names


@cache
def _get_dataset_arg_names():
    """Get argument names from dataset factory parser structure.

    The result is cached since parser structure doesn't change during runtime.
    """
    # Import here to avoid circular imports
    from dataset_factory import add_common_sample_dataset_args, add_unified_function_dataset_args
    parser = argparse.ArgumentParser()
    add_common_sample_dataset_args(parser)
    common_arg_names = set(get_arg_names(parser))
    common_arg_names.add('dataset_type')

    groups_arg_names = add_unified_function_dataset_args(
        parser=parser, thing_used_for="function samples", name_prefix="")

    return {
        'common': list(common_arg_names),
        'dataset_specific': groups_arg_names
    }


def get_cmd_options_sample_dataset(options: dict[str, Any]):
    """Extract dataset command options from a config dict.

    Args:
        options: Dictionary of configuration options

    Returns:
        Dictionary of command options for dataset creation
    """
    # Extract dataset_type to determine which parameters to include
    dataset_type = options.get('dataset_type', 'gp')

    arg_structure = _get_dataset_arg_names()

    # Start with common dataset parameters
    cmd_opts_sample_dataset = {
        k: options.get(k)
        for k in arg_structure['common']
    }

    # Add dimension for dataset types that use it (gp and cancer_dosage)
    if dataset_type in {'gp', 'cancer_dosage'}:
        cmd_opts_sample_dataset['dimension'] = options.get('dimension')

    # Add dataset-specific parameters based on dataset_type
    if dataset_type in arg_structure['dataset_specific']:
        dataset_args = arg_structure['dataset_specific'][dataset_type]
        cmd_opts_sample_dataset.update({
            k: options.get(k)
            for k in dataset_args
        })
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return cmd_opts_sample_dataset
