import argparse
from datasets.hpob_dataset import get_hpob_dataset
from datasets.acquisition_dataset_manager import AcquisitionDatasetManager


class HPOBAcquisitionDatasetManager(AcquisitionDatasetManager):
    def __init__(self, device: str = "cpu"):
        super().__init__("HPO-B", device)
    
    def create_function_samples_dataset(self, **kwargs):
        name = kwargs.pop('name')
        kwargs['dataset_type'] = {'train': 'train', 'test': 'validation'}[name]
        return get_hpob_dataset(**kwargs)
    
    def get_function_samples_config(self, args: argparse.Namespace, device=None):
        """Get HPO-B-specific function samples configuration."""
        return dict(
            search_space_id=args.hpob_search_space_id,
        )
    
    def get_outcome_transform(self, args: argparse.Namespace, device=None):
        """Get HPO-B-specific outcome transform (always None)."""
        return None  # No outcome transform for HPO-B
    
    def get_train_test_true_stats_flags(self):
        """HPO-B-specific stats flags - no GP stats."""
        return False, False


def add_hpob_args(parser: argparse.ArgumentParser):
    """Add HPO-B-specific arguments to the parser."""
    parser.add_argument(
        '--hpob_search_space_id', type=str, default='5970',
        help="HPO-B search space ID (default: '5970')")
