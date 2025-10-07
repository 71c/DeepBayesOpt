import argparse
from datasets.cancer_dosage_dataset import CancerDosageDataset
from datasets.acquisition_dataset_manager import AcquisitionDatasetManager


class CancerDosageAcquisitionDatasetManager(AcquisitionDatasetManager):
    def __init__(self, device: str = "cpu"):
        super().__init__("cancer_dosage", device)
    
    def create_function_samples_dataset(self, **kwargs):
        name = kwargs.pop('name')
        kwargs['seed'] = {'train': 0, 'test': 1}[name]
        return CancerDosageDataset(**kwargs)
    
    def get_function_samples_config(self, args: argparse.Namespace, device=None):
        return dict(
            dim_x=args.dimension,
            dim_features=args.dim_features,
            nnz_per_row=args.nnz_per_row,
            scale_intercept=args.scale_intercept,
            scale_coef=args.scale_coef,
            noise_std=args.noise_std,
            is_simplex=args.is_simplex,
            #### Dataset size
            train_samples_size=args.train_samples_size,
            test_samples_size=args.test_samples_size,
        )
    
    def get_outcome_transform(self, args: argparse.Namespace, device=None):
        return None
    
    def get_train_test_true_stats_flags(self):
        return False, False


def add_cancer_dosage_args(
        parser: argparse.ArgumentParser, thing_used_for: str,
        name_prefix=""):
    """Add cancer dosage dataset-specific arguments to the parser."""
    if name_prefix:
        name_prefix = f"{name_prefix}_"
    parser.add_argument(
        f'--{name_prefix}dim_features',
        type=int,
        help=f'Number of random features for generating parameters for the {thing_used_for}'
    )
    parser.add_argument(
        f'--{name_prefix}nnz_per_row',
        type=int,
        help=f'Number of non-zero entries per row in coefficient matrix for the {thing_used_for}'
    )
    parser.add_argument(
        f'--{name_prefix}scale_intercept',
        type=float,
        help=f'Scaling factor for intercept coefficients for the {thing_used_for}'
    )
    parser.add_argument(
        f'--{name_prefix}scale_coef',
        type=float,
        help=f'Scaling factor for linear coefficients for the {thing_used_for}'
    )
    parser.add_argument(
        f'--{name_prefix}noise_std',
        type=float,
        help=f'Standard deviation of Gaussian observation noise for the {thing_used_for}'
    )
    parser.add_argument(
        f'--{name_prefix}is_simplex',
        action='store_true',
        help=f' whether to use simplex x for the {thing_used_for}. If True, assumes x sums to at most 1. If False, assumes x is in [0, 1]^d'
    )
