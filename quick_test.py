#!/usr/bin/env python3
"""Quick test for integration"""

import argparse
from gp_acquisition_dataset_manager import (
    create_train_test_gp_acq_datasets_from_args as create_train_test_gp_datasets, 
    add_gp_acquisition_dataset_args,
)
from lr_acquisition_dataset_manager import (
    create_train_test_lr_acq_datasets_from_args as create_train_test_lr_datasets,
    add_logistic_regression_acquisition_dataset_args,
)
from acquisition_dataset_base import add_lamda_args

def test_gp():
    print("Testing GP dataset...")
    parser = argparse.ArgumentParser()
    add_gp_acquisition_dataset_args(parser)
    add_lamda_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args([
        '--dimension', '1',
        '--kernel', 'Matern52', 
        '--lengthscale', '0.1',
        '--train_samples_size', '5',
        '--test_samples_size', '3',
        '--train_acquisition_size', '10',
        '--train_n_candidates', '3',
        '--test_n_candidates', '5',
        '--min_history', '1',
        '--max_history', '3',
        '--lamda', '0.01'
    ])
    
    train_ds, _, _ = create_train_test_gp_datasets(
        args, check_cached=False, load_dataset=False)
    if train_ds is not None:
        print(f"GP dataset created: {len(train_ds)} samples")
    else:
        print("GP dataset creation returned None")

def test_gp_original():
    print("Testing original GP dataset...")
    from gp_acquisition_dataset import (
        create_train_test_gp_acq_datasets_from_args,
        add_gp_acquisition_dataset_args as add_gp_args_orig
    )
    parser = argparse.ArgumentParser()
    add_gp_args_orig(parser)
    add_lamda_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args([
        '--dimension', '1',
        '--kernel', 'Matern52', 
        '--lengthscale', '0.1',
        '--train_samples_size', '5',
        '--test_samples_size', '3',
        '--train_acquisition_size', '10',
        '--train_n_candidates', '3',
        '--test_n_candidates', '5',
        '--min_history', '1',
        '--max_history', '3',
        '--lamda', '0.01'
    ])
    
    train_ds, _, _ = create_train_test_gp_acq_datasets_from_args(
        args, check_cached=False, load_dataset=False)
    if train_ds is not None:
        print(f"Original GP dataset created: {len(train_ds)} samples")
    else:
        print("Original GP dataset creation returned None")

def test_lr():
    print("Testing LR dataset...")
    parser = argparse.ArgumentParser()
    add_logistic_regression_acquisition_dataset_args(parser)
    add_lamda_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args([
        '--train_samples_size', '3',
        '--test_samples_size', '2', 
        '--train_acquisition_size', '6',
        '--train_n_candidates', '2',
        '--test_n_candidates', '4',
        '--min_history', '1',
        '--max_history', '2',
        '--lr_n_samples_range', '20', '40',
        '--lr_n_features_range', '2', '4',
        '--lamda', '0.01'
    ])
    
    train_ds, _, _ = create_train_test_lr_datasets(
        args, check_cached=False, load_dataset=False)
    if train_ds is not None:
        print(f"LR dataset created: {len(train_ds)} samples")
    else:
        print("LR dataset creation returned None")

if __name__ == "__main__":
    test_gp()
    test_lr()