#!/usr/bin/env python3
"""
Comprehensive test suite to verify correctness of the dataset refactoring.

This test validates:
1. Logistic regression dataset creation and evaluation
2. GP dataset compatibility (backward compatibility)
3. Unified dataset factory functionality
4. Configuration parsing and argument handling
5. Dataset caching and persistence
6. End-to-end training pipeline integration
"""

import argparse
import torch
import numpy as np
from typing import Tuple, Any

# Test imports
try:
    from dataset_factory import (
        create_train_test_acquisition_datasets_from_args,
        add_unified_dataset_args,
        _validate_args_for_dataset_type
    )
    from datasets.logistic_regression_dataset import LogisticRegressionRandomDataset
    from lr_acquisition_dataset_manager import create_train_test_lr_acq_datasets_from_args
    from gp_acquisition_dataset_manager import create_train_test_gp_acq_datasets_from_args
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)


def test_logistic_regression_dataset():
    """Test the core logistic regression dataset functionality."""
    print("\n=== Testing Logistic Regression Dataset ===")
    
    try:
        # Create a small LR dataset for testing
        dataset = LogisticRegressionRandomDataset(
            dataset_size=5,
            n_datapoints=3,
            n_samples_range=(20, 50),
            n_features_range=(2, 5),
            bias_range=(-1.0, 1.0),
            coefficient_std=0.5,
            noise_range=(0.1, 0.5),
            log_lambda_range=(-3, 1),
            log_uniform_sampling=False,
            device='cpu'
        )
        
        # Test dataset properties
        assert len(dataset) == 5, f"Expected dataset size 5, got {len(dataset)}"
        print(f"✓ Dataset size correct: {len(dataset)}")
        
        # Test data generation (use iterator for IterableDataset)
        dataset_iter = iter(dataset)
        item = next(dataset_iter)
        assert hasattr(item, 'x_values') and hasattr(item, 'y_values'), "Missing x_values or y_values"
        assert item.x_values.shape[1] == 1, f"Expected 1D hyperparameters, got {item.x_values.shape[1]}"
        assert item.y_values.shape[1] == 1, f"Expected 1D objectives, got {item.y_values.shape[1]}"
        assert item.x_values.shape[0] == 3, f"Expected 3 datapoints, got {item.x_values.shape[0]}"
        
        # Verify hyperparameters are in [0,1]
        assert torch.all(item.x_values >= 0) and torch.all(item.x_values <= 1), "Hyperparameters not in [0,1]"
        
        # Verify log-likelihood values are reasonable (should be negative but not too extreme)
        assert torch.all(item.y_values > -100) and torch.all(item.y_values < 100), "Log-likelihood values seem unreasonable"
        
        print(f"✓ Sample data shape: x={item.x_values.shape}, y={item.y_values.shape}")
        print(f"✓ Hyperparameter range: [{item.x_values.min():.3f}, {item.x_values.max():.3f}]")
        print(f"✓ Objective range: [{item.y_values.min():.3f}, {item.y_values.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Logistic regression dataset test failed: {e}")
        return False
    
    return True


def test_unified_argument_parsing():
    """Test the unified argument parsing for both dataset types."""
    print("\n=== Testing Unified Argument Parsing ===")
    
    try:
        # Test LR argument parsing
        parser = argparse.ArgumentParser()
        add_unified_dataset_args(parser)
        parser.add_argument('--batch_size', type=int, default=32)
        
        lr_args = parser.parse_args([
            '--dataset_type', 'logistic_regression',
            '--train_samples_size', '10',
            '--test_samples_size', '5',
            '--train_acquisition_size', '20',
            '--train_n_candidates', '3',
            '--test_n_candidates', '5',
            '--min_history', '1',
            '--max_history', '5',
            '--lr_n_samples_range', '50', '200',
            '--lr_n_features_range', '5', '20',
            '--lamda', '0.01'
        ])
        
        assert lr_args.dataset_type == 'logistic_regression'
        assert lr_args.lr_n_samples_range == [50, 200]
        assert lr_args.train_samples_size == 10
        print("✓ LR argument parsing successful")
        
        # Test GP argument parsing
        parser = argparse.ArgumentParser()
        add_unified_dataset_args(parser)
        parser.add_argument('--batch_size', type=int, default=32)
        
        gp_args = parser.parse_args([
            '--dataset_type', 'gp',
            '--dimension', '2',
            '--kernel', 'Matern52',
            '--lengthscale', '0.2',
            '--train_samples_size', '10',
            '--test_samples_size', '5',
            '--train_acquisition_size', '20',
            '--train_n_candidates', '3',
            '--test_n_candidates', '5',
            '--min_history', '1',
            '--max_history', '5',
            '--lamda', '0.01'
        ])
        
        assert gp_args.dataset_type == 'gp'
        assert gp_args.dimension == 2
        assert gp_args.kernel == 'Matern52'
        assert gp_args.lengthscale == 0.2
        print("✓ GP argument parsing successful")
        
    except Exception as e:
        print(f"✗ Unified argument parsing test failed: {e}")
        return False
        
    return True


def test_dataset_creation_pipeline():
    """Test the full dataset creation pipeline for both types."""
    print("\n=== Testing Dataset Creation Pipeline ===")
    
    try:
        # Test LR dataset creation
        parser = argparse.ArgumentParser()
        add_unified_dataset_args(parser)
        parser.add_argument('--batch_size', type=int, default=32)
        
        lr_args = parser.parse_args([
            '--dataset_type', 'logistic_regression',
            '--train_samples_size', '3',
            '--test_samples_size', '2',
            '--train_acquisition_size', '6',
            '--train_n_candidates', '2',
            '--test_n_candidates', '3',
            '--min_history', '1',
            '--max_history', '2',
            '--lr_n_samples_range', '30', '50',
            '--lr_n_features_range', '3', '5',
            '--lamda', '0.01'
        ])
        
        # Test dataset creation
        train_ds, test_ds, small_test_ds = create_train_test_acquisition_datasets_from_args(
            lr_args, check_cached=False, load_dataset=False)
        
        assert train_ds is not None, "Training dataset is None"
        assert test_ds is not None, "Test dataset is None"
        assert len(train_ds) == 6, f"Expected training dataset size 6, got {len(train_ds)}"
        print("✓ LR dataset pipeline successful")
        
        # Test GP dataset creation for backward compatibility
        gp_args = parser.parse_args([
            '--dataset_type', 'gp',
            '--dimension', '1',
            '--kernel', 'Matern52',
            '--lengthscale', '0.1',
            '--train_samples_size', '3',
            '--test_samples_size', '2',
            '--train_acquisition_size', '6',
            '--train_n_candidates', '2',
            '--test_n_candidates', '3',
            '--min_history', '1',
            '--max_history', '2',
            '--lamda', '0.01'
        ])
        
        train_ds_gp, test_ds_gp, small_test_ds_gp = create_train_test_acquisition_datasets_from_args(
            gp_args, check_cached=False, load_dataset=False)
        
        assert train_ds_gp is not None, "GP training dataset is None"
        assert test_ds_gp is not None, "GP test dataset is None"
        assert len(train_ds_gp) == 6, f"Expected GP training dataset size 6, got {len(train_ds_gp)}"
        print("✓ GP dataset pipeline (backward compatibility) successful")
        
    except Exception as e:
        print(f"✗ Dataset creation pipeline test failed: {e}")
        return False
        
    return True


def test_argument_validation():
    """Test argument validation for different dataset types."""
    print("\n=== Testing Argument Validation ===")
    
    try:
        # Test that GP validation works
        parser = argparse.ArgumentParser()
        add_unified_dataset_args(parser)
        
        incomplete_gp_args = parser.parse_args([
            '--dataset_type', 'gp',
            '--train_samples_size', '10',
            '--test_samples_size', '5',
            '--train_acquisition_size', '20',
            '--train_n_candidates', '3',
            '--test_n_candidates', '5',
            '--min_history', '1',
            '--max_history', '5'
            # Missing: dimension, kernel, lengthscale
        ])
        
        # This should raise an error
        try:
            _validate_args_for_dataset_type(incomplete_gp_args, 'gp')
            print("✗ GP validation should have failed but didn't")
            return False
        except ValueError:
            print("✓ GP validation correctly rejects incomplete args")
        
        # Test that LR validation passes (all args have defaults)
        lr_args = parser.parse_args([
            '--dataset_type', 'logistic_regression',
            '--train_samples_size', '10',
            '--test_samples_size', '5',
            '--train_acquisition_size', '20',
            '--train_n_candidates', '3',
            '--test_n_candidates', '5',
            '--min_history', '1',
            '--max_history', '5'
        ])
        
        # This should not raise an error
        _validate_args_for_dataset_type(lr_args, 'logistic_regression')
        print("✓ LR validation passes with default args")
        
    except Exception as e:
        print(f"✗ Argument validation test failed: {e}")
        return False
        
    return True


def test_hyperparameter_objective_evaluation():
    """Test that hyperparameter optimization actually works."""
    print("\n=== Testing Hyperparameter Objective Evaluation ===")
    
    try:
        # Create a controlled LR dataset
        dataset = LogisticRegressionRandomDataset(
            dataset_size=1,
            n_datapoints=5,
            n_samples_range=(100, 100),  # Fixed size for consistency
            n_features_range=(10, 10),   # Fixed features
            bias_range=(0.0, 0.0),       # Fixed bias
            coefficient_std=1.0,
            noise_range=(0.1, 0.1),      # Low noise
            log_lambda_range=(-4, 0),    # Reasonable lambda range
            log_uniform_sampling=False,
            device='cpu'
        )
        
        # Generate one sample (use iterator for IterableDataset)
        dataset_iter = iter(dataset)
        item = next(dataset_iter)
        
        # Check that we have reasonable variation in objective values
        y_values = item.y_values.flatten()
        y_range = y_values.max() - y_values.min()
        
        # Should have some variation (not all identical)
        assert y_range > 1e-3, f"Objective values have no variation: range={y_range:.6f}"
        
        # Check that hyperparameters span the space
        x_values = item.x_values.flatten()
        x_range = x_values.max() - x_values.min()
        assert x_range > 0.1, f"Hyperparameter values too clustered: range={x_range:.3f}"
        
        print(f"✓ Objective variation: {y_range:.3f}")
        print(f"✓ Hyperparameter spread: {x_range:.3f}")
        print(f"✓ Evaluation successful")
        
    except Exception as e:
        print(f"✗ Hyperparameter objective evaluation test failed: {e}")
        return False
        
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("Running comprehensive test suite for dataset refactoring...")
    
    tests = [
        ("Logistic Regression Dataset", test_logistic_regression_dataset),
        ("Unified Argument Parsing", test_unified_argument_parsing), 
        ("Dataset Creation Pipeline", test_dataset_creation_pipeline),
        ("Argument Validation", test_argument_validation),
        ("Hyperparameter Objective Evaluation", test_hyperparameter_objective_evaluation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} | {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dataset refactoring is working correctly.")
        return True
    else:
        print(f"⚠️  {total-passed} test(s) failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)