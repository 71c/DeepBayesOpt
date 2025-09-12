# Technical Project Proposal: Logistic Regression Hyperparameter Tuning Dataset

## Overview

This proposal outlines the integration of a new hyperparameter tuning dataset for logistic regression classification into the existing DeepBayesOpt codebase. The dataset will serve as a realistic hyperparameter optimization benchmark alongside the current Gaussian Process-based synthetic datasets.

## Dataset Specification

### Problem Setup
- **Target Hyperparameter**: Regularization parameter λ (lambda) of logistic regression classifier
- **Objective**: Log-likelihood on validation set (to be maximized)
- **Search Space**: Normalized to [0, 1], then affinely mapped to log-space (e.g., x ∈ [0,1] → [-6, 2] → λ = 10^x)

### Synthetic Data Generation
Each source dataset is generated with:
- **Covariates**: `X ~ N(0, I_d)` where d is the feature dimension
- **Labels**: `Y_i ~ Bernoulli(σ(b + c^T X_i + ε_i))` where:
  - `σ(u) = 1/(1+e^(-u))` (sigmoid function)
  - `b`: bias term (scalar)
  - `c`: coefficient vector (d-dimensional)
  - `ε_i ~ N(0, σ_y^2)`: noise term

### Dataset Parameters (Randomized)
- **Sample size (n)**: Log-uniformly distributed, e.g., [50, 2000]
- **Feature dimension (d)**: Log-uniformly distributed, e.g., [5, 100]
- **Bias term (b)**: Uniformly distributed, e.g., [-2, 2]
- **Coefficients (c)**: Each component ~ N(0, 1)
- **Noise level (σ_y)**: Log-uniformly distributed, e.g., [0.01, 1.0]

## Architecture Integration

### New Components

#### 1. Synthetic Dataset Generator (`datasets/logistic_regression_dataset.py`)
```python
class LogisticRegressionDatasetItem:
    """Single logistic regression dataset instance"""
    - X: torch.Tensor  # (n_samples, n_features)
    - y: torch.Tensor  # (n_samples,) binary labels
    - params: dict     # {b, c, sigma_y, n_samples, n_features}

class LogisticRegressionRandomDataset(FunctionSamplesDataset):
    """Generates random logistic regression datasets"""
    - Inherits from existing FunctionSamplesDataset infrastructure
    - Randomizes n_samples, n_features, b, c, sigma_y per sample
```

#### 2. Hyperparameter Optimization Function (`datasets/logistic_regression_objective.py`)
```python
def logistic_regression_objective(x: float, dataset_item: LogisticRegressionDatasetItem) -> float:
    """Evaluates logistic regression with normalized hyperparameter x ∈ [0,1]"""
    - Maps x ∈ [0,1] → log-space → λ = 10^(a*x + b) where a,b define the range
    - Fits LogisticRegression(C=1/lambda_val) using sklearn
    - Returns validation log-likelihood (to be maximized)
    - Uses cross-validation for robust evaluation
```

#### 3. Integration with Existing GP Dataset Infrastructure
Following the existing `gp_acquisition_dataset.py` pattern:
```python
# Extend existing dataset creation functions to support multiple dataset types
def create_hyperparameter_acquisition_dataset(..., dataset_type="gp"):
    """Extended version of create_gp_acquisition_dataset supporting multiple types"""
    if dataset_type == "gp":
        # Existing GP dataset creation logic
    elif dataset_type == "logistic_regression":
        # New logistic regression dataset creation
    # Easily extensible for future dataset types

# Reuse existing FunctionSamplesAcquisitionDataset and related infrastructure
# No need for separate LogisticRegressionAcquisitionDataset class
```

### Integration Points

#### Configuration System
- Extend `config/train_acqf.yml` with logistic regression dataset options
- Add new experiment configs for hyperparameter tuning scenarios
- Maintain compatibility with existing GP-based configurations

#### Dataset Factory
- Extend `gp_acquisition_dataset.py` to `hyperparameter_acquisition_dataset.py` supporting multiple dataset types
- Add dataset type parameter (`dataset_type: ["gp", "logistic_regression", ...]`)
- Preserve existing GP dataset functionality and maximize code reuse
- Design for easy extensibility to future dataset types (e.g., neural network hyperparameters, SVM parameters)

#### Training Pipeline
- Extend `run_train.py` to handle logistic regression datasets
- No changes needed to neural network architectures
- Reuse existing training methods (Gittins, MSE, policy gradient)

## Implementation Strategy

### Phase 1: Core Dataset Implementation
1. Implement `LogisticRegressionRandomDataset` following existing patterns
2. Create hyperparameter evaluation function with proper cross-validation
3. Ensure dataset caching and serialization work correctly

### Phase 2: Acquisition Dataset Integration  
1. Extend existing acquisition dataset infrastructure (reuse FunctionSamplesAcquisitionDataset)
2. Extend configuration system for new dataset type
3. Add command-line arguments and argument parsing

### Phase 3: Testing and Validation
1. Create small-scale test experiments
2. Validate against known hyperparameter optimization results
3. Ensure seamless integration with existing training pipeline

### Phase 4: Documentation and Examples
1. Add usage examples to existing experiment configurations
2. Document new parameters in CLAUDE.md
3. Create sample experiment configs for hyperparameter tuning

## Benefits

1. **Realistic Benchmarking**: Provides real hyperparameter optimization problems vs synthetic GP functions
2. **Modular Design**: Leverages existing infrastructure without breaking changes
3. **Configurable Complexity**: Adjustable dataset difficulty via parameter ranges
4. **Research Applicability**: Enables comparison of acquisition functions on practical ML problems

## Compatibility

- **Backward Compatible**: All existing GP-based experiments continue unchanged
- **Modular Integration**: New functionality isolated in separate modules
- **Configuration Driven**: Dataset type selection through existing YAML config system
- **Caching Compatible**: Reuses existing content-based dataset caching

## Next Steps

Upon approval, implementation would proceed through the four phases outlined above, with each phase including appropriate testing and validation to ensure robust integration with the existing codebase.