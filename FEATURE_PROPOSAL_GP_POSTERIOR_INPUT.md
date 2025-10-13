# Technical Feature Proposal: GP Posterior Input to Neural Network Acquisition Functions

## Overview

This proposal outlines the implementation of GP posterior parameters (mean μ and log-standard-deviation ln(σ)) as input features to neural network acquisition functions. This Phase 1 approach uses simple concatenation with existing acqf_params, with a dictionary-based routing system deferred to Phase 2.

## Key Design Insight

Start with the simplest approach: concatenate GP posterior (μ, ln(σ)) with existing acqf_params and treat them identically in terms of routing to local NN, final MLP, or both. This avoids the complexity of a dictionary-based system while still providing the core functionality.

## Motivation

The current neural network architecture handles acquisition function parameters (like lambda for Gittins index) as concatenated tensors. Adding GP posterior (μ, ln(σ)) as input features could help the neural network learn better acquisition functions by providing direct access to the GP's uncertainty estimates.

**Phase 1 Goals:**
1. **GP posterior input**: Add μ and ln(σ) as input features via simple concatenation
2. **Minimal complexity**: Use existing acqf_params_input routing (local_only, final_only, local_and_final)  
3. **Backward compatibility**: Existing lambda/cost parameters continue to work unchanged
4. **Foundation for future**: Set up infrastructure for more sophisticated routing in Phase 2

## Architecture Changes

### 1. Simple GP Posterior Concatenation

**Current API (unchanged):**
- `acqf_params_input`: `"local_only"` | `"final_only"` | `"local_and_final"` | `null`

**New Command-Line Argument:**
```bash
--include_gp_posterior     # Enable GP posterior (μ, ln(σ)) concatenation to acqf_params
```

**Simple Concatenation Approach:**
- When `--include_gp_posterior` is enabled, GP posterior (μ, ln(σ)) is concatenated to existing acqf_params
- The combined tensor uses the same routing as specified by `acqf_params_input`
- No dictionary structure - everything treated as a single concatenated tensor

### 2. Implementation in `AcquisitionFunctionBodyPointnetV1and2`

**Changes to `__init__`:**
```python
def __init__(self,
             # ... existing parameters ...
             include_gp_posterior=False,                         # NEW: enable GP posterior concatenation
             ):
    self.include_gp_posterior = include_gp_posterior
    
    # Calculate parameter dimensions (existing logic + GP posterior)
    effective_n_acqf_params = n_acqf_params
    if include_gp_posterior:
        effective_n_acqf_params += 2  # Add μ, ln(σ) dimensions
    
    # Use existing dimension calculation logic with effective_n_acqf_params
    # All parameter handling (including GP posterior) uses the same routing as acqf_params_input
```

**Changes to `forward`:**
```python
def forward(self, x_hist, y_hist, x_cand, acqf_params=None, gp_posterior=None, hist_mask=None, cand_mask=None, **kwargs):
    # Concatenate GP posterior to acqf_params if both are provided
    if self.include_gp_posterior and gp_posterior is not None:
        if acqf_params is not None:
            acqf_params = torch.cat([acqf_params, gp_posterior], dim=-1)
        else:
            acqf_params = gp_posterior
    
    # Continue with existing forward logic using the combined acqf_params
    # No other changes needed - existing routing logic handles everything
```

## Dataset and Training Integration

### 1. Simple GP Posterior Addition to Training

**Key Insight:** For training datasets, we fit GPs to the **original function samples** from `FunctionSamplesDataset`, not to the sampled acquisition history. This gives the NN access to the "true" posterior uncertainty.

**Location:** `nn_af/train_acquisition_function_net.py`

```python
def train_acquisition_function_net(nn_model, train_dataset, ...):
    # Check if NN expects GP posterior input
    if _nn_uses_gp_posterior(nn_model):
        # Add GP posterior to datasets via simple concatenation
        train_dataset = _add_gp_posterior_to_dataset(train_dataset, verbose=verbose)
        if test_dataset is not None:
            test_dataset = _add_gp_posterior_to_dataset(test_dataset, verbose=verbose)
        # ... handle small_test_dataset similarly
    
    # Continue with existing training loop
```

### 2. Helper Functions

```python
def _nn_uses_gp_posterior(nn_model) -> bool:
    """Check if NN model expects GP posterior input"""
    # Navigate to the body to check for GP posterior flag
    if hasattr(nn_model, 'base_model'):
        model = nn_model.base_model
    else:
        model = nn_model
    
    if hasattr(model, 'af_body') and hasattr(model.af_body, 'include_gp_posterior'):
        return model.af_body.include_gp_posterior
    
    return False

def _add_gp_posterior_to_dataset(dataset, verbose=False):
    """Add GP posterior to dataset items via concatenation with acqf_params.
    
    This handles both fixed datasets (in-place modification) and sampled datasets (wrapper).
    """
    if verbose:
        if dataset.data_is_fixed:
            print(f"Adding GP posterior to fixed dataset ({len(dataset)} items)...")
        else:
            print(f"Creating GP posterior wrapper for sampled dataset...")
    
    if dataset.data_is_fixed:
        # For fixed datasets: modify items in-place 
        # Create copy if dataset has models to preserve original
        if dataset.has_models:
            dataset = dataset.copy_with_new_size(None)
        
        for item in (tqdm(dataset) if verbose else dataset):
            # Fit GP to original function samples
            gp_model = _fit_gp_to_function_samples(item)
            with torch.no_grad():
                posterior = gp_model.posterior(item.x_cand)
                mu = posterior.mean.squeeze(-1)  # (n_cand,)
                log_sigma = posterior.variance.sqrt().log().squeeze(-1)  # (n_cand,)
                gp_posterior = torch.stack([mu, log_sigma], dim=-1)  # (n_cand, 2)
            
            # Concatenate with existing acqf_params
            if hasattr(item, 'acqf_params') and item.acqf_params is not None:
                item.acqf_params = torch.cat([item.acqf_params, gp_posterior], dim=-1)
            else:
                item.acqf_params = gp_posterior
        
        return dataset
    else:
        # For sampled datasets: use wrapper class that adds GP posterior on-the-fly
        return GPPosteriorDatasetWrapper(dataset, verbose)

class GPPosteriorDatasetWrapper:
    """Wrapper for sampled datasets that adds GP posterior via concatenation on-the-fly.
    
    This class wraps acquisition datasets that generate items dynamically (data_is_fixed=False)
    and adds GP posterior to acqf_params during iteration via simple concatenation.
    """
    def __init__(self, base_dataset, verbose=False):
        self.base_dataset = base_dataset
        self.verbose = verbose
        
        # Forward all attributes to the base dataset
        self.__dict__.update({
            attr: getattr(base_dataset, attr) 
            for attr in dir(base_dataset) 
            if not attr.startswith('_') and attr not in ['__iter__']
        })
    
    def __getattr__(self, name):
        """Forward any missing attributes to the base dataset"""
        return getattr(self.base_dataset, name)
    
    def __iter__(self):
        """Iterate over base dataset and add GP posterior via concatenation"""
        for item in self.base_dataset:
            # Fit GP to original function samples
            gp_model = _fit_gp_to_function_samples(item)
            with torch.no_grad():
                posterior = gp_model.posterior(item.x_cand)
                mu = posterior.mean.squeeze(-1)  # (n_cand,)
                log_sigma = posterior.variance.sqrt().log().squeeze(-1)  # (n_cand,)
                gp_posterior = torch.stack([mu, log_sigma], dim=-1)  # (n_cand, 2)
            
            # Concatenate with existing acqf_params
            if hasattr(item, 'acqf_params') and item.acqf_params is not None:
                item.acqf_params = torch.cat([item.acqf_params, gp_posterior], dim=-1)
            else:
                item.acqf_params = gp_posterior
            
            yield item
    
    def copy_with_new_size(self, size=None):
        """Create a copy with new size, preserving the wrapper"""
        base_copy = self.base_dataset.copy_with_new_size(size)
        return GPPosteriorDatasetWrapper(base_copy, self.verbose)
    
    def random_split(self, lengths):
        """Split the dataset while preserving the wrapper"""
        base_splits = self.base_dataset.random_split(lengths)
        return [GPPosteriorDatasetWrapper(split, self.verbose) for split in base_splits]

def _fit_gp_to_function_samples(acquisition_item):
    """Fit GP to original function samples, preserving original GP for non-GP datasets"""
    # For GP datasets: use the original underlying GP model if available
    if hasattr(acquisition_item, 'model') and acquisition_item.model is not None:
        # Return the original GP model - this preserves the "true" GP statistics
        return acquisition_item.model
    
    # For non-GP datasets (logistic regression, HPO-B): fit MAP GP to function samples
    # Get the original function samples from the acquisition dataset item
    if hasattr(acquisition_item, 'function_sample'):
        function_x = acquisition_item.function_sample.x_values
        function_y = acquisition_item.function_sample.y_values
    else:
        # Fallback: construct from x_hist + x_cand and their function values
        # This requires access to the original function evaluation
        raise NotImplementedError("Need access to original function samples for non-GP datasets")
    
    # Fit MAP GP with RBF kernel to the function samples
    gp_model = _create_map_gp(function_x, function_y)
    _fit_gp_map(gp_model, function_x, function_y)
    return gp_model

def _create_map_gp(train_x, train_y):
    """Create GP model for MAP fitting"""
    from utils.utils import get_gp, get_kernel
    # Use RBF kernel for MAP fitting
    kernel = get_kernel(
        dimension=train_x.shape[-1],
        kernel="RBF",
        add_priors=False,  # MAP fitting without priors
        lengthscale=None,  # Let it be learned
        device=train_x.device
    )
    return get_gp(
        dimension=train_x.shape[-1],
        observation_noise=False,
        covar_module=kernel,
        device=train_x.device,
        outcome_transform=None
    )

def _fit_gp_map(gp_model, train_x, train_y):
    """Fit GP using MAP estimation"""
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    
    gp_model.set_train_data(train_x, train_y, strict=False)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)
```

## Bayesian Optimization Integration

### 1. Modify `NNAcquisitionOptimizer` for Simple GP Posterior

**Key Insight:** For BO loops, we fit MAP GP to the **evaluation history** (`self.x`, `self.y`) to get the posterior at candidate points. This is different from training where we use the entire function samples.

```python
class NNAcquisitionOptimizer:
    def get_model(self):
        y = self.y if self.maximize else -self.y
        nn_device = next(self.model.parameters()).device
        
        # Check if we need GP posterior input
        if _nn_uses_gp_posterior(self.model):
            # Fit GP to evaluation history for BO posterior estimates
            gp_model = self._fit_gp_to_evaluation_history()
            return AcquisitionFunctionNetModelWithGP(
                self.model, self.x.to(nn_device), y.to(nn_device), gp_model=gp_model
            )
        else:
            # Legacy approach without GP posterior
            return AcquisitionFunctionNetModel(
                self.model, self.x.to(nn_device), y.to(nn_device)
            )
    
    def _fit_gp_to_evaluation_history(self):
        """Fit MAP GP to the current evaluation history"""
        y = self.y if self.maximize else -self.y
        gp_model = _create_map_gp(self.x, y)
        _fit_gp_map(gp_model, self.x, y)
        return gp_model

class AcquisitionFunctionNetModelWithGP(AcquisitionFunctionNetModel):
    def __init__(self, model, train_X, train_Y, gp_model=None):
        super().__init__(model, train_X, train_Y)
        self.gp_model = gp_model
    
    def forward(self, X, **kwargs):
        # Add GP posterior via concatenation if GP model available
        if self.gp_model is not None:
            with torch.no_grad():
                posterior = self.gp_model.posterior(X)
                mu = posterior.mean.squeeze(-1)
                log_sigma = posterior.variance.sqrt().log().squeeze(-1)
                gp_posterior = torch.stack([mu, log_sigma], dim=-1)  # (n_cand, 2)
            
            # Concatenate with existing acqf_params
            if 'acqf_params' in kwargs and kwargs['acqf_params'] is not None:
                kwargs['acqf_params'] = torch.cat([kwargs['acqf_params'], gp_posterior], dim=-1)
            else:
                kwargs['acqf_params'] = gp_posterior
        
        return self.model(self.train_X, self.train_Y, X, **kwargs)
```

## Configuration and Command-Line Interface

### 1. New Command-Line Argument

**Location:** `nn_af/acquisition_function_net_save_utils.py`

```python
nn_architecture_group.add_argument(
    '--include_gp_posterior',
    action='store_true',
    default=False,
    help='Include GP posterior (mean, log-std) as input features concatenated to acqf_params. '
         'A GP will be fitted using MAP to function samples during training and evaluation history during BO.'
)
```

### 2. Update Model Creation

```python
def _get_model(args):
    # ... existing code ...
    
    if architecture == "pointnet":
        # Add GP posterior flag to model initialization
        if hasattr(args, 'include_gp_posterior') and args.include_gp_posterior:
            af_body_init_params['include_gp_posterior'] = True
        
        # Continue with existing parameter handling for acqf_params_input
        # No changes needed - GP posterior will be concatenated to acqf_params
```

### 3. YAML Configuration

**Location:** `config/train_acqf.yml`

```yaml
architecture:
  values:
  - value: pointnet
    parameters:
      # ... existing parameters ...
      include_gp_posterior:
        values: [false, true]
```

## Dataset-Specific GP Handling

### 1. GP Datasets: Preserving Original Statistics

**Critical Requirement:** When the original dataset is GP-based (`dataset_type: gp`), we must preserve the **original GP model** for computing GP statistics, while also providing the **MAP-fitted GP posterior** as input to the neural network.

```python
def _fit_gp_to_function_samples(acquisition_item):
    """Fit GP to original function samples, preserving original GP for non-GP datasets"""
    # For GP datasets: use the original underlying GP model if available
    if hasattr(acquisition_item, 'model') and acquisition_item.model is not None:
        # Return the original GP model - this preserves the "true" GP statistics
        return acquisition_item.model
    
    # For non-GP datasets: fit MAP GP to function samples  
    # ... (as shown above)
```

**Key Points:**
1. **GP Datasets**: Use the original GP model, which maintains the exact GP that generated the function samples
2. **Non-GP Datasets**: Fit MAP GP with RBF kernel to the function samples
3. **BO Loops**: Always fit MAP GP to evaluation history regardless of original dataset type

### 3. Dual GP Usage Pattern

For GP datasets, this creates a **dual GP usage pattern**:

1. **Original GP Model**: Used for computing "true" GP statistics (Expected Improvement, UCB, etc.) for comparison and evaluation
2. **MAP-fitted GP Posterior**: Used as input features to the neural network acquisition function

This allows the NN to learn from GP posterior information while preserving the ability to compute exact GP-based acquisition function values for comparison.

### 2. Backward Compatibility

- **Existing tensor-based `acqf_params` still works**: Forward method handles both `acqf_params` and `acqf_params_dict`
- **Existing `acqf_params_input` still works**: Used when parameter routing dictionary is not specified
- **Lambda/cost parameters preserved**: Existing lambda and cost parameters work without changes
- **Original GP statistics preserved**: GP datasets continue to use original GP models for statistical evaluation
- **Gradual migration**: Can add parameter types incrementally without breaking existing functionality

## Implementation Plan

### Phase 1: Core Architecture Extension (1-2 hours)
1. Add `include_gp_posterior` flag to `AcquisitionFunctionBodyPointnetV1and2.__init__()`
2. Update dimension calculations to account for GP posterior (+2 dimensions when enabled)
3. Minimal changes to `forward()` method - just concatenate GP posterior to acqf_params

### Phase 2: Dataset Integration (2-3 hours)
1. Implement `_add_gp_posterior_to_dataset()` function with simple concatenation
2. Implement `GPPosteriorDatasetWrapper` for sampled datasets
3. Implement GP fitting helpers (`_fit_gp_to_function_samples`, `_create_map_gp`, `_fit_gp_map`)
4. Integrate with training pipeline in `train_acquisition_function_net()`

### Phase 3: BO Integration (1-2 hours)
1. Create `AcquisitionFunctionNetModelWithGP` class with simple concatenation
2. Update `NNAcquisitionOptimizer.get_model()` method

### Phase 4: CLI and Configuration (30 minutes)
1. Add `--include_gp_posterior` boolean flag  
2. Update model creation logic to pass the flag
3. Add YAML configuration option

### Phase 5: Testing (1-2 hours)
1. Test GP posterior concatenation with enabled/disabled
2. Verify backward compatibility with existing lambda/cost parameters
3. Test BO integration with GP posterior

**Total estimated time:** 5-9 hours (significantly reduced from dictionary approach)

## Example Usage

```bash
# Train with GP posterior concatenated to acqf_params
python run_train.py \
    --include_gp_posterior \
    --acqf_params_input "final_only" \
    --dimension 1 --lengthscale 0.05 --kernel Matern52 \
    --method gittins --lamda 1e-2 \
    # ... other args

# Train without GP posterior (default behavior) 
python run_train.py \
    --acqf_params_input "final_only" \
    --dimension 1 --lengthscale 0.05 --kernel Matern52 \
    --method gittins --lamda 1e-2 \
    # ... other args

# BO loop with GP posterior-trained model
python run_bo.py \
    --nn_model_name v2/model_[hash_with_gp_posterior] \
    --n_iter 20 --objective_dimension 1 \
    # ... other args
```

## Key Advantages of This Simplified Approach

### 1. Simplicity and Ease of Implementation  
- **Minimal code changes**: Simple concatenation requires very few modifications
- **Single boolean flag**: Easy to understand and configure (`--include_gp_posterior`)
- **Reuses existing routing**: Leverages current `acqf_params_input` system
- **Clear conceptual model**: GP posterior is just "more parameters" to concatenate

### 2. Dataset Handling Robustness
- **Fixed datasets**: Items modified in-place for efficiency (preserves memory and caching)
- **Sampled datasets**: Wrapper class adds GP posterior on-the-fly without breaking sampling
- **Transparent operation**: The wrapper perfectly mimics the original dataset interface  
- **Memory efficient**: No need to pre-compute GP posteriors for infinite/large datasets

### 3. Full Backward Compatibility
- **Existing lambda/cost parameters work unchanged**: All current functionality preserved
- **No breaking changes**: Can enable/disable GP posterior without affecting existing code
- **Simple migration**: Single flag controls the new functionality

### 4. GP Handling Robustness
- **Dataset-aware GP fitting**: Uses original GP for GP datasets, MAP fitting for others
- **Dual GP usage**: Preserves original GP statistics while providing NN with GP posterior features  
- **Efficient computation**: GP fitting happens on-demand during iteration, not pre-computed

### 5. Foundation for Future Enhancement
- **Easy to extend**: This approach sets up the infrastructure for more sophisticated parameter routing later (Phase 2)
- **Proven pattern**: Builds on existing concatenation approach that already works
- **Low risk**: Minimal changes reduce the chance of introducing bugs

This simplified approach provides the core functionality with minimal complexity, while serving as a solid foundation for more advanced parameter routing systems in the future.