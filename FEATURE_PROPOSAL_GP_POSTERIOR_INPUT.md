# Technical Feature Proposal: Flexible Acquisition Function Parameter Types

## Overview

This proposal outlines the implementation of a flexible system for handling different types of acquisition function parameters, starting with GP posterior parameters (mean μ and log-standard-deviation ln(σ)). The system allows arbitrary parameter types to be routed to the local NN, final MLP, or both using a generic dictionary-based approach.

## Key Design Insight

Instead of concatenating different parameter types into a single tensor, we use a dictionary-based approach where different parameter types can be routed independently to different parts of the network. The configuration remains flat (YAML/command-line), but internally the system uses a clean dictionary structure for maximum flexibility.

## Motivation

The current neural network architecture handles acquisition function parameters (like lambda for Gittins index, or lambda*cost for cost-aware optimization) as a single concatenated tensor. However, different parameter types may benefit from being routed to different parts of the network. This proposal enables:

1. **Flexible routing**: GP posterior (μ, ln(σ)) to final MLP, lambda parameters to local NN, etc.
2. **Type-aware processing**: Each parameter type is handled according to its semantic meaning
3. **Easy extensibility**: Adding new parameter types requires no hardcoded changes
4. **Backward compatibility**: Existing lambda/cost parameters continue to work unchanged

## Architecture Changes

### 1. New Flexible Parameter Routing Configuration

**Current API (backward compatible):**
- `acqf_params_input`: `"local_only"` | `"final_only"` | `"local_and_final"` | `null`

**New Flat Configuration Format:**
```yaml
acqf_params_input:
  gp_posterior: "final_only"       # GP posterior μ, ln(σ) routing
  lambda: "local_and_final"        # lambda parameter routing (existing)
  cost: "final_only"               # cost parameter routing (existing)
  default: "final_only"            # default routing for unspecified types
```

### 2. New Command-Line Arguments

```bash
--gp_posterior_routing "final_only"     # Where to route GP posterior (null/unspecified = disabled)
--param_routing_default "final_only"    # Default routing for parameter types
```

### 3. Dictionary-Based Internal Structure

Instead of concatenating parameters, the forward method receives:
```python
acqf_params_dict = {
    "gp_posterior": tensor,  # shape: (..., 2) for μ, ln(σ)
    "lambda": tensor,        # shape: (..., 1) for lambda values  
    "cost": tensor,          # shape: (..., 1) for cost values
    # ... other parameter types
}
```

### 4. Implementation in `AcquisitionFunctionBodyPointnetV1and2`

**Changes to `__init__`:**
```python
def __init__(self,
             # ... existing parameters ...
             param_routing=None,                                 # NEW: dict mapping param types to routing
             param_routing_default="final_only",                 # NEW: default routing for unspecified types  
             ):
    # Store parameter routing configuration
    self.param_routing = param_routing or {}                    # {"gp_posterior": "final_only", "lambda": "local_and_final", ...}
    self.param_routing_default = param_routing_default
    
    # Calculate total dimensions for each parameter type that will be used
    total_local_param_dim = 0
    total_final_param_dim = 0
    
    # Handle GP posterior dimensions (enabled when routing is specified)
    gp_routing = self.param_routing.get("gp_posterior", None)
    if gp_routing is not None:
        if gp_routing in ["local_only", "local_and_final"]:
            total_local_param_dim += 2  # μ, ln(σ)
        if gp_routing in ["final_only", "local_and_final"]:
            total_final_param_dim += 2  # μ, ln(σ)
    
    # Handle existing acqf_params (lambda, cost, etc.) with default routing
    if n_acqf_params > 0:
        default_routing = param_routing_default
        if default_routing in ["local_only", "local_and_final"]:
            total_local_param_dim += n_acqf_params
        if default_routing in ["final_only", "local_and_final"]:
            total_final_param_dim += n_acqf_params
    
    # Calculate input dimensions
    input_dim = dimension + n_hist_out + int(include_best_y) * n_hist_out \
        + (dimension if input_xcand_to_local_nn else 0) \
        + total_local_param_dim
    
    # Calculate features dimension for final MLP  
    self._features_dim = encoded_history_dim + \
        (dimension if input_xcand_to_final_mlp else 0) \
        + total_final_param_dim
```

**Changes to `forward`:**
```python
def forward(self, x_hist, y_hist, x_cand, acqf_params_dict=None, acqf_params=None, hist_mask=None, cand_mask=None, **kwargs):
    # Handle backward compatibility: convert old acqf_params tensor to dict
    if acqf_params_dict is None and acqf_params is not None:
        # Legacy mode: treat as default parameter type with default routing
        acqf_params_dict = {"default": acqf_params}
    
    # Build local NN input components  
    local_components = []
    final_components = []
    
    if acqf_params_dict is not None:
        for param_type, param_tensor in acqf_params_dict.items():
            routing = self.param_routing.get(param_type, self.param_routing_default)
            
            if routing in ["local_only", "local_and_final"]:
                local_components.append(param_tensor)
            if routing in ["final_only", "local_and_final"]:
                final_components.append(param_tensor)
    
    # Concatenate local parameter components for local NN
    local_acqf_params = torch.cat(local_components, dim=-1) if local_components else None
    
    # ... existing local NN processing with local_acqf_params ...
    
    # Build final MLP inputs
    final_items = [out]  # encoded history features
    if final_components:
        final_items.extend(final_components)
    if self.input_xcand_to_final_mlp:
        final_items.append(x_cand)
    
    if len(final_items) > 1:
        out = torch.cat(final_items, dim=-1)
    
    return out
```

## Dataset and Training Integration

### 1. Function Samples Dataset GP Posterior Computation

**Key Insight:** For training datasets, we need to fit GPs to the **original function samples** from `FunctionSamplesDataset`, not to the sampled acquisition history. This is because:

1. **Training**: Each acquisition dataset item is derived from a function sample, and we want the GP posterior from the **entire function** for the NN to learn optimal acquisition behavior
2. **BO loops**: We fit MAP GP to the **evaluation history** to get the posterior at candidate points

**Location:** `nn_af/train_acquisition_function_net.py`

```python
def train_acquisition_function_net(nn_model, train_dataset, ...):
    # Check if NN expects parameter dictionary format
    if _nn_expects_param_dict(nn_model):
        # Convert datasets to parameter dictionary format
        train_dataset = _convert_dataset_to_param_dict(train_dataset, nn_model, verbose=verbose)
        if test_dataset is not None:
            test_dataset = _convert_dataset_to_param_dict(test_dataset, nn_model, verbose=verbose)
        # ... handle small_test_dataset similarly
    
    # Continue with existing training loop
```

### 2. Helper Functions

```python
def _nn_expects_param_dict(nn_model) -> bool:
    """Check if NN model expects parameter dictionary format"""
    # Navigate to the body to check for new parameter routing system
    if hasattr(nn_model, 'base_model'):
        model = nn_model.base_model
    else:
        model = nn_model
    
    if hasattr(model, 'af_body') and hasattr(model.af_body, 'param_routing'):
        return True
    
    return False

def _convert_dataset_to_param_dict(dataset, nn_model, verbose=False):
    """Convert acqf_params tensor to dictionary format and add GP posterior if needed.
    
    This handles both fixed datasets (where items can be modified in-place) and
    sampled datasets (where items are generated on-the-fly) by using a wrapper class.
    """
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from utils.utils import get_gp
    
    if verbose:
        if dataset.data_is_fixed:
            print(f"Converting fixed dataset to parameter dictionary format for {len(dataset)} items...")
        else:
            print(f"Creating parameter dictionary wrapper for sampled dataset...")
    
    # Get model configuration
    af_body = nn_model.base_model.af_body if hasattr(nn_model, 'base_model') else nn_model.af_body
    gp_routing = getattr(af_body, 'param_routing', {}).get("gp_posterior", None)
    
    if dataset.data_is_fixed:
        # For fixed datasets: modify items in-place (original approach)
        # Create copy if dataset has models to preserve original
        if dataset.has_models:
            dataset = dataset.copy_with_new_size(None)
        
        for item in (tqdm(dataset) if verbose else dataset):
            # Start building parameter dictionary
            param_dict = {}
            
            # Add GP posterior if routing is specified (enabled)
            if gp_routing is not None:
                # **CRITICAL**: For training, fit GP to ORIGINAL function samples, not acquisition history
                gp_model = _fit_gp_to_function_samples(item)
                with torch.no_grad():
                    posterior = gp_model.posterior(item.x_cand)
                    mu = posterior.mean.squeeze(-1)  # (n_cand,)
                    log_sigma = posterior.variance.sqrt().log().squeeze(-1)  # (n_cand,)
                    gp_posterior = torch.stack([mu, log_sigma], dim=-1)  # (n_cand, 2)
                param_dict["gp_posterior"] = gp_posterior
            
            # Add existing acqf_params as default type (for lambda, cost, etc.)
            if hasattr(item, 'acqf_params') and item.acqf_params is not None:
                param_dict["lambda"] = item.acqf_params  # Assume existing params are lambda/cost
            
            # Replace acqf_params with dictionary
            item.acqf_params_dict = param_dict
            item.acqf_params = None  # Clear old tensor format for clarity
        
        return dataset
    else:
        # For sampled datasets: use wrapper class that converts on-the-fly
        return ParamDictDatasetWrapper(dataset, gp_routing, verbose)

class ParamDictDatasetWrapper:
    """Wrapper for sampled datasets that converts acqf_params to dictionary format on-the-fly.
    
    This class wraps acquisition datasets that generate items dynamically (data_is_fixed=False)
    and converts acqf_params to the new parameter dictionary format during iteration.
    It preserves all original dataset properties and methods while adding GP posterior computation.
    """
    def __init__(self, base_dataset, gp_routing, verbose=False):
        self.base_dataset = base_dataset
        self.gp_routing = gp_routing
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
        """Iterate over base dataset and convert items to parameter dictionary format"""
        for item in self.base_dataset:
            # Start building parameter dictionary
            param_dict = {}
            
            # Add GP posterior if routing is specified (enabled)
            if self.gp_routing is not None:
                # **CRITICAL**: For training, fit GP to ORIGINAL function samples, not acquisition history
                gp_model = _fit_gp_to_function_samples(item)
                with torch.no_grad():
                    posterior = gp_model.posterior(item.x_cand)
                    mu = posterior.mean.squeeze(-1)  # (n_cand,)
                    log_sigma = posterior.variance.sqrt().log().squeeze(-1)  # (n_cand,)
                    gp_posterior = torch.stack([mu, log_sigma], dim=-1)  # (n_cand, 2)
                param_dict["gp_posterior"] = gp_posterior
            
            # Add existing acqf_params as default type (for lambda, cost, etc.)
            if hasattr(item, 'acqf_params') and item.acqf_params is not None:
                param_dict["lambda"] = item.acqf_params  # Assume existing params are lambda/cost
            
            # Create new item with parameter dictionary
            # Use the same class as the original item for consistency
            new_item = type(item)(
                item.x_hist, item.y_hist, item.x_cand, item.vals_cand,
                model=getattr(item, '_model', None),
                model_params=getattr(item, 'model_params', None),
                give_improvements=item.give_improvements
            )
            
            # Add the parameter dictionary
            new_item.acqf_params_dict = param_dict
            new_item.acqf_params = None  # Clear old tensor format for clarity
            
            yield new_item
    
    def copy_with_new_size(self, size=None):
        """Create a copy with new size, preserving the wrapper"""
        base_copy = self.base_dataset.copy_with_new_size(size)
        return ParamDictDatasetWrapper(base_copy, self.gp_routing, self.verbose)
    
    def random_split(self, lengths):
        """Split the dataset while preserving the wrapper"""
        base_splits = self.base_dataset.random_split(lengths)
        return [ParamDictDatasetWrapper(split, self.gp_routing, self.verbose) 
                for split in base_splits]

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

### 1. Modify `NNAcquisitionOptimizer`

**Key Insight:** For BO loops, we fit MAP GP to the **evaluation history** (`self.x`, `self.y`) to get the posterior at candidate points. This is different from training where we use the entire function samples.

```python
class NNAcquisitionOptimizer:
    def get_model(self):
        y = self.y if self.maximize else -self.y
        nn_device = next(self.model.parameters()).device
        
        # Check if we need parameter dictionary format
        if _nn_expects_param_dict(self.model):
            # Check if GP posterior is needed
            af_body = self.model.base_model.af_body if hasattr(self.model, 'base_model') else self.model.af_body
            gp_routing = getattr(af_body, 'param_routing', {}).get("gp_posterior", None)
            
            if gp_routing is not None:
                # **CRITICAL**: For BO, fit GP to EVALUATION HISTORY, not function samples
                gp_model = self._fit_gp_to_evaluation_history()
                return AcquisitionFunctionNetModelWithParamDict(
                    self.model, self.x.to(nn_device), y.to(nn_device), gp_model=gp_model
                )
            else:
                return AcquisitionFunctionNetModelWithParamDict(
                    self.model, self.x.to(nn_device), y.to(nn_device)
                )
        else:
            # Legacy tensor-based approach
            return AcquisitionFunctionNetModel(
                self.model, self.x.to(nn_device), y.to(nn_device)
            )
    
    def _fit_gp_to_evaluation_history(self):
        """Fit MAP GP to the current evaluation history"""
        y = self.y if self.maximize else -self.y
        gp_model = _create_map_gp(self.x, y)
        _fit_gp_map(gp_model, self.x, y)
        return gp_model

class AcquisitionFunctionNetModelWithParamDict(AcquisitionFunctionNetModel):
    def __init__(self, model, train_X, train_Y, gp_model=None):
        super().__init__(model, train_X, train_Y)
        self.gp_model = gp_model
    
    def forward(self, X, **kwargs):
        # Build parameter dictionary
        param_dict = {}
        
        # Add GP posterior if available
        if self.gp_model is not None:
            with torch.no_grad():
                posterior = self.gp_model.posterior(X)
                mu = posterior.mean.squeeze(-1)
                log_sigma = posterior.variance.sqrt().log().squeeze(-1)
                gp_posterior = torch.stack([mu, log_sigma], dim=-1)  # (n_cand, 2)
            param_dict["gp_posterior"] = gp_posterior
        
        # Add existing acqf_params as lambda type (backward compatibility)
        if 'acqf_params' in kwargs and kwargs['acqf_params'] is not None:
            param_dict["lambda"] = kwargs['acqf_params']
        
        # Replace acqf_params with dictionary
        if param_dict:
            kwargs['acqf_params_dict'] = param_dict
            kwargs.pop('acqf_params', None)  # Remove old format
        
        return self.model(self.train_X, self.train_Y, X, **kwargs)
```

## Configuration and Command-Line Interface

### 1. New Command-Line Arguments

**Location:** `nn_af/acquisition_function_net_save_utils.py`

```python
nn_architecture_group.add_argument(
    '--gp_posterior_routing',
    type=str,
    choices=['local_only', 'final_only', 'local_and_final'],
    default=None,
    help='Where to route GP posterior parameters. If not specified (null), GP posterior is disabled. '
         'A GP will be fitted using MAP to the history during training and BO when enabled.'
)

nn_architecture_group.add_argument(
    '--param_routing_default',
    type=str,
    choices=['local_only', 'final_only', 'local_and_final'],
    default='final_only',
    help='Default routing for parameter types not explicitly specified. Default is "final_only".'
)
```

### 2. Update Model Creation

```python
def _get_model(args):
    # ... existing code ...
    
    if architecture == "pointnet":
        # Build parameter routing configuration
        param_routing = {}
        
        # Add GP posterior routing if specified (not None)
        if getattr(args, 'gp_posterior_routing', None) is not None:
            param_routing["gp_posterior"] = args.gp_posterior_routing
        
        # Add parameter routing to model initialization
        if param_routing or hasattr(args, 'param_routing_default'):
            af_body_init_params['param_routing'] = param_routing
            af_body_init_params['param_routing_default'] = getattr(args, 'param_routing_default', 'final_only')
        else:
            # Use existing acqf_params_input system for backward compatibility
            extra_params = _POINTNET_X_CAND_INPUT_OPTIONS[args.x_cand_input]
            if args.acqf_params_input is not None:
                extra_params_acqf = POINTNET_ACQF_PARAMS_INPUT_OPTIONS[args.acqf_params_input]
                extra_params = dict(**extra_params, **extra_params_acqf)
            af_body_init_params = dict(**af_body_init_params_base, **extra_params)
```

### 3. YAML Configuration

**Location:** `config/train_acqf.yml`

```yaml
architecture:
  values:
  - value: pointnet
    parameters:
      # ... existing parameters ...
      gp_posterior_routing:
        values: [null, "final_only", "local_only", "local_and_final"]
      param_routing_default:
        values: ["final_only", "local_only", "local_and_final"]
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

### Phase 1: Core Architecture Extension (2-3 hours)
1. Extend `AcquisitionFunctionBodyPointnetV1and2` to support parameter routing dictionary
2. Add dictionary-based parameter handling in `forward()` method
3. Update dimension calculations for local NN and final MLP

### Phase 2: Dataset Integration (2-3 hours)
1. Implement `_convert_dataset_to_param_dict()` function
2. Implement GP fitting helpers for GP posterior computation
3. Integrate with training pipeline in `train_acquisition_function_net()`

### Phase 3: BO Integration (1-2 hours)
1. Create `AcquisitionFunctionNetModelWithParamDict` class
2. Update `NNAcquisitionOptimizer.get_model()` method

### Phase 4: CLI and Configuration (1 hour)
1. Add new command-line arguments for GP posterior and parameter routing
2. Update model creation logic to build parameter routing configuration
3. Add YAML configuration options

### Phase 5: Testing (1-2 hours)
1. Test parameter routing with GP posterior enabled/disabled
2. Verify backward compatibility with existing lambda/cost parameters
3. Test BO integration with parameter dictionary format

**Total estimated time:** 7-11 hours

## Example Usage

```bash
# GP posterior only to final MLP, other params use default routing
python run_train.py \
    --gp_posterior_routing "final_only" \
    --param_routing_default "final_only" \
    # ... other args

# GP posterior to both local and final, lambda params to local only  
python run_train.py \
    --gp_posterior_routing "local_and_final" \
    --param_routing_default "local_only" \
    # ... other args (lambda/cost will route to local_only)

# No GP posterior (disabled), only lambda/cost params with default routing
python run_train.py \
    --param_routing_default "final_only" \
    # ... other args (no --gp_posterior_routing means it's disabled)
```

## Key Advantages of This Approach

### 1. Flexible Dataset Handling
- **Fixed datasets**: Items modified in-place for efficiency (preserves memory and allows caching)
- **Sampled datasets**: Wrapper class converts items on-the-fly without breaking the sampling pattern
- **Transparent operation**: The wrapper perfectly mimics the original dataset interface
- **Memory efficient**: No need to pre-compute GP posteriors for infinite/large datasets

### 2. Architecture Flexibility
- **Maximum routing flexibility**: Any parameter type can be routed to any combination of network parts
- **Clean separation**: Each parameter type is handled independently 
- **Extensible**: Adding new parameter types requires no hardcoded changes
- **Type-aware**: Network understands semantic meaning of different parameter types

### 3. Backward Compatibility
- **Existing lambda/cost parameters work unchanged**: All current functionality preserved
- **Gradual migration**: Can add parameter types incrementally without breaking existing functionality
- **Simple configuration**: Flat YAML/CLI structure despite powerful internal dictionary handling

### 4. GP Handling Robustness
- **Dataset-aware GP fitting**: Uses original GP for GP datasets, MAP fitting for others
- **Dual GP usage**: Preserves original GP statistics while providing NN with GP posterior features
- **Efficient computation**: GP fitting happens on-demand during iteration, not pre-computed

This hybrid approach (in-place modification + wrapper class) handles the diversity of dataset types in the codebase while providing maximum flexibility and maintaining a clean, extensible architecture.