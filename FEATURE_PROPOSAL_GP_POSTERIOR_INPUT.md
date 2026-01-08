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

### 1. Dataset Parameter Dictionary Creation

**Location:** `utils_train/train_acquisition_function_net.py`

Add preprocessing step to convert tensor-based `acqf_params` to dictionary format and compute GP posterior when enabled:

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
    """Convert acqf_params tensor to dictionary format and add GP posterior if needed"""
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from utils.utils import get_gp
    
    if verbose:
        print(f"Converting dataset to parameter dictionary format for {len(dataset)} items...")
    
    # Get model configuration
    af_body = nn_model.base_model.af_body if hasattr(nn_model, 'base_model') else nn_model.af_body
    gp_routing = getattr(af_body, 'param_routing', {}).get("gp_posterior", None)
    
    # Create copy if dataset has models to preserve original
    if dataset.has_models:
        dataset = dataset.copy_with_new_size(None)
    
    for item in (tqdm(dataset) if verbose else dataset):
        # Start building parameter dictionary
        param_dict = {}
        
        # Add GP posterior if routing is specified (enabled)
        if gp_routing is not None:
            # Fit GP model and compute posterior
            gp_model = _fit_gp_model_for_item(item)
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
```

## Bayesian Optimization Integration

### 1. Modify `NNAcquisitionOptimizer`

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
                gp_model = self._fit_gp_for_posterior()
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

**Location:** `utils_train/model_save_utils.py`

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
def initialize_module_from_args(args):
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

## Backwards Compatibility

- **Existing tensor-based `acqf_params` still works**: Forward method handles both `acqf_params` and `acqf_params_dict`
- **Existing `acqf_params_input` still works**: Used when parameter routing dictionary is not specified
- **Lambda/cost parameters preserved**: Existing lambda and cost parameters work without changes
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
python single_train.py \
    --gp_posterior_routing "final_only" \
    --param_routing_default "final_only" \
    # ... other args

# GP posterior to both local and final, lambda params to local only  
python single_train.py \
    --gp_posterior_routing "local_and_final" \
    --param_routing_default "local_only" \
    # ... other args (lambda/cost will route to local_only)

# No GP posterior (disabled), only lambda/cost params with default routing
python single_train.py \
    --param_routing_default "final_only" \
    # ... other args (no --gp_posterior_routing means it's disabled)
```

## Key Advantages of This Approach

1. **Maximum flexibility**: Any parameter type can be routed to any combination of network parts
2. **Clean separation**: Each parameter type is handled independently 
3. **Extensible**: Adding new parameter types requires no hardcoded changes
4. **Backward compatible**: Existing lambda/cost parameters work unchanged
5. **Simple configuration**: Flat YAML/CLI structure despite powerful internal dictionary handling
6. **Type-aware**: Network understands semantic meaning of different parameter types

This dictionary-based approach provides maximum flexibility while maintaining a clean, extensible architecture.