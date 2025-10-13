# Technical Feature Proposal: GP Posterior (μ, ln(σ)) Input to Neural Network

## Overview

This proposal outlines the implementation of an optional feature to pass GP posterior parameters (mean μ and log-standard-deviation ln(σ)) as additional inputs to the acquisition function neural network. The GP will be fitted using Maximum A Posteriori (MAP) estimation on the evaluation history.

## Motivation

The current neural network architecture learns acquisition functions from history (x, y) pairs without explicit uncertainty estimates. By providing the NN with GP posterior statistics (μ, ln(σ)) at each candidate point, we can potentially improve the NN's ability to:
1. Distinguish between areas of high uncertainty and low uncertainty
2. Better approximate acquisition functions that rely on uncertainty (e.g., Expected Improvement, UCB)
3. Leverage both learned patterns from training and model-based uncertainty estimates during inference

## Key Design Decision: Consistent RBF GP Models for Training

When `input_gp_posterior=True`, the implementation will **always use fitted RBF GP models with default priors** for computing posterior statistics during training, regardless of the original dataset type:

- **GP Datasets** (already have models): **Create a copy** of the dataset and replace models with fitted RBF GP models in the copy
- **Non-GP Datasets** (no models): Fitted RBF GP models are **created and attached** (can modify in-place since no original models to preserve)

**Rationale:** This ensures consistent GP posterior statistics across all dataset types. The NN learns to use standardized uncertainty estimates rather than dataset-specific model characteristics.

**Important:** The original dataset remains unchanged - model replacement only affects the copy used during training.

## Architecture Changes

### 1. Neural Network Architecture (`nn_af/acquisition_function_net.py`)

#### 1.1 New Architecture Parameter

Add a new boolean parameter `input_gp_posterior` to the **acquisition function head** classes that control whether the NN expects GP posterior inputs.

**Key Design Decision:** The GP posterior (μ, ln(σ)) represents point-wise statistics about individual candidates and should be handled by the **head** (final MLP), not the body. The body extracts features from (x_hist, y_hist, x_cand), while the head uses those features plus optional per-candidate statistics to compute the final acquisition value.

**Affected Classes:**
- `AcquisitionFunctionHead` (abstract base class)
- `AcquisitionFunctionNetFinalMLP` (concrete implementation)
- `AcquisitionFunctionNetFinalMLPSoftmaxExponentiate` (concrete implementation)

**Changes to `AcquisitionFunctionHead`:**
```python
class AcquisitionFunctionHead(nn.Module, SaveableObject):
    @property
    @abstractmethod
    def input_gp_posterior(self) -> bool:
        """Returns whether this head expects GP posterior (mu, log_sigma) as input"""
        pass
```

**Changes to `AcquisitionFunctionNetFinalMLP.__init__`:**
- Add parameter: `input_gp_posterior: bool = False`
- Store as buffer: `self.register_buffer("input_gp_posterior", torch.as_tensor(input_gp_posterior))`
- Adjust `input_dim` to account for 2 additional features when `input_gp_posterior=True`:
  ```python
  def __init__(self,
               input_dim: int,
               hidden_dims: Sequence[int]=[256, 64],
               output_dim=1,
               activation="relu",
               layer_norm_before_end=False,
               layer_norm_at_end=False,
               dropout=None,
               input_gp_posterior: bool = False):  # NEW parameter
      super().__init__()

      # Adjust input dimension if GP posterior is expected
      effective_input_dim = input_dim + (2 if input_gp_posterior else 0)

      self.dense = Dense(effective_input_dim,  # Use adjusted dimension
                         hidden_dims,
                         output_dim,
                         activation=activation,
                         activation_at_end=False,
                         layer_norm_before_end=layer_norm_before_end,
                         layer_norm_at_end=False,
                         dropout=dropout,
                         dropout_at_end=False)

      self.register_buffer("_input_dim", torch.as_tensor(input_dim))
      self.register_buffer("_output_dim", torch.as_tensor(output_dim))
      self.register_buffer("layer_norm_at_end", torch.as_tensor(layer_norm_at_end))
      self.register_buffer("input_gp_posterior", torch.as_tensor(input_gp_posterior))
  ```

**Changes to `AcquisitionFunctionNetFinalMLP.forward`:**
- Add optional parameters: `gp_posterior_mu: Optional[Tensor] = None`, `gp_posterior_log_sigma: Optional[Tensor] = None`
- Add validation logic:
  ```python
  def forward(self, features,
              x_hist, y_hist, x_cand,
              hist_mask=None, cand_mask=None, stdvs=None,
              gp_posterior_mu: Optional[Tensor] = None,  # NEW
              gp_posterior_log_sigma: Optional[Tensor] = None,  # NEW
              **other_kwargs) -> Tensor:
      # Validate GP posterior inputs
      if self.input_gp_posterior:
          if gp_posterior_mu is None or gp_posterior_log_sigma is None:
              raise ValueError(
                  "gp_posterior_mu and gp_posterior_log_sigma must be provided "
                  "when input_gp_posterior=True")
          # Concatenate GP posterior to features
          # features shape: (*, n_cand, input_dim)
          # gp_posterior_mu shape: (*, n_cand, 1)
          # gp_posterior_log_sigma shape: (*, n_cand, 1)
          features = torch.cat([features, gp_posterior_mu, gp_posterior_log_sigma], dim=-1)
      else:
          if gp_posterior_mu is not None or gp_posterior_log_sigma is not None:
              raise ValueError(
                  "gp_posterior_mu and gp_posterior_log_sigma must be None "
                  "when input_gp_posterior=False")

      # Continue with existing logic
      acquisition_values = self.dense(features)
      # ... rest of existing code ...
  ```

**Similar changes for `AcquisitionFunctionNetFinalMLPSoftmaxExponentiate`:**
- Add the same `input_gp_posterior` parameter to `__init__`
- Adjust `input_dim` calculation
- Add the same validation and concatenation logic in `forward()`

#### 1.2 Propagate Changes Through Two-Part Architecture

**Changes to `TwoPartAcquisitionFunctionNet.__init__`:**
- No changes needed - the head's `input_gp_posterior` is already encapsulated in `af_head_init_params`

**Changes to `TwoPartAcquisitionFunctionNet.forward`:**
- Add optional parameters: `gp_posterior_mu: Optional[Tensor] = None`, `gp_posterior_log_sigma: Optional[Tensor] = None`
- Pass these parameters through to `self.af_head.forward()` call:
  ```python
  def forward(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
              acqf_params:Optional[Tensor]=None,
              hist_mask:Optional[Tensor]=None,
              cand_mask:Optional[Tensor]=None,
              gp_posterior_mu: Optional[Tensor] = None,  # NEW
              gp_posterior_log_sigma: Optional[Tensor] = None,  # NEW
              **kwargs) -> Tensor:
      # ... existing preprocessing ...

      # batch_shape x n_cand x features_dim
      features = self.af_body(x_hist, y_hist, x_cand,
                              acqf_params=acqf_params,
                              hist_mask=hist_mask, cand_mask=cand_mask)
      # batch_shape x n_cand x output_dim
      return self.af_head(features, x_hist, y_hist, x_cand,
                          hist_mask=hist_mask, cand_mask=cand_mask,
                          stdvs=stdvs,
                          gp_posterior_mu=gp_posterior_mu,  # NEW
                          gp_posterior_log_sigma=gp_posterior_log_sigma,  # NEW
                          **kwargs)
  ```

**Changes to wrapper classes:**
- `GittinsAcquisitionFunctionNet.forward`: Add the two optional GP posterior parameters and pass through to `self.base_model()`
- `ExpectedImprovementAcquisitionFunctionNet.forward`: Add the two optional GP posterior parameters and pass through to `self.base_model()`

### 2. Dataset Management (`datasets/`)

Each dataset item in the acquisition dataset already has a `model` attribute that can store GP models. The implementation will leverage this existing infrastructure.

**Key insight from code review:**
- `datasets/dataset_with_models.py` already provides infrastructure for attaching models to dataset items
- `datasets/acquisition_dataset.py` defines the `AcquisitionDataset` base class with `has_models` property
- Training loop in `train_acquisition_function_net.py` already accesses `batch.model` when models are available

#### 2.1 Pre-fit GP Models During Dataset Creation

**Location:** `nn_af/train_acquisition_function_net.py` - function `train_acquisition_function_net()`

Add a preprocessing step at the beginning of the function (before training loop):

```python
def train_acquisition_function_net(
        nn_model: AcquisitionFunctionNet,
        train_dataset: AcquisitionDataset,
        ...
    ):
    # Existing parameter validation...

    # NEW: Check if NN expects GP posterior input
    nn_expects_gp_posterior = _nn_expects_gp_posterior_input(nn_model)

    if nn_expects_gp_posterior:
        # Strategy for handling GP models:
        # For GP posterior input, we ALWAYS want to use fitted RBF MAP models,
        # regardless of what models the dataset originally had (if any).
        # This ensures consistent GP posterior statistics across all dataset types.
        #
        # Implementation:
        # - If dataset has models: Create a copy using built-in copy_with_new_size()
        #   and replace models in the copy
        # - If dataset has no models: Can modify in-place (no original models to preserve)

        if train_dataset.has_models:
            # Create a copy to avoid modifying the original dataset
            # copy_with_new_size(None) creates a copy with the same size
            train_dataset = train_dataset.copy_with_new_size(None)
        _replace_with_fitted_rbf_gp_models(train_dataset, verbose=verbose)

        if test_dataset is not None:
            if test_dataset.has_models:
                test_dataset = test_dataset.copy_with_new_size(None)
            _replace_with_fitted_rbf_gp_models(test_dataset, verbose=verbose)

        if small_test_dataset is not None:
            if small_test_dataset.has_models:
                small_test_dataset = small_test_dataset.copy_with_new_size(None)
            _replace_with_fitted_rbf_gp_models(small_test_dataset, verbose=verbose)

    # Continue with existing training loop...
```

**Helper function `_nn_expects_gp_posterior_input`:**

This function will be needed in multiple places:
- `nn_af/train_acquisition_function_net.py` (for training)
- `bayesopt/bayesopt.py` (for BO loops)

It should be defined in `train_acquisition_function_net.py` and also either duplicated or imported in `bayesopt.py`.

```python
def _nn_expects_gp_posterior_input(nn_model: AcquisitionFunctionNet) -> bool:
    """Check if NN model expects GP posterior (mu, log_sigma) as input"""
    # Navigate through the model hierarchy to find the head
    if hasattr(nn_model, 'base_model'):  # GittinsAcquisitionFunctionNet or ExpectedImprovementAcquisitionFunctionNet
        model = nn_model.base_model
    else:
        model = nn_model

    if hasattr(model, 'af_head'):  # TwoPartAcquisitionFunctionNet
        return model.af_head.input_gp_posterior

    return False
```

**Helper function `_replace_with_fitted_rbf_gp_models`:**
```python
def _replace_with_fitted_rbf_gp_models(dataset: AcquisitionDataset, verbose: bool = False):
    """Replace dataset models with fitted RBF GP models using MAP estimation.

    This modifies the dataset in-place by creating a new RBF GP model with default priors
    for each item, fitting it to that item's history using MAP estimation, and replacing
    any existing models with these fitted models.

    The purpose is to ensure consistent GP posterior statistics across all dataset types,
    regardless of what models (if any) the dataset originally had.

    IMPORTANT: If the dataset originally has models that should be preserved, the caller
    should pass a copy of the dataset to this function (e.g., `dataset.copy_with_new_size(None)`),
    not the original.

    Args:
        dataset: The acquisition dataset to replace models in (will be modified in-place)
        verbose: Whether to print progress information
    """
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.models.transforms.outcome import Standardize
    from utils.utils import get_gp
    import warnings

    if verbose:
        if dataset.has_models:
            print(f"Replacing existing models in {len(dataset)} dataset items with fitted RBF GP models using MAP estimation...")
        else:
            print(f"Creating and fitting RBF GP models for {len(dataset)} dataset items using MAP estimation...")

    # We need to get the dimension from the first item
    first_item = next(iter(dataset))
    dimension = first_item.x_hist.size(-1)

    n_fitted = 0
    n_failed = 0

    for i, item in enumerate(tqdm(dataset) if verbose else dataset):
        x_hist = item.x_hist  # shape: (n_hist, d)
        y_hist = item.y_hist  # shape: (n_hist, 1) or (n_hist,)
        hist_mask = item.hist_mask  # shape: (n_hist,) or (n_hist, 1) or None

        # Remove padding if there is a mask
        if hist_mask is not None:
            mask_flat = hist_mask.squeeze() if hist_mask.dim() > 1 else hist_mask
            x_hist = x_hist[mask_flat.bool()]
            y_hist = y_hist[mask_flat.bool()]

        # Ensure y_hist has correct shape for BoTorch
        if y_hist.dim() == 1:
            y_hist = y_hist.unsqueeze(-1)

        # Create a new RBF GP model with default priors and outcome standardization
        # This matches the default configuration used in BO loops
        model = get_gp(
            dimension=dimension,
            observation_noise=False,
            outcome_transform=Standardize(m=1)
        )

        # Set training data (this also handles outcome transforms if present)
        model.set_train_data_with_transforms(x_hist, y_hist, strict=False, train=True)

        # Fit hyperparameters using MAP (priors are kept by default in get_gp)
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        try:
            fit_gpytorch_mll(mll)
            n_fitted += 1
        except Exception as e:
            # If fitting fails, keep the model with initial hyperparameters
            n_failed += 1
            warnings.warn(
                f"Failed to fit GP for dataset item {i}: {e}. "
                f"Using initial hyperparameters.",
                RuntimeWarning)

        # Set to eval mode for posterior computation
        model.eval()

        # Replace the item's model with the newly fitted RBF GP model
        # This works whether the item previously had a model or not
        item._model = model
        item.model_params = None  # No randomized params since we fitted to data

    # If the dataset didn't have models before, we need to mark it as having models now
    # Note: If it already had models, has_models should still be True
    if not dataset.has_models:
        # Need to set up _model_sampler or equivalent to make has_models return True
        # This may require creating a dummy RandomModelSampler or similar
        # The exact mechanism depends on the dataset_with_models.py implementation
        pass  # TODO: Investigate and implement proper way to mark dataset as having models

    if verbose:
        print(f"GP model replacement complete: {n_fitted} models successfully fitted, {n_failed} fitting failures")
```

#### 2.2 Compute GP Posterior During Training

**Location:** `nn_af/train_acquisition_function_net.py` - function `train_or_test_loop()`

Modify the section where NN forward pass is called (around lines 636-652):

```python
if nn_model is not None:
    # ... existing code to prepare inputs ...

    # NEW: Compute GP posterior if NN expects it
    if _nn_expects_gp_posterior_input(nn_model):
        # Models are accessed from batch when has_models is True (line ~600-601)
        gp_posterior_mu_nn, gp_posterior_log_sigma_nn = _compute_gp_posterior_batch(
            x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, models, nn_device
        )
    else:
        gp_posterior_mu_nn = None
        gp_posterior_log_sigma_nn = None

    with torch.set_grad_enabled(train and not is_degenerate_batch):
        if method == 'gittins':
            # ... existing code ...
            nn_output = nn_model(
                x_hist_nn, y_hist_nn, x_cand_nn,
                lambda_cand=lambda_cand_nn,
                hist_mask=hist_mask_nn, cand_mask=cand_mask_nn,
                is_log=True,
                gp_posterior_mu=gp_posterior_mu_nn,
                gp_posterior_log_sigma=gp_posterior_log_sigma_nn
            )
        else:  # method = 'mse_ei' or 'policy_gradient'
            nn_output = nn_model(
                x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, cand_mask_nn,
                exponentiate=(method == "mse_ei"),
                softmax=(method == "policy_gradient"),
                gp_posterior_mu=gp_posterior_mu_nn,
                gp_posterior_log_sigma=gp_posterior_log_sigma_nn
            )
        # ... rest of existing code ...
```

**Helper function `_compute_gp_posterior_batch`:**
```python
def _compute_gp_posterior_batch(
        x_hist: Tensor,
        y_hist: Tensor,
        x_cand: Tensor,
        hist_mask: Optional[Tensor],
        models: List[GPyTorchModel],
        device
    ) -> Tuple[Tensor, Tensor]:
    """Compute GP posterior (mu, log_sigma) for a batch of candidates.

    Args:
        x_hist: shape (batch_size, n_hist, d)
        y_hist: shape (batch_size, n_hist, 1)
        x_cand: shape (batch_size, n_cand, d)
        hist_mask: shape (batch_size, n_hist, 1) or None
        models: list of fitted GP models, length batch_size
        device: torch device

    Returns:
        mu: shape (batch_size, n_cand, 1) - posterior mean
        log_sigma: shape (batch_size, n_cand, 1) - log of posterior standard deviation
    """
    batch_size = x_hist.size(0)
    n_cand = x_cand.size(1)

    mus = []
    log_sigmas = []

    for i in range(batch_size):
        x_hist_i = x_hist[i]  # (n_hist, d)
        y_hist_i = y_hist[i]  # (n_hist, 1)
        x_cand_i = x_cand[i]  # (n_cand, d)
        model_i = models[i]

        # Remove padding if mask exists
        if hist_mask is not None:
            mask_i = hist_mask[i].squeeze()  # (n_hist,)
            x_hist_i = x_hist_i[mask_i.bool()]
            y_hist_i = y_hist_i[mask_i.bool()]

        # Set model to eval mode
        model_i.eval()

        # Update training data (should already be set from pre-fitting, but just in case)
        # Note: set_train_data_with_transforms with train=False will not refit
        model_i.set_train_data_with_transforms(x_hist_i, y_hist_i, strict=False, train=False)

        # Compute posterior
        with torch.no_grad():
            posterior = model_i.posterior(x_cand_i)
            mu_i = posterior.mean  # shape (n_cand, 1)
            # Get standard deviation (not variance)
            sigma_i = posterior.variance.sqrt()  # shape (n_cand, 1)
            log_sigma_i = sigma_i.log()

        mus.append(mu_i)
        log_sigmas.append(log_sigma_i)

    mu = torch.stack(mus, dim=0).to(device)  # (batch_size, n_cand, 1)
    log_sigma = torch.stack(log_sigmas, dim=0).to(device)  # (batch_size, n_cand, 1)

    return mu, log_sigma
```

### 3. Bayesian Optimization Loop Integration (`bayesopt/bayesopt.py`)

#### 3.1 Modify NNAcquisitionOptimizer

**Changes to `NNAcquisitionOptimizer.get_model()`:**

Currently this method creates an `AcquisitionFunctionNetModel` wrapper. We need to extend it to compute GP posterior when needed.

```python
def get_model(self):
    y = self.y if self.maximize else -self.y
    nn_device = next(self.model.parameters()).device

    # Check if NN expects GP posterior
    gp_model = None
    if _nn_expects_gp_posterior_input(self.model):
        gp_model = self._fit_gp_for_posterior()

    return AcquisitionFunctionNetModel(
        self.model,
        self.x.to(nn_device),
        y.to(nn_device),
        gp_model=gp_model
    )
```

Add helper method to `NNAcquisitionOptimizer`:

```python
def _fit_gp_for_posterior(self):
    """Fit a GP model to current optimization history using MAP estimation.

    Returns:
        Fitted SingleTaskGP model
    """
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import warnings

    # Create GP model with outcome standardization (standard practice)
    gp_model = SingleTaskGP(
        self.x,
        self.y if self.maximize else -self.y,
        outcome_transform=Standardize(m=1)
    )

    # Fit using MAP (priors are kept by default)
    gp_model.train()
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        warnings.warn(f"GP fitting failed during BO: {e}. Using initial hyperparameters.", RuntimeWarning)

    gp_model.eval()
    return gp_model
```

#### 3.2 Modify AcquisitionFunctionNetModel and AcquisitionFunctionNetAcquisitionFunction

**Changes to `AcquisitionFunctionNetModel.__init__`:**

Store optional GP model for posterior computation:

```python
def __init__(self,
             model: AcquisitionFunctionNet,
             train_X: Optional[Tensor]=None,
             train_Y: Optional[Tensor]=None,
             gp_model: Optional[GPyTorchModel]=None):  # NEW parameter
    super().__init__()
    # ... existing code ...
    self.gp_model = gp_model  # NEW
```

**Changes to `AcquisitionFunctionNetModel.forward`:**

```python
def forward(self, X: Tensor, **kwargs) -> Tensor:
    if self.train_X is None:
        raise RuntimeError("Cannot make predictions without conditioning on data.")

    # NEW: Compute GP posterior if model expects it
    if _nn_expects_gp_posterior_input(self.model):
        if self.gp_model is None:
            raise RuntimeError("NN expects GP posterior input but no GP model was provided")

        # Compute posterior for the candidate points X
        with torch.no_grad():
            posterior = self.gp_model.posterior(X)
            gp_posterior_mu = posterior.mean  # shape (n_cand, 1)
            gp_posterior_log_sigma = posterior.variance.sqrt().log()  # shape (n_cand, 1)

        kwargs['gp_posterior_mu'] = gp_posterior_mu
        kwargs['gp_posterior_log_sigma'] = gp_posterior_log_sigma

    ret = self.model(self.train_X, self.train_Y, X, **kwargs)
    assert ret.shape[:-1] == X.shape[:-1]
    return ret
```

### 4. Configuration and Command-Line Interface

#### 4.1 Add Command-Line Argument

**Location:** `nn_af/acquisition_function_net_save_utils.py` - function `_get_run_train_parser()`

Add to the `nn_architecture_group`:

```python
nn_architecture_group.add_argument(
    '--input_gp_posterior',
    action='store_true',
    help=('Whether to input GP posterior (mu, log_sigma) to the NN final MLP. '
          'If enabled, a GP will be fitted using MAP to the history, and '
          'the posterior mean and log-standard-deviation at each candidate '
          'point will be concatenated with the features before the final MLP. '
          'Default is False.')
)
```

#### 4.2 Update Model Creation

**Location:** `nn_af/acquisition_function_net_save_utils.py` - function `_get_model()`

Update the head initialization parameters:

```python
af_head_init_params = dict(
    hidden_dims=hidden_dims,
    activation="relu",
    layer_norm_before_end=False,
    layer_norm_at_end=False,
    dropout=args.dropout,
    input_gp_posterior=args.input_gp_posterior,  # NEW
)
```

This applies to all acquisition function types (Gittins, MSE EI, Policy Gradient).

#### 4.3 Update Training Configuration

**Location:** `nn_af/acquisition_function_net_save_utils.py` - function `_get_training_config()`

No changes needed here - the architecture parameter is saved as part of the model's init dict, not the training config.

### 5. YAML Configuration Support

**Location:** `config/train_acqf.yml`

Add to the architecture section:

```yaml
architecture:
  values:
    # ... existing values ...
    input_gp_posterior:
      - false
      - true
```

## Implementation Plan

### Phase 1: Core NN Architecture Changes
1. Add `input_gp_posterior` property to `AcquisitionFunctionHead` base class
2. Implement in `AcquisitionFunctionNetFinalMLP`
   - Add parameter to `__init__`
   - Adjust `input_dim` for the Dense layer
   - Add validation and concatenation in `forward()`
3. Implement in `AcquisitionFunctionNetFinalMLPSoftmaxExponentiate`
4. Propagate through `TwoPartAcquisitionFunctionNet`, `GittinsAcquisitionFunctionNet`, `ExpectedImprovementAcquisitionFunctionNet`

### Phase 2: Training Infrastructure
1. Implement helper function `_nn_expects_gp_posterior_input()` in `train_acquisition_function_net.py`
2. Implement `_replace_with_fitted_rbf_gp_models()` function
   - Create RBF GP models using `get_gp()`
   - Fit to each item's history using MAP
   - Replace existing models (or add new ones) on items in-place
   - **Important:** If dataset didn't have models initially, mark it as `has_models=True`
3. Implement `_compute_gp_posterior_batch()` function
4. Modify `train_acquisition_function_net()` to:
   - Create a copy of datasets if they have models (using `dataset.copy_with_new_size(None)`)
   - Call `_replace_with_fitted_rbf_gp_models()` on the copy (or original if no models)
5. Modify `train_or_test_loop()` to compute and pass GP posterior

### Phase 3: BO Loop Integration
1. Add or import `_nn_expects_gp_posterior_input()` helper in `bayesopt.py`
2. Add `_fit_gp_for_posterior()` method to `NNAcquisitionOptimizer`
3. Modify `NNAcquisitionOptimizer.get_model()` to fit GP when needed
4. Add `gp_model` parameter to `AcquisitionFunctionNetModel.__init__`
5. Modify `AcquisitionFunctionNetModel.forward()` to compute and pass GP posterior

### Phase 4: Configuration and CLI
1. Add command-line argument `--input_gp_posterior`
2. Update `_get_model()` to pass the parameter to head init
3. Add to YAML config files
4. Test with example configurations

### Phase 5: Testing and Validation
1. Unit tests for NN forward pass with/without GP posterior
2. Integration tests for training with GP posterior input
3. Integration tests for BO loops with GP posterior input
4. Validation experiments comparing with/without GP posterior input

## Testing Strategy

### Unit Tests
1. Test NN `forward()` with `input_gp_posterior=True` and valid inputs
2. Test NN `forward()` with `input_gp_posterior=True` and missing inputs (should raise ValueError)
3. Test NN `forward()` with `input_gp_posterior=False` and GP inputs provided (should raise ValueError)
4. Test shape consistency of GP posterior computation

### Integration Tests
1. Train a small NN with `input_gp_posterior=True` on a small dataset
2. Run a BO loop with `input_gp_posterior=True`
3. Verify models save and load correctly with the new parameter
4. Test backwards compatibility (models without this feature still work)

### Validation Experiments
1. Compare BO performance with/without GP posterior input on synthetic functions
2. Analyze training curves to see if GP posterior helps convergence
3. Visualize learned acquisition functions to understand the effect

## Backwards Compatibility

- Default value of `input_gp_posterior=False` ensures existing code continues to work
- Models trained without this feature can still be loaded and used
- No changes to existing configurations required
- The feature is purely additive - no breaking changes

## Potential Issues and Mitigations

### Issue 1: GP Fitting Failures
**Mitigation:** Wrap `fit_gpytorch_mll()` in try-except, fall back to initial hyperparameters with warning

### Issue 2: Computational Overhead
**Mitigation:**
- Pre-fit GPs once per dataset item (not per batch/epoch)
- Posterior computation is fast (linear algebra operation)
- Can be disabled by setting `input_gp_posterior=False`

### Issue 3: Memory Overhead
**Mitigation:**
- Store only one GP model per dataset item
- Posterior computation doesn't require storing full covariance

### Issue 4: Dataset Copying and Model Replacement
**Mitigation:**
- Use built-in `dataset.copy_with_new_size(None)` method to copy dataset when it has models
- This method creates a proper copy by re-initializing with same parameters
- If dataset initially had no models, need to properly mark `dataset.has_models = True` after replacement
- Test that model replacement works correctly for both GP datasets and non-GP datasets

## Open Questions

1. **Should we support MLE instead of MAP?**
   - Proposal: Default to MAP (keep priors), add optional flag for MLE if needed later

2. **Should we cache GP hyperparameters across epochs?**
   - Proposal: Fit once before training starts, don't refit per epoch

3. **What if the dataset outcome transform differs from GP outcome transform?**
   - Proposal: Use same outcome transform (Standardize) for both dataset and GP fitting

4. **How to properly mark dataset as `has_models=True` after replacing/adding models?**
   - Need to investigate `dataset_with_models.py` to understand the relationship between `has_models` property and internal state
   - The property checks for `_model_sampler` attribute - may need to create a dummy `RandomModelSampler` or find another approach
   - Only relevant when dataset initially had `has_models=False` - if it was already True, replacement maintains that state

## Files to Modify

1. `nn_af/acquisition_function_net.py` (~150 lines of changes)
2. `nn_af/train_acquisition_function_net.py` (~150 lines of changes)
3. `nn_af/acquisition_function_net_save_utils.py` (~20 lines of changes)
4. `bayesopt/bayesopt.py` (~50 lines of changes)
5. `config/train_acqf.yml` (~5 lines of changes)

**Total estimated changes:** ~375 lines of code

## Timeline Estimate

- Phase 1: 2-3 hours
- Phase 2: 3-4 hours
- Phase 3: 2-3 hours
- Phase 4: 1 hour
- Phase 5: 3-4 hours

**Total estimated time:** 11-15 hours

## References

- BoTorch GP posterior computation: `botorch.models.model.Model.posterior()`
- GP fitting with MAP: `botorch.fit.fit_gpytorch_mll()` with priors
- Existing GP fitting code: `bayesopt/bayesopt.py:442-460`, `utils/exact_gp_computations.py:115-126`
