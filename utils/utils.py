from abc import abstractmethod, ABC
from collections import defaultdict
import hashlib
import itertools
import os
import random
import re
import json
import inspect
import copy
import math
from typing import Any, Set, TypeVar, Iterable, Sequence, List, Tuple, Dict, Optional, Union
# Python 3.11+: can do "from typing import Self"
# Before Python 3.11: need to do the following:
from typing_extensions import Self
import warnings
from functools import partial, lru_cache

import numpy as np
from scipy.optimize import root_scalar
import torch

torch.set_default_dtype(torch.float64)
from torch import Tensor

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.exceptions.errors import (
    BotorchTensorDimensionError,
    InputDataError,
)
from botorch.exceptions.warnings import (
    _get_single_precision_warning,
    BotorchTensorDimensionWarning,
    InputDataWarning,
)
from botorch.posteriors import Posterior
from botorch.models.transforms.outcome import OutcomeTransform, Standardize, Log, Power
from torch.nn import ModuleList

from botorch.models.model import Model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel, BatchedMultiOutputGPyTorchModel
from botorch.acquisition.analytic import _ei_helper


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print("Current device:", current_device)
    print("Current device name:", torch.cuda.get_device_name(current_device))


class _InverseOutcomeTransform(OutcomeTransform):
    def __init__(self, transform: OutcomeTransform):
        super().__init__()
        if not isinstance(transform, OutcomeTransform):
            raise ValueError("transform must be a OutcomeTransform instance")
        self._original_transform = transform
    
    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self._original_transform.untransform(Y, Yvar)

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        return self.__class__(self._original_transform.subset_output(idcs))

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self._original_transform.forward(Y, Yvar)
    
    @property
    def _is_linear(self) -> bool:
        return self._original_transform._is_linear

class _Unstandardize(_InverseOutcomeTransform):
    def __init__(self, standardizer: Standardize):
        super().__init__(standardizer)

        if not isinstance(standardizer, Standardize):
            raise ValueError("standardizer must be a Standardize instance")
        if not standardizer._is_trained:
            raise RuntimeError(
            "Can only invert a Standardize if it has been called on some outcome data")

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        standardizer = self._original_transform
        assert standardizer._is_trained
        new_tf = standardizer.__class__(
            m=standardizer._m,
            outputs=standardizer._outputs,
            batch_shape=standardizer._batch_shape,
            min_stdv=standardizer._min_stdv)

        new_stdvs = 1 / standardizer.stdvs
        new_tf.stdvs = new_stdvs
        new_tf._stdvs_sq = new_stdvs.pow(2)
        
        new_tf.means = -new_stdvs * standardizer.means
        new_tf._is_trained = standardizer._is_trained
        return new_tf.untransform_posterior(posterior)


class Affine(OutcomeTransform):
    def __init__(
        self,
        bias: Tensor,
        weight: Tensor,
        outputs: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        if bias.dim() == 0:
            bias = bias.unsqueeze(-1)
        if weight.dim() == 0:
            weight = weight.unsqueeze(-1)
        m = bias.shape[-1]
        if m != weight.shape[-1]:
            raise ValueError(
                "Both bias and weight should have the same output dimension m.")
        batch_shape = bias.shape[:-1]
        if batch_shape != weight.shape[:-1]:
            raise ValueError(
                "Both bias and weight should have the same batch shape.")
        
        # shape: batch_shape x 1 x m
        means = bias.unsqueeze(-2)
        stdvs = weight.unsqueeze(-2)

        standardize = Standardize(
            m=m, outputs=outputs, batch_shape=batch_shape)
        standardize.means = means
        standardize.stdvs = stdvs
        standardize._stdvs_sq = stdvs.pow(2)
        # put in eval mode -- no updating the means & stdvs
        standardize.eval()
        standardize._is_trained = torch.tensor(True)

        self._transform = _Unstandardize(standardize)
    
    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self._transform.forward(Y, Yvar)

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        ret = object.__new__(type(self))
        super(type(self), ret).__init__() # do the super().__init__() as in __init__
        ret._transform = self._transform.subset_output(idcs)
        return ret

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self._transform.untransform(Y, Yvar)
    
    @property
    def _is_linear(self) -> bool:
        return True

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        return self._transform.untransform_posterior(posterior)

    def _invert(self):
        ret = object.__new__(type(self))
        super(type(self), ret).__init__() # do the super().__init__() as in __init__
        ret._transform = self._transform._original_transform
        return ret


class Exp(_InverseOutcomeTransform):
    def __init__(self, outputs: Optional[List[int]] = None) -> None:
        super().__init__(Log(outputs))
    
    @classmethod
    def from_log(cls, log: Log):
        if not isinstance(log, Log):
            raise ValueError("log must be a Log instance")
        x = cls.__new__(cls)
        _InverseOutcomeTransform.__init__(x, log)
        return x
    
    # Could also implement untransform_posterior but it would be some work.


# BoTorch has ChainedOutcomeTransform but that's a dict
# but I don't care about having names for my transforms
# so I implemented a list version (basically the same)
class ChainedOutcomeTransformList(OutcomeTransform, ModuleList):
    r"""An outcome transform representing the chaining of individual transforms"""

    def __init__(self, transforms: Iterable[OutcomeTransform]) -> None:
        r"""Chaining of outcome transforms.

        Args:
            transforms: The transforms to chain. Internally, the names of the
                kwargs are used as the keys for accessing the individual
                transforms on the module.
        """
        super().__init__(transforms)

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Transform the outcomes in a model's training targets

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        for tf in self:
            Y, Yvar = tf.forward(Y, Yvar)
        return Y, Yvar

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        return self.__class__(
            [tf.subset_output(idcs=idcs) for tf in self]
        )

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform previously transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transfomred training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the training targets (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        for tf in reversed(self):
            Y, Yvar = tf.untransform(Y, Yvar)
        return Y, Yvar

    @property
    def _is_linear(self) -> bool:
        """
        A `ChainedOutcomeTransform` is linear only if all of the component transforms
        are linear.
        """
        return all((octf._is_linear for octf in self))

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform a posterior

        Args:
            posterior: A posterior in the transformed space.

        Returns:
            The un-transformed posterior.
        """
        for tf in reversed(self):
            posterior = tf.untransform_posterior(posterior)
        return posterior


def get_standardized_exp_transform(sigma: float, device=None):
    """Returns an appropriately scaled exp transform that works as follows:
    Given an input y ~ N(0,1),
    -- First calculate z = exp(sigma * y)
    -- Next, shift and scale z as z' = a + b z
    where a and b are values, dependent on sigma,
    such that the result has E(z') = 0 and Var(z') = 1."""
    transform_1 = Affine(
        bias=torch.tensor(0.0, device=device),
        weight=torch.tensor(sigma, device=device)
    )
    tmp1 = 1 / math.sqrt(math.expm1(sigma**2))
    tmp2 = math.exp(-0.5 * sigma**2) * tmp1
    transform_3 = Affine(
        bias=torch.tensor(-tmp1, device=device),
        weight=torch.tensor(tmp2, device=device)
    )
    return ChainedOutcomeTransformList(
        [transform_1, Exp(), transform_3]
    )


def _get_base_transforms(transform: OutcomeTransform):
    if isinstance(transform, ChainedOutcomeTransformList):
        ret = []
        for tf in transform:
            ret.extend(_get_base_transforms(tf))
        return ret
    return [transform]


def concatenate_outcome_transforms(*transforms: OutcomeTransform) -> OutcomeTransform:
    base_transforms = _get_base_transforms(ChainedOutcomeTransformList(transforms))
    return ChainedOutcomeTransformList(base_transforms)


def invert_outcome_transform(transform: OutcomeTransform):
    if isinstance(transform, ChainedOutcomeTransformList):
        return ChainedOutcomeTransformList([
            invert_outcome_transform(tf) for tf in reversed(transform)
        ])
    
    if isinstance(transform, _InverseOutcomeTransform):
        # This handles _Unstandardize and Exp automatically
        return transform._original_transform
    
    if isinstance(transform, Standardize):
        return _Unstandardize(transform)
    
    if isinstance(transform, Affine):
        return transform._invert()
    
    if isinstance(transform, Log):
        return Exp.from_log(transform)
    
    if isinstance(transform, Power):
        ret = Power(1.0 / transform.power, transform._outputs)
        ret.train(transform.training) # just in case, probably doesn't matter
        return ret
    
    # fallback
    return _InverseOutcomeTransform(transform)


def add_outcome_transform(gp, octf):
    octf_of_gp = invert_outcome_transform(octf)
    if hasattr(gp, 'outcome_transform'):
        gp.outcome_transform = concatenate_outcome_transforms(
            octf_of_gp, gp.outcome_transform)
    else:
        gp.outcome_transform = octf_of_gp
    return gp # Could alternatively return None; either would work


def get_kernel(
        dimension: int,
        kernel: str,
        add_priors: bool,
        lengthscale: float,
        device=None):
    r"""Constructs a kernel for Gaussian Processes with optional priors.

    Args:
        dimension (int):
            The number of dimensions for the kernel.
        kernel (str):
            The type of kernel to use. Options are 'RBF', 'Matern32', or 'Matern52'.
        add_priors (bool):
            If True, priors will be added to the kernel parameters.
        lengthscale (float):
            The lengthscale parameter for the kernel.
        device (optional):
            The device on which to place the tensors (e.g., 'cpu' or 'cuda').

    Returns:
        kernel: A kernel object configured with the specified parameters.
    """
    base_kernel_kwargs = dict(
        ard_num_dims=dimension,
        batch_shape=torch.Size()
    )
    if add_priors:
        alpha = 3.0
        # lengthscale == alpha / beta
        beta = alpha / lengthscale
        base_kernel_kwargs['lengthscale_prior'] = GammaPrior(alpha, beta)

    if kernel == 'RBF':
        base_kernel = RBFKernel(**base_kernel_kwargs)
    else:
        if kernel == 'Matern32':
            nu = 1.5
        elif kernel == 'Matern52':
            nu = 2.5
        else:
            raise ValueError(f"Invalid kernel {kernel}")
        base_kernel = MaternKernel(nu=nu, **base_kernel_kwargs)

    # Set the lengthscale. Note: This automatically sets it to this vaue for
    # all components.
    base_kernel.lengthscale = torch.tensor(lengthscale, device=device)

    scale_kernel_kwargs = dict(
        base_kernel=base_kernel,
        batch_shape=torch.Size()
    )
    if add_priors:
        # NOTE: This is not exactly what we want if outcome_transform=exp,
        # because in that case, the implied mean and std of the outcome value
        # will *not* be 0 and outputscale if there is a prior on outputscale.
        # Ideally, we would do outcome transform and then put scale kernel
        # with scale prior rather than scale kernel with scale prior and then
        # outcome transform, but that's too difficult to do with the current
        # setup.
        alpha = 3.0
        mean = 1.0
        scale_kernel_kwargs['outputscale_prior'] = GammaPrior(alpha, alpha / mean)
    kernel = ScaleKernel(**scale_kernel_kwargs)
    kernel.outputscale = torch.tensor(1.0, device=device)
    return kernel


def get_gp(train_X:Optional[Tensor]=None,
           train_Y:Optional[Tensor]=None,
           dimension:Optional[int]=None,
           observation_noise=False,
           likelihood=None,
           covar_module=None,
           mean_module=None,
           outcome_transform=None,
           input_transform=None,
           device=None):
    # Default: Matern 5/2 kernel with gamma priors on
    # lengthscale and outputscale, and noise level also if
    # observation_noise.
    
    if train_X is None and train_Y is None:
        has_data = False
        if dimension is None:
            raise ValueError(
                "dimension should be specified if train_X, train_Y are not specified")
        # SingleTaskGP doesn't support initializing with no data.
        # So initialize it like this.
        train_X = torch.zeros(2, dimension, device=device)
        # train_Y = torch.zeros(2, 1, device=device)
        # Make train_Y have unit variance
        # (unbiased 1/(n-1) calculation as torch.std uses)
        # so that SingleTaskGP constructor doesn't give warning.
        train_Y = torch.tensor([
            [-torch.sqrt(torch.tensor(0.5, device=device))],
            [torch.sqrt(torch.tensor(0.5, device=device))]])
    elif train_X is not None and train_Y is not None:
        has_data = True
        if dimension is not None:
            raise ValueError(
                "dimension should not be specified if train_X, train_Y are specified")
    else:
        raise ValueError("Either train_X and train_Y, or neither, should be specified")

    if type(observation_noise) is not bool:
        raise ValueError("observation_noise should be a bool")
    if not observation_noise:
        if likelihood is not None:
            raise ValueError(
                "likelihood should not be specified if observation_noise=False")
        likelihood = GaussianLikelihood(
            noise_prior=None, batch_shape=torch.Size(),
            noise_constraint=GreaterThan(
                0.0, transform=None, initial_value=0.0
            )
        )
        # Make it so likelihood can't change by gradient-based optimization.
        likelihood.noise_covar.raw_noise.requires_grad_(False)
    model = SingleTaskGP(
        train_X, train_Y, likelihood=likelihood, covar_module=covar_module,
        mean_module=mean_module, outcome_transform=outcome_transform,
        input_transform=input_transform).to(device)
    if not has_data:
        model.remove_data()
    return model


########### Code for Model.set_train_data_with_transforms function #############
def _transform_Y(self, Y: Optional[Tensor]=None, train:Optional[bool]=None):
    if Y is not None and hasattr(self, "outcome_transform"):
        outcome_transform = self.outcome_transform
        if train is None:
            train = outcome_transform.training
        switch_training = train != outcome_transform.training
        if switch_training:
            outcome_transform.train(not outcome_transform.training)
        Y, _ = outcome_transform(Y)
        if switch_training:
            outcome_transform.train(not outcome_transform.training)
    return Y
Model._transform_Y = _transform_Y


def _set_train_data_with_X_transform(self,
                                     X: Optional[Tensor]=None,
                                     Y: Optional[Tensor]=None,
                                     strict=True,
                                     train:Optional[bool]=None):
    # If only one is given, then it does make sense to make sure it has the
    # right shape...
    strict = strict or ((X is None) ^ (Y is None))
    self.set_train_data(inputs=X, targets=Y, strict=strict)

    if X is not None and hasattr(self, "input_transform"):
        assert self.training == (not self._has_transformed_inputs)
        if not self.training: # eval mode
            # Then self._has_transformed_inputs == True,
            # and self._original_train_inputs is wrong.
            if train is None or train == self.training: # train = None or False
                # We can set _has_transformed_inputs = False
                # to make it so that _set_transformed_inputs sets the
                # _original_train_inputs, transforms the inputs, and sets
                # self._has_transformed_inputs = True again.
                self._has_transformed_inputs = False
                self._set_transformed_inputs()
            else: # train = True
                # Same as above code _set_transformed_inputs except do learn
                # the bounds or coefficients
                self._original_train_inputs = self.train_inputs[0]
                # Learn any input transform parameters
                was_eval = not self.input_transform.training
                if was_eval:
                    self.input_transform.train()
                with torch.no_grad():
                    X_tf = self.input_transform(self.train_inputs[0])
                if was_eval:
                    self.input_transform.eval()
                self.set_train_data(X_tf, strict=False)
        else: # train mode
            if train is None or train == self.training: # train = None or True
                # Learn any input transform parameters
                # (Nothing will be done with this, just need to call it)
                with torch.no_grad():
                    X_tf = self.input_transform(self.train_inputs[0])

            # If in training mode, then train() doesn't do anything since
            # _has_transformed_inputs == False, and eval() makes
            # _original_train_inputs set correct. So don't need to do anything.
            # However, it doesn't hurt to just set it anyway for consistency:
            if self._original_train_inputs is not None:
                self._original_train_inputs = self.train_inputs[0]
Model._set_train_data_with_X_transform = _set_train_data_with_X_transform


def _set_train_data_with_transforms_Model(self,
                                          X: Optional[Tensor]=None,
                                          Y: Optional[Tensor]=None,
                                          strict=True,
                                          train:Optional[bool]=None):
    Y = self._transform_Y(Y, train)
    self._set_train_data_with_X_transform(X, Y, strict, train)
Model.set_train_data_with_transforms = _set_train_data_with_transforms_Model


def _set_train_data_with_transforms_GPyTorchModel(self,
                                                  X: Optional[Tensor]=None,
                                                  Y: Optional[Tensor]=None,
                                                  strict=True,
                                                  train:Optional[bool]=None):
    Y = self._transform_Y(Y, train)
    
    if not (X is None and Y is None):
        proposed_X = X if X is not None else self.train_inputs[0]

        if Y is None:
            proposed_Y = self.train_targets
            if (proposed_X.dim() - proposed_Y.dim() == 1) and \
                (proposed_X.shape[:-1] == proposed_Y.shape):
                # Then the targets doesn't have an explicit output dimension
                proposed_Y = proposed_Y.unsqueeze(-1)
        else:
            proposed_Y = Y
        
        self._validate_tensor_args(X=proposed_X, Y=proposed_Y)

    self._set_train_data_with_X_transform(X, Y, strict, train)
GPyTorchModel.set_train_data_with_transforms = \
    _set_train_data_with_transforms_GPyTorchModel


def _set_train_data_with_transforms_BatchedMultiOutputGPyTorchModel(self,
                                                  X: Optional[Tensor]=None,
                                                  Y: Optional[Tensor]=None,
                                                  strict=True,
                                                  train:Optional[bool]=None):
    Y = self._transform_Y(Y, train)
    
    both_are_given = X is not None and Y is not None
    neither_is_given = X is None and Y is None
    
    if both_are_given:
        self._validate_tensor_args(X=X, Y=Y)
        self._set_dimensions(train_X=X, train_Y=Y)
        X, Y, _ = self._transform_tensor_args(X=X, Y=Y)
    elif not neither_is_given:
        if X is not None:
            # X is given but not Y

            # Try to do something akin to _validate_tensor_args
            if X.dim() < 2:
                raise BotorchTensorDimensionError(
                    f"X has shape {X.shape} but needs to have at least 2 dimensions"
                )
            given_input_batch_shape = X.shape[:-2]
            if given_input_batch_shape != self.batch_shape:
                raise BotorchTensorDimensionError(
                    f"Expected X to have batch shape {self.batch_shape} but has"
                    f" batch shape {given_input_batch_shape}"
                )
            
            # Akin to _transform_tensor_args
            if self.num_outputs > 1:
                X = X.unsqueeze(-3).expand(
                    X.shape[:-2] + torch.Size([self.num_outputs]) + X.shape[-2:]
                )
        elif Y is not None:
            # Y is given but not X

            # Try to do something akin to _validate_tensor_args
            if Y.dim() < 2:
                raise BotorchTensorDimensionError(
                    f"Y has shape {Y.shape} but needs to have at least 2 dimensions"
                )
            given_input_batch_shape = Y.shape[:-2]
            given_num_outputs = Y.shape[-1]
            if given_input_batch_shape != self.batch_shape:
                raise BotorchTensorDimensionError(
                    f"Expected Y to have batch shape {self.batch_shape} but has"
                    f" batch shape {given_input_batch_shape}"
                )
            if given_num_outputs != self.num_outputs:
                raise BotorchTensorDimensionError(
                    f"Expected Y to have number of outputs {self.num_outputs} but has"
                    f" number of outputs {given_num_outputs}"
                )
            
            # Akin to _transform_tensor_args
            if self.num_outputs > 1:
                Y = Y.transpose(-1, -2)
            else:
                Y = Y.squeeze(-1)
    
    self._set_train_data_with_X_transform(X, Y, strict, train)

    if not both_are_given and not neither_is_given:
        # In the case that only one of X or Y was provided, check the dtypes
        # akin to the last part of _validate_tensor_args
        new_X, new_Y = self.train_inputs[0], self.train_targets
        if new_X.dtype != new_Y.dtype:
            raise InputDataError(
                "Expected all inputs to share the same dtype. Got "
                f"{new_X.dtype} for X, {new_Y.dtype} for Y."
            )
        if new_X.dtype != torch.float64:
            warnings.warn(
                _get_single_precision_warning(str(new_X.dtype)),
                InputDataWarning,
                stacklevel=3,  # Warn at model constructor call.
            )
BatchedMultiOutputGPyTorchModel.set_train_data_with_transforms = \
    _set_train_data_with_transforms_BatchedMultiOutputGPyTorchModel
################################################################################
def _condition_on_observations_with_transforms(
        self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
    assert not self.training, "Model should be in eval mode."
    # Since the model is in eval mode, the inputs are already transformed.
    
    # condition_on_observations already takes care of the outcome transform
    # but not the input transform. So we need to do that here.
    fantasy_model = self.condition_on_observations(
        X=self.transform_inputs(X), Y=Y, **kwargs)

    if hasattr(fantasy_model, "input_transform"):
        assert fantasy_model._has_transformed_inputs
        
        # Certainly not ideal but it should get the job done.
        # It doesn't really matter though if you're not gonna go back to
        # train mode...I'm not sure why BoTorch has this things.
        fantasy_model._original_train_inputs = \
            fantasy_model.input_transform.untransform(fantasy_model.train_inputs[0])
    return fantasy_model
Model.condition_on_observations_with_transforms = \
    _condition_on_observations_with_transforms


from botorch.sampling.pathwise.utils import get_train_inputs

def _remove_data(self):
    train_inputs = get_train_inputs(self, transformed=False)
    if len(train_inputs) > 1:
        raise NotImplementedError("Model has multiple inputs")
    train_input = train_inputs[0]
    
    # (shape is batch_shape x n x d), so remove the n dimension
    new_train_inputs_shape = list(train_input.shape)
    new_train_inputs_shape[-2] = 0
    # Make zeros
    new_train_inputs = torch.zeros(
        new_train_inputs_shape,
        dtype=train_input.dtype,
        device=train_input.device)
    self.train_inputs = (new_train_inputs,)

    if self._original_train_inputs is not None:
        self._original_train_inputs = new_train_inputs.clone()
    
    train_targets = self.train_targets
    train_targets_shape = list(train_targets.shape)
    assert train_targets_shape[-1] == train_input.size(-2)
    train_targets_shape[-1] = 0
    new_train_targets = torch.zeros(
        train_targets_shape,
        dtype=train_targets.dtype,
        device=train_targets.device)
    self.train_targets = new_train_targets

    if hasattr(self, "prediction_strategy"):
        self.prediction_strategy = None
Model.remove_data = _remove_data


def get_dimension(model: Model):
    (train_X,) = get_train_inputs(model, transformed=False)
    return train_X.shape[-1]


def remove_priors(module: gpytorch.module.Module) -> list:
    """Removes all priors from a GPyTorch Module, and also returns the
    equivalent of list(module.named_priors()) for convenience to be used with
    add_priors."""
    named_priors_tuple_list = []
    for name, parent_module, prior, closure, inv_closure in module.named_priors():
        named_priors_tuple_list.append(
            (name, parent_module, prior, closure, inv_closure))
        prior_variable_name = name.rsplit('.', 1)[-1]
        delattr(parent_module, prior_variable_name)
        del parent_module._priors[prior_variable_name]
    return named_priors_tuple_list


def add_priors(named_priors_tuple_list: List[Tuple]):
    """Add priors to a GPyTorch Module. Note that the module itself doesn't need
    to be specified because the parent modules of the priors (the children of
    the module) are already given.
    
    Example:
    ```
    # remove the priors
    named_priors_tuple_list = remove_priors(module)
    # add the priors back
    add_priors(named_priors_tuple_list)
    """
    for name, parent_module, prior, closure, inv_closure in named_priors_tuple_list:
        prior_variable_name = name.rsplit('.', 1)[-1]
        parent_module.register_prior(prior_variable_name, prior, closure, inv_closure)


def calculate_batch_improvement(y_hist_batch: torch.Tensor, y_cand_batch: Tensor, 
                                hist_mask: Optional[Tensor] = None, 
                                cand_mask: Optional[Tensor] = None):
    """Calculate the improvement values for a batch of y_hist and y_cand tensors with
    optional masking.
    
    Args:
        y_hist_batch (Tensor):
            Tensor of shape (batch_size, max_n_hist, 1) containing historical y values.
        y_cand_batch (Tensor):
            Tensor of shape (batch_size, max_n_cand, 1) containing candidate y values.
        hist_mask (Optional[Tensor]):
            Boolean tensor of shape (batch_size, max_n_hist, 1) indicating valid y
            values. If None, all values in y_hist_batch are considered valid.
        cand_mask (Optional[Tensor]):
            Boolean tensor of shape (batch_size, max_n_cand, 1) indicating valid y
            values. If None, all values in y_cand_batch are considered valid.
    
    Returns:
        Tensor: Tensor of improvement values with the same shape as y_cand_batch.
    """
    if hist_mask is None:
        # Special case: no hist_mask, all values are valid
        best_f_batch = y_hist_batch.amax(dim=1, keepdim=True)
    else:
        # General case: use the hist_mask to find the valid values
        y_hist_batch_masked = y_hist_batch.masked_fill(~hist_mask, float('-inf'))
        best_f_batch = y_hist_batch_masked.amax(dim=1, keepdim=True)
    
    improvement_values_batch = torch.nn.functional.relu(y_cand_batch - best_f_batch,
                                                        inplace=True)
    
    # Ensure padding with zeros where there were invalid (masked) values
    if cand_mask is not None:
        improvement_values_batch = improvement_values_batch * cand_mask
    
    return improvement_values_batch


def expand_dim(tensor, dim, k):
    new_shape = list(tensor.shape)
    new_shape[dim] = k
    return tensor.expand(*new_shape)


MIN_STDV = 1e-8

def standardize_y_hist(y_hist, hist_mask):
    max_n_hist = y_hist.size(-2)

    if hist_mask is None:
        means = y_hist.mean(dim=-2, keepdim=True)
    else:
        n_hists = hist_mask.sum(dim=-2, keepdim=True).double()
        if (n_hists == 0).any():
            raise ValueError(f"Can't standardize with no observations. {n_hists=}.")
        # shape batch_size x 1 x 1
        means = y_hist.sum(dim=-2, keepdim=True) / n_hists

    if max_n_hist < 1:
        raise ValueError(f"Can't standardize with no observations. {y_hist.shape=}.")
    elif max_n_hist == 1:
        stdvs = torch.ones(
                (*y_hist.shape[:-2], 1, y_hist.shape[-1]),
                dtype=y_hist.dtype, device=y_hist.device)
    else:
        if hist_mask is None:
            stdvs = y_hist.std(dim=-2, keepdim=True)
        else:
            # shape batch_size x max_n_hist x 1
            # Need to mask it
            means_expanded = expand_dim(
                    means, -2, max_n_hist) * hist_mask
            # Since both y_hist and means_expanded are mask with zeros,
            # (0 - 0)^2 = 0 so this is also mask with zeros, good.
            squared_differences = torch.nn.functional.mse_loss(
                    y_hist, means_expanded, reduction='none')
            denominators = (n_hists - 1.0).where(n_hists == 1.0,
                                                 torch.full_like(n_hists, 1.0))
            variances = squared_differences.sum(dim=-2, keepdim=True) / denominators
            variances = variances.where(n_hists == 1.0, torch.full_like(variances, 1.0))
            stdvs = torch.sqrt(variances)
        
    stdvs = stdvs.where(stdvs >= MIN_STDV, torch.full_like(stdvs, 1.0))
    y_hist = (y_hist - means) / stdvs
    return y_hist, stdvs


def add_tbatch_dimension(x: Tensor, x_name: str):
    if x.dim() < 2:
        raise ValueError(
            f"{x_name} must have at least 2 dimensions,"
            f" but has only {x.dim()} dimensions."
        )
    return x if x.dim() > 2 else x.unsqueeze(0)


def uniform_randint(min_val, max_val):
    return torch.randint(min_val, max_val+1, (1,), dtype=torch.int32).item()


def get_uniform_randint_generator(min_val, max_val):
    return partial(uniform_randint, min_val, max_val)


def loguniform_randint(min_val, max_val, size=1, pre_offset=0.0, offset=0):
    if not (isinstance(min_val, int)
            and isinstance(max_val, int) and isinstance(offset, int)):
        raise ValueError("min_val, max_val, and offset must be integers")
    if not (1 <= min_val <= max_val):
        raise ValueError("min_val must be between 1 and max_val")
    if not (pre_offset >= 0):
        raise ValueError("pre_offset must be non-negative")

    min_log = torch.log(torch.tensor(min_val + pre_offset))
    max_log = torch.log(torch.tensor(max_val+1 + pre_offset))
    random_log = torch.rand(size) * (max_log - min_log) + min_log
    ret = (torch.exp(random_log) - pre_offset).to(dtype=torch.int32) + offset
    if torch.numel(ret) == 1:
        return ret.item()
    return ret


def get_loguniform_randint_generator(min_val, max_val, pre_offset=0.0, offset=0):
    return partial(loguniform_randint, min_val, max_val,
                   pre_offset=pre_offset, offset=offset)


def _int_linspace_naive(start, stop, num):
    return np.unique(np.round(np.linspace(start, stop, num)).astype(int))


@lru_cache(maxsize=128) # Not necessary but why not.
def int_linspace(start, stop, num):
    if not (isinstance(start, int) and isinstance(stop, int)):
        raise ValueError("start and stop should be integers")

    if num > stop - start + 1:
        raise ValueError('num must be less than or equal to stop - start + 1')
    ret = _int_linspace_naive(start, stop, num)
    length = len(ret)
    
    if length < num:
        sol = root_scalar(
            lambda x: len(_int_linspace_naive(start, stop, int(x))) - num,
            method='secant', x0=num, x1=2*num, xtol=1e-12, rtol=1e-12)
        
        k = int(sol.root)
        ret = _int_linspace_naive(start, stop, k)

        if len(ret) != num:
            if len(ret) > num:
                while len(ret) > num:
                    k -= 1
                    ret = _int_linspace_naive(start, stop, k)
            else:
                while len(ret) < num:
                    k += 1
                    ret = _int_linspace_naive(start, stop, k)

    return ret


def sanitize_file_name(file_name: str) -> str:
    # Define a dictionary of characters to replace
    replacements = {
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
    }

    # Replace the characters based on the replacements dictionary
    sanitized_name = ''.join(replacements.get(c, c) for c in file_name)

    # Remove characters that are non-printable or not allowed
    sanitized_name = re.sub(r'[^\x20-\x7E]', '', sanitized_name)

    # Remove all whitespace characters
    sanitized_name = re.sub(r'\s+', '', sanitized_name)

    return sanitized_name

def _to_str(x, include_space=False) -> str:
    sep = ', ' if include_space else ','
    if type(x) is dict:
        return '(' + sep.join(
            key + '=' + _to_str(value)
            for key, value in sorted(x.items())
        ) + ')'
    if type(x) is list:
        return '[' + sep.join(map(_to_str, x)) + ']'
    if type(x) is str:
        return x
    return repr(x)

def dict_to_str(d: Dict[str, Any], include_space=False) -> str:
    if type(d) is not dict:
        raise ValueError(f"Expected a dictionary, got a {type(d).__name__} "
                            f"with value {d!r}")
    return _to_str(d, include_space=include_space)[1:-1]

def dict_to_fname_str(d: Dict[str, Any]) -> str:
    return sanitize_file_name(dict_to_str(d))

def str_to_hash(s: str) -> str:
    return hashlib.sha256(s.encode('ascii')).hexdigest()

def dict_to_hash(d: Dict[str, Any]) -> str:
    return str_to_hash(dict_to_str(d))


def hash_gpytorch_module(module,
                         include_priors=True,
                         include_str=True,
                         hash_str=False):
    if not include_priors:
        named_priors_tuple_list = remove_priors(module)
    
    serialized_state_dict = convert_to_json_serializable(
        module.state_dict(), hash_gpytorch_modules=True,
        include_priors=include_priors, hash_include_str=False,
        hash_str=True)
    
    if hash_str:
        hashed_result = dict_to_hash({
            'module': module,
            'state_dict': serialized_state_dict
        })
    else:
        hashed_result = dict_to_hash(serialized_state_dict)
    
    if include_str:
        ret = f'{module!r}_{hashed_result}'
    else:
        ret = hashed_result

    if not include_priors:
        add_priors(named_priors_tuple_list)
    
    return ret


def convert_to_json_serializable(data,
                                 include_priors=True,
                                 hash_gpytorch_modules=True,
                                 hash_include_str=True,
                                 hash_str=False):
    
    if isinstance(data, dict):
        return {k: convert_to_json_serializable(
            v, include_priors, hash_gpytorch_modules, hash_include_str, hash_str)
            for k, v in data.items()}
    if isinstance(data, np.ndarray):
        return data.tolist()
    if torch.is_tensor(data):
        return data.cpu().numpy().tolist()
    if isinstance(data, (list, tuple)):
        return [convert_to_json_serializable(
            x, include_priors, hash_gpytorch_modules, hash_include_str, hash_str)
            for x in data]
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
    if isinstance(data, gpytorch.Module):
        if hash_gpytorch_modules:
            return hash_gpytorch_module(
                data, include_priors, hash_include_str, hash_str)
        else:
            if not include_priors:
                named_priors_tuple_list = remove_priors(data)
            ret = {
                'module': str(data),
                'state_dict': convert_to_json_serializable(
                    data.state_dict(), include_priors=True,
                    hash_gpytorch_modules=False)
            }
            if not include_priors:
                add_priors(named_priors_tuple_list)
            return ret
    return str(data)

def _json_serializable_to_numpy(data: Any, array_keys: Optional[set]=None):
    if isinstance(data, dict):
        return {
            k: np.array(v) if isinstance(v, list) and
            (array_keys is None or k in array_keys)
            else _json_serializable_to_numpy(v, array_keys)
            for k, v in data.items()
        }
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (list, tuple)):
        return [_json_serializable_to_numpy(x, array_keys) for x in data]
    return data

def json_serializable_to_numpy(data: Any,
                               array_keys: Optional[Union[list,tuple,set]]=None):
    if array_keys is not None:
        array_keys = set(array_keys)
    return _json_serializable_to_numpy(data, array_keys)


def to_device(tensor, device):
    if tensor is None or device is None:
        return tensor
    return tensor.to(device)


def save_json(data, fname, **kwargs):
    already_exists = os.path.exists(fname)
    r = random.randint(0, 1_000_000_000)
    save_fname = fname + f'{r}.tmp' if already_exists else fname
    try:
        # Ensure the directory exists
        dirname = os.path.dirname(fname)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)
        
        # Write data to the (possibly temporary) file
        with open(save_fname, 'w') as json_file:
            json.dump(data, json_file, **kwargs)

        if already_exists:
            # Replace the original file with the temporary file
            os.replace(save_fname, fname)
    except Exception as e:
        if os.path.exists(save_fname):
            # Remove the written file if an error occurs
            os.remove(save_fname)
        raise e


def load_json(fname, **kwargs):
    with open(fname, 'r') as json_file:
        return json.load(json_file, **kwargs)


def dict_to_cmd_args(params, equals=False) -> list[str]:
    parts = []
    for key, value in sorted(params.items()):
        # If the value is a boolean, only include it if True.
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
        elif value is not None:
            if equals:
                parts.append(f"--{key}={value}")
            else:
                parts.append(f"--{key}")
                parts.append(str(value))
    return parts


K = TypeVar('K')
V = TypeVar('V')

def combine_dicts(dict_list: List[Dict[K, V]]) -> Dict[K, V]:
    return {k: v for d in dict_list for k, v in d.items()}

def dict_product(*lists_of_dicts: List[Dict[K, V]]) -> List[Dict[K, V]]:
    """
    Generate all possible combinations of dictionaries from the input lists.

    This function takes multiple lists of dictionaries and returns a list of all
    possible combinations, where each combination is a union of one dictionary
    from each input list.
    If there are duplicate keys, the value from the later dictionary in the
    combination takes precedence.

    Args:
        *lists_of_dicts:
            Variable number of lists, where each list contains dictionaries.

    Returns:
        A list of dictionaries, each representing a unique combination of the
        input dictionaries.
    """
    options_product = itertools.product(*lists_of_dicts)
    return [combine_dicts(dicts) for dicts in options_product]

def combine_nested_dicts(*dicts : Dict[str, Dict[K, V]]) -> Dict[str, Dict[K, V]]:
    """Combine nested dictionaries to create all possible combinations of their
    contents.

    This function takes multiple dictionaries, where each dictionary contains
    string keys mapped to sub-dictionaries. It generates all possible
    combinations of these sub-dictionaries across the input dictionaries,
    creating new combined dictionaries.

    The keys of the resulting dictionary are created by joining the
    corresponding keys from the input dictionaries with ', '.

    Args:
        *dicts: Variable number of dictionaries. Each dictionary should have
                string keys and dictionary values.

    Returns:
        A dictionary where:
        - Keys are strings created by joining the keys of the input dictionaries.
        - Values are dictionaries created by combining the corresponding
          sub-dictionaries from the input dictionaries.

    Example:
        >>> d1 = {'A': {'x': 1}, 'B': {'x': 2}}
        >>> d2 = {'C': {'y': 3}, 'D': {'y': 4}}
        >>> combine_nested_dicts(d1, d2)
        {'A, C': {'x': 1, 'y': 3}, 'A, D': {'x': 1, 'y': 4},
         'B, C': {'x': 2, 'y': 3}, 'B, D': {'x': 2, 'y': 4}}
    """
    return {
        ', '.join(names): combine_dicts([d[n] for d, n in zip(dicts, names)])
        for names in itertools.product(*dicts)
    }


def assert_has_type(x: object, name: str, t: type):
    if not isinstance(x, t):
        raise ValueError(f"Expected {name} to have type {t.__name__}, "
                         f"but it is of type {x.__class__.__name__}")


def assert_all_have_type(values: list, name: str, t: type):
    for i, v in enumerate(values):
        if not isinstance(v, t):
            raise ValueError(
                f"Expected all elements of {name} to have type {t.__name__}, "
                f"but {name}[{i}] is of type {v.__class__.__name__}")


def aggregate_stats_list(stats_list: Union[list[dict[str]], list[np.ndarray],
                                           list[float], list[int], list[str],
                                             list[bool], list[None]]):
    assert_has_type(stats_list, "stats_list", list)
    stats0 = stats_list[0]
    if isinstance(stats0, dict):
        assert_all_have_type(stats_list, "stats_list", dict)
        return {
            key: aggregate_stats_list([
                stats[key] for stats in stats_list
            ]) for key in stats0.keys()
        }
    if isinstance(stats0, (np.ndarray, float, int, str, bool, type(None))):
        assert_all_have_type(stats_list, "stats_list", type(stats0))
        return np.array(stats_list)
    raise ValueError(f"stats_list must be a list of dicts or list of ndarrays, "
                     f"but got stats_list[0]={stats0}")


def group_by(items, group_function=lambda x: x):
    """
    Groups items by a grouping function
    parameters:
        items: iterable containing things
        group_function: function that gives the same result when called on two
            items in the same group
    returns: a dict where the keys are results of the function and values are
        lists of items that when passed to group_function give that key
    """
    group_dict = {}
    for item in items:
        value = group_function(item)
        if value in group_dict:
            group_dict[value].append(item)
        else:
            group_dict[value] = [item]
    return group_dict


def iterate_nested(d):
    for key, value in d.items():
        yield key, value
        if isinstance(value, dict):
            yield from iterate_nested(value)  # Recursively process the nested dict


def get_values(d):
    if type(d) is dict:
        for v in d.values():
            yield from get_values(v)
    else:
        yield d


def are_all_disjoint(sets: Sequence[Set]) -> bool:
    """Checks if all sets in a list are pairwise disjoint (have no common elements).

    Args:
        sets: A list of sets.

    Returns:
        True if all sets are disjoint, False otherwise.
    """
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if not sets[i].isdisjoint(sets[j]):
                return False
    return True


def _group_by_nested_attrs(items: List[dict[K, Any]],
                        attrs_groups_list: List[Set[K]],
                        dict_to_str_func,
                        return_single=False,
                        indices=None):
    if indices is None:
        indices = list(range(len(items)))

    if len(attrs_groups_list) == 0:
        if return_single and len(indices) == 1:
            return indices[0]
        return indices
    initial_attrs = attrs_groups_list[0]
    initial_grouped_items = {}
    to_add = []
    for idx in indices:
        item = items[idx]
        d = {k: item[k] for k in initial_attrs if k in item}
        d = {k: v for k, v in d.items() if v is not None}
        # if not d:
        #     # counts = group_by(
        #     #     item.keys(),
        #     #     lambda k: sum(other_item[k] != item[k]
        #     #                   for j, other_item in enumerate(items) if j != idx)
        #     # )
        #     # counts_sorted = sorted(counts.items(), reverse=True)
        #     # highest_mismatch_count, candidates =  counts_sorted[0]
        #     # key = str(candidates[0])
        #     # print("Found an item that would have an empty description. Using the "
        #     #       f"description {key} as it is different from the others "
        #     #       f"{highest_mismatch_count}/{len(items)} times.")
        #     key = ""
        key = dict_to_str_func(d)
        if set(d.keys()) == initial_attrs:
            if key in initial_grouped_items:
                initial_grouped_items[key]['items'].append(idx)
            else:
                initial_grouped_items[key] = {
                    'items': [idx],
                    'vals': d
                }
        else:
            to_add.append((key, idx, d))

    new_grouped_items = {}
    for key_to_add, idx_to_add, d_to_add in to_add:
        item_to_add = items[idx_to_add]
        count = 0
        for key, value in initial_grouped_items.items():
            if all(k not in item_to_add or item_to_add[k] == v for k, v in value['vals'].items()):
                initial_grouped_items[key]['items'].append(idx_to_add)
                count += 1
        if count == 0:
            if key_to_add in new_grouped_items:
                new_grouped_items[key_to_add]['items'].append(idx_to_add)
            else:
                new_grouped_items[key_to_add] = {
                    'items': [idx_to_add],
                    'vals': d_to_add
                }
    
    # initial_grouped_items = {key: value['items']
    #                         for key, value in initial_grouped_items.items()}
    initial_grouped_items.update(new_grouped_items)

    next_attrs = attrs_groups_list[1:]
    return {
        k: {
            'items': _group_by_nested_attrs(items, next_attrs, dict_to_str_func,
                                indices=v['items'], return_single=return_single),
            'vals': v['vals']
        }
        for k, v in initial_grouped_items.items()
    }


def group_by_nested_attrs(items: List[dict[K, Any]],
                        attrs_groups_list: List[Set[K]],
                        dict_to_str_func=dict_to_str,
                        add_extra_index=-1):
    if not are_all_disjoint(attrs_groups_list):
        raise ValueError("Attributes in the groups are not disjoint")    
    keys = set().union(*[set(item.keys()) for item in items])
    
    for attrs in attrs_groups_list:
        if not attrs.issubset(keys):
            warnings.warn(
                f"A group of attributes is not in the items: {attrs}")

    # Remove those that we don't have
    attrs_groups_list = [
        attrs & keys for attrs in attrs_groups_list
    ]
    
    vals_dict = {
        k: {item[k] for item in items if k in item}
        for k in keys
    }
    constant_keys = {k for k in keys if len(vals_dict[k]) == 1}

    attrs_groups_list = [z - constant_keys for z in attrs_groups_list]
    attrs_groups_list = [z for z in attrs_groups_list if len(z) != 0]
    if len(attrs_groups_list) == 0:
        raise ValueError("No attributes to group by")
    
    ret = _group_by_nested_attrs(items, attrs_groups_list, dict_to_str_func)
    
    nonconstant_keys = set()
    keys_in_all = {u for u in keys}

    for key, value in iterate_nested(ret):
        if not (key == "items" and isinstance(value, list)):
            continue
        itmz = value
        nonconstant_keys_item = set()
        in_all_keys_item = set()
        for k in keys:
            is_in_all = True
            vals_taken = set()
            for idx in itmz:
                item = items[idx]
                if k in item:
                    vals_taken.add(item[k])
                else:
                    is_in_all = False
            if is_in_all:
                in_all_keys_item.add(k)
            if len(vals_taken) > 1: # if non-constant
                nonconstant_keys_item.add(k)
        nonconstant_keys |= nonconstant_keys_item
        keys_in_all &= in_all_keys_item
    
    nonconstant_keys -= {"index"}
    keys_in_all -= {"index"}
    
    nonconstant_keys_in_all = nonconstant_keys & keys_in_all
    nonconstant_keys_not_in_all = nonconstant_keys - nonconstant_keys_in_all

    # print(f"{nonconstant_keys_in_all=}, {nonconstant_keys_not_in_all=}")

    if len(nonconstant_keys_in_all) != 0:
        attrs_groups_list = [nonconstant_keys_in_all] + attrs_groups_list
        return group_by_nested_attrs(
            items, attrs_groups_list, dict_to_str_func, add_extra_index=add_extra_index)

    if len(nonconstant_keys_not_in_all) != 0:
        attrs_groups_list[add_extra_index] |= nonconstant_keys_not_in_all

    return _group_by_nested_attrs(
        items, attrs_groups_list, dict_to_str_func,
        return_single=True), attrs_groups_list




def pad_tensor(vec, length, dim, add_mask=False):
    """Pads a tensor 'vec' to a size 'length' in dimension 'dim' with zeros.
    args:
        vec - tensor to pad
        length - the size to pad to in dimension 'dim'
        dim - dimension to pad
        add_mask - whether to return the mask as well

    returns:
        If add_mask=True, return a tuple (padded, mask).
        Otherwise, return the padded tensor only.
    """
    pad_size = length - vec.size(dim)
    if pad_size < 0:
        raise ValueError("Tensor cannot be padded to length less than it already is")
    
    pad_shape = list(vec.shape)
    pad_shape[dim] = pad_size
    if pad_size == 0: # Could pad with nothing but that's unnecessary
        padded = vec
    else:
        padding = torch.zeros(*pad_shape, dtype=vec.dtype, device=vec.device)
        padded = torch.cat([vec, padding], dim=dim)

    if add_mask:
        mask_true = torch.ones(vec.shape, dtype=torch.bool, device=vec.device)
        mask_false = torch.zeros(*pad_shape, dtype=torch.bool, device=vec.device)
        mask = torch.cat([mask_true, mask_false], dim=dim)
        return padded, mask

    return padded


# Training took 5.717105 seconds
# Training took 4.495734 seconds
# Training took 4.565008 seconds
# Training took 4.320932 seconds
# def max_pad_tensors_batch(tensors, dim=0, add_mask=False):
#     """Pads a batch of tensors with zeros along a dimension to match the maximum
#     length.

#     Args:
#         tensors (List[torch.Tensor]): A list of tensors to be padded.
#         dim (int, default: 0): The dimension along which to pad the tensors.
#         add_mask (bool, optional, default: False):
#             Whether to also return a mask tensor

#     Returns:
#         If add_mask=True, return a tuple (padded, mask).
#         If all tensors have the same length, mask is None.
#         Otherwise, returns the padded tensor only.
#     """
#     lengths = [x.shape[dim] for x in tensors]
#     max_length = max(lengths)
#     if all(length == max_length for length in lengths):
#         stacked = torch.stack(tensors) # Don't pad if we don't need to
#         if add_mask:
#             mask = None
#     else:
#         if add_mask:
#             padded_tensors, masks = zip(*[
#                 pad_tensor(x, max_length, dim=dim, add_mask=True)
#                 for x in tensors])
#             mask = torch.stack(masks)
#         else:
#             padded_tensors = [
#                 pad_tensor(x, max_length, dim=dim, add_mask=False)
#                 for x in tensors]
#         stacked = torch.stack(padded_tensors)
    
#     if add_mask:
#         return stacked, mask
#     return stacked


def equals(a, b):
    try:
        iter(a)
        is_iterable = True
    except TypeError:
        is_iterable = False
    if is_iterable:
        if len(a) != len(b):
            return False
        return type(b) is type(a) and all(equals(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    return a == b


# Training took 4.215692 seconds
# Training took 3.932536 seconds
# Training took 4.084472 seconds
# Training took 4.148941 seconds
# Training took 4.345846 seconds
# Training took 4.416536 seconds
## I don't know maybe this is a bit faster
def max_pad_tensors_batch(tensors, add_mask=False):
    """Pads a batch of tensors with zeros along a dimension to match the maximum
    length.

    Args:
        tensors (List[torch.Tensor]): A list of tensors to be padded.
        add_mask (bool, optional, default: False):
            Whether to also return a mask tensor

    Returns:
        If add_mask=True, return a tuple (padded, mask).
        If all tensors have the same length, mask is None.
        Otherwise, returns the padded tensor only.
    """
    lengths = [x.shape[0] for x in tensors]
    max_length = max(lengths)
    if all(length == max_length for length in lengths):
        padded = torch.stack(tensors) # Don't pad if we don't need to
        if add_mask:
            mask = None
    else:
        if add_mask:
            ones_at_end = [1] * (tensors[0].dim() - 1)
            mask = torch.zeros(len(tensors), max_length, *ones_at_end,
                               dtype=torch.bool, device=tensors[0].device)
            for i, x in enumerate(tensors):
                mask[i, :x.shape[0], ...] = True
        
        padded = torch.zeros(len(tensors), max_length, *tensors[0].shape[1:],
                             dtype=tensors[0].dtype, device=tensors[0].device)
        for i, x in enumerate(tensors):
            padded[i, :x.shape[0], ...] = x
    
    if add_mask:
        ret = padded, mask
    else:
        ret = padded

    # Testing
    # assert equals(ret, max_pad_tensors_batch_old(tensors, dim=0, add_mask=add_mask))

    return ret


# Taken from
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
def get_lengths_from_proportions(total_length: int, proportions: Sequence[float]):
    subset_lengths: List[int] = []
    for i, frac in enumerate(proportions):
        if frac < 0 or frac > 1:
            raise ValueError(f"Fraction at index {i} is not between 0 and 1")
        n_items_in_split = int(math.floor(total_length * frac))
        subset_lengths.append(n_items_in_split)
    remainder = total_length - sum(subset_lengths)
    # add 1 to all the lengths in round-robin fashion until the remainder is 0
    for i in range(remainder):
        idx_to_add_at = i % len(subset_lengths)
        subset_lengths[idx_to_add_at] += 1
    lengths = subset_lengths
    for i, length in enumerate(lengths):
        if length == 0:
            warnings.warn(
                f"Length of split at index {i} is 0. "
                f"This might result in an empty dataset."
            )
    return lengths


def get_lengths_from_proportions_or_lengths(
        total_length: int, lengths: Sequence[Union[int, float]]):
    lengths_is_proportions = math.isclose(sum(lengths), 1) and sum(lengths) <= 1
    if lengths_is_proportions:
        return get_lengths_from_proportions(total_length, lengths)
    return lengths


class SizedIterableMixin(Iterable):
    """An abstract mixin class that provides functionality 'len()' for iterable objects.
    All subclasses should implement `__iter__` because this class inherits from
    the abstract base class `Iterable`.

    Attributes:
        _size (int or inf):
            The size of the iterable object. math.inf if the size is infinite.
            Subclasses must use this attribute to hold the length of the object.
    """
    def _len_or_inf(self):
        if not hasattr(self, "_size"):
            raise AttributeError(
                f"{self.__class__.__name__}, a subclass of SizedIterableMixin, "\
                    "must have attribute '_size' to hold the length.")
        size = self._size
        if size != math.inf and (not isinstance(size, int) or size < 0):
            raise ValueError(
                f"self._size should inf or a non-negative integer but got {size}")
        return size
    
    def __len__(self):
        size = self._len_or_inf()
        if size == math.inf:
            raise TypeError(f"Length of the {self.__class__.__name__} is infinite")
        return size


class SizedInfiniteIterableMixin(SizedIterableMixin):
    """An abstract mixin class that provides functionality for creating iterable objects
    with a specified size. If the size is inf, the object is considered to be
    infinite and so calling iter() then you can call next() indefinitely wihout
    any StopIteration exception.
    If the size is not inf, then the object is considered to be finite and
    calling iter() will return a generator that will yield the next element
    until the size is reached.

    Attributes:
        _size (int or inf):
            The size of the iterable object. math.inf if the size is infinite.
            Subclasses must use this attribute to hold the length of the object.
    """
    @abstractmethod
    def copy_with_new_size(self, size:int) -> "SizedInfiniteIterableMixin":
        """Creates a copy of the object with a new size.
        Should set the _size attribute of the new object to the specified size.

        Args:
            size (int): The new size for the object.

        Returns:
            A new instance of the object with the specified size.
        """
        pass  # pragma: no cover
    
    @abstractmethod
    def _next(self):
        """Returns the next element in the iterable."""
        pass  # pragma: no cover

    def __iter__(self):
        if self._len_or_inf() == math.inf:
            return self
        # Must separate this in a different function because otherwise,
        # iter will always return a generator, even if self._size == math.inf
        return self._finite_iterator()
    
    def _finite_iterator(self):
        for _ in range(len(self)):
            yield self._next()

    def __next__(self):
        if self._len_or_inf() == math.inf:
            return self._next()
        raise TypeError(
            f"Cannot call __next__ on a finitely sized {type(self)}. Use iter() first.")


class _ResizedIterable(Iterable):
    """
    Creates an iterable that resizes given iterable to desired size

    If allow_repeats = False, then:
    Takes any iterable and an integer 'n', and provides an iterator
    that yields the first 'n' elements of the given iterable. If the original
    iterable contains fewer than 'n' elements, the iterator will yield only the
    available  elements without raising an error.
    If allow_repeats = False, then:
    repeats the given iterable until the desired number n.

    Args:
        iterable (iterable): The iterable to wrap.
        n (int): The number of elements to yield from the iterable.
    """
    def __init__(self, iterable, n, allow_repeats=False):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n should be a positive integer.")
        self.iterable = iterable
        self.n = n
        self.allow_repeats = allow_repeats
    
    def __iter__(self):
        allow_repeats = self.allow_repeats

        n = self.n
        i = 0
        iterator = iter(self.iterable)
        while i < n:
            try:
                yield next(iterator)
                i += 1
            except StopIteration:
                if allow_repeats:
                    iterator = iter(self.iterable)
                else:
                    break
    
    def __len__(self):
        if self.allow_repeats:
            return self.n
        return min(len_or_inf(self.iterable), self.n)


def len_or_inf(x):
    try:
        l = len(x)
        if l != math.inf and (not isinstance(l, int) or l < 0):
            raise ValueError(
                f"len(x) should be inf or a non-negative integer but got {l}")
        return l
    except TypeError:
        # Then it has no length, so we can only say it's infinite if
        # it is an iterable.
        try: # Check if it is an iterable
            iter(x)
            return math.inf
        except TypeError:
            raise TypeError(
                f"Object of type {type(x).__name__} is not iterable so has no length")


def iterable_is_finite(x):
    return len_or_inf(x) != math.inf


def resize_iterable(it, new_len: Optional[int] = None, allow_repeats=False):
    if new_len is not None:
        if not isinstance(new_len, int) or new_len <= 0:
            raise ValueError("new_len should be a positive integer")
        original_len = len_or_inf(it)
        if new_len != original_len:
            # Weaker condition than `if isinstance(it, SizedInfiniteIterableMixin):`
            if callable(getattr(it, "copy_with_new_size", None)):
                it = it.copy_with_new_size(new_len)
            else:
                if not allow_repeats and new_len > original_len:
                    raise ValueError(f"{new_len=} should be <= len(it)={original_len} "
                                     "if it is not a SizedInfiniteIterableMixin")
                it = _ResizedIterable(it, new_len, allow_repeats=allow_repeats)
    return it


def safe_issubclass(obj, parent):
    """Returns whether `obj` is a class that is a subclass of `parent`.
    In contrast to `issubclass`, doesn't raise TypeError when `obj` is not a class."""
    return isinstance(obj, type) and issubclass(obj, parent)


# Dictionary to keep track of subclasses of SaveableObject
_CLASSES: dict[str, type] = {}

def _info_dict_to_instance(info_dict) -> tuple[type, dict]:
    """Creates an instance of a class from a dictionary containing its class name
    and arguments."""
    if set(info_dict.keys()) != {"class_name", "kwargs"}:
        raise ValueError("info_dict should have keys 'class_name' and 'kwargs'")
    
    class_name = info_dict["class_name"]
    try:
        loaded_cls = _CLASSES[class_name]
    except KeyError:
        raise RuntimeError(f"Subclass {class_name} of SaveableObject does not exist")
    
    kwargs = info_dict["kwargs"]
    if not isinstance(kwargs, dict):
        raise TypeError("'kwargs' should be a dictionary")
    for k, v in kwargs.items():
        if type(v) is str and v in _CLASSES:
            kwargs[k] = _CLASSES[v]
        elif isinstance(v, dict) and set(v.keys()) == {"class_name", "kwargs"}:
            v_cls, v_kwargs = _info_dict_to_instance(v)
            kwargs[k] = v_cls(**v_kwargs)
    return loaded_cls, kwargs

def _default_json_SaveableObject(o):
    if o.__class__ is type:
        the_type = o.__name__
        tmp = f"the type {the_type}"
    else:
        the_type = o.__class__.__name__
        tmp = f"of type {the_type}"
    msg = "Unable to save the file because one of the values is not JSON " \
        "serializable. Only JSON serializable values and SaveableObject " \
        "subclasses and instances are supported to be saved by SaveableObject. " \
        f"If you need to save this object ({tmp}), consider making {the_type} " \
        "a subclass of SaveableObject."
    raise TypeError(msg)

def _get_class_name(c: type):
    return f"{c.__module__}.{c.__name__}"

class SaveableObject(ABC):
    r"""An abstract base class for objects that can be saved and loaded with their
    initialization parameters.

    This class enables automatic serialization and deserialization of subclasses by 
    tracking their __init__ arguments. When a subclass is instantiated, its 
    constructor arguments are introspected and stored, allowing the object to be 
    reconstructed from saved data. Supports instances of `SaveableObject`
    as well as subclasses of `SaveableObject` as arguments to __init__.

    Subclass __init__ must not use `*args` (variadic positional arguments).
    
    Subclasses should subclass `SaveableObject` with `SaveableObject` as the last
    one, i.e., as in,
    ```class C(A, B, SaveableObject)
    ```
    
    *Note:* although internally, SaveableObject has an abstract method __init__, this is
    NOT for purpose of an interface/contract: since SaveableObject automatically assigns
    an explicit __init__ method whenever a new subclass is created, subclasses are NOT
    required to implement __init__.
    Rather, the purpose of this abstract __init__ method is to have at least one
    @abstractmethod in the abstract base class SaveableObject so that SaveableObject is
    prevented from being instantiated directly.
    """
    @classmethod
    def load_init(cls, folder: str) -> Self:
        r"""Loads an instance from a JSON file.

        This method reads the JSON file 'init.json' from the specified folder,
        reconstructs the initialization parameters stored therein, and returns a new
        instance of the appropriate class.

        Usage:
        - To load any `SaveableObject` instance using the base class:
                ```instance = SaveableObject.load_init("/path/to/folder")
                ```
        - To load an instance of a specific subclass `MySubClass`:
                ```instance = MySubClass.load_init("/path/to/folder")
                ```
        In the latter case, ensure that the JSON file contains data for a `MySubClass`
        instance; otherwise, a TypeError will be raised.

        Args:
            folder (str): The directory containing the 'init.json' file.

        Returns:
            The loaded instance.

        Raises:
            RuntimeError:
                If the JSON file could not be loaded (e.g., file not found,
                JSON decoding error, or Unicode decode error).
            TypeError:
                If the JSON file contains an instance of a different class than
                expected.
        """
        try:
            info_dict = load_json(os.path.join(folder, "init.json"))
        except (FileNotFoundError,
                json.decoder.JSONDecodeError, UnicodeDecodeError) as e:
            raise RuntimeError(f"Could not load {cls.__name__} instance") from e
        typ, kwargs = _info_dict_to_instance(info_dict)
        if not issubclass(typ, cls):
            raise TypeError(
                f"Trying to load a {cls.__name__} instance but the JSON file "
                f"contains a {typ.__name__} instance. Instead of calling "
                f"{cls.__name__}.load_init, you could call "
                f"SaveableObject.load_init or {typ.__name__}.load_init"
            )
        return typ(**kwargs)

    def save_init(self, folder: str):
        r"""Saves initialization parameters to a JSON file.
        Args:
            folder (str): The directory to save the JSON file.
        """
        os.makedirs(folder, exist_ok=True)
        save_json(self.get_info_dict(), os.path.join(folder, "init.json"),
                  default=_default_json_SaveableObject)

    def get_info_dict(self) -> dict[str, Union[str, dict[str, Any]]]:
        r"""Returns a dictionary representation of initialization parameters."""
        return {
            "class_name": _get_class_name(self.__class__),
            "kwargs": self._init_kwargs
        }

    @abstractmethod
    def __init__(self, *args, **kwargs):
        # Default __init__ for SaveableObject that does nothing.
        # Prevents SaveableObject from being instantiated directly.
        # Although this is an abstractmethod, subclasses do NOT need to
        # implement __init__.
        pass

    def __init_subclass__(cls, **kwargs):
        # Preserve the original __init__ method.
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # If we are constructing THIS exact class (and not a subclass further down
            # the hierarchy). i.e., check if if we are calling the same __init__ as the
            # __init__ defined directly in the object that is being constructed,
            # not a superclass' __init__.
            is_same_class = self.__class__ is cls

            if is_same_class:
                # Copy the original args and kwargs, just in case original_init modifies
                # them
                original_args = copy.deepcopy(args)
                original_kwargs = copy.deepcopy(kwargs)

            original_init(self, *args, **kwargs) # Call the original __init__ method

            if is_same_class:
                # Convert args to kwargs
                sig = inspect.signature(original_init)
                
                # Can use either bind or bind_partial;
                # we already ensured that all required arguments are passed
                # because we already called the original __init__ method.
                # Need to remember to put 'self' in the arguments.
                bound_args = sig.bind(self, *original_args, **original_kwargs)
                bound_args.apply_defaults()
                all_kwargs = bound_args.arguments

                # Detect VAR_KEYWORD parameters (i.e. **kwargs) and flatten them
                # to fix the problem that all_kwargs = {..., 'kwargs': {...}}
                # arises due to apply_defaults when there's ** kind of parameters.
                # Also, make sure that there are no *args type of parameters
                # because we don't support that (makes it too complicated).
                for p_name, param in sig.parameters.items():
                    if param.kind == param.VAR_KEYWORD:
                        if p_name in all_kwargs:
                            # all_kwargs[p_name] is the dict that ended up in **whatever
                            var_kw_dict = all_kwargs.pop(p_name) # remove from top-level
                            # Flatten all items inside that dict:
                            all_kwargs.update(var_kw_dict)
                    elif param.kind == param.VAR_POSITIONAL:
                        value = all_kwargs[p_name]
                        assert type(value) is tuple
                        if len(value) > 0:
                            raise ValueError(
                                "SaveableObject does not support *args in __init__")
                        all_kwargs.pop(p_name)

                # Remove 'self' from the kwargs
                all_kwargs.pop('self', None)

                for k, v in all_kwargs.items():
                    # If there are any subclasses of SaveableObject in the
                    # kwargs, replace them with their name
                    if safe_issubclass(v, SaveableObject):
                        all_kwargs[k] = _get_class_name(v)
                    # If there are any instances of SaveableObject in the
                    # kwargs, replace them with their info_dict
                    elif isinstance(v, SaveableObject):
                        all_kwargs[k] = v.get_info_dict()

                self._init_kwargs = all_kwargs

        # Replace the __init__ method with the new one
        cls.__init__ = new_init

        # Register the class in the _CLASSES dictionary
        _CLASSES[_get_class_name(cls)] = cls

        # Call the original __init_subclass__ method
        super().__init_subclass__(**kwargs)


# Based on
# https://docs.gpytorch.ai/en/stable/_modules/gpytorch/module.html#Module.initialize
def get_param_value(module, name):
    if "." in name:
        submodule, name = module._get_module_and_name(name)
        if isinstance(submodule, torch.nn.ModuleList):
            idx, name = name.split(".", 1)
            return get_param_value(submodule[int(idx)], name)
        else:
            return get_param_value(submodule, name)
    elif not hasattr(module, name):
        raise AttributeError(
            "Unknown parameter {p} for {c}".format(p=name, c=module.__class__.__name__))
    elif name not in module._parameters and name not in module._buffers:
        return getattr(module, name)
    else:
        return module.__getattr__(name)


# Print out all parameters of a random model:
# random_model = model.pyro_sample_from_prior()
# for name, param in model.named_parameters(): 
#     print(name)
#     print(get_param_value(random_model, name))
#     print()
## OR,
# random_model_params_dict = {
#     name: get_param_value(random_model, name)
#     for name, param in model.named_parameters()
# }


