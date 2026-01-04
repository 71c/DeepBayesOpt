import inspect
import math
from typing import List, Optional, Sequence, Tuple, Type, Union, Literal

import torch
from torch import nn
from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import UnsupportedError
from botorch.utils.transforms import t_batch_mode_transform, match_batch_shape
from abc import abstractmethod
from nn_af.acquisition_function_net_constants import POINTNET_ACQF_PARAMS_INPUT_DEFAULT, POINTNET_ACQF_PARAMS_INPUT_OPTIONS
from utils_general.saveable_object import SaveableObject
from utils.utils import standardize_y_hist
from utils_general.io_utils import safe_issubclass

from utils_general.nn_utils import Dense, expand_dim
from utils.nn_utils import (MultiLayerPointNet, PointNetLayer,
                      SoftmaxOrSoftplusLayer, add_neg_inf_for_max, check_xy_dims)

import logging

from utils_general.utils import to_device

# Set to True to enable debug logging
DEBUG = False

# Create a logger for your application
logger = logging.getLogger('acquisition_function_net')
# Configure the logging
logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)


def get_best_y(y_hist, hist_mask=None):
    if hist_mask is not None:
        neg_inf = torch.zeros_like(y_hist)
        neg_inf[~hist_mask] = float("-inf")
        return (y_hist + neg_inf).amax(-2, keepdim=True)
    return y_hist.amax(-2, keepdim=True)

def concat_y_hist_with_best_y(y_hist, hist_mask, subtract=False):
    best_f = get_best_y(y_hist, hist_mask).expand_as(y_hist)
    if subtract:
        return torch.cat((best_f, best_f - y_hist), dim=-1)
    return torch.cat((best_f, y_hist), dim=-1)


def check_class_for_init_params(cls, base_class_name:str, *required_params):
    if inspect.isabstract(cls):
        # Don't check abstract classes; only check concrete classes
        return
    init_sig = inspect.signature(cls.__init__)
    for param_name in required_params:
        if param_name not in init_sig.parameters:
            raise TypeError(
                f"Class {cls.__name__} is missing required init param '{param_name}' "
                f"since it is a subclass of {base_class_name}"
            )


class AcquisitionFunctionNet(nn.Module, SaveableObject):
    """Neural network model for the acquisition function in NN-based
    likelihood-free Bayesian optimization.

    Attributes:
        output_dim (int):
            The output dimension of the acquisition function.
    """
    @property
    @abstractmethod
    def output_dim(self) -> int:
        r"""Returns the output dimension of the acquisition function"""
        pass  # pragma: no cover

    @abstractmethod
    def forward(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
                hist_mask:Optional[Tensor]=None, cand_mask:Optional[Tensor]=None,
                **kwargs) -> Tensor:
        """Forward pass of the acquisition function network.

        Args:
            x_hist (torch.Tensor):
                A `batch_shape x n_hist x d` tensor of training features.
            y_hist (torch.Tensor):
                A `batch_shape x n_hist` or `batch_shape x n_hist x n_hist_out`
                tensor of training observations.
            x_cand (torch.Tensor):
                Candidate input tensor with shape `batch_shape x n_cand x d`.
            hist_mask (torch.Tensor, optional):
                Mask tensor for the history inputs with shape `batch_shape x n_hist`
                or `batch_shape x n_hist x 1`. If None, then mask is all ones.
            cand_mask (torch.Tensor, optional):
                Mask tensor for the candidate inputs with shape `batch_shape x n_cand`
                or `batch_shape x n_cand x 1`. If None, then mask is all ones.
            **kwargs:
                Any potential additional arguments.

        Note: It is assumed x_hist and y_hist are padded (with zeros), although
            that shouldn't matter since the mask will take care of it.

        Returns:
            torch.Tensor: A `batch_shape x n_cand x output_dim` tensor of acquisition
            values. (`output_dim` is 1 for most acquisition functions)
        """
        pass  # pragma: no cover

    def preprocess_inputs(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
                hist_mask:Optional[Tensor]=None, cand_mask:Optional[Tensor]=None,
                **kwargs):
        # Put on GPU if it's not already
        nn_device = next(self.parameters()).device
        x_hist = x_hist.to(nn_device)
        y_hist = y_hist.to(nn_device)
        x_cand = x_cand.to(nn_device)
        
        hist_mask = to_device(hist_mask, nn_device)
        cand_mask = to_device(cand_mask, nn_device)

        y_hist = check_xy_dims(x_hist, y_hist, "x_hist", "y_hist")
        hist_mask = check_xy_dims(x_hist, hist_mask,
                                  "x_hist", "hist_mask", expected_y_dim=1)
        cand_mask = check_xy_dims(x_cand, cand_mask,
                                  "x_cand", "cand_mask", expected_y_dim=1)

        return dict(
            x_hist=x_hist,
            y_hist=y_hist,
            x_cand=x_cand,
            hist_mask=hist_mask,
            cand_mask=cand_mask,
            **kwargs
        )


class ParameterizedAcquisitionFunctionNet(AcquisitionFunctionNet):
    r"""This is an abstract class that is meant to represent a generic
    AcquisitionFunctionNet that has some number of acquisition function parameters.

    Attributes:
        output_dim (int):
            The output dimension of the acquisition function.
        n_acqf_params (int):
            a non-negative integer representing the number of scalar variable paramters
            that can be passed in to the acquisition function.
            Required parameter in __init__.
    """
    def __init_subclass__(cls, **kwargs):
        check_class_for_init_params(
            cls, 'ParameterizedAcquisitionFunctionNet', 'n_acqf_params')
        super().__init_subclass__(**kwargs)
    
    @property
    @abstractmethod
    def n_acqf_params(self) -> int:
        r"""`n_acqf_params`: a non-negative integer representing the number of scalar
        variable paramters that can be passed in to the acquisition function. """
        pass  # pragma: no cover

    @abstractmethod
    def forward(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
                acqf_params:Optional[Tensor]=None,
                hist_mask:Optional[Tensor]=None,
                cand_mask:Optional[Tensor]=None,
                **kwargs) -> Tensor:
        """Forward pass of the acquisition function network.

        Args:
            x_hist (torch.Tensor):
                A `batch_shape x n_hist x d` tensor of training features.
            y_hist (torch.Tensor):
                A `batch_shape x n_hist` or `batch_shape x n_hist x n_hist_out`
                tensor of training observations.
            x_cand (torch.Tensor):
                Candidate input tensor with shape `batch_shape x n_cand x d`.
            acqf_params (torch.Tensor, optional):
                Tensor of shape `batch_shape x n_cand x n_acqf_params`. Represents any
                variable parameters for the acquisition function, for example
                lambda for the Gittins index or best_f for expected improvement.
            hist_mask (torch.Tensor, optional):
                Mask tensor for the history inputs with shape `batch_shape x n_hist`
                or `batch_shape x n_hist x 1`. If None, then mask is all ones.
            cand_mask (torch.Tensor, optional):
                Mask tensor for the candidate inputs with shape `batch_shape x n_cand`
                or `batch_shape x n_cand x 1`. If None, then mask is all ones.
            **kwargs:
                Any potential additional arguments.

        Note: It is assumed x_hist and y_hist are padded (with zeros), although
            that shouldn't matter since the mask will take care of it.

        Returns:
            torch.Tensor: A `batch_shape x n_cand x output_dim` tensor of acquisition
            values. (`output_dim` is 1 for most acquisition functions)
        """
        pass  # pragma: no cover

    def preprocess_inputs(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
                acqf_params:Optional[Tensor]=None,
                hist_mask:Optional[Tensor]=None,
                cand_mask:Optional[Tensor]=None):
        existing = super().preprocess_inputs(x_hist, y_hist, x_cand,
                                             hist_mask=hist_mask, cand_mask=cand_mask)
        nn_device = next(self.parameters()).device
        acqf_params = to_device(acqf_params, nn_device)
        acqf_params = check_xy_dims(x_cand, acqf_params, "x_cand", "acqf_params")
        existing['acqf_params'] = acqf_params
        return existing


class AcquisitionFunctionNetFixedHistoryOutputDim(AcquisitionFunctionNet):
    r"""This is an abstract class that is meant to represent a generic
    AcquisitionFunctionNet that has a fixed dimension of y values in the history
    (function outputs and other things).

    Attributes:
        output_dim (int):
            The output dimension of the acquisition function.
        n_hist_out (int):
            a positive integer representing the number of "y values" or outputs
            in the history. Required parameter in __init__.
    """
    def __init_subclass__(cls, **kwargs):
        check_class_for_init_params(
            cls, 'AcquisitionFunctionNetFixedHistoryOutputDim', 'n_hist_out')
        super().__init_subclass__(**kwargs)
    
    @property
    @abstractmethod
    def n_hist_out(self) -> int:
        r"""`n_hist_out`: a positive integer representing the number of "y values" or
        outputs in the history."""
        pass  # pragma: no cover


class AcquisitionFunctionNetGivenOutputDim(AcquisitionFunctionNet):
    r"""This is an abstract class that is meant to represent a generic
    AcquisitionFunctionNet that has an output dimension that is explicitly given
    in the constructor.

    Attributes:
        output_dim (int):
            The output dimension of the acquisition function.
            Required parameter in __init__.
    """
    def __init_subclass__(cls, **kwargs):
        check_class_for_init_params(
            cls, 'AcquisitionFunctionNetGivenOutputDim', 'output_dim')
        super().__init_subclass__(**kwargs)


class AcquisitionFunctionBody(nn.Module, SaveableObject):
    r"""This abstract class represents the computation of some features based on
    history and candidates, e.g. using PointNet or transformer neural process,
    which is then to be fed into probably some MLP to compute the acquisition function
    value.

    Attributes:
        output_dim (int):
            The input dimension (dimension of the features).
        n_acqf_params (int):
            The number of scalar variable paramters that can be passed in to the
            acquisition function. Required parameter in __init__."""
    @abstractmethod
    def forward(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
                acqf_params:Optional[Tensor]=None,
                hist_mask:Optional[Tensor]=None,
                cand_mask:Optional[Tensor]=None) -> Tensor:
        """Compute the features.

        Args:
            x_hist (torch.Tensor):
                A `batch_shape x n_hist x d` tensor of training features.
            y_hist (torch.Tensor):
                A `batch_shape x n_hist x n_hist_out` tensor of training observations.
            x_cand (torch.Tensor):
                A `batch_shape x n_cand x d` tensor of candidate points.
            acqf_params (torch.Tensor, optional):
                Tensor of shape `batch_shape x n_cand x n_params`. Represents any
                variable parameters for the acquisition function, for example
                lambda for the Gittins index or best_f for expected improvement.
            hist_mask (torch.Tensor):
                A `batch_shape x n_hist x 1` mask tensor for the history inputs.
                If None, then mask is all ones.
            cand_mask (torch.Tensor):
                A `batch_shape x n_cand x 1` mask tensor for the candidate
                inputs. If None, then mask is all ones.

        Returns:
            torch.Tensor: The `batch_shape x n_cand x output_dim`
            input tensor to the final MLP network.
        """
        pass  # pragma: no cover

    def __init_subclass__(cls, **kwargs):
        check_class_for_init_params(
            cls, 'AcquisitionFunctionBody', 'n_acqf_params')
        super().__init_subclass__(**kwargs)

    @property
    @abstractmethod
    def output_dim(self) -> int:
        r"""Returns the output dimension (dimension of the features)"""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def n_acqf_params(self) -> int:
        r"""`n_acqf_params`: a non-negative integer representing the number of scalar
        variable paramters that can be passed in to the acquisition function."""
        pass  # pragma: no cover


class AcquisitionFunctionBodyFixedHistoryOutputDim(AcquisitionFunctionBody):
    r"""This represents a AcquisitionFunctionBody where the `n_hist_out` is fixed
    and is provided in __init__.

    Attributes:
        output_dim (int):
            The input dimension (dimension of the features).
        n_acqf_params (int):
            The number of scalar variable paramters that can be passed in to the
            acquisition function. Required parameter in __init__.
        n_hist_out (int):
            The number of outputs in the history. Required parameter in __init__.
    """
    def __init_subclass__(cls, **kwargs):
        check_class_for_init_params(
            cls, 'AcquisitionFunctionBody', 'n_hist_out')
        super().__init_subclass__(**kwargs)
    
    @property
    @abstractmethod
    def n_hist_out(self) -> int:
        r"""The number of outputs in the history."""
        pass  # pragma: no cover


class AcquisitionFunctionHead(nn.Module, SaveableObject):
    r"""AcquisitionFunctionHead is an abstract base class for defining the head of an
    acquisition function network. It ensures that any subclass implements the required
    properties and methods for input and output dimensions, as well as the forward
    method for computing the acquisition function value.

    Attributes:
        input_dim (int):
            The input dimension (dimension of the features).
            Required parameter in __init__.
        output_dim (int):
            The output dimension (dimension of the acquisition function output).
            Required parameter in __init__."""
    def __init_subclass__(cls, **kwargs):
        check_class_for_init_params(
            cls, 'AcquisitionFunctionHead', 'input_dim', 'output_dim')
        super().__init_subclass__(**kwargs)

    @property
    @abstractmethod
    def input_dim(self) -> int:
        r"""Returns the input dimension (dimension of the features)"""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def output_dim(self) -> int:
        r"""Returns the output dimension (dimension of the AF output)"""
        pass  # pragma: no cover

    @abstractmethod
    def forward(self, features,
                x_hist, y_hist, x_cand,
                hist_mask=None, cand_mask=None, stdvs=None,
                **other_kwargs) -> Tensor:
        r"""Compute the acquisition function value from the features that were computed
        from the original inputs, and potentially also from the original inputs directly
        Args:
            features: shape (*, n_cand, input_dim)
            x_hist, y_hist, x_cand, hist_mask, cand_mask:
                Those args that were passed into AcquisitionFunctionNet.forward
            **other_kwargs:
                Any additional args that were passed into AcquisitionFunctionNet.forward
            stdvs (Tensor, optional):
                Standard deviations of y_hist, shape (*, 1, n_hist_out)
        """
        pass  # pragma: no cover


def check_subclass(cls, cls_var_name:str, super_cls):
    if not safe_issubclass(cls, super_cls):
        cls_str = cls.__name__ if isinstance(cls, type) else str(cls)
        raise ValueError(
            f"{cls_var_name}={cls_str} should be a subclass of {super_cls.__name__}")


class TwoPartAcquisitionFunctionNet(ParameterizedAcquisitionFunctionNet,
                                    AcquisitionFunctionNetGivenOutputDim):
    def __init__(self,
                 output_dim:int, n_acqf_params:int,
                 af_body_class:Type[AcquisitionFunctionBody],
                 af_head_class:Type[AcquisitionFunctionHead],
                 af_body_init_params:dict,
                 af_head_init_params:dict,
                 standardize_outcomes=False):
        check_subclass(af_body_class, "af_body_class", AcquisitionFunctionBody)
        
        if type(self) is TwoPartAcquisitionFunctionNet and \
            issubclass(af_body_class, AcquisitionFunctionBodyFixedHistoryOutputDim):
            raise ValueError(
                "TwoPartAcquisitionFunctionNet should not be used with a "
                "AcquisitionFunctionBodyFixedHistoryOutputDim subclass. "
                "Use TwoPartAcquisitionFunctionNetFixedHistoryOutputDim instead.")

        check_subclass(af_head_class, "af_head_class", AcquisitionFunctionHead)
        
        super().__init__()

        if 'n_acqf_params' in af_body_init_params:
            raise ValueError("n_acqf_params should not be in af_body_init_params"
                             " since it is already a required constructor parameter"
                             f" of {self.__class__.__name__}.")
        if 'input_dim' in af_head_init_params:
            raise ValueError("input_dim should not be in af_head_init_params"
                             " since it is inferred from af_body.output_dim "
                             f"in {self.__class__.__name__}.")
        if 'output_dim' in af_head_init_params:
            raise ValueError("output_dim should not be in af_head_init_params"
                             " since it is already a required constructor parameter"
                             f" of {self.__class__.__name__}.")
        
        self.af_body = af_body_class(n_acqf_params=n_acqf_params, **af_body_init_params)

        self.af_head = af_head_class(input_dim=self.af_body.output_dim,
                                     output_dim=output_dim, **af_head_init_params)
        
        self.register_buffer("standardize_outcomes",
                             torch.as_tensor(standardize_outcomes))

    @property
    def output_dim(self) -> int:
        return self.af_head.output_dim

    @property
    def n_acqf_params(self) -> int:
        return self.af_body.n_acqf_params

    def forward(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
                acqf_params:Optional[Tensor]=None,
                hist_mask:Optional[Tensor]=None,
                cand_mask:Optional[Tensor]=None,
                **kwargs) -> Tensor:
        preprocessed = self.preprocess_inputs(
            x_hist, y_hist, x_cand, acqf_params=acqf_params,
            hist_mask=hist_mask, cand_mask=cand_mask)
        
        x_hist = preprocessed['x_hist']
        y_hist = preprocessed['y_hist']
        x_cand = preprocessed['x_cand']
        hist_mask = preprocessed['hist_mask']
        cand_mask = preprocessed['cand_mask']
        acqf_params = preprocessed['acqf_params']
        
        if self.standardize_outcomes:
            y_hist, stdvs = standardize_y_hist(y_hist, hist_mask)
            preprocessed['y_hist'] = y_hist
        else:
            stdvs = None
        
        # batch_shape x n_cand x features_dim
        features = self.af_body(x_hist, y_hist, x_cand,
                                acqf_params=acqf_params,
                                hist_mask=hist_mask, cand_mask=cand_mask)
        # batch_shape x n_cand x output_dim
        return self.af_head(features, x_hist, y_hist, x_cand,
                            hist_mask=hist_mask, cand_mask=cand_mask,
                            stdvs=stdvs, **kwargs)


class TwoPartAcquisitionFunctionNetFixedHistoryOutputDim(
    TwoPartAcquisitionFunctionNet, AcquisitionFunctionNetFixedHistoryOutputDim):
    def __init__(self,
                 output_dim:int, n_acqf_params:int, n_hist_out:int,
                 af_body_class:Type[AcquisitionFunctionBodyFixedHistoryOutputDim],
                 af_head_class:Type[AcquisitionFunctionHead],
                 af_body_init_params:dict,
                 af_head_init_params:dict,
                 standardize_outcomes=False):
        check_subclass(af_body_class, "af_body_class",
                       AcquisitionFunctionBodyFixedHistoryOutputDim)

        if 'n_hist_out' in af_body_init_params:
            raise ValueError("n_hist_out should not be in af_body_init_params"
                             " since it is already a required constructor parameter"
                             f" of {self.__class__.__name__}.")
        
        af_body_init_params['n_hist_out'] = n_hist_out

        super().__init__(
            output_dim=output_dim, n_acqf_params=n_acqf_params,
            af_body_class=af_body_class, af_head_class=af_head_class,
            af_body_init_params=af_body_init_params,
            af_head_init_params=af_head_init_params,
            standardize_outcomes=standardize_outcomes)
    
    @property
    def n_hist_out(self) -> int:
        return self.af_body.n_hist_out


class AcquisitionFunctionNetFinalMLP(AcquisitionFunctionHead):
    """Class for an acquisition function network's final MLP layer."""
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int]=[256, 64],
                 output_dim=1,
                 activation="relu",
                 layer_norm_before_end=False,
                 layer_norm_at_end=False,
                 dropout=None):
        """Initializes the MLP layer at the end of the acquisition function.

        Args:
            input_dim (int):
                The dimensionality of the input (number of features)
            hidden_dims (Sequence[int], default: [256, 64]):
                A sequence of integers representing the sizes of the hidden
                layers of the final fully connected network.
        """
        super().__init__()

        self.dense = Dense(input_dim,
                           hidden_dims,
                           output_dim,
                           activation=activation,
                           activation_at_end=False,
                           layer_norm_before_end=layer_norm_before_end,
                           layer_norm_at_end=False,
                           dropout=dropout,
                           dropout_at_end=False)

        self.register_buffer("_input_dim",
                             torch.as_tensor(input_dim))
        self.register_buffer("_output_dim",
                             torch.as_tensor(output_dim))
        self.register_buffer("layer_norm_at_end",
                             torch.as_tensor(layer_norm_at_end))
    
    @property
    def input_dim(self) -> int:
        return self._input_dim.item()

    @property
    def output_dim(self) -> int:
        return self._output_dim.item()
    
    def _transform_acquisition_values(self, acquisition_values,
                                      x_hist, y_hist, x_cand,
                                      hist_mask=None, cand_mask=None, stdvs=None,
                                      **other_kwargs):
        return acquisition_values
    
    def forward(self, features,
                x_hist, y_hist, x_cand,
                hist_mask=None, cand_mask=None, stdvs=None,
                **other_kwargs) -> Tensor:
        # shape (*, n_cand, output_dim)
        acquisition_values = self.dense(features)

        if self.layer_norm_at_end:
            # This doesn't handle mask correctly; TODO
            # (only if I'll even end up using this which I probably won't)
            if cand_mask is not None:
                raise NotImplementedError(
                    "layer_norm_at_end doesn't handle mask correctly.")
            if acquisition_values.dim() > 2:
                mean = torch.mean(acquisition_values, dim=(-3, -2), keepdim=True)
                std = torch.std(acquisition_values, dim=(-3, -2), keepdim=True)
                acquisition_values = (acquisition_values - mean) / std
        
        acquisition_values = self._transform_acquisition_values(
            acquisition_values, x_hist, y_hist, x_cand,
            hist_mask=hist_mask, cand_mask=cand_mask, stdvs=stdvs, **other_kwargs)
        
        if cand_mask is not None:
            # Mask out the padded values
            acquisition_values = acquisition_values * cand_mask

        return acquisition_values


class AcquisitionFunctionNetFinalMLPSoftmaxExponentiate(AcquisitionFunctionNetFinalMLP):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int]=[256, 64],
                 output_dim=1,
                 activation="relu",
                 layer_norm_before_end=False,
                 layer_norm_at_end=False,
                 dropout=None,

                 include_alpha=False,
                 learn_alpha=False,
                 initial_alpha=1.0,
                 initial_beta=1.0,
                 learn_beta=False,
                 softplus_batchnorm=False,
                 softplus_batchnorm_momentum=0.1,
                 positive_linear_at_end=False,
                 gp_ei_computation=False):
        """Initializes the MLP layer at the end of the acquisition function.

        Args:
            input_dim (int):
                The dimensionality of the input.
            hidden_dims (Sequence[int], default: [256, 64]):
                A sequence of integers representing the sizes of the hidden
                layers of the final fully connected network.
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
        """
        if positive_linear_at_end and gp_ei_computation:
            raise ValueError(
                "positive_linear_at_end and gp_ei_computation can't both be True.")

        if positive_linear_at_end:
            if learn_beta:
                raise ValueError(
                    "positive_linear_at_end and learn_beta can't both be True.")
            hidden_dims_ = hidden_dims[:-1]
            dense_output_dim = hidden_dims[-1] * output_dim
        elif gp_ei_computation:
            if learn_beta:
                raise ValueError(
                    "gp_ei_computation and learn_beta can't both be True.")
            hidden_dims_ = hidden_dims
            dense_output_dim = 2 * output_dim
        else:
            hidden_dims_ = hidden_dims
            dense_output_dim = output_dim
        
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims_,
            output_dim=dense_output_dim,
            activation=activation,
            layer_norm_before_end=layer_norm_before_end,
            layer_norm_at_end=layer_norm_at_end,
            dropout=dropout)
        
        self.register_buffer("positive_linear_at_end",
                             torch.as_tensor(positive_linear_at_end))
        self.register_buffer("gp_ei_computation",
                             torch.as_tensor(gp_ei_computation))
        
        self.transform = SoftmaxOrSoftplusLayer(
            softmax_dim=-2,
            include_alpha=include_alpha,
            learn_alpha=learn_alpha,
            initial_alpha=initial_alpha,
            initial_beta=initial_beta,
            learn_beta=learn_beta,
            softplus_batchnorm=softplus_batchnorm,
            softplus_batchnorm_num_features=output_dim,
            softplus_batchnorm_dim=-1,
            softplus_batchnorm_momentum=softplus_batchnorm_momentum,
        )
    
    def get_alpha(self):
        return self.transform.get_alpha()

    def set_alpha(self, val):
        self.transform.set_alpha(val)
    
    def get_beta(self):
        return self.transform.softplus.beta

    @property
    def includes_alpha(self) -> bool:
        return self.transform.includes_alpha
    
    def _transform_acquisition_values(self, acquisition_values,
                                      x_hist, y_hist, x_cand,
                                      hist_mask=None, cand_mask=None, stdvs=None,
                                      exponentiate=None, softmax=None):
        if self.positive_linear_at_end or self.gp_ei_computation:
            last_hidden_dim = acquisition_values.shape[-1] // self.output_dim.item()

            # shape (*, n_cand, last_hidden_dim, output_dim)
            acquisition_values = acquisition_values.view(*acquisition_values.shape[:-1],
                                                        last_hidden_dim,
                                                        self.output_dim.item())
            
            if self.positive_linear_at_end:
                # # shape (*, 1, 1)
                # best_y = get_best_y(y_hist, hist_mask)
                # # shape (*, 1, 1, 1)
                # best_y = best_y.unsqueeze(-1)
                # acquisition_values = nn.functional.relu(
                #     acquisition_values - best_y, inplace=True)

                acquisition_values = nn.functional.relu(
                    acquisition_values, inplace=False)

                # shape (*, n_cand, output_dim)
                acquisition_values = acquisition_values.mean(dim=-2, keepdim=False)
            else:
                # shape (*, 1, 1)
                best_y = get_best_y(y_hist, hist_mask)
                means = acquisition_values[..., 0, :]
                sigmas = nn.functional.softplus(acquisition_values[..., 1, :], beta=1.)
                acquisition_values = sigmas * nn.functional.softplus(
                    (means - best_y) / sigmas, beta=1.77)
        
        if exponentiate is None or softmax is None:
            raise ValueError("Need to provide exponentiate and softmax parameters")

        acquisition_values = self.transform(
            acquisition_values, mask=cand_mask,
            exponentiate=exponentiate and not (
                self.positive_linear_at_end or self.gp_ei_computation),
            softmax=softmax
        )
        
        if stdvs is not None and exponentiate:
            # Assume that if exponentiate=True, then we are computing EI
            acquisition_values = acquisition_values * stdvs

        return acquisition_values


def _get_xy_hist_and_cand(x_hist, y_hist, x_cand, hist_mask=None, include_y=True):
    """Combines historical data and candidate data for Bayesian optimization.

    Args:
        x_hist (torch.Tensor):
            Historical input data with shape (*, n_hist, dim_hist).
        y_hist (torch.Tensor):
            Historical output data with shape (*, n_hist, n_hist_out).
        x_cand (torch.Tensor):
            Candidate input data with shape (*, n_cand, dim_cand).
        hist_mask (torch.Tensor, optional):
            Mask for historical data with shape (*, n_hist, 1).
        include_y (bool, default: True):
            Whether to include historical output data in the combined tensor.

    Returns:
        A tuple containing:
            - xy_hist (torch.Tensor):
                Combined historical input and output data with shape
                (*, n_hist, dim_hist+n_hist_out).
            - xy_hist_and_cand (torch.Tensor):
                Combined candidate and historical data with shape
                (*, n_cand, n_hist, dim_cand+dim_hist+n_hist_out) if include_y is True,
                otherwise with shape (*, n_cand, n_hist, dim_cand+dim_hist).
            - mask (torch.Tensor or None):
                Expanded mask for historical data with shape (*, n_cand, n_hist, 1) if
                hist_mask is provided, otherwise None.
    """
    # shape (*, n_hist, dim_hist+n_hist_out)
    xy_hist = torch.cat((x_hist, y_hist), dim=-1)

    n_hist = x_hist.size(-2)
    n_cand = x_cand.size(-2)
    # shape (*, n_cand, n_hist, dim_cand)
    x_cand_expanded = expand_dim(x_cand.unsqueeze(-2), -2, n_hist)

    # hist_mask has shape (*, n_hist, 1), so need to expand to match.
    # shape (*, n_cand, n_hist, 1)
    mask = None if hist_mask is None else expand_dim(hist_mask.unsqueeze(-3), -3, n_cand)

    if include_y:
        # shape (*, n_cand, n_hist, dim_hist+n_hist_out)
        xy_hist_expanded = expand_dim(xy_hist.unsqueeze(-3), -3, n_cand)

        # shape (*, n_cand, n_hist, dim_cand+dim_hist+n_hist_out)
        xy_hist_and_cand = torch.cat((x_cand_expanded, xy_hist_expanded), dim=-1)

        return xy_hist, xy_hist_and_cand, mask
    else:
        # shape (*, n_cand, n_hist, dim_hist)
        x_hist_expanded = expand_dim(x_hist.unsqueeze(-3), -3, n_cand)

        # shape (*, n_cand, n_hist, dim_cand+dim_hist)
        x_hist_and_cand = torch.cat((x_cand_expanded, x_hist_expanded), dim=-1)

        return xy_hist, x_hist_and_cand, mask


_POINTNET_ACQF_PARAMS_INPUT_DEFAULT_OPTIONS = \
    POINTNET_ACQF_PARAMS_INPUT_OPTIONS[POINTNET_ACQF_PARAMS_INPUT_DEFAULT]
_ACQF_PARAMS_LOCAL_DEFAULT = \
    _POINTNET_ACQF_PARAMS_INPUT_DEFAULT_OPTIONS["input_acqf_params_to_local_nn"]
_ACQF_PARAMS_FINAL_DEFAULT = \
    _POINTNET_ACQF_PARAMS_INPUT_DEFAULT_OPTIONS["input_acqf_params_to_final_mlp"]


class AcquisitionFunctionBodyPointnetV1and2(
    AcquisitionFunctionBodyFixedHistoryOutputDim):
    def __init__(self,
                 dimension:int, n_hist_out:int, n_acqf_params:int=0,
                #  history_encoder: Literal["pointnet", "transformer"] = "pointnet",  # TODO: work on this
                 
                 history_enc_hidden_dims=[256, 256],
                 pooling="max",
                 encoded_history_dim=1024,

                 input_xcand_to_local_nn=True,
                 input_xcand_to_final_mlp=False,
                
                 subtract_x_cand_from_x_hist=False,

                 input_acqf_params_to_local_nn=_ACQF_PARAMS_LOCAL_DEFAULT,
                 input_acqf_params_to_final_mlp=_ACQF_PARAMS_FINAL_DEFAULT,
                 
                 activation_at_end_pointnet=True,
                 layer_norm_pointnet=False,
                 dropout_pointnet=None,
                 activation_pointnet:str="relu",

                 include_best_y=False,
                 subtract_best_y=False,
                 n_pointnets=1,
                 max_history_input=None,):
        """
        Args:
            dimension (int):
                The dimensionality of the input space.
            n_hist_out (int):
                The number of outputs in the history.
            n_acqf_params (int, default: 0):
                The number of parameters for the acquisition function.
            history_enc_hidden_dims: sequence of integers representing the
                hidden layer dimensions of the history encoder network.
                Default is [256, 256].
            pooling (str): The pooling method used in the history encoder.
                Must be either "max", "mean", or "sum". Default is "max".
            encoded_history_dim (int): The dimensionality of the encoded history
                representation. Default is 1024.
            input_xcand_to_local_nn:
                Whether to input the candidate points to the local neural network.
            input_xcand_to_final_mlp:
                Whether to input the candidate points to the final MLP.
            subtract_x_cand_from_x_hist:
                Whether to subtract the candidate points from the historical points
                before passing them to the local neural network.
            activation_at_end_pointnet (bool, default: True):
                Whether to apply the activation function at the end of the PointNet.
            layer_norm_pointnet (bool, default: False):
                Whether to use layer normalization in the PointNet.
            dropout_pointnet (float, default: None):
                Dropout probability for the PointNet. Default is None.
            activation_pointnet:
                The activation function to use in the PointNet.
            include_best_y (bool, default: False):
                Whether to include the best y value in the input to the local neural
                network.
            subtract_best_y (bool, default: False):
                Whether to subtract the best y value from the historical y values
                before passing them to the local neural network.
            n_pointnets (int, default: 1):
                The number of PointNets to use. Default is 1.
            max_history_input (int, optional):
                The maximum number of historical points to consider. If None,
                all historical points are used.
        """
        super().__init__()

        assert isinstance(n_pointnets, int) and n_pointnets >= 1

        if not (input_xcand_to_local_nn or input_xcand_to_final_mlp \
                or subtract_x_cand_from_x_hist):
            raise ValueError("At least one of input_xcand_to_local_nn, "
                             "input_xcand_to_final_mlp, or subtract_x_cand_from_x_hist "
                              "must be True.")
        self.input_xcand_to_local_nn = input_xcand_to_local_nn
        self.input_xcand_to_final_mlp = input_xcand_to_final_mlp
        self.input_acqf_params_to_local_nn = input_acqf_params_to_local_nn
        self.input_acqf_params_to_final_mlp = input_acqf_params_to_final_mlp
        self.subtract_x_cand_from_x_hist = subtract_x_cand_from_x_hist

        if not (isinstance(n_acqf_params, int) and n_acqf_params >= 0):
            raise ValueError("n_acqf_params should be a non-negative integer.")
        self._n_acqf_params = n_acqf_params
        self._n_hist_out = n_hist_out

        input_dim = dimension + n_hist_out + int(include_best_y) * n_hist_out \
            + (dimension if input_xcand_to_local_nn else 0) \
            + (n_acqf_params if input_acqf_params_to_local_nn else 0)

        history_encoder = "pointnet" # Temporary
        if history_encoder == "pointnet":
            pointnet_kwargs = dict(activation_at_end=activation_at_end_pointnet,
                    layer_norm_before_end=layer_norm_pointnet,
                    layer_norm_at_end=layer_norm_pointnet,
                    dropout=dropout_pointnet,
                    dropout_at_end=True,
                    activation=activation_pointnet)
            if n_pointnets == 1:
                self.history_encoder_net = PointNetLayer(
                    input_dim, history_enc_hidden_dims,
                    encoded_history_dim, pooling, **pointnet_kwargs)
            else:
                kwargs_list = [pointnet_kwargs] * n_pointnets
                self.history_encoder_net = MultiLayerPointNet(
                    input_dim,
                    [history_enc_hidden_dims] * n_pointnets,
                    [encoded_history_dim] * n_pointnets,
                    [pooling] * n_pointnets,
                    kwargs_list, use_local_features=True)

            self._features_dim = encoded_history_dim + \
                (dimension if input_xcand_to_final_mlp else 0) \
                + (n_acqf_params if input_acqf_params_to_final_mlp else 0)
        self.dimension = dimension
        self.include_best_y = include_best_y
        self.subtract_best_y = subtract_best_y
        self.max_history_input = max_history_input
    
    def get_init_kwargs(self):
        ## Oh wait, I just realized that this is not needed because
        ## the AcquisitionFunctionNet body init params are stored explicitly
        ## rather than them extracted from the AcquisitionFunctionBody.
        ## (I wouldn't do it like this if I were to design it now, but it
        ## is too late to change it now because want to keep the already saved models.)
        ## But why not keep it like this just as an example -- it doesn't do anything.
        ret = super().get_init_kwargs()
        if not self.subtract_x_cand_from_x_hist:
            ret.pop("subtract_x_cand_from_x_hist", None)
        if self.max_history_input is None:
            ret.pop("max_history_input", None)
        return ret

    @property
    def output_dim(self) -> int:
        return self._features_dim
    
    @property
    def n_acqf_params(self) -> int:
        return self._n_acqf_params
    
    @property
    def n_hist_out(self) -> int:
        return self._n_hist_out

    def forward(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
                acqf_params:Optional[Tensor]=None,
                hist_mask:Optional[Tensor]=None,
                cand_mask:Optional[Tensor]=None) -> Tensor:
        y_hist = check_xy_dims(x_hist, y_hist, "x_hist", "y_hist",
                               expected_y_dim=self.n_hist_out)
        
        if self.include_best_y or self.subtract_best_y:
            best_f = get_best_y(y_hist, hist_mask).expand_as(y_hist)
            if self.subtract_best_y:
                y_hist = best_f - y_hist
            if self.include_best_y:
                y_hist = torch.cat((best_f, y_hist), dim=-1)
        
        n_hist = x_hist.size(-2)
        n_cand = x_cand.size(-2)
        
        if self.n_acqf_params > 0:
            if acqf_params is None:
                raise ValueError("acqf_params must be provided if n_acqf_params > 0.")
            if acqf_params.size(-1) != self.n_acqf_params:
                raise ValueError(f"acqf_params should have {self.n_acqf_params} values.")
        else:
            if acqf_params is not None:
                raise ValueError("acqf_params should not be provided if n_acqf_params=0.")
        
        if self.max_history_input is not None:
            if hist_mask is None:
                max_hist_length = n_hist
            else:
                max_hist_length = int(torch.sum(hist_mask, dim=-2).max().item())
            must_truncate_history_length = max_hist_length > self.max_history_input
        else:
            must_truncate_history_length = False
        
        input_xcand_local = self.input_xcand_to_local_nn
        input_acqf_params_local = self.n_acqf_params > 0 and \
            self.input_acqf_params_to_local_nn
        input_local = input_xcand_local or input_acqf_params_local \
            or self.subtract_x_cand_from_x_hist or must_truncate_history_length
        
        if input_local:
            # shape (*, n_cand, n_hist, dimension)
            x_cand_expanded = expand_dim(x_cand.unsqueeze(-2), -2, n_hist)

            # shape (*, n_cand, n_hist, dimension)
            x_hist_expanded = expand_dim(x_hist.unsqueeze(-3), -3, n_cand)

            # shape (*, n_cand, n_hist, n_hist_out)
            y_hist_expanded = expand_dim(y_hist.unsqueeze(-3), -3, n_cand)

            # shape (*, n_cand, n_hist, 1)
            hist_mask_expanded = None if hist_mask is None \
                else expand_dim(hist_mask.unsqueeze(-3), -3, n_cand)

        if must_truncate_history_length:
            # shape (*, n_cand, n_hist, 1)
            neg_squared_distances = -((x_hist_expanded - x_cand_expanded) ** 2).sum(dim=-1, keepdim=True)
            if hist_mask is not None:
                neg_squared_distances = add_neg_inf_for_max(neg_squared_distances, hist_mask_expanded)
            # shape (*, n_cand, max_history_input, 1)
            _, topk_indices = torch.topk(
                neg_squared_distances, self.max_history_input, dim=-2)

            n_hist = self.max_history_input
            x_cand_expanded = expand_dim(x_cand.unsqueeze(-2), -2, n_hist)
            x_hist_expanded = torch.gather(
                x_hist_expanded, -2,
                topk_indices.expand(*topk_indices.shape[:-1], x_hist_expanded.size(-1)))
            y_hist_expanded = torch.gather(
                y_hist_expanded, -2,
                topk_indices.expand(*topk_indices.shape[:-1], y_hist_expanded.size(-1)))
            hist_mask_expanded = None if hist_mask is None else torch.gather(
                hist_mask_expanded, -2,
                topk_indices.expand(*topk_indices.shape[:-1], hist_mask_expanded.size(-1)))

        if input_local:
            inputs = [
                x_hist_expanded - x_cand_expanded \
                    if self.subtract_x_cand_from_x_hist else x_hist_expanded,
                y_hist_expanded]
            if input_acqf_params_local:
                # shape (*, n_cand, n_hist, n_acqf_params)
                acqf_params_expanded = expand_dim(
                    acqf_params.unsqueeze(-2), -2, n_hist)
                inputs = [acqf_params_expanded] + inputs
            if input_xcand_local:
                inputs = [x_cand_expanded] + inputs
            
            # shape (*, n_cand, n_hist, input_dim)
            nn_input = torch.cat(inputs, dim=-1)
            
            # shape (*, n_cand, encoded_history_dim)
            out = self.history_encoder_net(nn_input, mask=hist_mask_expanded, keepdim=False)
            logger.debug(f"out.shape: {out.shape}")
        else:
            # shape (*, n_hist, dimension+n_hist_out)
            xy_hist = torch.cat((x_hist, y_hist), dim=-1)
            # shape (*, 1, encoded_history_dim)
            out = self.history_encoder_net(xy_hist, mask=hist_mask, keepdim=True)
            # Prepare input to the acquisition function network final dense layer
            # shape (*, n_cand, encoded_history_dim)
            out = expand_dim(out, -2, n_cand)
        
        add_x_cand = self.input_xcand_to_final_mlp
        add_acqf_params = self.n_acqf_params > 0 and self.input_acqf_params_to_final_mlp
        if add_x_cand or add_acqf_params:
            items = [out]
            if add_acqf_params:
                items = [acqf_params] + items
            if add_x_cand:
                items = [x_cand] + items
            out = torch.cat(items, dim=-1)

        return out


# THE FOLLOWING WAS WRITTEN BY CHATGPT (I don't know how it works or whether it is what
# I would have written).
class AcquisitionFunctionBodyTransformerNP(AcquisitionFunctionBodyFixedHistoryOutputDim):
    r"""AcquisitionFunctionBodyTransformerNP uses a transformer-based encoder
    (with cross-attention) to process history and candidate data for computing features
    for an acquisition function.
    
    Args:
        dimension (int):
            Dimensionality of the input x.
        n_hist_out (int):
            Number of outputs (i.e. function values) per historical point.
        n_acqf_params (int, optional):
            Number of additional parameters (e.g. from acqf_params) to be appended
            to each candidate. Default is 0.
        hidden_dim (int, optional):
            Internal hidden dimension used for both the history encoder and attention.
            Default is 128.
        num_heads (int, optional):
            Number of attention heads. Default is 4.
        num_layers (int, optional):
            Number of transformer encoder layers. Default is 2.
        dropout (float, optional):
            Dropout probability. Default is None.
        include_best_y (bool, optional):
            Whether to augment y_hist by concatenating the best y value (across history).
            Default is False.
        input_xcand_to_final_mlp (bool, optional):
            If True, the candidate's raw input (with acqf parameters, if any) is concatenated
            to the attended features. This is analogous to the flag in the PointNet version.
            Default is False.
    """
    def __init__(
        self,
        dimension: int,
        n_hist_out: int,
        n_acqf_params: int = 0,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: Optional[float] = 0.0,
        include_best_y: bool = False,
        input_xcand_to_final_mlp: bool = False,
    ):
        super().__init__()
        self.dimension = dimension
        self._n_hist_out = n_hist_out
        self._n_acqf_params = n_acqf_params
        self.include_best_y = include_best_y
        self.input_xcand_to_final_mlp = input_xcand_to_final_mlp

        # The context comes from concatenating x_hist and y_hist.
        # Its dimension is (dimension + n_hist_out).
        self.context_proj = nn.Linear(dimension + n_hist_out, hidden_dim)

        # Candidate representation: if n_acqf_params > 0, then x_cand
        # will be concatenated with acqf_params, making its input dim (dimension + n_acqf_params).
        candidate_in_dim = dimension + n_acqf_params
        self.candidate_proj = nn.Linear(candidate_in_dim, hidden_dim)

        # Transformer encoder layers over the history (context).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention: candidates (queries) attend over the encoded history (keys/values).
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        # Final feature dimension: if we choose to append the raw candidate representation,
        # the output feature dimension is hidden_dim + candidate_in_dim, otherwise just hidden_dim.
        if self.input_xcand_to_final_mlp:
            self._features_dim = hidden_dim + candidate_in_dim
        else:
            self._features_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        r"""Returns the feature dimension that will be fed to the final MLP."""
        return self._features_dim

    @property
    def n_acqf_params(self) -> int:
        r"""Number of extra acquisition function parameters."""
        return self._n_acqf_params

    @property
    def n_hist_out(self) -> int:
        r"""Number of history outputs."""
        return self._n_hist_out

    def forward(
        self,
        x_hist: Tensor,
        y_hist: Tensor,
        x_cand: Tensor,
        acqf_params: Optional[Tensor] = None,
        hist_mask: Optional[Tensor] = None,
        cand_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Ensure y_hist has the expected output dimension.
        y_hist = check_xy_dims(x_hist, y_hist, "x_hist", "y_hist", expected_y_dim=self.n_hist_out)

        if self.include_best_y:
            # Augment y_hist with its best value.
            y_hist = concat_y_hist_with_best_y(y_hist, hist_mask, subtract=False)

        if self.n_acqf_params > 0:
            if acqf_params is None:
                raise ValueError("acqf_params must be provided if n_acqf_params > 0.")
            acqf_params = check_xy_dims(x_cand, acqf_params, "x_cand", "acqf_params")
            x_cand = torch.cat((x_cand, acqf_params), dim=-1)
        elif acqf_params is not None:
            raise ValueError("acqf_params should not be provided if n_acqf_params == 0.")

        # Encode the history.
        # Concatenate x_hist and y_hist along the last dimension.
        context = torch.cat((x_hist, y_hist), dim=-1)  # shape: (*, n_hist, dimension+n_hist_out)
        # Project the context.
        context_encoded = self.context_proj(context)  # shape: (*, n_hist, hidden_dim)

        # Flatten the leading batch dimensions for the transformer.
        orig_batch_shape = context_encoded.shape[:-2]
        n_hist = context_encoded.shape[-2]
        hidden_dim = context_encoded.shape[-1]
        context_flat = context_encoded.reshape(-1, n_hist, hidden_dim)  # shape: (B, n_hist, hidden_dim)
        # Transformer expects (seq_len, batch, embed_dim)
        context_flat = context_flat.transpose(0, 1)  # shape: (n_hist, B, hidden_dim)
        encoded_context = self.transformer_encoder(context_flat)  # shape: (n_hist, B, hidden_dim)
        encoded_context = encoded_context.transpose(0, 1).reshape(*orig_batch_shape, n_hist, hidden_dim)

        # Prepare candidate representations.
        candidate_proj = self.candidate_proj(x_cand)  # shape: (*, n_cand, hidden_dim)
        orig_candidate = x_cand  # Save raw candidate (after concatenation if acqf_params were added)

        # Flatten candidate batch dims.
        cand_shape = candidate_proj.shape[:-2]
        n_cand = candidate_proj.shape[-2]
        candidate_flat = candidate_proj.reshape(-1, n_cand, hidden_dim).transpose(0, 1)  # (n_cand, B, hidden_dim)

        # Also flatten the encoded context.
        context_flat = encoded_context.reshape(-1, n_hist, hidden_dim).transpose(0, 1)  # (n_hist, B, hidden_dim)
        # Optionally, if hist_mask is provided, prepare a key_padding_mask.
        if hist_mask is not None:
            # Expected shape for key_padding_mask is (B, n_hist) with True for positions to mask.
            key_padding_mask = ~hist_mask.squeeze(-1).reshape(-1, n_hist)
        else:
            key_padding_mask = None

        # Use candidate as query to attend over context.
        attn_output, _ = self.attention(
            query=candidate_flat,
            key=context_flat,
            value=context_flat,
            key_padding_mask=key_padding_mask,
        )
        # Reshape attention output back to candidate shape.
        attn_output = attn_output.transpose(0, 1).reshape(*cand_shape, n_cand, hidden_dim)  # (*, n_cand, hidden_dim)

        # Optionally, concatenate raw candidate representation.
        if self.input_xcand_to_final_mlp:
            out = torch.cat((orig_candidate, attn_output), dim=-1)
        else:
            out = attn_output

        return out


def safe_log(x):
    if x is None:
        return None
    if isinstance(x, float):
        return math.log(x) if x > 0.0 else 0.0
    return torch.where(x > 0, torch.log(x), torch.zeros_like(x))


class GittinsAcquisitionFunctionNet(AcquisitionFunctionNet):
    r"""Gittins index acquisition function neural net.
    
    Attributes:
        output_dim (int):
            The output dimension of the acquisition function.
        variable_lambda (bool):
            Whether to use a variable lambda that is input to the NN.
        costs_in_history (bool):
            Whether the past costs are in the history.
        cost_is_input (bool):
            Whether the cost of a candidate is an input to the NN
            (in this case we can know the cost before evaluation).
    """
    def __init__(self,
                 af_class,
                 variable_lambda:bool,
                 costs_in_history:bool,
                 cost_is_input:bool,
                 assume_y_independent_cost:bool=False,
                 **init_kwargs):
        r"""Initialize the GittinsAcquisitionFunctionNet class.
        
        Args:
            af_class:
                The class of the acquisition function to use.
                Subclass of both AcquisitionFunctionNetFixedHistoryOutputDim
                and AcquisitionFunctionNetGivenOutputDim.
            variable_lambda (bool):
                Whether to use a variable lambda that is input to the NN.
            costs_in_history (bool):
                Whether the past costs are in the history.
            cost_is_input (bool):
                Whether the cost of a candidate is an input to the NN
                (in this case we can know the cost before evaluation).
            assume_y_independent_cost (bool, default: False):
                Whether to assume that the cost is independent of the function value
                conditioned on the history and candidate point,
                so that only lambda*cost is input rather than (lambda*cost, cost).
                ONLY applicable if both variable_lambda=True and cost_is_input=True.
            **init_kwargs:
                Arguments to pass to the acquisition function class __init__
                except for the dimension argument.
        
        `costs_in_history` and `cost_is_input` can be chosen based on the following
        rules, if `heterogeneous_costs` tells whether costs are heterogeneous and
        `known_cost` tells whether the cost is known before evaluation:
        if heterogeneous_costs:
            if known_cost:
                costs_in_history: Could be False since we already know the cost.
                    Could also set True if we think that past costs tell us
                    something about future y values.
                cost_is_input: True
            else:
                costs_in_history: True
                cost_is_input: False
        else:
            costs_in_history: False
            cost_is_input: False
        """
        super().__init__()

        if type(variable_lambda) is not bool:
            raise ValueError("variable_lambda should be bool")
        if type(costs_in_history) is not bool:
            raise ValueError("costs_in_history should be bool")
        if type(cost_is_input) is not bool:
            raise ValueError("cost_is_input should be bool")
        self.variable_lambda = variable_lambda
        self.costs_in_history = costs_in_history
        self.cost_is_input = cost_is_input

        if variable_lambda and cost_is_input:
            if type(assume_y_independent_cost) is not bool:
                raise ValueError("assume_y_independent_cost should be bool")
            n_acqf_params = 1 if assume_y_independent_cost else 2
            self.assume_y_independent_cost = assume_y_independent_cost
        else:
            n_acqf_params = variable_lambda + cost_is_input
        check_subclass(af_class, "af_class", AcquisitionFunctionNetFixedHistoryOutputDim)
        check_subclass(af_class, "af_class", AcquisitionFunctionNetGivenOutputDim)

        additional_kwargs = {
            'n_hist_out': 1 + costs_in_history, # for AcquisitionFunctionNetFixedHistoryOutputDim
            'output_dim': 1, # for AcquisitionFunctionNetGivenOutputDim
        }
        if safe_issubclass(af_class, ParameterizedAcquisitionFunctionNet):
            additional_kwargs['n_acqf_params'] = n_acqf_params
        elif n_acqf_params > 0:
            raise ValueError(
                f"af_class={af_class.__name__} should be a subclass of "
                "ParameterizedAcquisitionFunctionNet since "
                f"there are {n_acqf_params}>0 parameters")
        
        tmp = {'n_hist_out', 'output_dim', 'n_acqf_params'}
        for key in init_kwargs:
            if key in tmp:
                raise ValueError(
                    "GittinsAcquisitionFunctionNet.__init__: init_kwargs should not "
                    f"contain {key}")
        
        self.base_model = af_class(**additional_kwargs, **init_kwargs)
    
    @property
    def output_dim(self) -> int:
        return 1

    def forward(self,
                x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
                lambda_cand:Optional[Union[Tensor, float]]=None,
                cost_hist:Optional[Tensor]=None,
                cost_cand:Optional[Tensor]=None,
                hist_mask:Optional[Tensor]=None, cand_mask:Optional[Tensor]=None,
                is_log:bool=False):
        r"""Forward pass of the acquisition function network.
        Args:
            x_hist (torch.Tensor):
                A `batch_shape x n_hist x d` tensor of training features.
            y_hist (torch.Tensor):
                A `batch_shape x n_hist x 1` tensor of training observations.
            x_cand (torch.Tensor):
                Candidate input tensor with shape `batch_shape x n_cand x d`.
            lambda_cand (torch.Tensor or float, optional):
                Either a `batch_shape x n_cand` or `batch_shape x n_cand x 1` tensor
                of lambda values for the candidate points, or a float which means the
                same value of lambda should be used for all candidate points.
            cost_hist (torch.Tensor, optional):
                A `batch_shape x n_hist` or `batch_shape x n_hist x 1` tensor of costs
                for the history points.
            cost_cand (torch.Tensor, optional):
                A `batch_shape x n_cand` or `batch_shape x n_cand x 1` tensor of costs
                for the candidate points.
            hist_mask (torch.Tensor, optional):
                Mask tensor for the history inputs with shape `batch_shape x n_hist`
                or `batch_shape x n_hist x 1`. If None, then mask is all ones.
            cand_mask (torch.Tensor, optional):
                Mask tensor for the candidate inputs with shape `batch_shape x n_cand`
                or `batch_shape x n_cand x 1`. If None, then mask is all ones.
            is_log (bool, default: False):
                Whether lambda_cand, cost_hist, and cost_cand (if applicable) are
                already log-transformed.
                If True, they will be unchanged. If False, their log will be taken
                before passing them to the acquisition function network.

        Returns:
            torch.Tensor: A `batch_shape x n_cand x 1` tensor of acquisition
            values.
        """
        ## Handle the history of y and cost
        y_hist = check_xy_dims(x_hist, y_hist, "x_hist", "y_hist", expected_y_dim=1)
        if self.costs_in_history:
            if cost_hist is None:
                raise ValueError("cost_hist must be specified if costs_in_history=True")
            cost_hist = check_xy_dims(x_hist, cost_hist,
                                      "x_hist", "cost_hist", expected_y_dim=1)
            if not is_log:
                cost_hist = safe_log(cost_hist)
            y_hist = torch.cat((y_hist, cost_hist), dim=-1)
        elif cost_hist is not None:
            raise ValueError(
                "cost_hist should not be specified if costs_in_history=False")

        ## Make sure the lambda and cost are specified or not as expected
        if self.variable_lambda:
            if lambda_cand is None:
                raise ValueError(
                    "lambda_cand must be specified if variable_lambda=True")
            if not isinstance(lambda_cand, float): # should be a Tensor
                lambda_cand = check_xy_dims(x_cand, lambda_cand,
                                            "x_cand", "lambda_cand", expected_y_dim=1)
        elif lambda_cand is not None:
            raise ValueError(
                "lambda_cand should not be specified if variable_lambda=False")
        if self.cost_is_input:
            if cost_cand is None:
                raise ValueError("cost_cand must be specified if cost_is_input=True")
            cost_cand = check_xy_dims(x_cand, cost_cand,
                                      "x_cand", "cost_cand", expected_y_dim=1)
        elif cost_cand is not None:
            raise ValueError("cost_cand should not be specified if cost_is_input=False")
        
        call_kwargs = dict(
            x_hist=x_hist,
            y_hist=y_hist,
            x_cand=x_cand,
            hist_mask=hist_mask,
            cand_mask=cand_mask
        )

        if self.variable_lambda or self.cost_is_input:
            # Log-transform the lambda and cost if necessary
            if is_log:
                log_lambda_cand = lambda_cand
                log_cost_cand = cost_cand
            else:
                log_lambda_cand = safe_log(lambda_cand)
                log_cost_cand = safe_log(cost_cand)
            
            if isinstance(log_lambda_cand, float):
                log_lambda_cand = torch.full(
                    list(x_cand.shape[:-1]) + [1],
                    log_lambda_cand,
                    dtype=x_cand.dtype, layout=x_cand.layout, device=x_cand.device)

            # Set the necessary acqf_params
            if self.variable_lambda and self.cost_is_input:
                lambda_cost_cand = log_lambda_cand + log_cost_cand
                if self.assume_y_independent_cost:
                    acqf_params = lambda_cost_cand
                else:
                    acqf_params = torch.cat((lambda_cost_cand, log_cost_cand), dim=-1)
            elif self.variable_lambda:
                acqf_params = log_lambda_cand
            elif self.cost_is_input:
                acqf_params = log_cost_cand

            # Add acqf_params
            call_kwargs['acqf_params'] = acqf_params
        
        return self.base_model(**call_kwargs)


class ExpectedImprovementAcquisitionFunctionNet(AcquisitionFunctionNet):
    def __init__(
            self,
            af_body_class,
            af_body_init_params:dict,
            af_head_init_params:dict,
            standardize_outcomes=False):
        super().__init__()
        self.base_model = TwoPartAcquisitionFunctionNetFixedHistoryOutputDim(
            output_dim=1,
            n_acqf_params=0,
            n_hist_out=1,
            af_body_class=af_body_class,
            af_head_class=AcquisitionFunctionNetFinalMLPSoftmaxExponentiate,
            af_body_init_params=af_body_init_params,
            af_head_init_params=af_head_init_params,
            standardize_outcomes=standardize_outcomes
        )
    
    @property
    def output_dim(self) -> int:
        return 1
    
    def forward(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
                hist_mask:Optional[Tensor]=None, cand_mask:Optional[Tensor]=None,
                exponentiate=False, softmax=False) -> Tensor:
        return self.base_model(
            x_hist, y_hist, x_cand,
            hist_mask=hist_mask, cand_mask=cand_mask,
            exponentiate=exponentiate, softmax=softmax)
    
    def get_alpha(self):
        return self.base_model.af_head.get_alpha()

    def set_alpha(self, val):
        self.base_model.af_head.set_alpha(val)
    
    def get_beta(self):
        return self.base_model.af_head.get_beta()

    @property
    def includes_alpha(self) -> bool:
        """Whether the acquisition function includes an alpha parameter."""
        return self.base_model.af_head.includes_alpha
    
    @property
    def transform(self):
        return self.base_model.af_head.transform


""""The following two classes, AcquisitionFunctionNetModel and
AcquisitionFunctionNetAcquisitionFunction, define the Model and AcquisitionFunction
classes that are necessary to be defined so that the AcquisitionFunctionNet can be used
in the BoTorch API to run Bayesian optimization loops."""

class AcquisitionFunctionNetModel(Model):
    """In this case, the model is the acquisition function network itself.
    So it's kind of silly to have this intermediate between the NN and the
    acquisition function, but it's necessary for the BoTorch API."""
    
    def __init__(self,
                 model: AcquisitionFunctionNet,
                 train_X: Optional[Tensor]=None,
                 train_Y: Optional[Tensor]=None):
        """
        Args:
            model: The acquisition function network model.
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` or `batch_shape x n` tensor of
                training observations, where `m` is the number of outputs.
        """
        super().__init__()
        if not isinstance(model, AcquisitionFunctionNet):
            raise ValueError("model must be an instance of AcquisitionFunctionNet.")
        model.eval()
        self.model = model

        if train_X is not None and train_Y is not None:
            # Check that the dimensions are compatible, and add an output dimension
            # to train_Y if there is none
            train_Y = check_xy_dims(train_X, train_Y, "train_X", "train_Y")
            
            # Add a batch dimension to both if they don't have it
            # Don't think I need to do this actually
            # train_X = add_tbatch_dimension(train_X, "train_X")
            # train_Y = add_tbatch_dimension(train_Y, "train_Y")
            
            self.train_X = train_X
            self.train_Y = train_Y
        elif train_X is None and train_Y is None:
            self.train_X = None
            self.train_Y = None
        else:
            raise ValueError("Both train_X and train_Y must be provided or neither.")
    
    def posterior(self, *args, **kwargs):
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support posterior inference.")
    
    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1 # Only supporting 1 output (for now at least)
    
    def subset_output(self, idcs: Sequence[int]):
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support output subsetting.")

    def condition_on_observations(self, X: Tensor, Y: Tensor) -> Model:
        """This doesn't have the original utility from GPyTorch --
        it is just as efficient as just making a new model.
        But it is here for convenience just in"""
        if self.train_X is None:
            new_X, new_Y = X, Y
        else:
            # Check dimensions & add output dimension to Y if there is none
            Y = check_xy_dims(X, Y, "X", "Y")
            X = match_batch_shape(X, self.train_X)
            Y = match_batch_shape(Y, self.train_Y)
            new_X = torch.cat((self.train_X, X), dim=-2)
            new_Y = torch.cat((self.train_Y, Y), dim=-2)
        return self.__class__(self.model, new_X, new_Y)

    def forward(self, X: Tensor, **kwargs) -> Tensor:
        """Forward pass of the acquisition function network.

        Args:
            X (Tensor): The input tensor of shape `(batch_shape) x n_cand x d`.
            **kwargs: Keyword arguments to pass to the model's `forward` method.
                If any are unspecified, then the default values will be used.

        Returns:
            Tensor: The output tensor of shape `(batch_shape) x n_cand x output_dim`.

        Raises:
            RuntimeError: If the encoded history is not available.
        """
        if self.train_X is None:
            raise RuntimeError("Cannot make predictions without conditioning on data.")
        
        # Don't think I need to do this actually
        # (would also need to do same to train_Y I think)
        # train_X = self.train_X
        # if X.dim() > train_X.dim():
        #     train_X = match_batch_shape(train_X, X)
        # else:
        #     X = match_batch_shape(X, train_X)

        logger.debug(f"In {self.__class__.__name__}.forward, X.shape = {X.shape}")

        ret = self.model(self.train_X, self.train_Y, X, **kwargs)
        assert ret.shape[:-1] == X.shape[:-1]
        return ret


class AcquisitionFunctionNetAcquisitionFunction(AcquisitionFunction):
    r"""Acquisition function for a neural network model, used for BoTorch API."""
    def __init__(self, model: AcquisitionFunctionNetModel, **kwargs):
        """
        Args:
            model: The acquisition function network model.
            **kwargs: Any keyword arguments to pass to the AF's `forward` method.
        """
        super().__init__(model=model) # sets self.model = model
        self.kwargs = kwargs
    
    @classmethod
    def from_net(cls,
                 model: AcquisitionFunctionNet,
                 train_X: Optional[Tensor]=None,
                 train_Y: Optional[Tensor]=None,
                 **kwargs) -> "AcquisitionFunctionNetAcquisitionFunction":
        return cls(AcquisitionFunctionNetModel(model, train_X, train_Y), **kwargs)
    
    # They all do this
    # https://botorch.org/api/utils.html#botorch.utils.transforms.t_batch_mode_transform
    # https://botorch.org/api/_modules/botorch/utils/transforms.html#t_batch_mode_transform
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each, where q=1.

        Returns:
            A `(b)`-dim Tensor of acquisition function values at the given
            design points `X`.
        """
        logger.debug(f"In {self.__class__.__name__}.forward, X.shape = {X.shape}")
        assert X.size(-2) == 1 # Guaranteed by t_batch_mode_transform
        X = X.squeeze(-2) # Make shape (b) x d

        # shape (b)
        output = self.model(X, **self.kwargs)
        if output.shape[-1] != 1:
            raise UnsupportedError("Only one output dimension is supported")
        output = output.squeeze(-1)
        assert output.shape == X.shape[:-1]
        return output

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support pending points.")
