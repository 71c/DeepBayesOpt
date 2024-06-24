from collections import OrderedDict
from typing import List, Optional, Sequence, Union
import warnings
import torch
from torch import nn
from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import UnsupportedError
from botorch.utils.transforms import t_batch_mode_transform, match_batch_shape
from abc import ABC, abstractmethod

import logging

from utils import pad_tensor
logging.basicConfig(level=logging.WARNING)

# Set to True to enable debug logging
DEBUG = True

# Create a logger for your application
logger = logging.getLogger('acquisition_function_net')
# Configure the logging
logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)


def expand_dim(tensor, dim, k):
    new_shape = list(tensor.shape)
    new_shape[dim] = k
    return tensor.expand(*new_shape)


def check_xy_dims_add_y_output_dim(x: Tensor, y: Union[Tensor, None],
                                   x_name: str, y_name: str) -> Tensor:
    """Check that the dimensions of x and y are as expected, and add an output
    dimension to y if there is none.
    
    Args:
        x (Tensor): The input tensor x.
        y (Tensor or None): The input tensor y.
        x_name (str): The name of the x tensor.
        y_name (str): The name of the y tensor.
    
    Returns:
        Tensor: The modified y tensor, with an added output dimension
        if it was missing.
    """
    if y is None:
        return y
    if x.dim() < 2:
        raise ValueError(
            f"{x_name} must have at least 2 dimensions,"
            f" but has only {x.dim()} dimensions."
        )
    if x.dim() != y.dim():
        if (x.dim() - y.dim() == 1) and (x.shape[:-1] == y.shape):
            y = y.unsqueeze(-1)
        else:
            raise ValueError(f"{x_name} and {y_name} must have the same number of dimensions or {y_name} must have one fewer dimension than {x_name}.")
    if x.size(-2) != y.size(-2):
        raise ValueError(f"{x_name} and {y_name} must have the same number of points in the history dimension.")
    if y.size(-1) != 1:
        raise ValueError(f"{y_name} must have one output dimension.")
    return y


def add_tbatch_dimension(x: Tensor, x_name: str):
    if x.dim() < 2:
        raise ValueError(
            f"{x_name} must have at least 2 dimensions,"
            f" but has only {x.dim()} dimensions."
        )
    return x if x.dim() > 2 else x.unsqueeze(0)


# https://discuss.pytorch.org/t/apply-mask-softmax/14212/13
def masked_softmax(vec, mask, dim=-1):
    if mask is None:
        return nn.functional.softmax(vec, dim=dim)
    neg_inf = torch.zeros_like(vec)
    mask_expanded = expand_dim(mask, -1, vec.size(-1))
    neg_inf[~mask_expanded] = float("-inf")
    masked_vec = vec + neg_inf

    # masked_vec = vec * mask

    max_vec = torch.max(masked_vec, dim=dim, keepdim=True).values
    exps = torch.exp(vec-max_vec)
    masked_exps = exps * mask
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = (masked_sums == 0)
    masked_sums += zeros
    return masked_exps/masked_sums


class SoftmaxOrExponentiateLayer(nn.Module):
    def __init__(self, softmax_dim=-1,
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0):
        """Initialize the SoftmaxOrExponentiateLayer class.

        Args:
            softmax_dim (int, default: -1):
                The dimension along which to apply the softmax.
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
        """
        super().__init__()
        self.softmax_dim = softmax_dim
        self.includes_alpha = include_alpha
        if include_alpha:
            if learn_alpha:
                self._log_alpha = nn.Parameter(torch.tensor(0.0))
            else:
                # register a buffer which is _log_alpha but not a parameter
                self.register_buffer("_log_alpha", torch.tensor(0.0))
            self.set_alpha(initial_alpha)

    def get_alpha(self):
        if not self.includes_alpha:
            raise ValueError("Model does not include alpha.")
        return torch.exp(self._log_alpha)
    
    def set_alpha(self, val):
        if not self.includes_alpha:
            raise ValueError("Model does not include alpha.")
        val = torch.log(torch.as_tensor(val))
        try:
            self._log_alpha.data.copy_(val.expand_as(self._log_alpha))
        except RuntimeError:
            self._log_alpha.data = val
    
    def forward(self, x, mask=None, exponentiate=False, softmax=False):
        """Compute the acquisition function.

        Args:
            x (torch.Tensor):
                input tensor
            mask (torch.Tensor, optional):
                mask tensor
            exponentiate (bool, optional): Whether to exponentiate the output.
                Default is False. For EI, False corresponds to the log of the
                acquisition function (e.g. log EI), and True corresponds to
                the acquisition function itself (e.g. EI).
                Only applies if self.includes_alpha is False.
            softmax (bool, optional): Whether to apply softmax to the output.
                Only applies if self.includes_alpha is True. Default is False.

        Returns:
            torch.Tensor: Exponentiated or softmaxed tensor.
        """
        if softmax:
            if self.includes_alpha:
                x = x * self.get_alpha()
            if mask is None:
                x = nn.functional.softmax(x, dim=self.softmax_dim)
            else:
                x = masked_softmax(x, mask, dim=self.softmax_dim)
        elif self.training and self.includes_alpha:
            warnings.warn("The model is in training mode but softmax is False "
                          "and alpha is included in the model. "
                          "If this is unintentional, set softmax=True.")

        if exponentiate:
            if self.includes_alpha:
                raise ValueError(
                    "It doesn't make sense to exponentiate and use alpha at "
                    "the same time. Should set softmax=True instead of exponentiate=True.")
            if softmax:
                raise ValueError(
                    "It doesn't make sense to exponentiate and use softmax at "
                    "the same time. Should set softmax=False instead.")
            # x = torch.exp(x)
            x = nn.functional.softplus(x)

        return x


class Dense(nn.Sequential):
    """Dense neural network with ReLU activations."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]=[256, 64],
                 output_dim: int=1, activation_at_end=False,
                 layer_norm_before_end=False, layer_norm_at_end=False,
                 activation=nn.ReLU):
        """
        Args:
            input_dim (int):
                The dimensionality of the input.
            hidden_dims (Sequence[int], default: [256, 64]):
                A sequence of integers representing the sizes of the hidden
                layers.
            output_dim (int, default: 1):
                The dimensionality of the output.
            activation_at_end (bool, default: False):
                Whether to apply the activation function at the end.
        """
        layer_widths = [input_dim] + list(hidden_dims) + [output_dim]
        n_layers = len(layer_widths) - 1
        
        layers = []
        for i in range(n_layers):
            in_dim, out_dim = layer_widths[i], layer_widths[i+1]
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            add_layer_norm = layer_norm_at_end if i == n_layers - 1 else layer_norm_before_end
            if add_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            
            if i != n_layers - 1 or activation_at_end:
                # layers.append(nn.ReLU())
                # layers.append(nn.Softplus())
                # layers.append(nn.SELU())
                layers.append(activation())
        
        super().__init__(*layers)


def add_neg_inf_for_max(x, mask):
    neg_inf = torch.zeros_like(x)
    mask_expanded = expand_dim(mask, -1, x.size(-1))
    neg_inf[~mask_expanded] = float("-inf")
    return x + neg_inf


POOLING_METHODS = {"max", "sum", "experiment1", "experiment2", "experiment3"}
class PointNetLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]=[256, 256],
                 output_dim: int=1024, pooling="max", **dense_kwargs):
        super().__init__()

        if pooling == "experiment1":
            output_dim = output_dim * 2 + 3
        elif pooling == "experiment2":
            output_dim = output_dim + 2
        elif pooling == "experiment3":
            output_dim = output_dim + 1

        self.network = Dense(input_dim, hidden_dims, output_dim, **dense_kwargs)

        if pooling in POOLING_METHODS:
            self.pooling = pooling
        else:
            raise ValueError(f"pooling must be one of {POOLING_METHODS}")
    
    def forward(self, x, mask=None, keepdim=True, return_local_features=False):
        """Computes the output of the PointNet layer.

        Args:
            x (torch.Tensor): Input tensor with shape (*, n, input_dim).
            mask (torch.Tensor): Mask tensor with shape (*, n) or (*, n, 1).
                If None, then mask is all ones.
            keepdim (boolean): whether to remove the `n` dimension from output

        Returns:
            torch.Tensor: Output with shape (*, 1, output_dim) if keepdim=True
            or with shape (*, output_dim) if keepdim=False
        """
        # shape (*, n, output_dim)
        local_features = self.network(x)

        # for name, param in self.network.named_parameters():
        #     logger.debug(f"Parameter: {name}")
        #     logger.debug(f"Parameter shape: {param.shape}")
        #     logger.debug(f"Parameter value: {param}")

        # Mask out the padded values. It is sufficient to mask at the end.
        if mask is not None:
            # shape (*, n, 1)
            mask = check_xy_dims_add_y_output_dim(x, mask, "x", "mask")
            local_features = local_features * mask
        
        # "global feature"
        if self.pooling == "sum":
            ret = torch.sum(local_features, dim=-2, keepdim=keepdim)
        elif self.pooling == "max":
            ret = torch.max(
                # This works for maxing. If ReLU is applied at the end, then
                # we could instead just use the one for summing.
                local_features if mask is None else add_neg_inf_for_max(local_features, mask),
                dim=-2, keepdim=keepdim).values
            if mask is not None:
                ret[ret == float("-inf")] = 0.0
        elif self.pooling == "experiment1":
            output_dim = (local_features.size(-1) - 3) // 2
            softmax_in = local_features[..., :output_dim]
            vals = local_features[..., output_dim:-3]
            alphas_1 = nn.functional.softplus(local_features[..., -3:-2])
            alphas_2 = nn.functional.softplus(local_features[..., -2:-1])
            betas = nn.functional.sigmoid(local_features[..., -1:])
            
            # softmaxes_1 = masked_softmax(softmax_in * alphas_1, mask, dim=-2)
            # softmaxes_2 = masked_softmax(vals * alphas_2, mask, dim=-2)
            # tmp = betas * softmaxes_1 + (1 - betas) * softmaxes_2
            # ret = torch.sum(vals * tmp, dim=-2, keepdim=keepdim)

            tmp = betas * softmax_in * alphas_1 + (1 - betas) * vals * alphas_2
            softmaxes = masked_softmax(tmp, mask, dim=-2)
            ret = torch.sum(vals * softmaxes, dim=-2, keepdim=keepdim)
        elif self.pooling == "experiment2":
            output_dim = (local_features.size(-1) - 2) // 2
            softmax_in = local_features[..., :output_dim]
            vals = local_features[..., output_dim:-2]
            alphas_1 = nn.functional.softplus(local_features[..., -2:-1])
            alphas_2 = nn.functional.softplus(local_features[..., -1:])

            softmaxes_1 = masked_softmax(softmax_in * alphas_1, mask, dim=-2)
            softmaxes_2 = masked_softmax(vals * alphas_2, mask, dim=-2)
            out_1 = torch.sum(vals * softmaxes_1, dim=-2, keepdim=keepdim)
            out_2 = torch.sum(vals * softmaxes_2, dim=-2, keepdim=keepdim)
            ret = torch.cat((out_1, out_2), dim=-1)
        elif self.pooling == "experiment3":
            output_dim = (local_features.size(-1) - 1) // 2
            
            vals = local_features[..., :output_dim]
            ret_max = torch.max(
                vals if mask is None else add_neg_inf_for_max(vals, mask),
                dim=-2, keepdim=keepdim).values
            if mask is not None:
                ret_max[ret_max == float("-inf")] = 0.0
            
            softmax_in = local_features[..., output_dim:-1]
            alphas = nn.functional.softplus(local_features[..., -1:])
            softmaxes = masked_softmax(softmax_in * alphas, mask, dim=-2)
            out = torch.sum(vals * softmaxes, dim=-2, keepdim=keepdim)

            ret = torch.cat((ret_max, out), dim=-1)
        
        if return_local_features:
            return ret, local_features
        return ret


class MultiLayerPointNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[Sequence[int]],
                 output_dims: Sequence[int], poolings: Sequence[str],
                 dense_kwargs_list: List[dict], use_local_features=True):
        super().__init__()

        if not (len(hidden_dims) == len(output_dims) == len(poolings) == len(dense_kwargs_list)):
            raise ValueError("hidden_dims, output_dims, poolings, and dense_kwargs_list must have the same length.")
        
        n_layers = len(hidden_dims)
        
        if use_local_features:
            input_dims = [input_dim] + [output_dims[i-1] * 2 for i in range(1, n_layers)]
        else:
            input_dims = [input_dim] + [output_dims[i-1] + input_dim for i in range(1, n_layers)]
        self.use_local_features = use_local_features

        self.pointnets = nn.ModuleList([
            PointNetLayer(input_dims[i], hidden_dims[i], output_dims[i],
                          poolings[i], **dense_kwargs_list[i])
            for i in range(n_layers)
        ])
    
    def forward(self, x, mask=None, keepdim=True):
        """Computes the output of the PointNet layers.

        Args:
            x (torch.Tensor): Input tensor with shape (*, n, input_dim).
            mask (torch.Tensor): Mask tensor with shape (*, n) or (*, n, 1).
                If None, then mask is all ones.
            keepdim (boolean): whether to remove the `n` dimension from output

        Returns:
            torch.Tensor: Output with shape (*, 1, output_dim) if keepdim=True
            or with shape (*, output_dim) if keepdim=False
        """
        if self.use_local_features:
            global_feat, local_feat = self.pointnets[0](x, mask, keepdim=True, return_local_features=True)
        else:
            # shape (*, 1, output_dims[0])
            global_feat = self.pointnets[0](x, mask, keepdim=True)

        n = x.size(-2)

        for i in range(1, len(self.pointnets)):
            # shape (*, n, output_dims[i-1])
            expanded_global_feat = expand_dim(global_feat, -2, n)
            
            if self.use_local_features:
                # shape (*, n, output_dims[i-1] * 2)
                in_i = torch.cat((local_feat, expanded_global_feat), dim=-1)
            else:
                # shape (*, n, input_dim + output_dims[i-1])
                in_i = torch.cat((x, expanded_global_feat), dim=-1)

            # shape (*, 1, output_dims[i])
            keepdim_i = keepdim if i == len(self.pointnets) - 1 else True
            if self.use_local_features:
                global_feat, local_feat = self.pointnets[i](in_i, mask, keepdim=keepdim_i, return_local_features=True)
            else:
                global_feat = self.pointnets[i](in_i, mask, keepdim=keepdim_i)

        return global_feat


class AcquisitionFunctionNet(nn.Module, ABC):
    """Neural network model for the acquisition function in NN-based
    likelihood-free Bayesian optimization."""

    @abstractmethod
    def forward(self, x_hist, y_hist, x_cand, hist_mask=None, cand_mask=None,
                **kwargs):
        """Forward pass of the acquisition function network.

        Args:
            x_hist (torch.Tensor):
                A `batch_shape x n_hist x d` tensor of training features.
            y_hist (torch.Tensor):
                A `batch_shape x n_hist` or `batch_shape x n_hist x 1`
                tensor of training observations.
            x_cand (torch.Tensor):
                Candidate input tensor with shape `batch_shape x n_cand x d`.
            hist_mask (torch.Tensor): Mask tensor for the history inputs with
                shape `batch_shape x n_hist` or `batch_shape x n_hist x 1`.
                If None, then mask is all ones.
            cand_mask (torch.Tensor): Mask tensor for the candidate inputs with
                shape `batch_shape x n_cand` or `batch_shape x n_cand x 1`.
                If None, then mask is all ones.
            **kwargs: Additional arguments.

        Note: It is assumed x_hist and y_hist are padded (with zeros), although
            that shouldn't matter since the mask will take care of it.

        Returns:
            torch.Tensor: A `batch_shape x n_cand` tensor of acquisition values.
        """
        pass  # pragma: no cover


class AcquisitionFunctionNetWithSoftmaxAndExponentiate(AcquisitionFunctionNet):
    @abstractmethod
    def forward(self, x_hist, y_hist, x_cand, hist_mask=None, cand_mask=None,
                exponentiate=False, softmax=False):
        """Forward pass of the acquisition function network.

        Args:
            x_hist (torch.Tensor):
                A `batch_shape x n_hist x d` tensor of training features.
            y_hist (torch.Tensor):
                A `batch_shape x n_hist` or `batch_shape x n_hist x 1`
                tensor of training observations.
            x_cand (torch.Tensor):
                Candidate input tensor with shape `batch_shape x n_cand x d`.
            hist_mask (torch.Tensor): Mask tensor for the history inputs with
                shape `batch_shape x n_hist` or `batch_shape x n_hist x 1`.
                If None, then mask is all ones.
            cand_mask (torch.Tensor): Mask tensor for the candidate inputs with
                shape `batch_shape x n_cand` or `batch_shape x n_cand x 1`.
                If None, then mask is all ones.
            exponentiate (bool, default: False):
                Whether to exponentiate the output.
                For EI, False corresponds to the log of the acquisition function
                (e.g. log EI), and True corresponds to the acquisition function
                itself (e.g. EI).
                Only applies if self.includes_alpha is False.
            softmax (bool, default: False):
                Whether to apply softmax to the output.
                Only applies if self.includes_alpha is True.

        Returns:
            torch.Tensor: A `batch_shape x n_cand` tensor of acquisition values.
        """
        pass  # pragma: no cover


class AcquisitionFunctionNetWithFinalMLP(AcquisitionFunctionNetWithSoftmaxAndExponentiate):
    """Abstract class for an acquisition function network with a final MLP
    layer. Subclasses should implement the `_get_mlp_input` method."""
    
    def __init__(self, initial_modules: OrderedDict,
                 input_to_final_layer_dim: int,
                 aq_func_hidden_dims: Sequence[int]=[256, 64],
                 include_alpha=False,
                 learn_alpha=False,
                 initial_alpha=1.0,
                 layer_norm_before_end=False,
                 layer_norm_at_end=False,
                 **dense_kwargs):
        """Initializes the MLP layer at the end of the acquisition function.
        Subclasses should call this method in their `__init__` method.

        Args:
            initial_modules (OrderedDict): the initial modules that are to be
                used before the final MLP. It is a good idea to specify them
                here in super().__init__ rather than setting these attributes
                yourself because that way, they are listed first when the module
                is printed, which makes conceptual sense.
            input_to_final_layer_dim (int):
                The dimensionality of the input.
            aq_func_hidden_dims (Sequence[int], default: [256, 64]):
                A sequence of integers representing the sizes of the hidden
                layers of the final fully connected network.
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
        """
        super().__init__()

        if not isinstance(initial_modules, OrderedDict):
            raise ValueError("initial_modules must be an instance of OrderedDict.")
        for key, val in initial_modules.items():
            setattr(self, key, val)

        self.dense = Dense(input_to_final_layer_dim, aq_func_hidden_dims, 1,
                           activation_at_end=False,
                           layer_norm_before_end=layer_norm_before_end,
                           layer_norm_at_end=False, **dense_kwargs)
        
        self.layer_norm_at_end = layer_norm_at_end
        
        self.transform = SoftmaxOrExponentiateLayer(softmax_dim=-2,
                                            include_alpha=include_alpha,
                                            learn_alpha=learn_alpha,
                                            initial_alpha=initial_alpha)
    
    def get_alpha(self):
        return self.transform.get_alpha()

    def set_alpha(self, val):
        self.transform.set_alpha(val)

    @property
    def includes_alpha(self):
        return self.transform.includes_alpha
    
    @abstractmethod
    def _get_mlp_input(self, x_hist, y_hist, x_cand,
                       hist_mask=None, cand_mask=None):
        """Compute the input to the final MLP network.
        This method should be implemented in a subclass.

        Args:
            x_hist (torch.Tensor):
                A `batch_shape x n_hist x d` tensor of training features.
            y_hist (torch.Tensor):
                A `batch_shape x n_hist x 1` tensor of training observations.
            x_cand (torch.Tensor):
                A `batch_shape x n_cand x d` tensor of candidate points.
            hist_mask (torch.Tensor):
                A `batch_shape x n_hist x 1` mask tensor for the history inputs.
                If None, then mask is all ones.
            cand_mask (torch.Tensor):
                A `batch_shape x n_cand x 1` mask tensor for the candidate
                inputs. If None, then mask is all ones.

        Returns:
            torch.Tensor: The `batch_shape x n_cand x input_to_final_layer_dim`
            input tensor to the final MLP network.
        """
        pass  # pragma: no cover
    
    def forward(self, x_hist, y_hist, x_cand,
                hist_mask=None, cand_mask=None,
                exponentiate=False, softmax=False):
        y_hist = check_xy_dims_add_y_output_dim(x_hist, y_hist, "x_hist", "y_hist")
        hist_mask = check_xy_dims_add_y_output_dim(x_hist, hist_mask, "x_hist", "hist_mask")
        cand_mask = check_xy_dims_add_y_output_dim(x_cand, cand_mask, "x_cand", "cand_mask")

        # shape (*, n_cand, input_to_final_layer_dim)
        a = self._get_mlp_input(x_hist, y_hist, x_cand, hist_mask, cand_mask)

        # shape (*, n_cand, 1)
        acquisition_values = self.dense(a)

        if self.layer_norm_at_end:
            if acquisition_values.dim() > 2:
                acquisition_values = (acquisition_values - torch.mean(acquisition_values, dim=(-3, -2), keepdim=True)) / torch.std(acquisition_values, dim=(-3, -2), keepdim=True)

        acquisition_values = self.transform(acquisition_values, mask=cand_mask,
                                            exponentiate=exponentiate, softmax=softmax)

        if cand_mask is not None:
            # Mask out the padded values
            acquisition_values = acquisition_values * cand_mask

        return acquisition_values.squeeze(-1) # shape (*, n_cand)


def concat_y_hist_with_best_y(y_hist, hist_mask, subtract=False):
    if hist_mask is not None:
        neg_inf = torch.zeros_like(y_hist)
        neg_inf[~hist_mask] = float("-inf")
        best_f = (y_hist + neg_inf).amax(-2, keepdim=True)
    else:
        best_f = y_hist.amax(-2, keepdim=True)
    best_f = best_f.expand_as(y_hist)

    if subtract:
        return torch.cat((best_f, best_f - y_hist), dim=-1)
    return torch.cat((best_f, y_hist), dim=-1)


def _get_xy_hist_and_cand(x_hist, y_hist, x_cand, hist_mask=None, include_y=True):
    # shape (*, n_hist, dimension+1)
    xy_hist = torch.cat((x_hist, y_hist), dim=-1)

    n_hist = x_hist.size(-2)
    n_cand = x_cand.size(-2)
    # shape (*, n_cand, n_hist, dimension)
    x_cand_expanded = expand_dim(x_cand.unsqueeze(-2), -2, n_hist)

    # hist_mask has shape (*, n_hist, 1), so need to expand to match.
    # shape (*, n_cand, n_hist, 1)
    mask = None if hist_mask is None else expand_dim(hist_mask.unsqueeze(-3), -3, n_cand)

    if include_y:
        # shape (*, n_cand, n_hist, dimension+1)
        xy_hist_expanded = expand_dim(xy_hist.unsqueeze(-3), -3, n_cand)

        # shape (*, n_cand, n_hist, 2*dimension+1)
        xy_hist_and_cand = torch.cat((x_cand_expanded, xy_hist_expanded), dim=-1)

        return xy_hist, xy_hist_and_cand, mask
    else:
        # shape (*, n_cand, n_hist, dimension)
        x_hist_expanded = expand_dim(x_hist.unsqueeze(-3), -3, n_cand)

        # shape (*, n_cand, n_hist, 2*dimension)
        x_hist_and_cand = torch.cat((x_cand_expanded, x_hist_expanded), dim=-1)

        return xy_hist, x_hist_and_cand, mask


class AcquisitionFunctionNetV1and2(AcquisitionFunctionNetWithFinalMLP):
    def __init__(self,
                 dimension, history_enc_hidden_dims=[256, 256], pooling="max",
                 encoded_history_dim=1024, aq_func_hidden_dims=[256, 64],
                 input_xcand_to_local_nn=True,
                 input_xcand_to_final_mlp=False,
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0,
                 activation_at_end_pointnet=True,
                 layer_norm_pointnet=False,
                 layer_norm_before_end_mlp=False,
                 layer_norm_at_end_mlp=False,
                 include_best_y=False,
                 activation_pointnet=nn.ReLU,
                 activation_mlp=nn.ReLU,
                 n_pointnets=1):
        """
        Args:
            dimension (int): The dimensionality of the input space.
            history_enc_hidden_dims: sequence of integers representing the
                hidden layer dimensions of the history encoder network.
                Default is [256, 256].
            pooling (str): The pooling method used in the history encoder.
                Must be either "max" or "sum". Default is "max".
            encoded_history_dim (int): The dimensionality of the encoded history
                representation. Default is 1024.
            aq_func_hidden_dims: sequence of integers representing the hidden
                layer dimensions of the acquisition function network.
                Default is [256, 64].
            input_xcand_to_local_nn:
                Whether to input the candidate points to the local neural network.
            input_xcand_to_final_mlp:
                Whether to input the candidate points to the final MLP.
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
            activation_at_end_pointnet (bool, default: True):
                Whether to apply the activation function at the end of the PointNet.
            layer_norm_pointnet (bool, default: False):
                Whether to use layer normalization in the PointNet.
            layer_norm_before_end_mlp (bool, default: False):
                Whether to use layer normalization before the end of the MLP.
            layer_norm_at_end_mlp (bool, default: False):
                Whether to use layer normalization at the end of the MLP.
            include_best_y (bool, default: False):
                Whether to include the best y value in the input to the local neural network.
            activation_pointnet:
                The activation function to use in the PointNet.
            activation_mlp:
                The activation function to use in the final MLP.
            n_pointnets (int, default: 1):
                The number of PointNets to use. Default is 1.
        """
        assert isinstance(n_pointnets, int) and n_pointnets >= 1

        if not (input_xcand_to_local_nn or input_xcand_to_final_mlp):
            raise ValueError("At least one of input_xcand_to_local_nn and input_xcand_to_final_mlp must be True.")
        self.input_xcand_to_local_nn = input_xcand_to_local_nn
        self.input_xcand_to_final_mlp = input_xcand_to_final_mlp

        pointnet_input_dim = dimension + 1 + int(include_best_y) + (dimension if input_xcand_to_local_nn else 0)

        pointnet_kwargs = dict(activation_at_end=activation_at_end_pointnet,
                layer_norm_before_end=layer_norm_pointnet,
                layer_norm_at_end=layer_norm_pointnet,
                activation=activation_pointnet)
        
        if n_pointnets == 1:
            initial_modules = OrderedDict([
                ('pointnet', PointNetLayer(
                pointnet_input_dim, history_enc_hidden_dims,
                encoded_history_dim, pooling, **pointnet_kwargs)
                )])
        else:
            kwargs_list = [pointnet_kwargs] * n_pointnets
            initial_modules = OrderedDict([
                ('pointnet', MultiLayerPointNet(
                pointnet_input_dim,
                [history_enc_hidden_dims] * n_pointnets,
                [encoded_history_dim] * n_pointnets,
                [pooling] * n_pointnets,
                kwargs_list, use_local_features=True)
                )])

        final_layer_input_dim = encoded_history_dim + (dimension if input_xcand_to_final_mlp else 0)
    
        super().__init__(initial_modules, final_layer_input_dim,
                         aq_func_hidden_dims, include_alpha,
                         learn_alpha, initial_alpha,
                         layer_norm_before_end_mlp, layer_norm_at_end_mlp,
                         activation=activation_mlp)
        self.dimension = dimension
        self.include_best_y = include_best_y
    
    def _get_mlp_input(self, x_hist, y_hist, x_cand,
                       hist_mask=None, cand_mask=None):
        if self.include_best_y:
            y_hist = concat_y_hist_with_best_y(y_hist, hist_mask, subtract=False)

        if self.input_xcand_to_local_nn: # V2
            xy_hist, xy_hist_and_cand, mask = _get_xy_hist_and_cand(x_hist, y_hist, x_cand, hist_mask)
            # shape (*, n_cand, encoded_history_dim)
            out = self.pointnet(xy_hist_and_cand, mask=mask, keepdim=False)
        else: # V1
            xy_hist = torch.cat((x_hist, y_hist), dim=-1)
            # shape (*, 1, encoded_history_dim)
            out = self.pointnet(xy_hist, mask=hist_mask, keepdim=True)
            
            ## Prepare input to the acquisition function network final dense layer
            n_cand = x_cand.size(-2)
            # shape (*, n_cand, encoded_history_dim)
            out = expand_dim(out, -2, n_cand)
            # Maybe neeed to match dimensions (?): (TODO: test this)
            out = match_batch_shape(out, x_cand)
        
        if self.input_xcand_to_final_mlp: # V1
            # shape (*, n_cand, dimension+encoded_history_dim)
            out = torch.cat((x_cand, out), dim=-1)

        return out


class AcquisitionFunctionNetV3(AcquisitionFunctionNetWithFinalMLP):
    def __init__(self,
                 dimension, history_enc_hidden_dims=[256, 256], pooling="max",
                 encoded_history_dim=1024,
                 mean_enc_hidden_dims=[256, 256], mean_dim=1,
                 std_enc_hidden_dims=[256, 256], std_dim=1,
                 aq_func_hidden_dims=[256, 64], layer_norm=False,
                 layer_norm_at_end_mlp=False, include_y=False,
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0):
        """
        Args:
            dimension (int): The dimensionality of the input space.
            history_enc_hidden_dims: sequence of integers representing the
                hidden layer dimensions of the history encoder network.
                Default is [256, 256].
            pooling (str): The pooling method used in the history encoder.
                Must be either "max" or "sum". Default is "max".
            encoded_history_dim (int): The dimensionality of the encoded history
                representation. Default is 1024.
            mean_enc_hidden_dims: sequence of integers representing the hidden
                layer dimensions of the mean network.
            mean_dim: The dimensionality of the mean output.
            std_enc_hidden_dims: sequence of integers representing the hidden
                layer dimensions of the standard deviation network.
            std_dim: The dimensionality of the standard deviation output.
            aq_func_hidden_dims: sequence of integers representing the hidden
                layer dimensions of the acquisition function network.
                Default is [256, 64].
            layer_norm: Whether to use layer normalization in the networks.
            layer_norm_at_end_mlp: Whether to use layer normalization at the end
                of the MLP.
            include_y: Whether to include the output in the history encoder in
                the input to the mean and standard deviation networks.
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
        """
        self.include_y = include_y
        tmp = 2 * dimension + encoded_history_dim + (1 if include_y else 0)
        initial_modules = OrderedDict([
            ('history_encoder', PointNetLayer(
            dimension + 1, history_enc_hidden_dims, encoded_history_dim, pooling,
            activation_at_end=False,
            layer_norm_before_end=layer_norm, layer_norm_at_end=layer_norm)),
            ('mean_net', Dense(tmp,
                               mean_enc_hidden_dims, mean_dim,
                               activation_at_end=False,
                               layer_norm_before_end=layer_norm,
                               layer_norm_at_end=False)),
            ('std_net', PointNetLayer(
            tmp, std_enc_hidden_dims,
            std_dim, pooling, activation_at_end=True,
            layer_norm_before_end=layer_norm, layer_norm_at_end=False))
        ])
        super().__init__(initial_modules, mean_dim + std_dim,
                         aq_func_hidden_dims, include_alpha,
                         learn_alpha, initial_alpha,
                         layer_norm_before_end=layer_norm,
                         layer_norm_at_end=layer_norm_at_end_mlp)
        self.dimension = dimension
    
    def _get_mlp_input(self, x_hist, y_hist, x_cand,
                       hist_mask=None, cand_mask=None):
        # xy_hist shape (*, n_hist, dimension+1)
        # x_hist_and_cand shape (*, n_cand, n_hist, 2*dimension)
        # mask shape (*, n_cand, n_hist, 1)
        xy_hist, x_hist_and_cand, mask = _get_xy_hist_and_cand(
            x_hist, y_hist, x_cand, hist_mask, include_y=self.include_y)

        # shape (*, 1, encoded_history_dim)
        encoded_history = self.history_encoder(xy_hist, mask=hist_mask, keepdim=True)

        n_hist = x_hist.size(-2)
        n_cand = x_cand.size(-2)

        # shape (*, n_hist, encoded_history_dim)
        encoded_history = expand_dim(encoded_history, -2, n_hist)
        # shape (*, n_cand, n_hist, encoded_history_dim)
        encoded_history = expand_dim(encoded_history.unsqueeze(-3), -3, n_cand)

        # shape (*, n_cand, n_hist, 2*dimension + encoded_history_dim)
        x_hist_and_cand_and_enc_hist = torch.cat((x_hist_and_cand, encoded_history), dim=-1)

        # shape (*, n_cand, n_hist, mean_dim)
        mean_a = self.mean_net(x_hist_and_cand_and_enc_hist)

        # y_hist shape: (*, n_hist, 1)
        # shape (*, n_cand, n_hist, 1)
        y_hist_expanded = expand_dim(y_hist.unsqueeze(-3), -3, n_cand)

        # shpae (*, n_cand, n_hist, mean_dim)
        tmp = mean_a * y_hist_expanded
        if mask is not None:
            tmp = tmp * mask
        # sum along history dimension
        # shape (*, n_cand, mean_dim)
        mean_out = torch.sum(tmp, dim=-2, keepdim=False)

        # shape (*, n_cand, std_dim)
        std_out = self.std_net(x_hist_and_cand_and_enc_hist, mask=mask, keepdim=False)

        # shape (*, n_cand, mean_dim + std_dim)
        mean_and_std = torch.cat((mean_out, std_out), dim=-1)

        return mean_and_std


class AcquisitionFunctionNetV4(AcquisitionFunctionNetWithFinalMLP):
    def __init__(self,
                 dimension, history_enc_hidden_dims=[256, 256], pooling="max",
                 encoded_history_dim=1024,
                 mean_enc_hidden_dims=[256, 256], mean_dim=1,
                 std_enc_hidden_dims=[256, 256], std_dim=1,
                 aq_func_hidden_dims=[256, 64], layer_norm=False,
                 layer_norm_at_end_mlp=False, include_mean=True,
                 include_local_features=False,
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0):
        """
        Args:
            dimension (int): The dimensionality of the input space.
            history_enc_hidden_dims: sequence of integers representing the
                hidden layer dimensions of the history encoder network.
                Default is [256, 256].
            pooling (str): The pooling method used in the history encoder.
                Must be either "max" or "sum". Default is "max".
            encoded_history_dim (int): The dimensionality of the encoded history
                representation. Default is 1024.
            mean_enc_hidden_dims: sequence of integers representing the hidden
                layer dimensions of the mean network.
            mean_dim: The dimensionality of the mean output.
            std_enc_hidden_dims: sequence of integers representing the hidden
                layer dimensions of the standard deviation network.
            std_dim: The dimensionality of the standard deviation output.
            aq_func_hidden_dims: sequence of integers representing the hidden
                layer dimensions of the acquisition function network.
                Default is [256, 64].
            layer_norm: Whether to use layer normalization in the networks.
            layer_norm_at_end_mlp: Whether to use layer normalization at the end
                of the MLP.
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
        """
        std_input_dim = encoded_history_dim + (
            2 * dimension + 1 if include_local_features else 0)
        initial_modules = OrderedDict([
            ('history_encoder', PointNetLayer(
            3 * dimension + 2, history_enc_hidden_dims, encoded_history_dim, pooling,
            activation_at_end=False,
            layer_norm_before_end=layer_norm, layer_norm_at_end=layer_norm)),
            ('std_net', PointNetLayer(
            std_input_dim, std_enc_hidden_dims,
            std_dim, pooling, activation_at_end=include_mean,
            layer_norm_before_end=layer_norm,
            layer_norm_at_end=layer_norm and not include_mean))
        ])
        if include_mean:
            mean_input_dim = encoded_history_dim + (
                2 * dimension + 1 if include_local_features else 0)
            initial_modules['mean_net'] = Dense(mean_input_dim,
                               mean_enc_hidden_dims, mean_dim,
                               activation_at_end=False,
                               layer_norm_before_end=layer_norm,
                               layer_norm_at_end=False)
        
        super().__init__(initial_modules, std_dim + (mean_dim if include_mean else 0),
                         aq_func_hidden_dims, include_alpha,
                         learn_alpha, initial_alpha,
                         layer_norm_before_end=layer_norm,
                         layer_norm_at_end=layer_norm_at_end_mlp)
        self.dimension = dimension
        self.include_mean = include_mean
        self.include_local_features = include_local_features
    
    def _get_mlp_input(self, x_hist, y_hist, x_cand,
                       hist_mask=None, cand_mask=None):
        # xy_hist shape (*, n_hist, dimension+1)
        # xy_hist_and_cand shape (*, n_cand, n_hist, 2*dimension+1)
        # hist_and_cand_mask shape (*, n_cand, n_hist, 1)
        xy_hist, xy_hist_and_cand, hist_and_cand_mask = _get_xy_hist_and_cand(
            x_hist, y_hist, x_cand, hist_mask, include_y=True)

        n_hist = x_hist.size(-2)
        n_cand = x_cand.size(-2)

        # shape (*, n_hist, [n_hist], dimension+1)
        xy_hist_1 = expand_dim(xy_hist.unsqueeze(-2), -2, n_hist)

        # shape (*, [n_hist], n_hist, dimension+1)
        xy_hist_2 = expand_dim(xy_hist.unsqueeze(-3), -3, n_hist)

        # shape (*, n_hist, n_hist, 2*dimension+2)
        xy_hist_pairwise = torch.cat((xy_hist_1, xy_hist_2), dim=-1)

        # shape (*, n_cand, n_hist, n_hist, 2*dimension+2)
        xy_hist_pairwise_expanded = expand_dim(xy_hist_pairwise.unsqueeze(-4), -4, n_cand)

        # x_cand shape: (*, n_cand, dimension)

        # shape (*, n_cand, n_hist, dimension)
        x_cand_expanded = expand_dim(x_cand.unsqueeze(-2), -2, n_hist)
        # shape (*, n_cand, n_hist, n_hist, dimension)
        x_cand_expanded = expand_dim(x_cand_expanded.unsqueeze(-2), -2, n_hist)

        # shape (*, n_cand, n_hist, n_hist, 3*dimension+2)
        x_cand_and_xy_hist_pairwise = torch.cat(
            (x_cand_expanded, xy_hist_pairwise_expanded), dim=-1)
        assert x_cand_and_xy_hist_pairwise.size(-1) == 3 * self.dimension + 2
        assert x_cand_and_xy_hist_pairwise.size(-2) == n_hist
        assert x_cand_and_xy_hist_pairwise.size(-3) == n_hist
        assert x_cand_and_xy_hist_pairwise.size(-4) == n_cand

        if hist_mask is None:
            mask = None
        else:
            # hist_mask has shape (*, n_hist, 1), so need to expand to match.
            # shape (*, n_hist, [n_hist], 1)
            mask1 = expand_dim(hist_mask.unsqueeze(-2), -2, n_hist)
            # shape (*, [n_hist], n_hist, 1)
            mask2 = expand_dim(hist_mask.unsqueeze(-3), -3, n_hist)
            # shape (*, n_hist, n_hist, 1)
            mask = mask1 * mask2

            # shape (*, n_cand, n_hist, n_hist, 1)
            mask = expand_dim(mask.unsqueeze(-4), -4, n_cand)

            # # cand_mask: (*, n_cand, 1)
            # # shape (*, n_cand, n_hist, 1)
            # cand_mask_expanded = expand_dim(cand_mask.unsqueeze(-2), -2, n_hist)
            # # shape (*, n_cand, n_hist, n_hist, 1)
            # cand_mask_expanded = expand_dim(cand_mask_expanded.unsqueeze(-2), -2, n_hist)
            # print(cand_mask_expanded.shape)

        # print(x_cand_and_xy_hist_pairwise - x_cand_and_xy_hist_pairwise * mask)
        # print(x_cand_and_xy_hist_pairwise)
        # exit()

        # shape (*, n_cand, n_hist, encoded_history_dim)
        out1 = self.history_encoder(
            x_cand_and_xy_hist_pairwise, mask=mask, keepdim=False)
        
        if self.include_local_features:
            # xy_hist_and_cand shape (*, n_cand, n_hist, 2*dimension+1)
            # out1             shape (*, n_cand, n_hist, encoded_history_dim)
            out1 = torch.cat((out1, xy_hist_and_cand), dim=-1)

        ### Compute Std
        # shape (*, n_cand, n_hist, 1)
        # mask_std = None if hist_mask is None else expand_dim(hist_mask.unsqueeze(-3), -3, n_cand)
        # shape (*, n_cand, std_dim)
        std_out = self.std_net(out1, mask=hist_and_cand_mask, keepdim=False)
        
        if self.include_mean:
            ### Compute Mean
            # shape (*, n_cand, n_hist, mean_dim)
            mean_a = self.mean_net(out1)
            # y_hist shape: (*, n_hist, 1)
            # shape (*, n_cand, n_hist, 1)
            y_hist_expanded = expand_dim(y_hist.unsqueeze(-3), -3, n_cand)
            # shpae (*, n_cand, n_hist, mean_dim)
            tmp = mean_a * y_hist_expanded
            # sum along history dimension
            # shape (*, n_cand, mean_dim)
            mean_out = torch.sum(tmp, dim=-2, keepdim=False)

            # shape (*, n_cand, mean_dim + std_dim)
            return torch.cat((mean_out, std_out), dim=-1)
        else:
            return std_out


def flatten_last_two_dimensions(tensor):
    return tensor.reshape(tensor.shape[:-2] + (-1,))


class AcquisitionFunctionNetDense(AcquisitionFunctionNetWithFinalMLP):
    def __init__(self, dimension, max_history, hidden_dims=[256, 64],
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0):
        """
        Args:
            dimension (int): The dimensionality of the input space.
            hidden_dims: sequence of integers representing the hidden
                layer dimensions of the acquisition function network.
                Default is [256, 64].
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
        """
        self.max_history = max_history
        self.input_dim = max_history * (2 * dimension + 1)
        super().__init__(OrderedDict(), self.input_dim, hidden_dims,
                         include_alpha, learn_alpha, initial_alpha)
        self.dimension = dimension
    
    def _get_mlp_input(self, x_hist, y_hist, x_cand,
                       hist_mask=None, cand_mask=None):
        # Pad things to the max history length
        x_hist = pad_tensor(x_hist, self.max_history, -2)
        y_hist = pad_tensor(y_hist, self.max_history, -2)
        hist_mask = None if hist_mask is None else pad_tensor(hist_mask, self.max_history, -2)

        # xy_hist_and_cand shape: (*, n_cand, n_hist, 2*dimension+1)
        # mask shape: (*, n_cand, n_hist, 1)
        xy_hist, xy_hist_and_cand, mask = _get_xy_hist_and_cand(x_hist, y_hist, x_cand, hist_mask)

        # shape (*, n_cand, n_hist, 2*dimension+1)
        # mask = None if mask is None else expand_dim(mask, -1, xy_hist_and_cand.size(-1))

        # Make both hsve shape (*, n_cand, n_hist*(2*dimension+1))
        xy_hist_and_cand = flatten_last_two_dimensions(xy_hist_and_cand)
        # mask = flatten_last_two_dimensions(mask)

        assert xy_hist_and_cand.size(-1) == self.input_dim
        assert xy_hist_and_cand.size(-2) == x_cand.size(-2)

        return xy_hist_and_cand


class AcquisitionFunctionNetModel(Model):
    """In this case, the model is the acquisition function network itself.
    So it's kind of silly to have this intermediate between the NN and the
    acquisition function, but it's necessary for the BoTorch API."""
    
    def __init__(self, model: AcquisitionFunctionNet,
                 train_X: Optional[Tensor]=None,
                 train_Y: Optional[Tensor]=None):
        """
        Args:
            model: The acquisition function network model.
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x 1` or `batch_shape x n` tensor of training observations.
        """
        super().__init__()
        if not isinstance(model, AcquisitionFunctionNet):
            raise ValueError("model must be an instance of AcquisitionFunctionNet.")
        model.eval()
        self.model = model

        if train_X is not None and train_Y is not None:
            # Check that the dimensions are compatible, and add an output dimension to train_Y if there is none
            train_Y = check_xy_dims_add_y_output_dim(train_X, train_Y, "train_X", "train_Y")
            
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
        raise UnsupportedError("AcquisitionFunctionNetModel does not support posterior inference.")
    
    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1 # Only supporting 1 output (for now at least)
    
    def subset_output(self, idcs: Sequence[int]):
        raise UnsupportedError("AcquisitionFunctionNetModel does not support output subsetting.")

    def condition_on_observations(self, X: Tensor, Y: Tensor) -> Model:
        if self.train_X is None:
            new_X, new_Y = X, Y
        else:
            # Check dimensions & add output dimension to Y if there is none
            Y = check_xy_dims_add_y_output_dim(X, Y, "X", "Y")
            X = match_batch_shape(X, self.train_X)
            Y = match_batch_shape(Y, self.train_Y)
            new_X = torch.cat(self.train_X, X, dim=-2)
            new_Y = torch.cat(self.train_Y, Y, dim=-2)
        return AcquisitionFunctionNetModel(self.model, new_X, new_Y)

    def forward(self, X: Tensor, **kwargs) -> Tensor:
        """Forward pass of the acquisition function network.

        Args:
            X (Tensor): The input tensor of shape `(batch_shape) x n_cand x d`.
            **kwargs: Keyword arguments to pass to the model's `forward` method.
                If any are unspecified, then the default values will be used.

        Returns:
            Tensor: The output tensor of shape `(batch_shape) x n_cand`.

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

        logger.debug(f"In AcquisitionFunctionNetModel.forward, X.shape = {X.shape}")

        ret = self.model(self.train_X, self.train_Y, X, **kwargs)
        assert ret.shape == X.shape[:-1]
        return ret


class LikelihoodFreeNetworkAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: AcquisitionFunctionNetModel, **kwargs):
        """
        Args:
            model: The acquisition function network model.
        """
        super().__init__(model=model) # sets self.model = model
        self.kwargs = kwargs
    
    @classmethod
    def from_net(cls, model: AcquisitionFunctionNet,
                 train_X: Optional[Tensor]=None,
                 train_Y: Optional[Tensor]=None,
                 **kwargs) -> "LikelihoodFreeNetworkAcquisitionFunction":
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
        logger.debug(f"In LikelihoodFreeNetworkAcquisitionFunction.forward, X.shape = {X.shape}")
        assert X.size(-2) == 1 # Guaranteed by t_batch_mode_transform
        X = X.squeeze(-2) # Make shape (b) x d

        # shape (b)
        output = self.model(X, **self.kwargs)
        assert output.shape == X.shape[:-1]
        return output

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError("AcquisitionFunctionNetModel does not support pending points.")
