from typing import Optional, Sequence, Union, List, Any
import warnings
import torch
from torch import nn
from torch import Tensor
from gpytorch.utils.transforms import inv_softplus
from botorch.utils.transforms import normalize_indices
from botorch.utils.safe_math import fatmax, smooth_amax
from utils.utils import expand_dim

import logging

# Set to True to enable debug logging
DEBUG = False

# Create a logger for your application
logger = logging.getLogger('nn_utils')
# Configure the logging
logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


ACTIVATIONS = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "selu": nn.SELU
}


def get_initialized_linear(in_dim, out_dim, activation):
    linear = nn.Linear(in_dim, out_dim)
    if activation == "softplus":
        activation = "relu" # approximate
    nn.init.kaiming_uniform_(linear.weight, nonlinearity=activation)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


class Dense(nn.Sequential):
    """Dense neural network with ReLU activations."""
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int]=[256, 64],
                 output_dim: int=1, 
                 activation:str="relu", activation_at_end=False,
                 layer_norm_before_end=False, layer_norm_at_end=False,
                 dropout:Optional[float]=None, dropout_at_end=True):
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
        if not isinstance(activation, str):
            raise ValueError("activation must be a string.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS.keys()}")
        activation_func = ACTIVATIONS[activation]

        layer_widths = [input_dim] + list(hidden_dims) + [output_dim]
        n_layers = len(layer_widths) - 1
        
        layers = []
        for i in range(n_layers):
            in_dim, out_dim = layer_widths[i], layer_widths[i+1]

            apply_activation = i != n_layers - 1 or activation_at_end
            
            # layers.append(nn.Linear(in_dim, out_dim))
            layers.append(get_initialized_linear(
                in_dim, out_dim,
                activation=activation if apply_activation else 'linear'
            ))
            
            add_layer_norm = layer_norm_at_end if i == n_layers - 1 else layer_norm_before_end
            if add_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            
            if apply_activation:
                layers.append(activation_func())
            
            if dropout is not None and (i != n_layers - 1 or dropout_at_end):
                layers.append(nn.Dropout(dropout))
        
        super().__init__(*layers)


class PositiveTensor(nn.Module):
    r"""A learnable or fixed positive tensor parameter."""
    def __init__(self,
                 initial_val:Union[float, Tensor],
                 learnable:bool=True,
                 softplus:bool=True):
        r"""
        Args:
            initial_val (float or Tensor): Initial value of the tensor. Must be positive
            learnable (bool): If True, the tensor is learnable.
            softplus (bool): If True, use softplus transformation. Otherwise,
                use exponential transformation. Only applicable if learnable=True.
        """
        super().__init__()
        self.register_buffer("learnable", torch.as_tensor(learnable))
        initial_val = self._check_value(initial_val)
        if learnable:
            self.register_buffer("softplus", torch.as_tensor(softplus))
            untf_value = self._get_untransformed_value(initial_val)
            self._untransformed_value = nn.Parameter(untf_value)
        else:
            self.register_buffer("_value", initial_val)
    
    def get_value(self) -> Tensor:
        r"""Returns the current value of the tensor."""
        if self.learnable:
            if self.softplus:
                return nn.functional.softplus(self._untransformed_value)
            else:
                return torch.exp(self._untransformed_value)
        else:
            return self._value
    
    def set_value(self, value: Union[float, Tensor], strict:bool=False):
        r"""Sets a new value for the tensor.
        
        Args:
            value (float or Tensor): New value.
            strict (bool): If True, the new value must have the same shape as the
                current value. Otherwise, the new value is expanded to the shape
                of the current value.
        """
        value = self._check_value(value)
        if self.learnable:
            untf_value = self._get_untransformed_value(value)
            if not strict:
                untf_value = untf_value.expand_as(self._untransformed_value)
            self._untransformed_value.data.copy_(untf_value)
        else:
            if not strict:
                value = value.expand_as(self._value)
            self._value.data.copy_(value)
    
    def _get_untransformed_value(self, value: Tensor) -> Tensor:
        return inv_softplus(value) if self.softplus else torch.log(value)

    @classmethod
    def _check_value(cls, value: Union[float, Tensor]):
        value = torch.as_tensor(value)
        if not torch.all(value > 0):
            raise ValueError("value should be positive")
        return value


class PositiveScalar(PositiveTensor):
    r"""A learnable or fixed positive scalar parameter."""
    def __init__(self,
                 initial_val:Union[float, Tensor]=1.0,
                 learnable:bool=True,
                 softplus:bool=True):
        r"""
        Args:
            initial_val (float or Tensor): Initial value of the scalar. Must be positive
                and have a single element.
            learnable (bool): If True, the scalar is learnable.
            softplus (bool): If True, use softplus transformation. Otherwise,
                use exponential transformation. Only applicable if learnable=True.
        """
        super().__init__(initial_val, learnable, softplus)
    
    def get_value(self) -> Tensor:
        r"""Returns the current value of the scalar."""
        return super().get_value()
    
    def set_value(self, value: Union[float, Tensor]):
        r"""Sets a new value for the scalar.
        
        Args:
            value (float or Tensor): New value.
        """
        super().set_value(value, strict=True)

    @classmethod
    def _check_value(cls, value: Union[float, Tensor]):
        value = super()._check_value(value)
        if value.dim() != 0:
            value = value.squeeze()
        if value.numel() != 1:
            raise ValueError("value should be a scalar")
        return value


def get_all_dims_except(input: Tensor, dim: int):
    dim = normalize_indices([dim], input.dim())[0]
    return list(range(dim)) + list(range(dim+1, input.dim()))


class PositiveBatchNorm(nn.Module):
    def __init__(self,
                 num_features: int,
                 dim: int,
                 momentum: Optional[float] = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = PositiveTensor(torch.ones(num_features, **factory_kwargs),
                                         learnable=True, softplus=True)
        self.register_buffer("running_mean", None)
        if self.track_running_stats:
            self.running_mean: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("num_batches_tracked", None)
    
    def forward(self, input: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        r"""Forward pass of the PositiveBatchNorm module.
        
        Args:
            input (Tensor): The input tensor, of shape
                batch_shape x num_features.
            mask (Tensor, optional): A binary mask tensor of shape that is
                broadcastable with input. The mask can have a shape which has the
                features dimension removed in which case it is added.
                If provided, the mean is computed only over the unmasked
                elements. The input is not assumed to have zeros in the masked
                part, however the output is *not* masked with zeros again
                (because if it did already have zeros there then applying this would
                not change that since it is multiplying by a number, no adding).
        """
        if input.size(self.dim) != self.num_features:
            raise ValueError("Input tensor has incorrect number of features.")

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        
        if self.training or self.running_mean is None:
            dims = get_all_dims_except(input, self.dim)
            if mask is None:
                mean = input.mean(dim=dims)
            else:
                if input.dim() - mask.dim() == 1:
                    mask = mask.unsqueeze(self.dim)
                if mask.dim() != input.dim():
                    raise ValueError("Mask tensor has incorrect number of dimensions.")
                if not (input.shape[:self.dim] == mask.shape[:self.dim]
                        and input.shape[self.dim+1:] == mask.shape[self.dim+1:]):
                    raise ValueError("Mask tensor has incorrect shape.")
                mean = (input * mask).sum(dim=dims) / mask.sum(dim=dims)

        if self.training:
            if self.track_running_stats:
                if self.num_batches_tracked is not None:  # type: ignore[has-type]
                    self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

                with torch.no_grad():
                    if self.running_mean is None:
                        self.running_mean = mean
                    else:
                        self.running_mean = exponential_average_factor * mean\
                            + (1 - exponential_average_factor) * self.running_mean
        elif self.running_mean is not None:
            mean = self.running_mean
        
        input = input / mean
        if self.affine:
            input = input * self.weight.get_value()
        
        return input


def custom_softplus(input: Tensor, beta: Tensor, threshold: float = 20.) -> Tensor:
    r"""Need to re-implement softplus because nn.functional.softplus
    expects beta to be a fixed number so can't take gradient w.r.t. beta in that case"""
    scaled_input = input * beta
    return torch.where(
        scaled_input > threshold,
        input,
        scaled_input.exp().log1p() / beta
    )


class LearnableSoftplus(nn.Module):
    r"""A custom PyTorch module that applies the softplus function with a learnable
    beta parameter."""
    
    def __init__(self,
                 initial_beta: Union[float, Tensor] = 1.0,
                 threshold: float = 20.0) -> None:
        r"""
        Args:
            initial_beta (float): The initial value for the learnable beta parameter.
            threshold (float): The threshold value for the softplus function."""
        super().__init__()
        self.threshold = threshold
        self._beta = PositiveTensor(
            initial_val=initial_beta, learnable=True, softplus=True)

    @property
    def beta(self) -> Tensor:
        r"""Returns the current value of the learnable beta parameter."""
        return self._beta.get_value()
    
    def forward(self, input: Tensor) -> Tensor:
        r"""Forward pass of the LearnableSoftplus module.
        
        Args:
            input (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor after applying the softplus function with the
            learnable beta parameter.
        """
        return custom_softplus(input, self._beta.get_value(), self.threshold)


class SoftmaxOrSoftplusLayer(nn.Module):
    def __init__(self, softmax_dim=-1,
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0,
                 initial_beta=1.0, learn_beta=False,
                 softplus_batchnorm=False,
                 softplus_batchnorm_num_features=1,
                 softplus_batchnorm_dim=-1,
                 softplus_batchnorm_momentum=0.1):
        """Initialize the SoftmaxOrSoftplusLayer class.

        Args:
            softmax_dim (int, default: -1):
                The dimension along which to apply the softmax.
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
            initial_beta (float, default: 1.0):
                The initial value for the beta parameter.
            learn_beta (bool, default: False):
                Whether to learn the beta parameter.
        """
        super().__init__()
        self.softmax_dim = softmax_dim

        self.includes_alpha = include_alpha
        if include_alpha:
            self._alpha = PositiveScalar(initial_val=initial_alpha,
                                         learnable=learn_alpha,
                                         softplus=False)
        
        self.learn_beta = learn_beta
        if learn_beta:
            self.softplus = LearnableSoftplus(initial_beta)
        else:
            self.softplus = nn.Softplus(initial_beta)
        
        self.register_buffer("softplus_batchnorm",
                             torch.as_tensor(softplus_batchnorm))
        if softplus_batchnorm:
            self.batchnorm = PositiveBatchNorm(
                num_features=softplus_batchnorm_num_features,
                dim=softplus_batchnorm_dim,
                momentum=softplus_batchnorm_momentum,
                affine=True,
                track_running_stats=True)

    def get_alpha(self):
        if not self.includes_alpha:
            raise ValueError("Model does not include alpha.")
        return self._alpha.get_value()
    
    def set_alpha(self, val):
        if not self.includes_alpha:
            raise ValueError("Model does not include alpha.")
        self._alpha.set_value(val)
    
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
            
            if self.learn_beta:
                warnings.warn("softmax is True but learn_beta is also True. "
                              "This is probably unintentional.")

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
            x = self.softplus(x)

            if self.softplus_batchnorm:
                x = self.batchnorm(x, mask=mask)

        return x


def check_xy_dims(x: Tensor, y: Union[Tensor, None],
                  x_name: str, y_name: str,
                  expected_y_dim:Optional[int]=None) -> Tensor:
    """If y is None, return y. Otherwise, check that the dimensions of x and y are as
    expected, add an output dimension to y if there is none, and return y.
    
    Args:
        x (Tensor):
            The input tensor x, `batch_shape x n x d`
        y (Tensor or None):
            The input tensor y, `batch_shape x n x m` or `batch_shape x n`
        x_name (str):
            The name of the x tensor.
        y_name (str):
            The name of the y tensor.
        expected_y_dim (int or None):
            The expected number of dimensions for y. None means that the number
            of dimensions is not checked.
    
    Returns:
        Tensor: The given y tensor, with an added output dimension if it was missing.
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
        raise ValueError(f"{x_name} and {y_name} must have the same number of points in the dimension -2.")
    if expected_y_dim is not None:
        if y.size(-1) != expected_y_dim:
            raise ValueError(
                f"{y_name} must have {expected_y_dim} output dimension"
                    + ('s' if expected_y_dim > 1 else ''))
    elif y.size(-1) < 1:
        raise ValueError(f"{y_name} must have at least one output dimension.")
    return y


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


def add_neg_inf_for_max(x, mask):
    # neg_inf = torch.zeros_like(x)
    # mask_expanded = expand_dim(mask, -1, x.size(-1))
    # neg_inf[~mask_expanded] = float("-inf")
    # return x + neg_inf

    mask_expanded = expand_dim(mask, -1, x.size(-1))
    return x.masked_fill(~mask_expanded, float("-inf"))


# doesn't work gives nan
# class SmoothMax(nn.Module):
#     def __init__(self, ndims, initial_tau=0.05):
#         super().__init__()
#         # self._log_tau = nn.Parameter(
#         #     torch.full((ndims,), math.log(initial_tau))
#         # )
#         # self._log_tau = nn.Parameter(
#         #     torch.tensor(math.log(initial_tau))
#         # )
#         self._log_tau = nn.Parameter(
#             torch.tensor(0.0)
#         )
#         self.ndims = ndims
    
#     def forward(self, x, keepdim=False):
#         if x.size(-1) != self.ndims:
#             raise ValueError("incorrect dimensions")
#         # tau = torch.exp(self._log_tau)
#         # tau = torch.nn.functional.softplus(self._log_tau)
#         tau = torch.sigmoid(self._log_tau)

#         # new_shape = [1] * (x.ndim - 1) + [self.ndims]
#         # tau = tau.view(*new_shape)
#         ret = smooth_amax(x, dim=-2, keepdim=keepdim, tau=tau)
#         print("tau", self._log_tau, tau, ret)
#         return ret


POOLING_METHODS = {
    "max", "sum", "fatmax",
    "experiment1", "experiment2", "experiment3"}
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
        # elif pooling == "fatmax":  # Doesn't work gives nan
        #     self.smoothmax_module = SmoothMax(ndims=output_dim)

        if 'dropout_at_end' not in dense_kwargs:
            dense_kwargs['dropout_at_end'] = True
        
        logger.debug(f"In PointNetLayer.__init__, dense_kwargs: {dense_kwargs}")

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
        logger.debug(f"x.shape: {x.shape}")
        local_features = self.network(x)

        # for name, param in self.network.named_parameters():
        #     logger.debug(f"Parameter: {name}")
        #     logger.debug(f"Parameter shape: {param.shape}")
        #     logger.debug(f"Parameter value: {param}")

        # Mask out the padded values. It is sufficient to mask at the end.
        if mask is not None:
            # shape (*, n, 1)
            mask = check_xy_dims(x, mask, "x", "mask", expected_y_dim=1)
            local_features = local_features * mask
        
        # "global feature"
        if self.pooling == "sum":
            ret = torch.sum(local_features, dim=-2, keepdim=keepdim)
        elif self.pooling == "max" or self.pooling == "fatmax":
            # This works for maxing. If ReLU is applied at the end, then
            # we could instead just use the one for summing.
            tmp = local_features if mask is None else add_neg_inf_for_max(local_features, mask)
            
            if self.pooling == "max":
                ret = torch.max(tmp, dim=-2, keepdim=keepdim).values
            else:
                # ret = fatmax(tmp, dim=-2, keepdim=keepdim) # gives nan
                # ret = self.smoothmax_module(tmp, keepdim=keepdim) # gives nan
                ret = smooth_amax(tmp, dim=-2, keepdim=keepdim, tau=0.1) # works
                
            
            # print("x has any infs:", torch.isinf(x).any().item())
            # print("x has any nans:", torch.isnan(x).any().item())
            # print("x:", x)
            # print("local_features:", local_features)
            # print("tmp:", tmp)
            # print("ret:", ret)
            # print()

            if mask is not None:
                # ret[ret == float("-inf")] = 0.0
                # ret = ret.masked_fill(maxvals.isinf(), 0.0)
                ret = ret.masked_fill(ret.isinf(), 0.0)
            
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



if __name__ == "__main__":
    def test_positive_scalar():
        fixed_scalar = PositiveScalar(initial_val=2.0, learnable=False)
        assert torch.isclose(fixed_scalar.get_value(), torch.tensor(2.0)), "Fixed scalar value mismatch."

        fixed_scalar.set_value(4.0)
        assert torch.isclose(fixed_scalar.get_value(), torch.tensor(4.0)), "Fixed scalar set value mismatch."

        for softplus in [False, True]:
            learnable_scalar = PositiveScalar(initial_val=2.0, learnable=True, softplus=softplus)
            assert torch.isclose(learnable_scalar.get_value(), torch.tensor(2.0)), "Learnable scalar initial value mismatch."

            learnable_scalar.set_value(3.0)
            assert torch.isclose(learnable_scalar.get_value(), torch.tensor(3.0)), "Learnable scalar set value mismatch."

    def test_positive_tensor():
        # Test fixed tensor
        fixed_tensor = PositiveTensor(initial_val=torch.tensor([2.0, 3.0, 4.0]), learnable=False)
        assert torch.allclose(fixed_tensor.get_value(), torch.tensor([2.0, 3.0, 4.0])), "Fixed tensor value mismatch."

        fixed_tensor.set_value(torch.tensor([5.0, 6.0, 7.0]))
        assert torch.allclose(fixed_tensor.get_value(), torch.tensor([5.0, 6.0, 7.0])), "Fixed tensor set value mismatch."

        # Test learnable tensor with softplus transformation
        learnable_tensor_sp = PositiveTensor(initial_val=torch.tensor([2.0, 3.0, 4.0]), learnable=True, softplus=True)
        assert torch.allclose(learnable_tensor_sp.get_value(), torch.tensor([2.0, 3.0, 4.0])), "Learnable tensor (softplus) initial value mismatch."

        learnable_tensor_sp.set_value(torch.tensor([3.0, 4.0, 5.0]))
        assert torch.allclose(learnable_tensor_sp.get_value(), torch.tensor([3.0, 4.0, 5.0])), "Learnable tensor (softplus) set value mismatch."

        learnable_tensor_sp.set_value(1.8)
        assert torch.allclose(learnable_tensor_sp.get_value(), torch.tensor([1.8, 1.8, 1.8])), "Learnable tensor (softplus) set value mismatch."

        learnable_tensor_sp.set_value(torch.tensor([1.8]))
        assert torch.allclose(learnable_tensor_sp.get_value(), torch.tensor([1.8, 1.8, 1.8])), "Learnable tensor (softplus) set value mismatch."

        try:
            learnable_tensor_sp.set_value(torch.tensor([1.8, 3.4]))
            assert False, "Expected RuntimeError for mismatched tensor shapes."
        except RuntimeError:
            pass

        # Test learnable tensor with exponential transformation
        learnable_tensor_exp = PositiveTensor(initial_val=torch.tensor([2.0, 3.0, 4.0]), learnable=True, softplus=False)
        assert torch.allclose(learnable_tensor_exp.get_value(), torch.tensor([2.0, 3.0, 4.0])), "Learnable tensor (exp) initial value mismatch."

        learnable_tensor_exp.set_value(torch.tensor([3.0, 4.0, 5.0]))
        assert torch.allclose(learnable_tensor_exp.get_value(), torch.tensor([3.0, 4.0, 5.0])), "Learnable tensor (exp) set value mismatch."


    def test_learnable_softplus():
        input_values = torch.linspace(-50, 50, steps=100)  # Test a range of input values
        initial_beta = 2.0
        threshold = 20.0

        # Instantiate LearnableSoftplus
        custom_softplus = LearnableSoftplus(initial_beta=initial_beta, threshold=threshold)
        
        # Compute custom softplus output
        custom_output = custom_softplus(input_values)

        # Compute torch softplus output
        beta_value = custom_softplus._beta.get_value().item()
        torch_output = nn.functional.softplus(input_values, beta=beta_value, threshold=threshold)

        # Check if the outputs are close
        assert torch.isclose(custom_output, torch_output).all(), "Outputs do not match!"

    test_positive_tensor()
    test_positive_scalar()
    test_learnable_softplus()
    print("All tests passed.")
