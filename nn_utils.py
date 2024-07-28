from typing import Optional, Sequence, Union
import torch
from torch import nn
from torch import Tensor
from gpytorch.utils.transforms import inv_softplus
from botorch.utils.transforms import normalize_indices


ACTIVATIONS = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "selu": nn.SELU
}

class Dense(nn.Sequential):
    """Dense neural network with ReLU activations."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]=[256, 64],
                 output_dim: int=1, activation_at_end=False,
                 layer_norm_before_end=False, layer_norm_at_end=False,
                 dropout:Optional[float]=None, dropout_at_end=True,
                 activation:str="relu"):
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
        activation = ACTIVATIONS[activation]

        layer_widths = [input_dim] + list(hidden_dims) + [output_dim]
        n_layers = len(layer_widths) - 1
        
        layers = []
        for i in range(n_layers):
            in_dim, out_dim = layer_widths[i], layer_widths[i+1]
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            add_layer_norm = layer_norm_at_end if i == n_layers - 1 else layer_norm_before_end
            if add_layer_norm:
                layers.append(nn.LayerNorm(out_dim))

                # if i == 0:
                #     layers.append(nn.LayerNorm(out_dim))
            
            if i != n_layers - 1 or activation_at_end:
                # layers.append(nn.ReLU())
                # layers.append(nn.Softplus())
                # layers.append(nn.SELU())
                layers.append(activation())
            
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
