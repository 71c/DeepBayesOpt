from typing import Optional, Sequence, Union
import torch
from torch import nn
from torch import Tensor
from gpytorch.utils.transforms import inv_softplus


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


class PositiveScalar(nn.Module):
    r"""A learnable or fixed positive scalar parameter."""
    def __init__(self,
                 learnable:bool=True,
                 initial_val:Union[float, Tensor]=1.0,
                 softplus:bool=True):
        r"""
        Args:
            learnable (bool): If True, the scalar is learnable.
            initial_val (float or Tensor): Initial value of the scalar.
            softplus (bool): If True, use softplus transformation. Otherwise,
                use exponential transformation. Only applicable if learnable=True.
        """
        super().__init__()
        self.register_buffer("learnable", torch.as_tensor(learnable))
        initial_val = self._get_positive_scalar_tensor(initial_val)
        if learnable:
            self.register_buffer("softplus", torch.as_tensor(softplus))
            untf_value = self._get_untransformed_value(initial_val)
            self._untransformed_value = nn.Parameter(untf_value)
        else:
            self.register_buffer("_value", initial_val)
    
    def get_value(self) -> Tensor:
        r"""Returns the current value of the scalar."""
        if self.learnable:
            if self.softplus:
                return nn.functional.softplus(self._untransformed_value)
            else:
                return torch.exp(self._untransformed_value)
        else:
            return self._value
    
    def set_value(self, value: Union[float, Tensor]):
        r"""Sets a new value for the scalar."""
        value = self._get_positive_scalar_tensor(value)
        if self.learnable:
            untf_value = self._get_untransformed_value(value)
            self._untransformed_value.data.copy_(untf_value)
        else:
            self._value.data.copy_(value)

    @staticmethod
    def _get_positive_scalar_tensor(value: Union[float, Tensor]):
        value = torch.as_tensor(value)
        if value.dim() != 0:
            value = value.squeeze()
        if value.numel() != 1:
            raise ValueError("value should be a scalar")
        if value.item() <= 0:
            raise ValueError("value should be positive")
        return value
    
    def _get_untransformed_value(self, value: Tensor) -> Tensor:
        return inv_softplus(value) if self.softplus else torch.log(value)


def custom_softplus(input: Tensor, beta: Tensor, threshold: float) -> Tensor:
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
    
    def __init__(self, initial_beta: float = 1.0, threshold: float = 20.0) -> None:
        r"""
        Args:
            initial_beta (float): The initial value for the learnable beta parameter.
            threshold (float): The threshold value for the softplus function."""
        super().__init__()
        self.threshold = threshold
        self._beta = PositiveScalar(
            learnable=True, initial_val=initial_beta, softplus=True)
    
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
        fixed_scalar = PositiveScalar(learnable=False, initial_val=2.0)
        assert torch.isclose(fixed_scalar.get_value(), torch.tensor(2.0)), "Fixed scalar value mismatch."

        fixed_scalar.set_value(4.0)
        assert torch.isclose(fixed_scalar.get_value(), torch.tensor(4.0)), "Fixed scalar set value mismatch."

        for softplus in [False, True]:
            learnable_scalar = PositiveScalar(learnable=True, initial_val=2.0, softplus=softplus)
            assert torch.isclose(learnable_scalar.get_value(), torch.tensor(2.0)), "Learnable scalar initial value mismatch."

            learnable_scalar.set_value(3.0)
            assert torch.isclose(learnable_scalar.get_value(), torch.tensor(3.0)), "Learnable scalar set value mismatch."

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

    test_positive_scalar()
    test_learnable_softplus()
    print("All tests passed.")
