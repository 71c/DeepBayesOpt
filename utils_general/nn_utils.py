from typing import Optional, Sequence
from torch import nn


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "selu": nn.SELU
}


def _get_initialized_linear(in_dim, out_dim, activation):
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
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS.keys()}")
        activation_func = _ACTIVATIONS[activation]

        layer_widths = [input_dim] + list(hidden_dims) + [output_dim]
        n_layers = len(layer_widths) - 1
        
        layers = []
        for i in range(n_layers):
            in_dim, out_dim = layer_widths[i], layer_widths[i+1]

            apply_activation = i != n_layers - 1 or activation_at_end
            
            # layers.append(nn.Linear(in_dim, out_dim))
            layers.append(_get_initialized_linear(
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
