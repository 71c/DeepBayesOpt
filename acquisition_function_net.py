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
logging.basicConfig(level=logging.WARNING)

# Set to True to enable debug logging
DEBUG = False

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


# Unfortunately, torch.masked does not support torch.addmm
# (which is used in nn.Linear), so we have to do the masking manually and
# can't support MaskedTensor.


class AcquisitionNetFinalLayer(nn.Module):
    """Final layer of the acquisition function network. This is a simple
    feedforward neural network with a single output node."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]=[256, 64],
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0):
        """Initialize the AcquisitionNetFinalLayer class.

        Args:
            input_dim (int):
                The dimensionality of the input.
            hidden_dims (Sequence[int], default: [256, 64]):
                A sequence of integers representing the sizes of the hidden
                layers.
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
        """
        super().__init__()

        aqnet_layer_widths = [input_dim] + list(hidden_dims) + [1]
        n_layers = len(aqnet_layer_widths) - 1
        network = nn.Sequential()
        for i in range(n_layers):
            in_dim, out_dim = aqnet_layer_widths[i], aqnet_layer_widths[i+1]
            
            network.append(nn.Linear(in_dim, out_dim))

            # if i != n_layers - 1:
            #     network.append(nn.LayerNorm(out_dim))
            
            # if i == n_layers - 1:
            #     network.append(nn.LayerNorm(out_dim))
            
            if i != n_layers - 1: # want to be linear in output
                network.append(nn.ReLU())
        self.network = network

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
    
    def forward(self, x, cand_mask=None, exponentiate=False, softmax=False):
        """Compute the acquisition function.

        Args:
            x (torch.Tensor):
                Fixed-dimension input of shape (*, n_cand, input_dim)
            cand_mask (torch.Tensor): Mask tensor for the candidate inputs with
                shape (*, n_cand) or (*, n_cand, 1). If None, mask is all ones.
            exponentiate (bool, optional): Whether to exponentiate the output.
                Default is False. For EI, False corresponds to the log of the
                acquisition function (e.g. log EI), and True corresponds to
                the acquisition function itself (e.g. EI).
                Only applies if self.includes_alpha is False.
            softmax (bool, optional): Whether to apply softmax to the output.
                Only applies if self.includes_alpha is True. Default is False.

        Returns:
            torch.Tensor: Acquisition values tensor with shape (*, n_cand).
        """
        # shape (*, n_cand, 1)
        acquisition_values = self.network(x)

        if softmax:
            if self.includes_alpha:
                acquisition_values = acquisition_values * self.get_alpha()
            acquisition_values = nn.functional.softmax(acquisition_values, dim=-2)
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
            acquisition_values = torch.exp(acquisition_values)
        
        if cand_mask is not None:
            # get to shape (*, n_cand, 1)
            cand_mask = check_xy_dims_add_y_output_dim(x, cand_mask, "x", "cand_mask")
            # Mask out the padded values
            acquisition_values = acquisition_values * cand_mask

        return acquisition_values.squeeze(-1) # shape (*, n_cand)


class PointNetLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]=[256, 256],
                 output_dim: int=1024, pooling="max"):
        super().__init__()

        layer_widths = [input_dim] + list(hidden_dims) + [output_dim]
        n_layers = len(layer_widths) - 1
        network = nn.Sequential()
        for i in range(n_layers):
            in_dim, out_dim = layer_widths[i], layer_widths[i+1]
            
            network.append(nn.Linear(in_dim, out_dim))
            
            # network.append(nn.LayerNorm(out_dim))
            
            if i != n_layers - 1:
                network.append(nn.ReLU())
            # network.append(nn.ReLU())
        self.network = network

        if pooling == "max" or pooling == "sum":
            self.pooling = pooling
        else:
            raise ValueError("pooling must be either 'max' or 'sum'")
    
    def forward(self, x, mask=None, keepdim=True):
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

        # Mask out the padded values. It is sufficient to mask at the end.
        if mask is not None:
            # shape (*, n, 1)
            mask = check_xy_dims_add_y_output_dim(x, mask, "x", "mask")

            if self.pooling == "sum":
                # This works for summing
                local_features = local_features * mask
            else: # self.pooling == "max"
                # This works for maxing. If ReLU is applied at the end, then
                # we could instead just use the above.
                neg_inf = torch.zeros_like(local_features)
                hist_mask_expanded = expand_dim(mask, -1, local_features.size(-1))
                neg_inf[~hist_mask_expanded] = float("-inf")
                local_features = local_features + neg_inf
        
        # "global feature"
        if self.pooling == "sum":
            # shape (*, 1, output_dim)
            return torch.sum(local_features, dim=-2, keepdim=keepdim)
        else: # self.pooling == "max"
            # shape (*, output_dim)
            return torch.max(local_features, dim=-2, keepdim=keepdim).values


class AcquisitionFunctionNet(nn.Module, ABC):
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
    
    def __init__(self, input_to_final_layer_dim: int,
                 aq_func_hidden_dims: Sequence[int]=[256, 64],
                 include_alpha=False,
                 learn_alpha=False,
                 initial_alpha=1.0):
        """Initializes the MLP layer at the end of the acquisition function.
        Subclasses should call this method in their `__init__` method.

        Args:
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
        self.acquisition_function_net = AcquisitionNetFinalLayer(
            input_to_final_layer_dim, aq_func_hidden_dims,
            include_alpha=include_alpha, learn_alpha=learn_alpha,
            initial_alpha=initial_alpha)
    
    def get_alpha(self):
        return self.acquisition_function_net.get_alpha()

    def set_alpha(self, val):
        self.acquisition_function_net.set_alpha(val)

    @property
    def includes_alpha(self):
        return self.acquisition_function_net.includes_alpha
    
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
        
        # Compute the acquisition function value, shape (*, n_cand)
        ret = self.acquisition_function_net(a, cand_mask, exponentiate, softmax)
        assert ret.shape == x_cand.shape[:-1]
        return ret


class AcquisitionFunctionNetV1(AcquisitionFunctionNetWithFinalMLP):
    """Neural network model for the acquisition function in NN-based
    likelihood-free Bayesian optimization."""

    def __init__(self,
                 dimension, history_enc_hidden_dims=[256, 256], pooling="max",
                 encoded_history_dim=1024, aq_func_hidden_dims=[256, 64],
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
            aq_func_hidden_dims: sequence of integers representing the hidden
                layer dimensions of the acquisition function network.
                Default is [256, 64].
            include_alpha (bool, default: False):
                Whether to include an alpha parameter.
            learn_alpha (bool, default: False):
                Whether to learn the alpha parameter.
            initial_alpha (float, default: 1.0):
                The initial value for the alpha parameter.
        """
        super().__init__(dimension + encoded_history_dim, aq_func_hidden_dims,
                         include_alpha, learn_alpha, initial_alpha)
        self.dimension = dimension
        self.history_encoder = PointNetLayer(
            dimension+1, history_enc_hidden_dims, encoded_history_dim, pooling)
    
    def _get_mlp_input(self, x_hist, y_hist, x_cand,
                       hist_mask=None, cand_mask=None):
        ## Encode the history inputs into a global feature
        # shape (*, n_hist, dimension+1)
        xy_hist = torch.cat((x_hist, y_hist), dim=-1)
        # shape (*, 1, encoded_history_dim)
        encoded_history = self.history_encoder(xy_hist, mask=hist_mask, keepdim=True)

        ## Prepare input to the acquisition function network final dense layer
        n_cand = x_cand.size(-2)
        logger.debug(f"In AcquisitionFunctionNet.forward, n_cand = {n_cand}")
        # shape (*, n_cand, encoded_history_dim)
        encoded_history_expanded = expand_dim(encoded_history, -2, n_cand)
        logger.debug(f"In AcquisitionFunctionNet.forward, encoded_history_expanded.shape = {encoded_history_expanded.shape}")
        # Maybe neeed to match dimensions (?): (TODO: test this)
        encoded_history_expanded = match_batch_shape(encoded_history_expanded, x_cand)
        logger.debug(f"In AcquisitionFunctionNet.forward, encoded_history_expanded.shape = {encoded_history_expanded.shape}")
        # shape (*, n_cand, dimension+encoded_history_dim)
        x_cand_encoded_history = torch.cat((x_cand, encoded_history_expanded), dim=-1)

        return x_cand_encoded_history

# class AcquisitionFunctionNet(nn.Module):
#     def forward(self, x_hist, y_hist, x_cand, hist_mask=None, cand_mask=None,
#                 **kwargs):

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

