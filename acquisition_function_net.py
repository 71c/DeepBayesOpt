from typing import List, Optional
import warnings
import torch
from torch import nn
from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import UnsupportedError
from botorch.utils.transforms import t_batch_mode_transform, match_batch_shape


def expand_dim(tensor, dim, k):
    new_shape = list(tensor.shape)
    new_shape[dim] = k
    return tensor.expand(*new_shape)


def make_dimensions_compatible(x: Tensor, y: Tensor, x_name: str, y_name: str) -> Tensor:
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


# Unfortunately, torch.masked does not support torch.addmm
# (which is used in nn.Linear), so we have to do the masking manually and
# can't support MaskedTensor.


class AcquisitionNetFinalLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 64], learn_alpha=False):
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

        if learn_alpha:
            self.includes_alpha = True
            self.log_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.includes_alpha = False
    
    @property
    def alpha(self):
        if self.includes_alpha:
            return torch.exp(self.log_alpha)
        else:
            raise ValueError("Model does not include alpha.")
    
    def forward(self, x, cand_mask=None, exponentiate=False, softmax=False):
        """Compute the acquisition function.

        Args:
            x (torch.Tensor):
                Fixed-dimension input of shape (*, n_cand, input_dim)
            cand_mask (torch.Tensor): Mask tensor for the candidate inputs with
                shape (*, n_cand). If None, then mask is all ones.
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
                acquisition_values = acquisition_values * self.alpha
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
            cand_mask = cand_mask.unsqueeze(-1) # shape (*, n_cand, 1)
            # Mask out the padded values
            acquisition_values = acquisition_values * cand_mask

        return acquisition_values.squeeze(-1) # shape (*, n_cand)


class PointNetLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 256], output_dim=1024, pooling="max"):
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
        """Computes the output

        Args:
            x (torch.Tensor): Input tensor with shape (*, n, input_dim).
            mask (torch.Tensor): Mask tensor with shape (*, n).
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
            mask = make_dimensions_compatible(x, mask, "x", "mask")

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


class AcquisitionFunctionNet(nn.Module):
    """
    Neural network model for the acquisition function in NN-based
        likelihood-free Bayesian optimization.

    Args:
        dimension (int): The dimensionality of the input space.
        history_encoder_hidden_dims (list): List of integers representing the
            hidden layer dimensions of the history encoder network.
            Default is [256, 256].
        encoded_history_dim (int): The dimensionality of the encoded history
            representation. Default is 1024.
        aq_func_hidden_dims (list): List of integers representing the hidden
            layer dimensions of the acquisition function network.
            Default is [256, 64].
    """

    def __init__(self,
                 dimension,
                 history_enc_hidden_dims=[256, 256],
                 encoded_history_dim=1024,
                 aq_func_hidden_dims=[256, 64],
                 learn_alpha=False,
                 pooling="max"):
        super().__init__()

        self.dimension = dimension

        self.history_encoder = PointNetLayer(
            dimension+1, history_enc_hidden_dims, encoded_history_dim, pooling)

        self.acquisition_function_net = AcquisitionNetFinalLayer(
            dimension + encoded_history_dim, aq_func_hidden_dims, learn_alpha)
    
    @property
    def alpha(self):
        return self.acquisition_function_net.alpha

    @property
    def log_alpha(self):
        return self.acquisition_function_net.log_alpha

    @property
    def includes_alpha(self):
        return self.acquisition_function_net.includes_alpha
    
    def encode_history(self, x_hist, y_hist, hist_mask=None):
        """Encodes the history inputs into a global feature.

        Args:
            x_hist (torch.Tensor):
                Input history tensor with shape (*, n_hist, dimension).
            y_hist (torch.Tensor):
                Output history tensor with shape (*, n_hist) or (*, n_hist, 1)
            hist_mask (torch.Tensor): Mask tensor for the history inputs with
                shape (*, n_hist). If None, then mask is all ones.

        Returns:
            torch.Tensor: Encoded history tensor with shape (*, 1, encoded_history_dim).
        """
        # Make y_hist have shape (*, n_hist, 1)
        y_hist = make_dimensions_compatible(x_hist, y_hist, "x_hist", "y_hist")

        # shape (*, n_hist, dimension+1)
        xy_hist = torch.cat((x_hist, y_hist), dim=-1)

        # shape (*, 1, encoded_history_dim)
        return self.history_encoder(xy_hist, mask=hist_mask, keepdim=True)

    def add_to_encoded_history(self, original_encoded_history, x_hist, y_hist, hist_mask=None):
        """Add new history to the encoded history.

        Args:
            original_encoded_history (torch.Tensor):
                Original encoded history tensor with shape (*, 1, encoded_history_dim).
            x_hist (torch.Tensor):
                Input history tensor with shape (*, n_hist, dimension).
            y_hist (torch.Tensor):
                Output history tensor with shape (*, n_hist).
            hist_mask (torch.Tensor): Mask tensor for the history inputs with
                shape (*, n_hist). If None, then mask is all ones.

        Returns:
            torch.Tensor: Updated encoded history tensor with shape (*, 1, encoded_history_dim).
        """
        # shape (*, 1, encoded_history_dim)
        add_enc_hist = self.encode_history(x_hist, y_hist, hist_mask)

        # shape (*, 2, encoded_history_dim)
        concat_hist = torch.cat((original_encoded_history, add_enc_hist), dim=-2)

        # shape (*, 1, encoded_history_dim)
        if self.pooling == "sum":
            new_enc_hist = torch.sum(concat_hist, dim=-2, keepdim=True)
        else: # self.pooling == "max"
            new_enc_hist = torch.max(concat_hist, dim=-2, keepdim=True).values

        return new_enc_hist

    def compute_acquisition_with_encoded_history(
            self, encoded_history, x_cand, cand_mask=None,
            exponentiate=False, softmax=False):
        """Compute the acquisition function with the encoded history.

        Args:
            encoded_history (torch.Tensor):
                Encoded history tensor with shape (*, 1, encoded_history_dim).
            x_cand (torch.Tensor):
                Candidate input tensor with shape (*, n_cand, dimension).
            cand_mask (torch.Tensor): Mask tensor for the candidate inputs with
                shape (*, n_cand). If None, then mask is all ones.
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
        # shape (*, n_cand, encoded_history_dim)
        n_cand = x_cand.size(-2)
        encoded_history_expanded = expand_dim(encoded_history, -2, n_cand)

        # shape (*, n_cand, dimension+encoded_history_dim)
        x_cand_encoded_history = torch.cat((x_cand, encoded_history_expanded), dim=-1)

        # shape (*, n_cand)
        return self.acquisition_function_net(x_cand_encoded_history,
                                             cand_mask=cand_mask,
                                             exponentiate=exponentiate,
                                             softmax=softmax)

    def forward(self, x_hist, y_hist, x_cand,
                hist_mask=None, cand_mask=None,
                exponentiate=False, softmax=False):
        """Forward pass of the acquisition function network.

        Args:
            x_hist (torch.Tensor):
                Input history tensor with shape (*, n_hist, dimension).
            y_hist (torch.Tensor):
                Output history tensor with shape (*, n_hist).
            x_cand (torch.Tensor):
                Candidate input tensor with shape (*, n_cand, dimension).
            hist_mask (torch.Tensor): Mask tensor for the history inputs with
                shape (*, n_hist). If None, then mask is all ones.
            cand_mask (torch.Tensor): Mask tensor for the candidate inputs with
                shape (*, n_cand). If None, then mask is all ones.
            exponentiate (bool, optional): Whether to exponentiate the output.
                Default is False. For EI, False corresponds to the log of the
                acquisition function (e.g. log EI), and True corresponds to
                the acquisition function itself (e.g. EI).
                Only applies if self.includes_alpha is False.
            softmax (bool, optional): Whether to apply softmax to the output.
                Only applies if self.includes_alpha is True. Default is False.

        Note: It is assumed x_hist and y_hist are padded (with zeros), although
            that shouldn't matter since the mask will take care of it.

        Returns:
            torch.Tensor: Acquisition values tensor with shape (*, n_cand).
        """
        # "global feature", shape (*, 1, encoded_history_dim)
        encoded_history = self.encode_history(x_hist, y_hist, hist_mask)

        return self.compute_acquisition_with_encoded_history(
            encoded_history, x_cand, cand_mask, exponentiate, softmax)


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
            train_Y: A `batch_shape x n x m` tensor of training observations.
        """
        super().__init__()
        model.eval()
        self.model = model

        if train_X is not None and train_Y is not None:
            self._encoded_history = model.encode_history(train_X, train_Y)
        elif train_X is None and train_Y is None:
            self._encoded_history = None
        else:
            raise ValueError("Both train_X and train_Y must be provided or neither.")
    
    def posterior(self, *args, **kwargs):
        raise UnsupportedError("AcquisitionFunctionNetModel does not support posterior inference.")
    
    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1 # Only supporting 1 output (for now at least)
    
    def subset_output(self, idcs: List[int]):
        raise UnsupportedError("AcquisitionFunctionNetModel does not support output subsetting.")

    def condition_on_observations(self, X: Tensor, Y: Tensor) -> Model:
        if self._encoded_history is None:
            return AcquisitionFunctionNetModel(self.model, X, Y)
        else:
            new_encoded_history = self.model.add_to_encoded_history(
                self._encoded_history, X, Y)
            ret = AcquisitionFunctionNetModel(self.model, train_X=None, train_Y=None)
            ret._encoded_history = new_encoded_history
            return ret

    def forward(self, X: Tensor, exponentiate=False) -> Tensor:
        """
        Forward pass of the acquisition function network.

        Args:
            X (Tensor): The input tensor of shape `(batch_shape) x q x d`.
            exponentiate (bool): Whether to exponentiate the output.

        Returns:
            Tensor: The output tensor of shape `(batch_shape) x q`.

        Raises:
            RuntimeError: If the encoded history is not available.
        """
        if self._encoded_history is None:
            raise RuntimeError("Cannot make predictions without conditioning on data.")
        encoded_history = match_batch_shape(self._encoded_history, X)
        return self.model.compute_acquisition_with_encoded_history(
            encoded_history, X, exponentiate=exponentiate)


class LikelihoodFreeNetworkAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: AcquisitionFunctionNetModel,
                 exponentiate: bool = False):
        """
        Args:
            model: The acquisition function network model.
        """
        super().__init__(model=model)
        self.exponentiate = exponentiate
    
    @classmethod
    def from_net(cls, model: AcquisitionFunctionNet,
                 train_X: Optional[Tensor]=None,
                 train_Y: Optional[Tensor]=None,
                 exponentiate: bool = False) -> "LikelihoodFreeNetworkAcquisitionFunction":
        return cls(AcquisitionFunctionNetModel(model, train_X, train_Y),
                   exponentiate=exponentiate)
    
    # They all do this
    # https://botorch.org/api/utils.html#botorch.utils.transforms.t_batch_mode_transform
    # https://botorch.org/api/_modules/botorch/utils/transforms.html#t_batch_mode_transform
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of acquisition function values at the given
            design points `X`.
        """
        # shape (batch_shape) x q=1
        output = self.model(X, exponentiate=self.exponentiate)
        assert output.size(-1) == 1
        return output.squeeze(-1)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError("AcquisitionFunctionNetModel does not support pending points.")

