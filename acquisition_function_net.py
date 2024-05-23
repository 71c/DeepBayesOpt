from typing import List, Optional
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


# Unfortunately, torch.masked does not support torch.addmm
# (which is used in nn.Linear), so we have to do the masking manually and
# can't support MaskedTensor.

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
                 history_encoder_hidden_dims=[256, 256],
                 encoded_history_dim=1024,
                 aq_func_hidden_dims=[256, 64]):
        super().__init__()

        self.dimension = dimension

        hist_enc_widths = [dimension+1] + list(history_encoder_hidden_dims) + [encoded_history_dim]
        n_hist_enc_layers = len(hist_enc_widths) - 1
        history_encoder = nn.Sequential()
        for i in range(n_hist_enc_layers):
            in_dim, out_dim = hist_enc_widths[i], hist_enc_widths[i+1]
            
            history_encoder.append(nn.Linear(in_dim, out_dim))
            
            # It was fund that LayerNorm is detrimental to the performance
            # history_encoder.append(nn.LayerNorm(out_dim))
            
            if i != n_hist_enc_layers - 1:
                history_encoder.append(nn.ReLU())
            # history_encoder.append(nn.ReLU())
        self.history_encoder = history_encoder

        aqnet_layer_widths = [dimension + encoded_history_dim] + list(aq_func_hidden_dims) + [1]
        n_aqnet_layers = len(aqnet_layer_widths) - 1
        acquisition_function_net = nn.Sequential()
        for i in range(n_aqnet_layers):
            in_dim, out_dim = aqnet_layer_widths[i], aqnet_layer_widths[i+1]
            
            acquisition_function_net.append(nn.Linear(in_dim, out_dim))

            # It was fund that LayerNorm is detrimental to the performance
            # acquisition_function_net.append(nn.LayerNorm(out_dim))
            
            if i != n_aqnet_layers - 1:
                acquisition_function_net.append(nn.ReLU())
        self.acquisition_function_net = acquisition_function_net
    
    def encode_history(self, x_hist, y_hist, hist_mask=None):
        """Encodes the history inputs into a global feature.

        Args:
            x_hist (torch.Tensor):
                Input history tensor with shape (*, n_hist, dimension).
            y_hist (torch.Tensor):
                Output history tensor with shape (*, n_hist).
            hist_mask (torch.Tensor): Mask tensor for the history inputs with
                shape (*, n_hist). If None, then mask is all ones.

        Returns:
            torch.Tensor: Encoded history tensor with shape (*, 1, encoded_history_dim).
        """
        if x_hist.dim() != y_hist.dim():
            if (x_hist.dim() - y_hist.dim() == 1) and (x_hist.shape[:-1] == y_hist.shape):
                y_hist = y_hist.unsqueeze(-1)
            else:
                raise ValueError("x_hist and y_hist must have the same number of dimensions or y_hist must have one fewer dimension than x_hist.")
        if x_hist.size(-2) != y_hist.size(-2):
            raise ValueError("x_hist and y_hist must have the same number of points in the history dimension.")
        if y_hist.size(-1) != 1:
            raise ValueError("y_hist must have one output dimension.")

        # shape (*, n_hist, dimension+1)
        xy_hist = torch.cat((x_hist, y_hist), dim=-1)

        # shape (*, n_hist, encoded_history_dim)
        local_features = self.history_encoder(xy_hist)

        # Mask out the padded values. It is sufficient to mask at the end.
        if hist_mask is not None:
            hist_mask = hist_mask.unsqueeze(-1) # shape (*, n_hist, 1)

            # This would work for summing
            # local_features = local_features * hist_mask

            # This works for maxing. If ReLU is applied at the end, then
            # we could instead just use the above.
            neg_inf = torch.zeros_like(local_features)
            hist_mask_expanded = expand_dim(hist_mask, -1, local_features.size(-1))
            neg_inf[~hist_mask_expanded] = float("-inf")
            local_features = local_features + neg_inf
        
        # "global feature", shape (*, 1, encoded_history_dim)
        encoded_history = torch.max(local_features, dim=-2, keepdim=True).values
        return encoded_history

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
        new_encoded_history = self.encode_history(x_hist, y_hist, hist_mask)

        # shape (*, 1, encoded_history_dim)
        updated_encoded_history = torch.max(
            torch.cat((original_encoded_history, new_encoded_history), dim=-2),
            dim=-2, keepdim=True).values

        return updated_encoded_history

    def compute_acquisition_with_encoded_history(self, encoded_history, x_cand, cand_mask=None, exponentiate=False):
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

        Returns:
            torch.Tensor: Acquisition values tensor with shape (*, n_cand).
        """
        # shape (*, n_cand, encoded_history_dim)
        n_cand = x_cand.size(-2)
        encoded_history_expanded = expand_dim(encoded_history, -2, n_cand)

        # shape (*, n_cand, dimension+encoded_history_dim)
        x_cand_encoded_history = torch.cat((x_cand, encoded_history_expanded), dim=-1)

        # shape (*, n_cand, 1)
        acquisition_values = self.acquisition_function_net(x_cand_encoded_history)

        if exponentiate:
            acquisition_values = torch.exp(acquisition_values)
        
        if cand_mask is not None:
            cand_mask = cand_mask.unsqueeze(-1) # shape (*, n_cand, 1)
            # Mask out the padded values
            acquisition_values = acquisition_values * cand_mask

        return acquisition_values.squeeze(-1) # shape (*, n_cand)

    def forward(self, x_hist, y_hist, x_cand, hist_mask=None, cand_mask=None, exponentiate=False):
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

        Note: It is assumed x_hist and y_hist are padded (with zeros), although
            that shouldn't matter since the mask will take care of it.

        Returns:
            torch.Tensor: Acquisition values tensor with shape (*, n_cand).
        """
        # "global feature", shape (*, 1, encoded_history_dim)
        encoded_history = self.encode_history(x_hist, y_hist, hist_mask)

        return self.compute_acquisition_with_encoded_history(
            encoded_history, x_cand, cand_mask, exponentiate)


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

