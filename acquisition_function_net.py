import torch
from torch import nn

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
        history_encoder = nn.Sequential()
        for in_dim, out_dim in zip(hist_enc_widths[:-1], hist_enc_widths[1:]):
            history_encoder.append(nn.Linear(in_dim, out_dim))
            history_encoder.append(nn.ReLU())
        self.history_encoder = history_encoder

        aqnet_layer_widths = [dimension + encoded_history_dim] + list(aq_func_hidden_dims) + [1]
        n_aqnet_layers = len(aqnet_layer_widths) - 1
        acquisition_function_net = nn.Sequential()
        for i in range(n_aqnet_layers):
            in_dim, out_dim = aqnet_layer_widths[i], aqnet_layer_widths[i+1]
            acquisition_function_net.append(nn.Linear(in_dim, out_dim))
            if i != n_aqnet_layers - 1:
                acquisition_function_net.append(nn.ReLU())
        self.acquisition_function_net = acquisition_function_net

    def forward(self, x_hist, y_hist, x_cand, hist_mask=None, cand_mask=None):
        """Forward pass of the acquisition function network.

        Args:
            x_hist (torch.Tensor):
                Input history tensor with shape (*, n_hist, dimension).
            y_hist (torch.Tensor):
                Output history tensor with shape (*, n_hist).
            x_cand (torch.Tensor):
                Candidate input tensor with shape (*, n_cand, dimension).
            hist_mask (torch.Tensor): Mask tensor for the history inputs with
                shape (*, n_hist). Defaults to have all ones.
            cand_mask (torch.Tensor): Mask tensor for the candidate inputs with
                shape (*, n_cand). Defaults to have all ones.

        Note: It is assumed x_hist and y_hist are padded (with zeros), although
            that shouldn't matter since the mask will take care of it.

        Returns:
            torch.Tensor: Acquisition values tensor with shape (*, n_cand).
        """
        if hist_mask is None:
            hist_mask = torch.ones_like(x_hist[..., 0], dtype=torch.bool)
        hist_mask = hist_mask.unsqueeze(-1) # shape (*, n_hist, 1)

        if cand_mask is None:
            cand_mask = torch.ones_like(x_cand[..., 0], dtype=torch.bool)
        cand_mask = cand_mask.unsqueeze(-1) # shape (*, n_cand, 1)

        # shape (*, n_hist, dimension+1)
        xy_hist = torch.cat((x_hist, y_hist.unsqueeze(-1)), dim=-1)

        # shape (*, n_hist, encoded_history_dim)
        local_features = self.history_encoder(xy_hist)
        # Mask out the padded values. It is sufficient to mask at the end.
        local_features = local_features * hist_mask
        
        # "global feature", shape (*, 1, encoded_history_dim)
        # The masking zeros will not affect the max operation because 
        # ReLU was applied at the end so all values are >= 0.
        encoded_history = torch.max(local_features, dim=-2, keepdim=True).values
        
        # (*, n_cand, encoded_history_dim)
        encoded_history_new_shape = list(encoded_history.shape)
        n_cand = x_cand.size(-2)
        encoded_history_new_shape[-2] = n_cand

        # shape (*, n_cand, encoded_history_dim)
        encoded_history_expanded = encoded_history.expand(*encoded_history_new_shape)

        # shape (*, n_cand, dimension+encoded_history_dim)
        x_cand_encoded_history = torch.cat((x_cand, encoded_history_expanded), dim=-1)

        # shape (*, n_cand, 1)
        acquisition_values = self.acquisition_function_net(x_cand_encoded_history)
        
        # Mask out the padded values
        acquisition_values = acquisition_values * cand_mask

        return acquisition_values.squeeze(-1) # shape (*, n_cand)
