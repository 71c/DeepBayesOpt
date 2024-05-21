import torch
from torch import nn


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
        if hist_mask is not None:
            hist_mask = hist_mask.unsqueeze(-1) # shape (*, n_hist, 1)

        if cand_mask is not None:
            cand_mask = cand_mask.unsqueeze(-1) # shape (*, n_cand, 1)

        # shape (*, n_hist, dimension+1)
        xy_hist = torch.cat((x_hist, y_hist.unsqueeze(-1)), dim=-1)

        # shape (*, n_hist, encoded_history_dim)
        local_features = self.history_encoder(xy_hist)

        # Mask out the padded values. It is sufficient to mask at the end.
        if hist_mask is not None:        
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
            # Mask out the padded values
            acquisition_values = acquisition_values * cand_mask

        return acquisition_values.squeeze(-1) # shape (*, n_cand)
