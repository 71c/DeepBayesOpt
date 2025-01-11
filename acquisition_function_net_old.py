class AcquisitionFunctionNetWithFinalMLP(ParameterizedAcquisitionFunctionNet):
    """Abstract class for an acquisition function network with a final MLP
    layer. Subclasses should implement the `_get_mlp_input` method."""
    
    def __init__(self, initial_modules: OrderedDict,
                 input_to_final_layer_dim: int,
                 aq_func_hidden_dims: Sequence[int]=[256, 64],
                 output_dim=1,
                 layer_norm_before_end=False,
                 layer_norm_at_end=False,
                 standardize_outcomes=False,
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
        """
        super().__init__()

        if not isinstance(initial_modules, OrderedDict):
            raise ValueError("initial_modules must be an instance of OrderedDict.")
        for key, val in initial_modules.items():
            setattr(self, key, val)

        self.dense = Dense(input_to_final_layer_dim,
                           aq_func_hidden_dims,
                           output_dim,
                           activation_at_end=False,
                           layer_norm_before_end=layer_norm_before_end,
                           layer_norm_at_end=False,
                           dropout_at_end=False,
                           **dense_kwargs)
        
        self.register_buffer("output_dim",
                             torch.as_tensor(output_dim))
        self.register_buffer("layer_norm_at_end",
                             torch.as_tensor(layer_norm_at_end))
        self.register_buffer("standardize_outcomes",
                             torch.as_tensor(standardize_outcomes))

    @abstractmethod
    def _get_mlp_input(self, x_hist, y_hist, x_cand, acqf_params=None,
                       hist_mask=None, cand_mask=None) -> Tensor:
        """Compute the input to the final MLP network.
        This method should be implemented in a subclass.

        Args:
            x_hist (torch.Tensor):
                A `batch_shape x n_hist x d` tensor of training features.
            y_hist (torch.Tensor):
                A `batch_shape x n_hist x n_out` tensor of training observations.
            x_cand (torch.Tensor):
                A `batch_shape x n_cand x d` tensor of candidate points.
            acqf_params (torch.Tensor, optional):
                Tensor of shape `batch_shape x n_cand x n_params`. Represents any
                variable parameters for the acquisition function, for example
                lambda for the Gittins index or best_f for expected improvement.
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
        pass  # pragma: no cove

    def _get_acquisition_values_and_stdvs(
            self, x_hist, y_hist, x_cand, acqf_params=None,
            hist_mask=None, cand_mask=None):
        preprocessed = self.preprocess_inputs(
            x_hist, y_hist, x_cand, acqf_params=acqf_params,
            hist_mask=hist_mask, cand_mask=cand_mask)

        if self.standardize_outcomes:
            y_hist, stdvs = standardize_y_hist(y_hist, hist_mask)
            preprocessed['y_hist'] = y_hist

        # shape (*, n_cand, input_to_final_layer_dim)
        a = self._get_mlp_input(**preprocessed)

        # shape (*, n_cand, output_dim)
        acquisition_values = self.dense(a)

        if self.layer_norm_at_end:
            # This doesn't handle mask correctly; TODO
            # (only if I'll even end up using this which I probably won't)
            if cand_mask is not None:
                raise NotImplementedError("layer_norm_at_end doesn't handle mask correctly.")
            if acquisition_values.dim() > 2:
                acquisition_values = (acquisition_values - torch.mean(acquisition_values, dim=(-3, -2), keepdim=True)) / torch.std(acquisition_values, dim=(-3, -2), keepdim=True)
        
        return acquisition_values, stdvs
    
    def forward(self, x_hist, y_hist, x_cand, acqf_params=None,
                hist_mask=None, cand_mask=None) -> Tensor:
        acquisition_values, stdvs = self._get_acquisition_values_and_stdvs(
            x_hist, y_hist, x_cand, acqf_params, hist_mask, cand_mask)

        if cand_mask is not None:
            # Mask out the padded values
            acquisition_values = acquisition_values * cand_mask

        return acquisition_values


class AcquisitionFunctionNetWithFinalMLPSoftmaxExponentiate(AcquisitionFunctionNetWithFinalMLP):
    """Abstract class for an acquisition function network with a final MLP
    layer. Subclasses should implement the `_get_mlp_input` method."""
    
    def __init__(self, initial_modules: OrderedDict,
                 input_to_final_layer_dim: int,
                 aq_func_hidden_dims: Sequence[int]=[256, 64],
                 output_dim=1,
                 layer_norm_before_end=False,
                 layer_norm_at_end=False,
                 standardize_outcomes=False,

                 include_alpha=False,
                 learn_alpha=False,
                 initial_alpha=1.0,
                 initial_beta=1.0,
                 learn_beta=False,
                 softplus_batchnorm=False,
                 softplus_batchnorm_momentum=0.1,
                 positive_linear_at_end=False,
                 gp_ei_computation=False,

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
        if positive_linear_at_end and gp_ei_computation:
            raise ValueError(
                "positive_linear_at_end and gp_ei_computation can't both be True.")

        if positive_linear_at_end:
            if learn_beta:
                raise ValueError(
                    "positive_linear_at_end and learn_beta can't both be True.")
            hidden_dims = aq_func_hidden_dims[:-1]
            dense_output_dim = aq_func_hidden_dims[-1] * output_dim
        elif gp_ei_computation:
            if learn_beta:
                raise ValueError(
                    "gp_ei_computation and learn_beta can't both be True.")
            hidden_dims = aq_func_hidden_dims
            dense_output_dim = 2 * output_dim
        else:
            hidden_dims = aq_func_hidden_dims
            dense_output_dim = output_dim
        
        super().__init__(
            initial_modules=initial_modules,
            input_to_final_layer_dim=input_to_final_layer_dim,
            aq_func_hidden_dims=hidden_dims,
            output_dim=dense_output_dim,
            layer_norm_before_end=layer_norm_before_end,
            layer_norm_at_end=layer_norm_at_end,
            standardize_outcomes=standardize_outcomes,
            **dense_kwargs)
        
        self.register_buffer("positive_linear_at_end",
                             torch.as_tensor(positive_linear_at_end))
        self.register_buffer("gp_ei_computation",
                             torch.as_tensor(gp_ei_computation))
        
        self.transform = SoftmaxOrSoftplusLayer(
            softmax_dim=-2,
            include_alpha=include_alpha,
            learn_alpha=learn_alpha,
            initial_alpha=initial_alpha,
            initial_beta=initial_beta,
            learn_beta=learn_beta,
            softplus_batchnorm=softplus_batchnorm,
            softplus_batchnorm_num_features=output_dim,
            softplus_batchnorm_dim=-1,
            softplus_batchnorm_momentum=softplus_batchnorm_momentum,
        )
    
    def get_alpha(self):
        return self.transform.get_alpha()

    def set_alpha(self, val):
        self.transform.set_alpha(val)
    
    def get_beta(self):
        return self.transform.softplus.beta

    @property
    def includes_alpha(self):
        return self.transform.includes_alpha
    
    def forward(self, x_hist, y_hist, x_cand, acqf_params=None,
                hist_mask=None, cand_mask=None,
                exponentiate=False, softmax=False) -> Tensor:
        acquisition_values, stdvs = self._get_acquisition_values_and_stdvs(
            x_hist, y_hist, x_cand, acqf_params=acqf_params,
            hist_mask=hist_mask, cand_mask=cand_mask)
        
        if self.positive_linear_at_end or self.gp_ei_computation:
            last_hidden_dim = acquisition_values.shape[-1] // self.output_dim.item()

            # shape (*, n_cand, last_hidden_dim, output_dim)
            acquisition_values = acquisition_values.view(*acquisition_values.shape[:-1],
                                                        last_hidden_dim,
                                                        self.output_dim.item())
            
            if self.positive_linear_at_end:
                # # shape (*, 1, 1)
                # best_y = get_best_y(y_hist, hist_mask)
                # # shape (*, 1, 1, 1)
                # best_y = best_y.unsqueeze(-1)
                # acquisition_values = nn.functional.relu(
                #     acquisition_values - best_y, inplace=True)

                acquisition_values = nn.functional.relu(
                    acquisition_values, inplace=False)

                # shape (*, n_cand, output_dim)
                acquisition_values = acquisition_values.mean(dim=-2, keepdim=False)
            else:
                # shape (*, 1, 1)
                best_y = get_best_y(y_hist, hist_mask)
                means = acquisition_values[..., 0, :]
                sigmas = nn.functional.softplus(acquisition_values[..., 1, :], beta=1.)
                acquisition_values = sigmas * nn.functional.softplus(
                    (means - best_y) / sigmas, beta=1.77)

        acquisition_values = self.transform(
            acquisition_values, mask=cand_mask,
            exponentiate=exponentiate and not (self.positive_linear_at_end or self.gp_ei_computation),
            softmax=softmax
        )
        
        if self.standardize_outcomes and exponentiate:
            # Assume that if exponentiate=True, then we are computing EI
            acquisition_values = acquisition_values * stdvs

        if cand_mask is not None:
            # Mask out the padded values
            acquisition_values = acquisition_values * cand_mask

        return acquisition_values

    # def __init__(self,
    #              input_to_final_layer_dim: int,
    #              aq_func_hidden_dims: Sequence[int]=[256, 64],
    #              output_dim=1,
    #              layer_norm_before_end=False,
    #              layer_norm_at_end=False,
    #              **dense_kwargs):

    #     @abstractmethod
    # def forward(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
    #             hist_mask:Optional[Tensor]=None, cand_mask:Optional[Tensor]=None,
    #             **kwargs) -> Tensor:




        # def __init__(self,
        #          input_to_final_layer_dim: int,
        #          aq_func_hidden_dims: Sequence[int]=[256, 64],
        #          output_dim=1,
        #          layer_norm_before_end=False,
        #          layer_norm_at_end=False,
        #          **dense_kwargs):









class AcquisitionFunctionNetV1and2(AcquisitionFunctionNetWithFinalMLP):
    def __init__(self,
                 dimension, history_enc_hidden_dims=[256, 256], pooling="max",
                 encoded_history_dim=1024, aq_func_hidden_dims=[256, 64],
                 output_dim=1,
                 input_xcand_to_local_nn=True,
                 input_xcand_to_final_mlp=False,
                 
                 activation_at_end_pointnet=True,
                 layer_norm_pointnet=False,
                 dropout_pointnet=None,
                 layer_norm_before_end_mlp=False,
                 layer_norm_at_end_mlp=False,
                 dropout_mlp=None,
                 standardize_outcomes=False,
                 include_best_y=False,
                 activation_pointnet:str="relu",
                 activation_mlp:str="relu",
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
                dropout=dropout_pointnet,
                dropout_at_end=True,
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
                         aq_func_hidden_dims, output_dim,
                         layer_norm_before_end_mlp, layer_norm_at_end_mlp,
                         standardize_outcomes=standardize_outcomes,
                         activation=activation_mlp,
                         dropout=dropout_mlp)
        self.dimension = dimension
        self.include_best_y = include_best_y
    
    def _get_mlp_input(self, x_hist, y_hist, x_cand,
                       hist_mask=None, cand_mask=None):
        if self.include_best_y:
            y_hist = concat_y_hist_with_best_y(y_hist, hist_mask, subtract=False)

        if self.input_xcand_to_local_nn: # V2
            # xy_hist_and_cand shape: (*, n_cand, n_hist, 2*dimension+1)
            xy_hist, xy_hist_and_cand, mask = _get_xy_hist_and_cand(x_hist, y_hist, x_cand, hist_mask)
            # shape (*, n_cand, encoded_history_dim)
            out = self.pointnet(xy_hist_and_cand, mask=mask, keepdim=False)
            logger.debug(f"out.shape: {out.shape}")
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


class AcquisitionFunctionNetV3(AcquisitionFunctionNetWithFinalMLPSoftmaxExponentiate):
    def __init__(self,
                 dimension, history_enc_hidden_dims=[256, 256], pooling="max",
                 encoded_history_dim=1024,
                 mean_enc_hidden_dims=[256, 256], mean_dim=1,
                 std_enc_hidden_dims=[256, 256], std_dim=1,
                 aq_func_hidden_dims=[256, 64], output_dim=1,
                 layer_norm=False,
                 layer_norm_at_end_mlp=False, include_y=False,
                 standardize_outcomes=False,
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0,
                 initial_beta=1.0, learn_beta=False,
                 softplus_batchnorm=False, softplus_batchnorm_momentum=0.1,
                 activation_pointnet:str="relu",
                 activation_mlp:str="relu"):
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
                               layer_norm_at_end=False,
                               activation=activation_pointnet)),
            ('std_net', PointNetLayer(
            tmp, std_enc_hidden_dims,
            std_dim, pooling, activation_at_end=True,
            layer_norm_before_end=layer_norm, layer_norm_at_end=False,
            activation=activation_pointnet))
        ])
        super().__init__(initial_modules, mean_dim + std_dim,
                         aq_func_hidden_dims, output_dim,
                         include_alpha, learn_alpha, initial_alpha,
                         initial_beta, learn_beta,
                         softplus_batchnorm, softplus_batchnorm_momentum,
                         layer_norm_before_end=layer_norm,
                         layer_norm_at_end=layer_norm_at_end_mlp,
                         standardize_outcomes=standardize_outcomes,
                         activation=activation_mlp)
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


class AcquisitionFunctionNetV4(AcquisitionFunctionNetWithFinalMLPSoftmaxExponentiate):
    def __init__(self,
                 dimension, history_enc_hidden_dims=[256, 256], pooling="max",
                 encoded_history_dim=1024,
                 mean_enc_hidden_dims=[256, 256], mean_dim=1,
                 std_enc_hidden_dims=[256, 256], std_dim=1,
                 aq_func_hidden_dims=[256, 64],
                 output_dim=1,
                 layer_norm=False,
                 layer_norm_at_end_mlp=False,
                 standardize_outcomes=False,
                 include_mean=True,
                 include_local_features=False,
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0,
                 initial_beta=1.0, learn_beta=False,
                 softplus_batchnorm=False, softplus_batchnorm_momentum=0.1,
                 activation_pointnet:str="relu",
                 activation_mlp:str="relu"):
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
            layer_norm_before_end=layer_norm, layer_norm_at_end=layer_norm,
            activation=activation_pointnet)),
            ('std_net', PointNetLayer(
            std_input_dim, std_enc_hidden_dims,
            std_dim, pooling, activation_at_end=include_mean,
            layer_norm_before_end=layer_norm,
            layer_norm_at_end=layer_norm and not include_mean,
            activation=activation_pointnet))
        ])
        if include_mean:
            mean_input_dim = encoded_history_dim + (
                2 * dimension + 1 if include_local_features else 0)
            initial_modules['mean_net'] = Dense(mean_input_dim,
                               mean_enc_hidden_dims, mean_dim,
                               activation_at_end=False,
                               layer_norm_before_end=layer_norm,
                               layer_norm_at_end=False,
                               activation=activation_pointnet)
        
        super().__init__(initial_modules, std_dim + (mean_dim if include_mean else 0),
                         aq_func_hidden_dims, output_dim,
                         include_alpha, learn_alpha, initial_alpha,
                         initial_beta, learn_beta,
                         softplus_batchnorm, softplus_batchnorm_momentum,
                         layer_norm_before_end=layer_norm,
                         layer_norm_at_end=layer_norm_at_end_mlp,
                         standardize_outcomes=standardize_outcomes,
                         activation=activation_mlp)
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
            # logger.debug(cand_mask_expanded.shape)

        # logger.debug(x_cand_and_xy_hist_pairwise - x_cand_and_xy_hist_pairwise * mask)
        # logger.debug(x_cand_and_xy_hist_pairwise)
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


class AcquisitionFunctionNetDense(AcquisitionFunctionNetWithFinalMLPSoftmaxExponentiate):
    def __init__(self, dimension, max_history, hidden_dims=[256, 64],
                 output_dim=1,
                 include_alpha=False, learn_alpha=False, initial_alpha=1.0,
                 initial_beta=1.0, learn_beta=False,
                 softplus_batchnorm=False, softplus_batchnorm_momentum=0.1,
                 standardize_outcomes=False,
                 activation:str="relu"):
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
                         output_dim,
                         include_alpha, learn_alpha, initial_alpha,
                         initial_beta, learn_beta,
                         softplus_batchnorm, softplus_batchnorm_momentum,
                            layer_norm_before_end=False,
                            layer_norm_at_end=False,
                            standardize_outcomes=standardize_outcomes,
                            activation=activation)
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


    # @abstractmethod
    # def forward(self, x_hist:Tensor, y_hist:Tensor, x_cand:Tensor,
    #             acqf_params:Optional[Tensor]=None,
    #             hist_mask:Optional[Tensor]=None,
    #             cand_mask:Optional[Tensor]=None) -> Tensor:


def _get_xy_hist_and_cand(x_hist, y_hist, x_cand, hist_mask=None, include_y=True):
    """Combines historical data and candidate data for Bayesian optimization.

    Args:
        x_hist (torch.Tensor):
            Historical input data with shape (*, n_hist, dim_hist).
        y_hist (torch.Tensor):
            Historical output data with shape (*, n_hist, n_out).
        x_cand (torch.Tensor):
            Candidate input data with shape (*, n_cand, dim_cand).
        hist_mask (torch.Tensor, optional):
            Mask for historical data with shape (*, n_hist, 1).
        include_y (bool, default: True):
            Whether to include historical output data in the combined tensor.

    Returns:
        A tuple containing:
            - xy_hist (torch.Tensor):
                Combined historical input and output data with shape
                (*, n_hist, dim_hist+n_out).
            - xy_hist_and_cand (torch.Tensor):
                Combined candidate and historical data with shape
                (*, n_cand, n_hist, dim_cand+dim_hist+n_out) if include_y is True,
                otherwise with shape (*, n_cand, n_hist, dim_cand+dim_hist).
            - mask (torch.Tensor or None):
                Expanded mask for historical data with shape (*, n_cand, n_hist, 1) if
                hist_mask is provided, otherwise None.
    """





        # Args:
        #     x_hist (torch.Tensor):
        #         A `batch_shape x n_hist x d` tensor of training features.
        #     y_hist (torch.Tensor):
        #         A `batch_shape x n_hist` or `batch_shape x n_hist x n_out`
        #         tensor of training observations.
        #     x_cand (torch.Tensor):
        #         Candidate input tensor with shape `batch_shape x n_cand x d`.
        #     acqf_params (torch.Tensor, optional):
        #         Tensor of shape `batch_shape x n_cand x n_acqf_params`. Represents any
        #         variable parameters for the acquisition function, for example
        #         lambda for the Gittins index or best_f for expected improvement.
        #     hist_mask (torch.Tensor, optional):
        #         Mask tensor for the history inputs with shape `batch_shape x n_hist`
        #         or `batch_shape x n_hist x 1`. If None, then mask is all ones.
        #     cand_mask (torch.Tensor, optional):
        #         Mask tensor for the candidate inputs with shape `batch_shape x n_cand`
        #         or `batch_shape x n_cand x 1`. If None, then mask is all ones.




class GittinsAcquisitionFunctionNet(AcquisitionFunctionNet):
    def __init__(self, acquisition_function_class,
                 variable_lambda:bool,
                 costs_in_history:bool, cost_is_input:bool,
                 assume_y_independent_cost:bool=False,
                 **init_kwargs):
        r"""Initialize the GittinsAcquisitionFunctionNet class.
        
        Args:
            acquisition_function_class (subclass of AcquisitionFunctionNetFixedHistoryOutputDim):
                The class of the acquisition function to use.
            variable_lambda (bool):
                Whether to use a variable lambda that is input to the NN.
            costs_in_history (bool):
                Whether the past costs are in the history.
            cost_is_input (bool):
                Whether the cost of a candidate is an input to the NN
                (in this case we can know the cost before evaluation).
            assume_y_independent_cost (bool, default: False):
                Whether to assume that the cost is independent of the function value
                conditioned on the history and candidate point,
                so that only lambda*cost is input rather than (lambda*cost, cost).
                ONLY applicable if both variable_lambda=True and cost_is_input=True.
            **init_kwargs:
                Arguments to pass to the acquisition function class __init__
                except for the dimension argument.
        
        `costs_in_history` and `cost_is_input` can be chosen based on the following
        rules, if `heterogeneous_costs` tells whether costs are heterogeneous and
        `known_cost` tells whether the cost is known before evaluation:
        if heterogeneous_costs:
            if known_cost:
                costs_in_history: Could be False since we already know the cost.
                    Could also set True if we think that past costs tell us
                    something about future y values.
                cost_is_input: True
            else:
                costs_in_history: True
                cost_is_input: False
        else:
            costs_in_history: False
            cost_is_input: False
        """
        super().__init__()

        if type(variable_lambda) is not bool:
            raise ValueError("variable_lambda should be bool")
        if type(costs_in_history) is not bool:
            raise ValueError("costs_in_history should be bool")
        if type(cost_is_input) is not bool:
            raise ValueError("cost_is_input should be bool")
        self.variable_lambda = variable_lambda
        self.costs_in_history = costs_in_history
        self.cost_is_input = cost_is_input

        if variable_lambda and cost_is_input:
            if type(assume_y_independent_cost) is not bool:
                raise ValueError("assume_y_independent_cost should be bool")
            n_acqf_params = 1 if assume_y_independent_cost else 2
            self.assume_y_independent_cost = assume_y_independent_cost
        else:
            n_acqf_params = variable_lambda + cost_is_input
        
        if not issubclass(acquisition_function_class, AcquisitionFunctionNetFixedHistoryOutputDim):
            raise ValueError(
                "acquisition_function_class should be a subclass of AcquisitionFunctionNetFixedHistoryOutputDim")
        additional_kwargs = {
            'n_hist_out': 1 + costs_in_history # for AcquisitionFunctionNetFixedHistoryOutputDim
        }
        if issubclass(acquisition_function_class, ParameterizedAcquisitionFunctionNet):
            additional_kwargs['n_acqf_params'] = n_acqf_params
        elif n_acqf_params > 0:
            raise ValueError(
                "acquisition_function_class should be a subclass of "
                f"ParameterizedAcquisitionFunctionNet since there are {n_acqf_params} parameters")
        
        self.base_model = acquisition_function_class(**additional_kwargs, **init_kwargs)

    def forward(self, x_hist, y_hist, x_cand,
                cost_hist=None, lambda_cand=None, cost_cand=None,
                hist_mask=None, cand_mask=None):
        r"""Forward pass of the acquisition function network.
        Args:
            x_hist (torch.Tensor):
                A `batch_shape x n_hist x d` tensor of training features.
            y_hist (torch.Tensor):
                A `batch_shape x n_hist x output_dim` tensor of training observations.
            x_cand (torch.Tensor):
                Candidate input tensor with shape `batch_shape x n_cand x d`.
            cost_hist (torch.Tensor, optional):
                A `batch_shape x n_hist` or `batch_shape x n_hist x 1` tensor of costs
                for the history points.
            lambda_cand (torch.Tensor, optional):
                A `batch_shape x n_cand` or `batch_shape x n_cand x 1` tensor of lambda
                values for the candidate points.
            cost_cand (torch.Tensor, optional):
                A `batch_shape x n_cand` or `batch_shape x n_cand x 1` tensor of costs
                for the candidate points.
            hist_mask (torch.Tensor, optional):
                Mask tensor for the history inputs with shape `batch_shape x n_hist`
                or `batch_shape x n_hist x 1`. If None, then mask is all ones.
            cand_mask (torch.Tensor, optional):
                Mask tensor for the candidate inputs with shape `batch_shape x n_cand`
                or `batch_shape x n_cand x 1`. If None, then mask is all ones.
        Returns:
            torch.Tensor: A `batch_shape x n_cand x output_dim` tensor of acquisition
            values. (`output_dim` is 1 for most acquisition functions)
        """
        ## Handle the history of y and cost
        y_hist = check_xy_dims(x_hist, y_hist, "x_hist", "y_hist", expected_y_dim=1)
        if self.costs_in_history:
            if cost_hist is None:
                raise ValueError("cost_hist must be specified if costs_in_history=True")
            cost_hist = check_xy_dims(x_hist, cost_hist, "x_hist", "cost_hist", expected_y_dim=1)
            y_hist = torch.cat((y_hist, cost_hist), dim=-1)
        elif cost_hist is not None:
            raise ValueError("cost_hist should not be specified if costs_in_history=False")

        ## Make sure the lambda and cost are specified or not as expected
        if self.variable_lambda:
            if lambda_cand is None:
                raise ValueError("lambda_cand must be specified if variable_lambda=True")
            lambda_cand = check_xy_dims(x_cand, lambda_cand, "x_cand", "lambda_cand", expected_y_dim=1)
        elif lambda_cand is not None:
            raise ValueError("lambda_cand should not be specified if variable_lambda=False")
        if self.cost_is_input:
            if cost_cand is None:
                raise ValueError("cost_cand must be specified if cost_is_input=True")
            cost_cand = check_xy_dims(x_cand, cost_cand, "x_cand", "cost_cand", expected_y_dim=1)
        elif cost_cand is not None:
            raise ValueError("cost_cand should not be specified if cost_is_input=False")
        
        ## Set the necessary acqf_params
        # (if at least one of variable_lambda or cost_is_input)
        if self.variable_lambda and self.cost_is_input:
            lambda_cost_cand = lambda_cand * cost_cand
            if self.assume_y_independent_cost:
                acqf_params = lambda_cost_cand
            else:
                acqf_params = torch.cat((lambda_cost_cand, cost_cand), dim=-1)
        elif self.variable_lambda:
            acqf_params = lambda_cand
        elif self.cost_is_input:
            acqf_params = cost_cand
        
        call_kwargs = dict(
            x_hist=x_hist,
            y_hist=y_hist,
            x_cand=x_cand,
            hist_mask=hist_mask,
            cand_mask=cand_mask
        )

        # Add acqf_params if necessary
        if self.variable_lambda or self.cost_is_input:
            call_kwargs['acqf_params'] = acqf_params
        
        return self.base_model(**call_kwargs)


def check_class_for_init_params(cls, base_class_name:str, *required_params):
    init_sig = inspect.signature(cls.__init__)
    for param_name in required_params:
        if param_name not in init_sig.parameters:
            raise TypeError(
                f"Class {cls.__name__} is missing required init param '{param_name}' "
                f"since it is a subclass of {base_class_name}"
            )
