from botorch.models.gp_regression import SingleTaskGP
from gpytorch.models import ExactGP
from botorch.exceptions import UnsupportedError
import torch
import utils # just to make sure that the additions to the BoTorch Model classes are loaded


def _gp_model_has_no_data(model: ExactGP):
    """Input is assumed to no-batch, single-output"""
    if model.train_inputs is None:
        assert model.train_targets is None
        return True

    assert len(model.train_inputs) == 1 # should always be 1, I think
    current_X = model.train_inputs[0]
    current_Y = model.train_targets
    assert current_X.dim() == 2 and current_Y.dim() == 1

    n_points = current_X.size(0)
    assert n_points == current_Y.size(0)

    return n_points == 0


# Can consider using botorch.models.utils.assorted.consolidate_duplicates instead
def get_unique(x):
    unique_indices_list = []
    for input_index in range(x.size(0)):
        input_point = x[input_index]
        
        found_matching_point = False
        for unique_point_index, original_indices in enumerate(unique_indices_list):
            unique_point = x[original_indices[0]]
            if torch.equal(input_point, unique_point):
                unique_indices_list[unique_point_index].append(input_index)
                found_matching_point = True
        
        if not found_matching_point:
            unique_indices_list.append([input_index])
    
    return x[[inds[0] for inds in unique_indices_list]], unique_indices_list


# Can consider using botorch.models.model.FantasizeMixin and `fantasize` instead
# (SingleTaskGP is a subclass of FantasizeMixin)
# But then maybe need to worry about input transform which I don't understand
class RandomGPFunction:
    def __init__(self, model: SingleTaskGP, observation_noise:bool=False):
        if not isinstance(model, SingleTaskGP):
            raise UnsupportedError(f"model should be a SingleTaskGP instance.")
        # Verify that the model is single-batch
        if len(model.batch_shape) != 0:
                raise UnsupportedError("model must be single-batch, but "
                                       f"model has batch shape {model.batch_shape}")
        # Verify that the model is single-output
        if model.num_outputs != 1:
            raise UnsupportedError("model must be single-output, but "
                                   f"model is {model.num_outputs}-output")
        
        self.model = model

        if not isinstance(observation_noise, bool):
            raise TypeError("observation_noise must be a boolean value.")
        self.observation_noise = observation_noise

    def __call__(self, x):
        """
        Args:
            x:
                torch.Tensor of shape (dimension,) or (n_evals, dimension)
        
        Returns:
            The corresponding output of the GP realization, of shape
            (n_evals,) if x is 2-dimensional or a scalar tensor if
            the x is 1-dimensional.
        """
        input_is_dim1 = False
        if x.dim() == 1:
            input_is_dim1 = True
            x = x.unsqueeze(0) # add n_evals dimension 
        if x.dim() != 2:
            raise ValueError(f"x should be a 2D tensor of shape "
                             "(n_evals, dimension) or (dimension,).")

        n_input_points = x.size(0)
        assert n_input_points > 0
        
        model = self.model
        model_is_empty = _gp_model_has_no_data(model)

        x_unique, unique_indices = get_unique(x)
        n_unique_points = x_unique.size(0)

        if model_is_empty:
            new_unique_input_indices = list(range(n_unique_points))
            n_new_points = n_unique_points
            x_unique_new = x_unique
        else:
            current_X = model.train_inputs[0]
            current_Y = model.train_targets

            if x.size(1) != current_X.size(1):
                raise ValueError("The dimension of the model does not match")
            
            n_history_points = current_X.size(0)

            new_unique_input_indices = []
            already_computed_unique_input_indices = []
            already_computed_history_indices = []
            for unique_input_index in range(n_unique_points):
                input_point = x_unique[unique_input_index]

                found_matching_point = False
                for history_index in range(n_history_points):
                    history_point = current_X[history_index]
                    if torch.equal(input_point, history_point):
                        already_computed_unique_input_indices.append(unique_input_index)
                        already_computed_history_indices.append(history_index)
                        found_matching_point = True
                        break
                
                if not found_matching_point:
                    new_unique_input_indices.append(unique_input_index)

            n_new_points = len(new_unique_input_indices)

            if n_new_points != 0:
                x_unique_new = x_unique[new_unique_input_indices]
            else:
                the_dtype = current_Y.dtype
                the_device = current_Y.device

        if n_new_points != 0:
            # Compute new values
            posterior = model.posterior(
                x_unique_new, observation_noise=self.observation_noise)
            # shape (n_evals, 1)
            y_unique_new_with_output_dim = posterior.sample(torch.Size([]))        
            assert y_unique_new_with_output_dim.dim() == 2 and \
                y_unique_new_with_output_dim.size(1) == 1
            y_unique_new = y_unique_new_with_output_dim.squeeze(1)

            the_dtype = y_unique_new.dtype
            the_device = y_unique_new.device
        
            # Add new data to the GP model
            if model_is_empty:
                # Then x_unique == x_unique_new and y_unique == y_unique_new
                model.set_train_data_with_transforms(x_unique_new, y_unique_new_with_output_dim, strict=False, train=False)
            else:
                self.model = model.condition_on_observations_with_transforms(
                    x_unique_new, y_unique_new_with_output_dim)

        if n_new_points == n_unique_points:
            # None of the points that user asked for have been computed yet
            y_unique = y_unique_new
        else:
            # There are some points that are new and some points that have
            # already been computed.
            y_unique = torch.empty(x_unique.size(0), dtype=the_dtype, device=the_device)
            
            # Store the values that have already been computed
            for unique_input_index, history_index in zip(
                already_computed_unique_input_indices, already_computed_history_indices):
                y_unique[unique_input_index] = current_Y[history_index]
            
            # Store the new values
            if n_new_points != 0:
                for i, new_unique_index in enumerate(new_unique_input_indices):
                    y_unique[new_unique_index] = y_unique_new[i]

        # Build the array that will be returned (that includes potential duplicates)
        ret_Y = torch.empty(n_input_points, dtype=the_dtype, device=the_device)
        for unique_input_index, indices_of_unique_index in enumerate(unique_indices):
            for input_index in indices_of_unique_index:
                ret_Y[input_index] = y_unique[unique_input_index]
        
        if input_is_dim1:
            # Return a scalar if input shape was (dimension,) i.e. one point
            return ret_Y.squeeze()
        return ret_Y
