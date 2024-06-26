from utils import to_device, unsupported_improvements
from train_acquisition_function_net import myopic_policy_gradient_ei, mse_loss, get_average_normalized_entropy, max_one_hot
from acquisition_dataset import AcquisitionDataset
from tqdm import tqdm
import torch
from predict_EI_simple import calculate_EI_GP_padded_batch


def compute_stats(dataloader, compute_gp_stats=True, fit_map_gp=False,
                  verbose=True, desc=None):
    dataset = dataloader.dataset

    if not isinstance(dataset, AcquisitionDataset):
        raise ValueError("The dataloader must contain an AcquisitionDataset")
    
    has_stats = hasattr(dataset, "_cached_stats")
    if dataset.data_is_fixed:
        if has_stats:
            return dataset._cached_stats
    else:
        assert not has_stats
    
    has_models = dataset.has_models
    
    mse_always_predict_0 = 0.
    ei_ideal = 0.
    ei_random_search = 0.

    if has_models and compute_gp_stats:
        mse_true_gp = 0.
        ei_true_gp = 0.
        if fit_map_gp:
            mse_map_gp = 0.
            ei_map_gp = 0.

    it = dataloader
    if verbose:
        it = tqdm(dataloader, desc=desc)
    for batch in unsupported_improvements(it):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask = batch.tuple_no_model

        if has_models:
            models = batch.model
        
        with torch.no_grad():
            if has_models and compute_gp_stats:
                # Calculate true GP EI values
                ei_values_true_model = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models)

                # Calculate the MSE loss of the true GP model
                mse_true_gp += mse_loss(ei_values_true_model, improvements, cand_mask).item()

                # Calculate true GP actual E(I) of slecting the point with maximum EI
                probabilities_true_model = max_one_hot(ei_values_true_model, cand_mask)
                ei_true_gp += myopic_policy_gradient_ei(probabilities_true_model, improvements).item()

            # Calculate the E(I) of selecting a point at random
            if cand_mask is None:
                random_search_probabilities = torch.ones_like(improvements) / improvements.size(1)
            else:
                random_search_probabilities = cand_mask.double() / cand_mask.sum(dim=1, keepdim=True).double()
            ei_random_search += myopic_policy_gradient_ei(random_search_probabilities, improvements).item()
            
            # Calculate the MSE loss of always predicting 0
            mse_always_predict_0 += mse_loss(torch.zeros_like(improvements), improvements, cand_mask).item()
            
            # Calculate the E(I) of selecting the point with the maximum I (cheating)
            ideal_probabilities = max_one_hot(improvements, cand_mask)
            ei_ideal += myopic_policy_gradient_ei(ideal_probabilities, improvements).item()
        
        if has_models and fit_map_gp and compute_gp_stats:
            # Calculate the MAP GP EI values
            ei_values_map = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models, fit_params=True)

            # Calculate the MSE loss of the MAP GP model
            mse_map_gp += mse_loss(ei_values_map, improvements, cand_mask).item()
            
            # Calculate MAP GP actual E(I) of slecting the point with maximum EI
            probabilities_ei_map = max_one_hot(ei_values_map, cand_mask)
            ei_map_gp += myopic_policy_gradient_ei(probabilities_ei_map, improvements).item()

    n_batches = len(dataloader)
    
    mse_always_predict_0 /= n_batches
    ei_ideal /= n_batches
    ei_random_search /= n_batches

    ret = {
        "mse_always_predict_0": mse_always_predict_0,
        "ei_ideal": ei_ideal,
        "ei_random_search": ei_random_search
    }

    if has_models and compute_gp_stats:
        mse_true_gp /= n_batches
        ei_true_gp /= n_batches
        ret.update({
            "mse_true_gp": mse_true_gp,
            "ei_true_gp": ei_true_gp})

        if fit_map_gp:
            mse_map_gp /= n_batches
            ei_map_gp /= n_batches
            ret["mse_map_gp"] = mse_map_gp
            ret["ei_map_gp"] = ei_map_gp
    
    if dataloader.dataset.data_is_fixed:
        dataloader.dataset._cached_stats = ret
    
    return ret

