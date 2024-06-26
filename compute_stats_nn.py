from utils import to_device, unsupported_improvements
from train_acquisition_function_net import myopic_policy_gradient_ei, mse_loss, get_average_normalized_entropy, max_one_hot
from acquisition_dataset import AcquisitionDataset
from tqdm import tqdm
import torch


def compute_stats_nn(dataloader, model, policy_gradient=False,
                     verbose=True, desc=None,
                     nn_device=None):
    if not isinstance(dataloader.dataset, AcquisitionDataset):
        raise ValueError("The dataloader must contain an AcquisitionDataset")
    
    model.eval()

    test_loss = 0.
    test_ei_max = 0.
    if policy_gradient:
        avg_normalized_entropy = 0.
    
    it = dataloader
    if verbose:
        it = tqdm(it, desc=desc)
    for batch in unsupported_improvements(it):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask = batch.tuple_no_model
        
        # # For testing
        # test_ei_true_gp += 1. # to avoid division by zero
        # continue
        
        with torch.no_grad():
            x_hist_nn = to_device(x_hist, nn_device)
            y_hist_nn = to_device(y_hist, nn_device)
            x_cand_nn = to_device(x_cand, nn_device)
            improvements_nn = to_device(improvements, nn_device)
            hist_mask_nn = to_device(hist_mask, nn_device)
            cand_mask_nn = to_device(cand_mask, nn_device)

            if policy_gradient:
                # Calculate the softmax EI of the NN
                probabilities_nn = model(x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, cand_mask_nn, exponentiate=False, softmax=True)
                avg_normalized_entropy += get_average_normalized_entropy(probabilities_nn, mask=cand_mask).item()
                test_loss += myopic_policy_gradient_ei(probabilities_nn, improvements_nn).item()

                probabilities_nn_max = max_one_hot(probabilities_nn, cand_mask)
            else:
                # Calculate the MSE of the NN
                ei_values_nn = model(x_hist_nn, y_hist_nn, x_cand_nn, hist_mask_nn, cand_mask_nn, exponentiate=True)
                test_loss += mse_loss(ei_values_nn, improvements_nn, cand_mask_nn).item()

                probabilities_nn_max = max_one_hot(ei_values_nn, cand_mask_nn)
            
            test_ei_max += myopic_policy_gradient_ei(probabilities_nn_max, improvements_nn).item()

    n_batches = len(dataloader)
    
    ret = {}

    if policy_gradient:
        test_loss /= n_batches
        ret["ei_softmax"] = test_loss
        
        avg_normalized_entropy /= n_batches
        ret["avg_normalized_entropy"] = avg_normalized_entropy
    else:
        test_loss /= n_batches
        ret["mse"] = test_loss

    test_ei_max /= n_batches
    ret["ei_max"] = test_ei_max

    return ret
