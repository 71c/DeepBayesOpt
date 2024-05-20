import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from generate_gp_data import GaussianProcessRandomDataset, TrainAcquisitionFunctionDataset
from utils import get_uniform_randint_generator, get_loguniform_randint_generator, pad_tensor
from acquisition_function_net import AcquisitionFunctionNet


def check_model(dataloader, model):
    params_with_differing_grads = set()
    all_outputs_same = True

    for x_hist, y_hist, x_cand, y_cand, hist_mask, cand_mask, models in dataloader:
        output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask)

        dummy_loss = output.sum()
        dummy_loss.backward()
        grads_dict = {name: param.grad for name, param in model.named_parameters()}

        n_hist = x_hist.size(-2)
        n_cand = x_cand.size(-2)

        model.zero_grad()

        outputs = []
        for i in range(BATCH_SIZE):
            x_hist_i_padded = x_hist[i] # shape (n_hist, dimension)
            y_hist_i_padded = y_hist[i] # shape (n_hist,)
            x_cand_i_padded = x_cand[i] # shape (n_cand, dimension)
            hist_mask_i_padded = None if hist_mask is None else hist_mask[i] # shape (n_hist,)
            cand_mask_i_padded = None if cand_mask is None else cand_mask[i] # shape (n_cand,)

            n_hist_i = n_hist if hist_mask is None else hist_mask_i_padded.sum().item()
            n_cand_i = n_cand if cand_mask is None else cand_mask_i_padded.sum().item()

            x_hist_i = x_hist_i_padded[:n_hist_i]
            y_hist_i = y_hist_i_padded[:n_hist_i]
            x_cand_i = x_cand_i_padded[:n_cand_i]
            hist_mask_i = None if hist_mask_i_padded is None else hist_mask_i_padded[:n_hist_i]
            cand_mask_i = None if cand_mask_i_padded is None else cand_mask_i_padded[:n_cand_i]

            output_i = model(x_hist_i, y_hist_i, x_cand_i, hist_mask_i, cand_mask_i)
            output_i_padded = pad_tensor(output_i, n_cand, 0, add_mask=False)
            outputs.append(output_i_padded)
        output_unbatched = torch.stack(outputs)
        
        dummy_loss_unbatched = output_unbatched.sum()
        dummy_loss_unbatched.backward()
        grads_dict_unbatched = {
            name: param.grad for name, param in model.named_parameters()}

        output_same = torch.allclose(output, output_unbatched)
        all_outputs_same = all_outputs_same and output_same

        for name in grads_dict:
            if not torch.allclose(grads_dict[name], grads_dict_unbatched[name]):
                if name not in params_with_differing_grads:
                    print(f"Gradients for {name} are different.")
                    print(grads_dict[name])
                    print(grads_dict_unbatched[name])
                    params_with_differing_grads.add(name)
        
        model.zero_grad()
    
    return all_outputs_same, params_with_differing_grads


MAX_HISTORY = 30 # roughly the maximum history length
BATCH_SIZE = 5
N_BATCHES = 10
DIMENSION = 6

history_length_gen = get_loguniform_randint_generator(
    2, MAX_HISTORY, pre_offset=3.0, offset=0)

def n_datapoints_random_gen(size=1):
    return 2 * history_length_gen(size)

dataset = GaussianProcessRandomDataset(
    dimension=DIMENSION, n_datapoints_random_gen=n_datapoints_random_gen,
    observation_noise=False, set_random_model_train_data=False,
    xvalue_distribution="normal",
    device=device,
    dataset_size=BATCH_SIZE * N_BATCHES)

aq_dataset = TrainAcquisitionFunctionDataset(
    dataset, n_candidate_points="binomial", n_samples="all",
    give_improvements=False, min_n_candidates=2)

dataloader = aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)


model = AcquisitionFunctionNet(DIMENSION).to(device)
print(model)


all_outputs_same, params_with_differing_grads = check_model(dataloader, model)

if all_outputs_same:
    print("All outputs are the same.")
else:
    print("Not all outputs are the same.")

if len(params_with_differing_grads) == 0:
    print("All gradients are the same.")
else:
    print("Not all gradients are the same. Those that differ are:")
    print(params_with_differing_grads)

