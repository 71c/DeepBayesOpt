import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from generate_gp_data import GaussianProcessRandomDataset, TrainAcquisitionFunctionDataset
from utils import get_uniform_randint_generator, get_loguniform_randint_generator
from acquisition_function_net import AcquisitionFunctionNet, LikelihoodFreeNetworkAcquisitionFunction
from predict_EI_simple import calculate_EI_GP_padded_batch, calculate_EI_GP
from botorch.acquisition.analytic import ExpectedImprovement

import os
from tqdm import tqdm


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



def mse_loss_with_mask(output, target, mask):
    """Calculate the MSE loss. Handle padding with mask for the case that
    there is padding. This works because the padded values are both zero
    so (0 - 0)^2 = 0. Equivalent to reduction="mean" if no padding.
    """
    if mask is None:
        return F.mse_loss(output, target, reduction="mean")
    return F.mse_loss(output, target, reduction="sum") / mask.sum()


def train_loop_ei(dataloader, model, optimizer, every_n_batches=10):
    model.train()

    n_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask, models = batch

        # print(type(models))
        # for gp_model in models:
        #     print(gp_model)
        #     for name, param in gp_model.named_parameters():
        #         print(name, param)
        #     print()
        # print()
        # exit()

        # shape (batch_size, n_cand)
        output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=True)

        loss = mse_loss_with_mask(output, improvements, cand_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % every_n_batches == 0:
            print(f"loss: {loss.item():>7f}  [{i+1:>4d}/{n_batches:>4d}]")


def test_loop_ei(dataloader, model):
    model.eval()

    test_loss = 0.
    test_loss_true_gp = 0.
    test_loss_gp_map = 0.
    always_predict_0_loss = 0.
    
    for batch in tqdm(dataloader):
        
        
        with torch.no_grad():
            x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask, models = batch

            # for gp_model in models:
            #     print(list(gp_model.named_parameters()))

            ei_values_nn = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=True)
            test_loss += mse_loss_with_mask(ei_values_nn, improvements, cand_mask).item()
            always_predict_0_loss += mse_loss_with_mask(torch.zeros_like(ei_values_nn), improvements, cand_mask).item()
            
            ei_values_true_model = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models)
            test_loss_true_gp += mse_loss_with_mask(ei_values_true_model, improvements, cand_mask).item()

        ei_values_map = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models, fit_params=True)
        test_loss_gp_map += mse_loss_with_mask(ei_values_map, improvements, cand_mask).item()


    n_batches = len(dataloader)
    test_loss /= n_batches
    test_loss_true_gp /= n_batches
    test_loss_gp_map /= n_batches
    always_predict_0_loss /= n_batches

    print(f"Test Error:\n NN improvement MSE (loss): {test_loss:>8f}\n True GP improvement MSE: {test_loss_true_gp:>8f}\n MAP GP improvement MSE: {test_loss_gp_map:>8f}\n Always predict 0 MSE: {always_predict_0_loss:>8f}\n")


N_CANDIDATES = 10
MAX_HISTORY = 50
HISTORY_LOGUNIFORM = True
BATCH_SIZE = 32 # 64
N_BATCHES = 100 # 300
DIMENSION = 6
RANDOMIZE_PARAMS = True
XVALUE_DISTRIBUTION = "normal"

TRAIN = True


if HISTORY_LOGUNIFORM:
    n_datapoints_random_gen = get_loguniform_randint_generator(
        1, MAX_HISTORY, pre_offset=3.0, offset=N_CANDIDATES)
else:
    n_datapoints_random_gen = get_uniform_randint_generator(
        N_CANDIDATES+1, N_CANDIDATES+MAX_HISTORY)

# print(n_datapoints_random_gen(30))

dataset = GaussianProcessRandomDataset(
    dimension=DIMENSION, n_datapoints_random_gen=n_datapoints_random_gen,
    observation_noise=False, set_random_model_train_data=False,
    xvalue_distribution=XVALUE_DISTRIBUTION, device=device,
    dataset_size=BATCH_SIZE * N_BATCHES,
    randomize_params=RANDOMIZE_PARAMS)

aq_dataset = TrainAcquisitionFunctionDataset(
    dataset, n_candidate_points=N_CANDIDATES, n_samples="all",
    give_improvements=True)

train_aq_dataset, test_aq_dataset = aq_dataset.random_split([0.95, 0.05])

train_aq_dataloader = train_aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)
test_aq_dataloader = test_aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)


script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = f"acquisition_function_net_{DIMENSION}d_{'random' if RANDOMIZE_PARAMS else 'fixed'}_kernel_{XVALUE_DISTRIBUTION}_x_upto_{MAX_HISTORY}_{'loguniform' if HISTORY_LOGUNIFORM else 'uniform'}_ei_{N_CANDIDATES}.pth"
model_path = os.path.join(script_dir, file_name)



model = AcquisitionFunctionNet(DIMENSION,
                            #    history_encoder_hidden_dims=[256, 512, 512], encoded_history_dim=1024, aq_func_hidden_dims=[512, 256, 64]
                               ).to(device)
print(model)
print("Number of trainable parameters:", count_trainable_parameters(model))
print("Number of parameters:", count_parameters(model))

if TRAIN:
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 10
    every_n_batches = 10

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop_ei(train_aq_dataloader, model, optimizer, every_n_batches)
        test_loop_ei(test_aq_dataloader, model)

    print("Done training!")

    # Save the model
    torch.save(model.state_dict(), model_path)
else:
    # Load the model
    model.load_state_dict(torch.load(model_path))
    model.eval()




test_aq_dataset_big = test_aq_dataset \
    .copy_with_new_size(len(aq_dataset))


# test_aq_dataloader_big = test_aq_dataset_big \
#     .get_dataloader(batch_size=BATCH_SIZE, drop_last=True)
# test_loop_ei(test_aq_dataloader_big, model)

print(model)

it = iter(test_aq_dataset_big)
x_hist, y_hist, x_cand, improvements, gp_model = next(it)
aq_fn = LikelihoodFreeNetworkAcquisitionFunction.from_net(
    model, x_hist, y_hist, exponentiate=True)

x_cand = torch.randn(100, DIMENSION)


ei_nn = aq_fn(x_cand.unsqueeze(1))


ei_true = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand)
ei_map = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, fit_params=True)




print(ei_true)
print(ei_nn)
print(ei_map)


import matplotlib.pyplot as plt

plt.scatter(ei_true.detach().numpy(), ei_nn.detach().numpy())
plt.xlabel('EI True')
plt.ylabel('EI NN')
plt.title('EI True vs EI NN')
plt.loglog()

plt.figure()
plt.scatter(ei_true.detach().numpy(), ei_map.detach().numpy())
plt.xlabel('EI True')
plt.ylabel('EI MAP')
plt.title('EI True vs EI MAP')
plt.loglog()

plt.show()

