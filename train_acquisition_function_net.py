import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from generate_gp_data import GaussianProcessRandomDataset, TrainAcquisitionFunctionDataset
from utils import get_uniform_randint_generator, get_loguniform_randint_generator
from acquisition_function_net import AcquisitionFunctionNet
from predict_EI_simple import calculate_EI_GP_padded_batch


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
    test_ei_loss = 0.
    test_loss_true_gp = 0.
    always_predict_0_loss = 0.
    with torch.no_grad():
        for batch in dataloader:
            x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask, models = batch
            
            output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=True)
            test_loss += mse_loss_with_mask(output, improvements, cand_mask).item()
            always_predict_0_loss += mse_loss_with_mask(torch.zeros_like(output), improvements, cand_mask).item()

            # Calculate the EI values using the GP models
            ei_values = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models)
            test_ei_loss += mse_loss_with_mask(output, ei_values, cand_mask).item()
            test_loss_true_gp += mse_loss_with_mask(ei_values, improvements, cand_mask).item()

    n_batches = len(dataloader)
    test_loss /= n_batches
    test_ei_loss /= n_batches
    test_loss_true_gp /= n_batches
    always_predict_0_loss /= n_batches

    print(f"Test Error:\n NN improvement MSE (loss): {test_loss:>8f}\n True GP improvement MSE: {test_loss_true_gp:>8f}\n NN EI - GP EI MSE: {test_ei_loss:>8f}\n Always predict 0 MSE: {always_predict_0_loss:>8f}\n")


N_CANDIDATES = 1
MAX_HISTORY = 50
HISTORY_LOGUNIFORM = True
BATCH_SIZE = 256
N_BATCHES = 200
DIMENSION = 6

if HISTORY_LOGUNIFORM:
    n_datapoints_random_gen = get_loguniform_randint_generator(
        1, MAX_HISTORY, pre_offset=3.0, offset=N_CANDIDATES)
else:
    n_datapoints_random_gen = get_uniform_randint_generator(
        N_CANDIDATES+1, N_CANDIDATES+MAX_HISTORY)

dataset = GaussianProcessRandomDataset(
    dimension=DIMENSION, n_datapoints_random_gen=n_datapoints_random_gen,
    observation_noise=False, set_random_model_train_data=False,
    xvalue_distribution="normal", device=device,
    dataset_size=BATCH_SIZE * N_BATCHES)

aq_dataset = TrainAcquisitionFunctionDataset(
    dataset, n_candidate_points=N_CANDIDATES, n_samples="all",
    give_improvements=True)

train_aq_dataset, test_aq_dataset = aq_dataset.random_split([0.8, 0.2])

train_aq_dataloader = train_aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)
test_aq_dataloader = test_aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)

model = AcquisitionFunctionNet(DIMENSION).to(device)
print(model)


learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop_ei(train_aq_dataloader, model, optimizer)
    test_loop_ei(test_aq_dataloader, model)

