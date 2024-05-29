import torch
import torch.nn.functional as F
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from generate_gp_data import GaussianProcessRandomDataset, TrainAcquisitionFunctionDataset
from utils import get_uniform_randint_generator, get_loguniform_randint_generator
from acquisition_function_net import AcquisitionFunctionNet, LikelihoodFreeNetworkAcquisitionFunction
from predict_EI_simple import calculate_EI_GP_padded_batch, calculate_EI_GP
from botorch.acquisition.analytic import ExpectedImprovement
import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm import tqdm


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def max_one_hot(values, mask=None):
    if mask is not None:
        neg_inf = torch.zeros_like(values)
        neg_inf[~mask] = float("-inf")
        values = values + neg_inf
    return F.one_hot(torch.argmax(values, dim=1),
                     num_classes=values.size(1)).double()


def get_average_normalized_entropy(probabilities, mask=None):
    """
    Calculates the average normalized entropy of a probability distribution,
    where a uniform distribution would give a value of 1.

    Args:
        probabilities (torch.Tensor):
            A tensor representing the probability distribution.
            Entries corresponding to masked out values are assumed to be zero.
        mask (torch.Tensor, optional):
            A tensor representing a mask to apply on the probabilities. 
            Defaults to None, meaning no mask is applied.

    Returns:
        torch.Tensor: The average normalized entropy.
    """
    entropy = Categorical(probs=probabilities).entropy()
    if mask is None:
        counts = torch.tensor(probabilities.size(1), dtype=torch.double)
    else:
        counts = mask.sum(dim=1).double()
    return (entropy / torch.log(counts)).mean()


def myopic_policy_gradient_loss(probabilities, improvements):
    """Calculate the policy gradient loss.

    Args:
        probabilities (Tensor): The output tensor from the model, assumed to be
            softmaxed. Shape (batch_size, n_cand)
       improvements (Tensor): The improvements tensor.
            Shape (batch_size, n_cand)
        Both tensors are assumed to be padded with zeros.
        Note: A mask is not needed because the padded values are zero and the
        computation works out even if there is a mask.
    """
    expected_improvements_per_batch = torch.sum(probabilities * improvements, dim=1)
    return -expected_improvements_per_batch.mean()


def mse_loss(pred_improvements, improvements, mask):
    """Calculate the MSE loss. Handle padding with mask for the case that
    there is padding. This works because the padded values are both zero
    so (0 - 0)^2 = 0. Equivalent to reduction="mean" if no padding.

    Args:
        pred_improvements (Tensor): The output tensor from the model,
            assumed to be exponentiated. Shape (batch_size, n_cand)
        improvements (Tensor): The improvements tensor.
            Shape (batch_size, n_cand)
        mask (Tensor): The mask tensor, shape (batch_size, n_cand)
    """
    if mask is None:
        return F.mse_loss(pred_improvements, improvements, reduction="mean")
    return F.mse_loss(pred_improvements, improvements, reduction="sum") / mask.sum()


def train_loop(dataloader, model, optimizer, every_n_batches=10, policy_gradient=False):
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

        if policy_gradient:
            output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=False, softmax=True)
            loss = myopic_policy_gradient_loss(output, improvements)
        else:
            output = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=True, softmax=False)
            loss = mse_loss(output, improvements, cand_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model.includes_alpha:
            d_alpha = 0.01
            d_log_alpha = d_alpha / model.alpha.data
            model.log_alpha.data = model.log_alpha.data + d_log_alpha

        if i % every_n_batches == 0:
            prefix = "Expected 1-step improvement" if policy_gradient else "MSE"
            
            suffix = ""
            if policy_gradient:
                avg_normalized_entropy = get_average_normalized_entropy(output, mask=cand_mask).item()
                suffix += f", avg normalized entropy={avg_normalized_entropy:>7f}"
            if model.includes_alpha:
                suffix += f", alpha={model.alpha:>7f}"
            
            loss_value = -loss.item() if policy_gradient else loss.item()
            print(f"{prefix}: {loss_value:>7f}{suffix}  [{i+1:>4d}/{n_batches:>4d}]")


def test_loop(dataloader, model, policy_gradient=False, fit_map_gp=False):
    model.eval()

    test_loss = 0.
    test_loss_true_gp = 0.
    if fit_map_gp:
        test_loss_gp_map = 0.
    always_predict_0_loss = 0.
    if policy_gradient:
        ideal_loss = 0.
        avg_normalized_entropy_nn = 0.
    
    for batch in tqdm(dataloader):
        x_hist, y_hist, x_cand, improvements, hist_mask, cand_mask, models = batch
        
        with torch.no_grad():
            ei_values_true_model = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models)

            if policy_gradient:
                probabilities_nn = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=False, softmax=True)
                avg_normalized_entropy_nn += get_average_normalized_entropy(probabilities_nn, mask=cand_mask).item()
                # print(probabilities_nn)
                test_loss += myopic_policy_gradient_loss(probabilities_nn, improvements).item()

                if cand_mask is None:
                    always_predict_0_probabilities = torch.ones_like(probabilities_nn) / probabilities_nn.size(1)
                else:
                    always_predict_0_probabilities = cand_mask.double() / cand_mask.sum(dim=1, keepdim=True).double()
                # print(always_predict_0_probabilities)
                always_predict_0_loss += myopic_policy_gradient_loss(always_predict_0_probabilities, improvements).item()

                probabilities_true_model = max_one_hot(ei_values_true_model, cand_mask)
                test_loss_true_gp += myopic_policy_gradient_loss(probabilities_true_model, improvements).item()

                ideal_probabilities = max_one_hot(improvements, cand_mask)
                ideal_loss += myopic_policy_gradient_loss(ideal_probabilities, improvements).item()
            else:
                ei_values_nn = model(x_hist, y_hist, x_cand, hist_mask, cand_mask, exponentiate=True)
                test_loss += mse_loss(ei_values_nn, improvements, cand_mask).item()

                always_predict_0_loss += mse_loss(torch.zeros_like(ei_values_nn), improvements, cand_mask).item()

                test_loss_true_gp += mse_loss(ei_values_true_model, improvements, cand_mask).item()

        if fit_map_gp:
            ei_values_map = calculate_EI_GP_padded_batch(x_hist, y_hist, x_cand, hist_mask, cand_mask, models, fit_params=True)
            
            if policy_gradient:
                probabilities_ei_map = max_one_hot(ei_values_map, cand_mask)
                # print(ei_values_map)
                # print(probabilities_ei_map)
                test_loss_gp_map += myopic_policy_gradient_loss(probabilities_ei_map, improvements).item()
            else:
                test_loss_gp_map += mse_loss(ei_values_map, improvements, cand_mask).item()


    n_batches = len(dataloader)
    multiplier = -1 if policy_gradient else 1
    
    test_loss /= multiplier * n_batches
    test_loss_true_gp /= multiplier * n_batches
    if fit_map_gp:
        test_loss_gp_map /= multiplier * n_batches
    always_predict_0_loss /= multiplier * n_batches
    if policy_gradient:
        ideal_loss /= multiplier * n_batches
        avg_normalized_entropy_nn /= n_batches

    mse_desc = "Expected 1-step improvement" if policy_gradient else"Improvement MSE"
    map_str = f" MAP GP: {test_loss_gp_map:>8f}\n" if fit_map_gp else ""
    naive_desc = "Random search" if policy_gradient else "Always predict 0"
    eval_str = f"Test {mse_desc}:\n NN (loss): {test_loss:>8f}\n True GP: {test_loss_true_gp:>8f}\n{map_str} {naive_desc}: {always_predict_0_loss:>8f}\n"
    if policy_gradient:
        eval_str += f" Ideal: {ideal_loss:>8f}\n NN avg normalized entropy: {avg_normalized_entropy_nn:>8f}\n"
    print(eval_str)


# Number of candidate points. More relevant for the policy gradient case.
# Doesn't matter that much for the MSE EI case; for MSE, could just set to 1.
N_CANDIDATES = 16

MIN_HISTORY = 1
MAX_HISTORY = 8
HISTORY_LOGUNIFORM = True

BATCH_SIZE = 64
N_BATCHES = 200
EPOCHS = 5

DIMENSION = 1 # dimension of the optimization problem
RANDOMIZE_PARAMS = False # whether to randomize the GP parameters for training data
XVALUE_DISTRIBUTION = "uniform" # choose either "uniform" or "normal" (or a custom distribution)
POLICY_GRADIENT = False # True for the softmax thing, False for MSE
INCLUDES_ALPHA = True # Only relevant if POLICY_GRADIENT is True
FIT_MAP_GP = False

TRAIN = True
LOAD_SAVED_MODEL_TO_TRAIN = False


if HISTORY_LOGUNIFORM:
    n_datapoints_random_gen = get_loguniform_randint_generator(
        MIN_HISTORY, MAX_HISTORY, pre_offset=3.0, offset=N_CANDIDATES)
else:
    n_datapoints_random_gen = get_uniform_randint_generator(
        N_CANDIDATES+MIN_HISTORY, N_CANDIDATES+MAX_HISTORY)

sample_n_points = n_datapoints_random_gen(30)
print("Examples of history lengths:", sample_n_points - N_CANDIDATES)

dataset = GaussianProcessRandomDataset(
    dimension=DIMENSION, n_datapoints_random_gen=n_datapoints_random_gen,
    observation_noise=False, set_random_model_train_data=False,
    xvalue_distribution=XVALUE_DISTRIBUTION, device=device,
    dataset_size=BATCH_SIZE * N_BATCHES,
    randomize_params=RANDOMIZE_PARAMS)

aq_dataset = TrainAcquisitionFunctionDataset(
    dataset, n_candidate_points=N_CANDIDATES, n_samples="all",
    give_improvements=True)

train_aq_dataset, test_aq_dataset = aq_dataset.random_split([0.9, 0.1])

train_aq_dataloader = train_aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)
test_aq_dataloader = test_aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)


script_dir = os.path.dirname(os.path.abspath(__file__))
# training_info = f"batchsize{BATCH_SIZE}_batches_per_epoch{N_BATCHES}_epochs{EPOCHS}"
file_name = f"acquisition_function_net_{DIMENSION}d_{'random' if RANDOMIZE_PARAMS else 'fixed'}_kernel_{XVALUE_DISTRIBUTION}_x_history{MIN_HISTORY}-{MAX_HISTORY}_{'loguniform' if HISTORY_LOGUNIFORM else 'uniform'}_{'policy_gradient_myopic' if POLICY_GRADIENT else 'ei'}_{N_CANDIDATES}cand.pth"
print(f"Model file: {file_name}")
model_path = os.path.join(script_dir, file_name)

model = AcquisitionFunctionNet(DIMENSION,
                               learn_alpha=INCLUDES_ALPHA and POLICY_GRADIENT,
                               history_enc_hidden_dims=[32, 32], encoded_history_dim=32, aq_func_hidden_dims=[32, 32]
                               ).to(device)
print(model)
print("Number of trainable parameters:", count_trainable_parameters(model))
print("Number of parameters:", count_parameters(model))


if TRAIN:
    if LOAD_SAVED_MODEL_TO_TRAIN:
        # Load the model
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    learning_rate = 1e-4
    # could also try RMSProp
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    every_n_batches = 10

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_aq_dataloader, model, optimizer, every_n_batches, policy_gradient=POLICY_GRADIENT)
        test_loop(test_aq_dataloader, model,
                  policy_gradient=POLICY_GRADIENT, fit_map_gp=FIT_MAP_GP)

    print("Done training!")

    # Save the model
    torch.save(model.state_dict(), model_path)
else:
    # Load the model
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

print(model)


test_aq_dataset_big = test_aq_dataset \
    .copy_with_new_size(len(aq_dataset))


# test_aq_dataloader_big = test_aq_dataset_big \
#     .get_dataloader(batch_size=BATCH_SIZE, drop_last=True)
# test_loop(test_aq_dataloader_big, model,
#           policy_gradient=POLICY_GRADIENT,
#           fit_map_gp=FIT_MAP_GP)


n_candidates = 100

it = iter(test_aq_dataset_big)
x_hist, y_hist, x_cand, improvements, gp_model = next(it)

# for x_hist, y_hist, x_cand, improvements, gp_model in test_aq_dataset_big:
#     if x_hist.size(0) != 2:
#         continue
#     if x_hist[0, 0].item() > 0.05 or x_hist[1, 0].item() < 0.95:
#         continue
#     break


print(f"Number of history points: {x_hist.size(0)}")
x_cand = torch.rand(n_candidates, DIMENSION)


aq_fn = LikelihoodFreeNetworkAcquisitionFunction.from_net(
    model, x_hist, y_hist, exponentiate=not POLICY_GRADIENT)
ei_nn = aq_fn(x_cand.unsqueeze(1))


if DIMENSION == 1:
    gp_model.set_train_data(x_hist, y_hist.squeeze(-1), strict=False)
    posterior_true = gp_model.posterior(x_cand, observation_noise=False)

ei_true = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, log=False)
ei_map = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, fit_params=True, log=False)

# Normalize so they have the same scale
if POLICY_GRADIENT:
    ei_nn = (ei_nn - ei_nn.mean()) / ei_nn.std()
    ei_true = (ei_true - ei_true.mean()) / ei_true.std()
    ei_map = (ei_map - ei_map.mean()) / ei_map.std()

name = "acquisition" if POLICY_GRADIENT else "EI"

print(f"{name} True:")
print(ei_true)
print(f"{name} NN:")
print(ei_nn)
print(f"{name} MAP:")
print(ei_map)


plt.scatter(ei_true.detach().numpy(), ei_nn.detach().numpy())
plt.xlabel(f'{name} True')
plt.ylabel(f'{name} NN')
plt.title(f'{name} True vs {name} NN')

plt.figure()
plt.scatter(ei_true.detach().numpy(), ei_map.detach().numpy())
plt.xlabel(f'{name} True')
plt.ylabel(f'{name} MAP')
plt.title(f'{name} True vs {name} MAP')



def plot_gp_posterior(ax, posterior, test_x, train_x, train_y, color, name=None):
    lower, upper = posterior.mvn.confidence_region()
    mean = posterior.mean.detach().squeeze().cpu().numpy()
    lower = lower.detach().squeeze().cpu().numpy()
    upper = upper.detach().squeeze().cpu().numpy()

    train_x = train_x.detach().squeeze().cpu().numpy()
    train_y = train_y.detach().squeeze().cpu().numpy()
    test_x = test_x.detach().squeeze().cpu().numpy()
    
    sorted_indices = np.argsort(test_x)
    test_x = test_x[sorted_indices]
    mean = mean[sorted_indices]
    lower = lower[sorted_indices]
    upper = upper[sorted_indices]

    extension = '' if name is None else f' {name}'

    # Plot training points as black stars
    ax.plot(train_x, train_y, f'{color}*', label=f'Observed Data{extension}')
    # Plot posterior means as blue line
    ax.plot(test_x, mean, color, label=f'Mean{extension}')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x, lower, upper, color=color, alpha=0.5, label=f'Confidence{extension}')


if DIMENSION == 1:
    fig, ax = plt.subplots()

    sorted_indices = np.argsort(x_cand.detach().numpy().flatten())
    sorted_x_cand = x_cand.detach().numpy().flatten()[sorted_indices]
    sorted_ei_true = ei_true.detach().numpy().flatten()[sorted_indices]
    sorted_ei_nn = ei_nn.detach().numpy().flatten()[sorted_indices]
    sorted_ei_map = ei_map.detach().numpy().flatten()[sorted_indices]

    plt.plot(sorted_x_cand, sorted_ei_true, label="True")
    plt.plot(sorted_x_cand, sorted_ei_nn, label="NN")
    plt.plot(sorted_x_cand, sorted_ei_map, label="MAP")

    plot_gp_posterior(ax, posterior_true, x_cand, x_hist, y_hist, 'b', name='True')

    plt.xlabel("x")
    plt.ylabel(f'{name}')
    plt.title(f'{name} vs x')
    plt.legend()

plt.show()

