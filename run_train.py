import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from generate_gp_data import GaussianProcessRandomDataset, TrainAcquisitionFunctionDataset
from utils import get_uniform_randint_generator, get_loguniform_randint_generator
from acquisition_function_net import AcquisitionFunctionNetV1, LikelihoodFreeNetworkAcquisitionFunction
from predict_EI_simple import calculate_EI_GP
import numpy as np
import matplotlib.pyplot as plt
import os

from train_acquisition_function_net import train_loop, test_loop, count_trainable_parameters, count_parameters


# Number of candidate points. More relevant for the policy gradient case.
# Doesn't matter that much for the MSE EI case; for MSE, could just set to 1.
N_CANDIDATES = 16

MIN_HISTORY = 1
MAX_HISTORY = 8
HISTORY_LOGUNIFORM = True

BATCH_SIZE = 8 # 64
N_BATCHES = 10 # 100
EVERY_N_BATCHES = 5
EPOCHS = 5

# dimension of the optimization problem
DIMENSION = 1
# whether to randomize the GP parameters for training data
RANDOMIZE_PARAMS = False
# choose either "uniform" or "normal" (or a custom distribution)
XVALUE_DISTRIBUTION = "uniform"

# True for the softmax thing, False for MSE
POLICY_GRADIENT = True
# Only used if POLICY_GRADIENT is True
INCLUDE_ALPHA = True
# Following 3 are only used if both POLICY_GRADIENT and INCLUDE_ALPHA are True
LEARN_ALPHA = True
INITIAL_ALPHA = 1.0
ALPHA_INCREMENT = 0.01

# Whether to fit maximum a posteriori GP for testing
FIT_MAP_GP = False

# Whether to train the model. If False, will load a saved model.
TRAIN = True
# Whether to load a saved model to train
LOAD_SAVED_MODEL_TO_TRAIN = False

# Initialize the acquisition function network
model = AcquisitionFunctionNetV1(DIMENSION,
                                 pooling="max",
                                 history_enc_hidden_dims=[32, 32],
                                 encoded_history_dim=32,
                                 aq_func_hidden_dims=[32, 32],
                                 include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
                                 learn_alpha=LEARN_ALPHA,
                                 initial_alpha=INITIAL_ALPHA).to(device)


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

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_aq_dataloader, model, optimizer,
                   every_n_batches=EVERY_N_BATCHES,
                   policy_gradient=POLICY_GRADIENT,
                   alpha_increment=ALPHA_INCREMENT)
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
    model, x_hist, y_hist, exponentiate=not POLICY_GRADIENT, softmax=False)
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

