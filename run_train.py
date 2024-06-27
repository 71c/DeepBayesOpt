import math
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
from torch import nn

from gp_acquisition_dataset import create_gp_acquisition_dataset
from acquisition_function_net import (
    AcquisitionFunctionNetV1and2, AcquisitionFunctionNetV3, AcquisitionFunctionNetV4,
    AcquisitionFunctionNetDense, LikelihoodFreeNetworkAcquisitionFunction)
from predict_EI_simple import calculate_EI_GP
import numpy as np
import matplotlib.pyplot as plt
import os

from train_acquisition_function_net import (
    train_acquisition_function_net, count_trainable_parameters, count_parameters)

import torch.distributions as dist

import cProfile


##################### Settings for this script #################################
# Whether to train the model. If False, will load a saved model.
TRAIN = True
# Whether to load a saved model to train
LOAD_SAVED_MODEL_TO_TRAIN = False
# Whether to fit maximum a posteriori GP for testing
FIT_MAP_GP = False


############################ FIXED DATASET SETTINGS ############################
########## Set number of history and candidate points generation ###############
# This means whether n history points or whether the total number of points
# is log-uniform
LOGUNIFORM = True
PRE_OFFSET = 3.0

# Whether to fix the number of candidate points (as opposed to randomized)
FIX_N_CANDIDATES = True

# If FIX_N_CANDIDATES is True, then the following are used:
TEST_N_CANDIDATES = 50
MIN_HISTORY = 1
MAX_HISTORY = 8

# If FIX_N_CANDIDATES is False, then the following are used:
MIN_N_CANDIDATES = 2
MAX_POINTS = 30

###################### GP realization characteristics ##########################
# Dimension of the optimization problem
DIMENSION = 1
# whether to randomize the GP parameters for training data
RANDOMIZE_PARAMS = False
# choose either "uniform" or "normal" (or a custom distribution)
XVALUE_DISTRIBUTION = "uniform"

####################### Other fixed dataset settings ###########################
## How many times bigger the big test dataset is than the train dataset, > 0
# TEST_FACTOR = 3.0
TEST_FACTOR = 0.2
## The proportion of the test dataset that is used for evaluating the model after
## each epoch, between 0 and 1
# SMALL_TEST_PROPORTION_OF_TEST = 0.04
SMALL_TEST_PROPORTION_OF_TEST = 1.0

# The following two should be kept as they are -- ALWAYS want to fix the test.
# As long as the acqisition dataset is fixed, then whether the function samples
# dataset is fixed doesn't matter.
FIX_TEST_SAMPLES_DATASET = False
FIX_TEST_ACQUISITION_DATASET = True
# The following two are not important.
LAZY_TRAIN = False
LAZY_TEST = False


################## Settings for dataset size and generation ####################
# The size of the training acquisition dataset
TRAIN_ACQUISITION_SIZE = 200_000
# The amount that the dataset is expanded to save compute of GP realizations
EXPANSION_FACTOR = 4
# Whether and how to fix the training dataset
FIX_TRAIN_SAMPLES_DATASET = False
FIX_TRAIN_ACQUISITION_DATASET = True

# Number of candidate points for training. For MSE EI, could just set to 1.
# Only used if FIX_N_CANDIDATES is True.
TRAIN_N_CANDIDATES = 50
############################# Settings for training ############################
POLICY_GRADIENT = True # True for the softmax thing, False for MSE EI
BATCH_SIZE = 128
LEARNING_RATE = 3e-5
EPOCHS = 3

# Only used if POLICY_GRADIENT is True
INCLUDE_ALPHA = True
# Following 3 are only used if both POLICY_GRADIENT and INCLUDE_ALPHA are True
LEARN_ALPHA = True
INITIAL_ALPHA = 1.0
ALPHA_INCREMENT = None # equivalent to 0.0
################################################################################

######################### Neural network architecture ##########################

# model = AcquisitionFunctionNetV4(DIMENSION,
#                                  history_enc_hidden_dims=[32, 32], pooling="max",
#                  include_local_features=True,
#                  encoded_history_dim=4, include_mean=False,
#                  mean_enc_hidden_dims=[32, 32], mean_dim=1,
#                  std_enc_hidden_dims=[32, 32], std_dim=32,
#                  aq_func_hidden_dims=[32, 32], layer_norm=True,
#                  layer_norm_at_end_mlp=False,
#                  include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
#                                  learn_alpha=LEARN_ALPHA,
#                                  initial_alpha=INITIAL_ALPHA).to(device)

# model = AcquisitionFunctionNetV3(DIMENSION,
#                                  history_enc_hidden_dims=[32, 32], pooling="max",
#                  encoded_history_dim=32,
#                  mean_enc_hidden_dims=[32, 32], mean_dim=1,
#                  std_enc_hidden_dims=[32, 32], std_dim=16,
#                  aq_func_hidden_dims=[32, 32], layer_norm=False,
#                  layer_norm_at_end_mlp=False, include_y=True,
#                  include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
#                                  learn_alpha=LEARN_ALPHA,
#                                  initial_alpha=INITIAL_ALPHA).to(device)

model = AcquisitionFunctionNetV1and2(DIMENSION,
                                 pooling="max",
                                 history_enc_hidden_dims=[32, 32],
                                 encoded_history_dim=32,
                                 aq_func_hidden_dims=[32, 32],
                                 input_xcand_to_local_nn=False,
                                 input_xcand_to_final_mlp=True,
                                 include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
                                 learn_alpha=LEARN_ALPHA,
                                 initial_alpha=INITIAL_ALPHA,
                                 activation_at_end_pointnet=True,
                                 layer_norm_pointnet=False,
                                 layer_norm_before_end_mlp=False,
                                 layer_norm_at_end_mlp=False,
                                 include_best_y=False,
                                 activation_pointnet=nn.ReLU,
                                 activation_mlp=nn.ReLU).to(device)


# model = AcquisitionFunctionNetDense(DIMENSION, MAX_HISTORY,
#                                     hidden_dims=[128, 128, 64, 32],
#                                     include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
#                                     learn_alpha=LEARN_ALPHA,
#                                     initial_alpha=INITIAL_ALPHA).to(device)
################################################################################


####################### Make the train and test datasets #######################
TRAIN_SAMPLES_SIZE = math.ceil(TRAIN_ACQUISITION_SIZE / EXPANSION_FACTOR)

DATASET_SIZE = math.ceil(TRAIN_SAMPLES_SIZE * (1 + TEST_FACTOR))

#### Calculate test size
TEST_SAMPLES_SIZE = DATASET_SIZE - TRAIN_SAMPLES_SIZE
## Could alternatively calculate test size like this if going by
## proportions of an original dataset:
# TEST_PROPORTION = TEST_FACTOR / (1 + TEST_FACTOR)
# TRAIN_PROPORTION = 1 - TEST_PROPORTION
# TRAIN_SAMPLES_SIZE, TEST_SAMPLES_SIZE = get_lengths_from_proportions(DATASET_SIZE, [TRAIN_PROPORTION, TEST_PROPORTION])

print(f"Small test proportion of test: {SMALL_TEST_PROPORTION_OF_TEST:.4f}")
# SMALL_TEST_PROPORTION_OF_TEST = 1 / ((1 / SMALL_TEST_PROPORTION_OF_TRAIN_AND_SMALL_TEST - 1) * TEST_FACTOR)
SMALL_TEST_PROPORTION_OF_TRAIN_AND_SMALL_TEST = 1 / (1 + 1 / (SMALL_TEST_PROPORTION_OF_TEST * TEST_FACTOR))
print(f"Small test proportion of train + small test: {SMALL_TEST_PROPORTION_OF_TRAIN_AND_SMALL_TEST:.4f}")

# Generating the random GP realizations is faster on CPU than on GPU.
# This is likely because the random GP realizations are generated one-by-one
# rather than in batches since the number of points is random so it's difficult
# to batch this. Hence we set device="cpu".
# Also, making the padded batches (the creation of zeros, concatenating, and
# stacking) on CPU rather than on GPU is much faster.
common_kwargs = dict(dimension=DIMENSION, randomize_params=RANDOMIZE_PARAMS,
    device="cpu", observation_noise=False,
    xvalue_distribution=XVALUE_DISTRIBUTION, expansion_factor=EXPANSION_FACTOR,
    loguniform=LOGUNIFORM, pre_offset=PRE_OFFSET if LOGUNIFORM else None)

if FIX_N_CANDIDATES:
    train_n_points_kwargs = dict(min_history=MIN_HISTORY, max_history=MAX_HISTORY,
                                 n_candidates=TRAIN_N_CANDIDATES)
    test_n_points_kwargs = dict(min_history=MIN_HISTORY, max_history=MAX_HISTORY,
                                n_candidates=TEST_N_CANDIDATES)
else:
    train_n_points_kwargs = dict(min_n_candidates=MIN_N_CANDIDATES, max_points=MAX_POINTS)
    test_n_points_kwargs = train_n_points_kwargs

train_dataset, train_aq_dataset = create_gp_acquisition_dataset(
    TRAIN_SAMPLES_SIZE, **common_kwargs, **train_n_points_kwargs,
    fix_gp_samples=FIX_TRAIN_SAMPLES_DATASET,
    fix_acquisition_samples=FIX_TRAIN_ACQUISITION_DATASET, lazy=LAZY_TRAIN)
test_dataset, test_aq_dataset = create_gp_acquisition_dataset(
    TEST_SAMPLES_SIZE, **common_kwargs, **test_n_points_kwargs,
    fix_gp_samples=FIX_TEST_SAMPLES_DATASET,
    fix_acquisition_samples=FIX_TEST_ACQUISITION_DATASET, lazy=LAZY_TEST)

small_test_aq_dataset, _ = test_aq_dataset.random_split(
    [SMALL_TEST_PROPORTION_OF_TEST, 1 - SMALL_TEST_PROPORTION_OF_TEST])

# print("Train acquisition dataset:")
# print(train_aq_dataset)
# print("\nTest acquisition dataset:")
# print(test_aq_dataset)
# print("\nSmall test acquisition dataset:")
# print(small_test_aq_dataset)
# print("\n")

print(model)
print("Number of trainable parameters:", count_trainable_parameters(model))
print("Number of parameters:", count_parameters(model))


########################## Get model path ######################################

script_dir = os.path.dirname(os.path.abspath(__file__))
model_class_name = model.__class__.__name__
loss_str = 'policy_gradient_myopic' if POLICY_GRADIENT else 'ei'
file_name = f"acquisition_function_net_{model_class_name}_{DIMENSION}d_{loss_str}_{'random' if RANDOMIZE_PARAMS else 'fixed'}_kernel_{XVALUE_DISTRIBUTION}_x"
if FIX_N_CANDIDATES:
    file_name += f"_history{MIN_HISTORY}-{MAX_HISTORY}_{'loguniform' if LOGUNIFORM else 'uniform'}_{TRAIN_N_CANDIDATES}cand.pth"
else:
    min_points = MIN_N_CANDIDATES + 1
    file_name += f"_points{min_points}-{MAX_POINTS}_{'loguniform' if LOGUNIFORM else 'uniform'}.pth"

# file_name = "acquisition_function_net_AcquisitionFunctionNetV4_1d_policy_gradient_myopic_fixed_kernel_uniform_x_history1-8_loguniform_50cand_200epochs.pth"

print(f"Model file: {file_name}")
model_path = os.path.join(script_dir, file_name)


######################## Train the model #######################################

print("Training function samples dataset size:", len(train_dataset))
print("Original training acquisition dataset size parameter:", TRAIN_ACQUISITION_SIZE)
print("Training acquisition dataset size:", len(train_aq_dataset),
      "number of batches:", len(train_aq_dataset) // BATCH_SIZE, len(train_aq_dataset) % BATCH_SIZE)

print("Test function samples dataset size:", len(test_dataset))
print("Test acquisition dataset size:", len(test_aq_dataset),
      "number of batches:", len(test_aq_dataset) // BATCH_SIZE, len(test_aq_dataset) % BATCH_SIZE)
if small_test_aq_dataset != test_aq_dataset:
    print("Small test acquisition dataset size:", len(small_test_aq_dataset),
            "number of batches:", len(small_test_aq_dataset) // BATCH_SIZE, len(small_test_aq_dataset) % BATCH_SIZE)


import json

import cProfile, pstats, io
from pstats import SortKey



if TRAIN:    
    if LOAD_SAVED_MODEL_TO_TRAIN:
        # Load the model
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    pr = cProfile.Profile()
    pr.enable()

    data = train_acquisition_function_net(
        model, train_aq_dataset, optimizer, POLICY_GRADIENT, EPOCHS, BATCH_SIZE,
        device, ALPHA_INCREMENT, verbose=True, n_train_printouts_per_epoch=10,
        test_dataset=test_aq_dataset, small_test_dataset=small_test_aq_dataset,
        get_train_stats_while_training=True,
        get_train_stats_after_training=True,
        ## These both default to reasonable values depending on whether the
        ## acquisition datasets are fixed
        get_train_true_gp_stats=False,
        get_test_true_gp_stats=False
    )

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    exit()

    print(json.dumps(data, indent=4))

    print("Done training!")

    # Save the model
    # torch.save(model.state_dict(), model_path)
else:
    # Load the model
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))


######################## Plot performance of model #############################

def plot_gp_posterior(ax, posterior, test_x, train_x, train_y, color, name=None):#
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

# decided actually don't want to use this
normal = dist.Normal(0, 1)
def normalize_by_quantile(x, dim=-1):
    indices = torch.argsort(torch.argsort(x, dim=dim), dim=dim)
    max_index = x.size(dim) - 1
    quantiles = indices / max_index
    return normal.icdf(quantiles)

def plot_nn_vs_gp_acquisition_function_1d_grid(
        aq_dataset, n_candidates, nrows, ncols, min_x=0., max_x=1.,
        plot_map=True, nn_device=None):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5*ncols, 2.5*nrows),
                            sharex=True, sharey=False)

    it = iter(aq_dataset)
    for row in range(nrows):
        for col in range(ncols):
            item = next(it)

            gp_model = item.model

            x_hist, y_hist, x_cand, improvements = item.tuple_no_model
            x_cand = torch.linspace(0, 1, n_candidates).unsqueeze(1)
            item.x_cand = x_cand

            x_hist_nn, y_hist_nn, x_cand_nn, improvements_nn = item.to(nn_device).tuple_no_model

            aq_fn = LikelihoodFreeNetworkAcquisitionFunction.from_net(
                model, x_hist_nn, y_hist_nn, exponentiate=not POLICY_GRADIENT, softmax=False)
            ei_nn = aq_fn(x_cand_nn.unsqueeze(1))
            ei_nn = ei_nn.cpu()

            gp_model.set_train_data(x_hist, y_hist.squeeze(-1), strict=False)
            posterior_true = gp_model.posterior(x_cand, observation_noise=False)

            ei_true = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, log=False)

            if plot_map:
                ei_map = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, fit_params=True, log=False)

            # Normalize so they have the same scale
            if POLICY_GRADIENT:
                ei_nn = (ei_nn - ei_nn.mean()) / ei_nn.std()
                ei_true = (ei_true - ei_true.mean()) / ei_true.std()
                if plot_map:
                    ei_map = (ei_map - ei_map.mean()) / ei_map.std()

                # ei_nn = normalize_by_quantile(ei_nn)
                # ei_true = normalize_by_quantile(ei_true)
                # if plot_map:
                #     ei_map = normalize_by_quantile(ei_map)

            ax = axs[row, col]

            sorted_indices = np.argsort(x_cand.detach().numpy().flatten())
            sorted_x_cand = x_cand.detach().numpy().flatten()[sorted_indices]
            sorted_ei_true = ei_true.detach().numpy().flatten()[sorted_indices]
            sorted_ei_nn = ei_nn.detach().numpy().flatten()[sorted_indices]
            if plot_map:
                sorted_ei_map = ei_map.detach().numpy().flatten()[sorted_indices]

            ax.plot(sorted_x_cand, sorted_ei_true, label="True GP")
            ax.plot(sorted_x_cand, sorted_ei_nn, label="NN")
            if plot_map:
                ax.plot(sorted_x_cand, sorted_ei_map, label="MAP")

            plot_gp_posterior(ax, posterior_true, x_cand, x_hist, y_hist, 'b', name='True')

            # ax.set_title(f"History: {x_hist.size(0)}")
            ax.set_xlim(min_x, max_x)
    
    # Add a single legend for all plots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    
    # axs[0, ncols - 1].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    axs[0, ncols - 1].legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1))
    fig.tight_layout(rect=[0.02, 0.02, 1, 1])
    # fig.suptitle(f'{name} vs x', fontsize=16)
    fig.supxlabel("x", fontsize=10)
    fig.supylabel(f'{name}', fontsize=10)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    
    return fig, axs


n_candidates = 2_000
name = "acquisition" if POLICY_GRADIENT else "EI"
PLOT_MAP = False

if DIMENSION == 1:
    nrows, ncols = 5, 5
    fig, axs = plot_nn_vs_gp_acquisition_function_1d_grid(
        test_aq_dataset, n_candidates, nrows=nrows, ncols=ncols,
        plot_map=PLOT_MAP, nn_device=device)
    fname = f'acqusion_function_net_vs_gp_acquisition_function_1d_grid_{nrows}x{ncols}.pdf'
    fig.savefig(fname, bbox_inches='tight')
else:
    it = iter(test_aq_dataset)
    x_hist, y_hist, x_cand, improvements, gp_model = next(it)
    print(f"Number of history points: {x_hist.size(0)}")

    x_cand = torch.rand(n_candidates, DIMENSION)

    aq_fn = LikelihoodFreeNetworkAcquisitionFunction.from_net(
        model, x_hist, y_hist, exponentiate=not POLICY_GRADIENT, softmax=False)
    ei_nn = aq_fn(x_cand.unsqueeze(1))

    ei_true = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, log=False)
    if PLOT_MAP:
        ei_map = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, fit_params=True, log=False)

    # print(f"{name} True:")
    # print(ei_true)
    # print(f"{name} NN:")
    # print(ei_nn)
    # print(f"{name} MAP:")
    # print(ei_map)

    plt.scatter(ei_true.detach().numpy(), ei_nn.detach().numpy())
    plt.xlabel(f'{name} True')
    plt.ylabel(f'{name} NN')
    plt.title(f'{name} True vs {name} NN')

    if PLOT_MAP:
        plt.figure()
        plt.scatter(ei_true.detach().numpy(), ei_map.detach().numpy())
        plt.xlabel(f'{name} True')
        plt.ylabel(f'{name} MAP')
        plt.title(f'{name} True vs {name} MAP')


plt.show()

