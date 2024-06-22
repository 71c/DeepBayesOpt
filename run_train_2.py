import math
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
from torch import nn

from generate_gp_data import GaussianProcessRandomDataset, FunctionSamplesMapDataset, LazyFunctionSamplesMapDataset, FunctionSamplesAcquisitionDataset
from utils import get_uniform_randint_generator, get_loguniform_randint_generator, get_lengths_from_proportions
from acquisition_function_net import AcquisitionFunctionNetV1, AcquisitionFunctionNetV2, AcquisitionFunctionNetV3, AcquisitionFunctionNetV4, AcquisitionFunctionNetDense, LikelihoodFreeNetworkAcquisitionFunction
from predict_EI_simple import calculate_EI_GP
import numpy as np
import matplotlib.pyplot as plt
import os

from train_acquisition_function_net import train_loop, test_loop, count_trainable_parameters, count_parameters

import torch.distributions as dist

import cProfile
from tictoc import tic, toc, tocl


# Dimension of the optimization problem
DIMENSION = 1


########## Set number of history and candidate points generation ###############

# This means whether n history points is log-uniform
# or whether the total number of points is log-uniform
LOGUNIFORM = True
# Whether to fix the number of candidate poitns (as opposed to randomized)
FIX_N_CANDIDATES = True
if FIX_N_CANDIDATES:
    # Number of candidate points. More relevant for the policy gradient case.
    # Doesn't matter that much for the MSE EI case; for MSE, could just set to 1.
    TRAIN_N_CANDIDATES = 1
    MIN_HISTORY = 1
    MAX_HISTORY = 8

    TEST_N_CANDIDATES = 50

    if LOGUNIFORM:
        train_n_datapoints_random_gen = get_loguniform_randint_generator(
            MIN_HISTORY, MAX_HISTORY, pre_offset=3.0, offset=TRAIN_N_CANDIDATES)
        test_n_datapoints_random_gen = get_loguniform_randint_generator(
            MIN_HISTORY, MAX_HISTORY, pre_offset=3.0, offset=TEST_N_CANDIDATES)
    else:
        train_n_datapoints_random_gen = get_uniform_randint_generator(
            TRAIN_N_CANDIDATES+MIN_HISTORY, TRAIN_N_CANDIDATES+MAX_HISTORY)
        test_n_datapoints_random_gen = get_uniform_randint_generator(
            TEST_N_CANDIDATES+MIN_HISTORY, TEST_N_CANDIDATES+MAX_HISTORY)
else:
    MIN_N_CANDIDATES = 2
    MIN_POINTS = MIN_N_CANDIDATES + 1
    MAX_POINTS = 30

    if LOGUNIFORM:
        train_n_datapoints_random_gen = get_loguniform_randint_generator(
            MIN_POINTS, MAX_POINTS, pre_offset=3.0, offset=0)
    else:
        train_n_datapoints_random_gen = get_uniform_randint_generator(
            MIN_POINTS, MAX_POINTS)
        test_n_datapoints_random_gen = train_n_datapoints_random_gen


####################### Make the train and test datasets #######################

TRAIN_SIZE = 128 * 100
# How many times bigger the big test dataset is than the train dataset, any
# value > 0.
TEST_FACTOR = 3.0

# The proportion of the test dataset that is used for evaluating the model after
# each epoch, between 0 and 1
SMALL_TEST_PROPORTION_OF_TEST = 0.04

FIX_TRAIN_DATASET = True
FIX_TEST_DATASET = True

# whether to randomize the GP parameters for training data
RANDOMIZE_PARAMS = False
# choose either "uniform" or "normal" (or a custom distribution)
XVALUE_DISTRIBUTION = "uniform"


## 1 - TRAIN_SIZE / DATASET_SIZE == TEST_SIZE
# DATASET_SIZE = math.ceil(TRAIN_SIZE / (1 - TEST_SIZE))
DATASET_SIZE = math.ceil(TRAIN_SIZE * (1 + TEST_FACTOR))

#### Calculate test size
TEST_SIZE = DATASET_SIZE - TRAIN_SIZE
## Could alternatively calculate test size like this if going by
## proportions of an original dataset:
# TEST_PROPORTION = TEST_FACTOR / (1 + TEST_FACTOR)
# TRAIN_PROPORTION = 1 - TEST_PROPORTION
# TRAIN_SIZE, TEST_SIZE = get_lengths_from_proportions(DATASET_SIZE, [TRAIN_PROPORTION, TEST_PROPORTION])

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
dataset_kwargs = dict(dimension=DIMENSION, observation_noise=False,
    set_random_model_train_data=False, xvalue_distribution=XVALUE_DISTRIBUTION,
    device="cpu", randomize_params=RANDOMIZE_PARAMS)

train_dataset = GaussianProcessRandomDataset(
    **dataset_kwargs, dataset_size=TRAIN_SIZE,
    n_datapoints_random_gen=train_n_datapoints_random_gen)
test_dataset = GaussianProcessRandomDataset(
    **dataset_kwargs, dataset_size=TEST_SIZE,
    n_datapoints_random_gen=test_n_datapoints_random_gen)

if FIX_TRAIN_DATASET:
    train_dataset = FunctionSamplesMapDataset.from_iterable_dataset(train_dataset)
if FIX_TEST_DATASET:
    ## Either way works, it just depends on when you wanna wait longer --
    ## preemptively vs when you actually need the data. Same time spent.
    # test_dataset = FunctionSamplesMapDataset.from_iterable_dataset(test_dataset)
    test_dataset = LazyFunctionSamplesMapDataset(test_dataset)


#### Make train-acquisition-function dataset from function samples dataset #####

if FIX_N_CANDIDATES:
    train_aq_dataset = FunctionSamplesAcquisitionDataset(
        train_dataset, n_candidate_points=TRAIN_N_CANDIDATES,
        n_samples="all", give_improvements=True)
    test_aq_dataset = FunctionSamplesAcquisitionDataset(
        test_dataset,
        n_candidate_points=TEST_N_CANDIDATES,
        n_samples="all", give_improvements=True)
else:
    train_aq_dataset = FunctionSamplesAcquisitionDataset(
        train_dataset, n_candidate_points="uniform", n_samples="all",
        give_improvements=True, min_n_candidates=MIN_N_CANDIDATES)
    test_aq_dataset = FunctionSamplesAcquisitionDataset(
        test_dataset, n_candidate_points="uniform", n_samples="all",
        give_improvements=True, min_n_candidates=MIN_N_CANDIDATES)

sample_n_points = test_n_datapoints_random_gen(30)
n_samples_and_candidates_examples = [
    test_aq_dataset._pick_random_n_samples_and_n_candidates(n)
    for n in sample_n_points]
n_hist_and_candidates_examples = [
    (n_samples - n_candidates, n_candidates)
    for n_samples, n_candidates in n_samples_and_candidates_examples]
print(n_hist_and_candidates_examples)

small_test_aq_dataset, _ = test_aq_dataset.random_split(
    [SMALL_TEST_PROPORTION_OF_TEST, 1 - SMALL_TEST_PROPORTION_OF_TEST])

# True for the softmax thing, False for MSE
POLICY_GRADIENT = False

################# Initialize the acquisition function network #################

# Only used if POLICY_GRADIENT is True
INCLUDE_ALPHA = True
# Following 3 are only used if both POLICY_GRADIENT and INCLUDE_ALPHA are True
LEARN_ALPHA = True
INITIAL_ALPHA = 1.0
ALPHA_INCREMENT = None # equivalent to 0.0

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

model = AcquisitionFunctionNetV2(DIMENSION,
                                 pooling="max",
                                 history_enc_hidden_dims=[32, 32],
                                 encoded_history_dim=32,
                                 aq_func_hidden_dims=[32, 32],
                                #  history_enc_hidden_dims=[64, 64],
                                #  encoded_history_dim=128,
                                #  aq_func_hidden_dims=[64, 64],
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

# model = AcquisitionFunctionNetV1(DIMENSION,
#                                  pooling="max",
#                                  history_enc_hidden_dims=[32, 32],
#                                  encoded_history_dim=32,
#                                  aq_func_hidden_dims=[32, 32],
#                                  include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
#                                  learn_alpha=LEARN_ALPHA,
#                                  initial_alpha=INITIAL_ALPHA,
#                                  layer_norm_pointnet=False,
#                                  layer_norm_before_end_mlp=False,
#                                  layer_norm_at_end_mlp=False,
#                                  include_best_y=True,
#                                  activation_pointnet=nn.ReLU,
#                                  activation_mlp=nn.ReLU).to(device)

# model = AcquisitionFunctionNetDense(DIMENSION, MAX_HISTORY,
#                                     hidden_dims=[128, 128, 64, 32],
#                                     include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
#                                     learn_alpha=LEARN_ALPHA,
#                                     initial_alpha=INITIAL_ALPHA).to(device)

print(model)
print("Number of trainable parameters:", count_trainable_parameters(model))
print("Number of parameters:", count_parameters(model))


########################## Get model path ######################################

script_dir = os.path.dirname(os.path.abspath(__file__))
# training_info = f"batchsize{BATCH_SIZE}_batches_per_epoch{N_BATCHES}_epochs{EPOCHS}"
model_class_name = model.__class__.__name__
loss_str = 'policy_gradient_myopic' if POLICY_GRADIENT else 'ei'
file_name = f"acquisition_function_net_{model_class_name}_{DIMENSION}d_{loss_str}_{'random' if RANDOMIZE_PARAMS else 'fixed'}_kernel_{XVALUE_DISTRIBUTION}_x"
if FIX_N_CANDIDATES:
    file_name += f"_history{MIN_HISTORY}-{MAX_HISTORY}_{'loguniform' if LOGUNIFORM else 'uniform'}_{TRAIN_N_CANDIDATES}cand.pth"
else:
    file_name += f"_points{MIN_POINTS}-{MAX_POINTS}_{'loguniform' if LOGUNIFORM else 'uniform'}.pth"

# file_name = "acquisition_function_net_AcquisitionFunctionNetV4_1d_policy_gradient_myopic_fixed_kernel_uniform_x_history1-8_loguniform_50cand_200epochs.pth"

print(f"Model file: {file_name}")
model_path = os.path.join(script_dir, file_name)


######################## Train the model #######################################

BATCH_SIZE = 128

train_aq_dataloader = train_aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True, device=device)
small_test_aq_dataloader = small_test_aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)

# Whether to train the model. If False, will load a saved model.
TRAIN = True
# Whether to load a saved model to train
LOAD_SAVED_MODEL_TO_TRAIN = False
# Whether to fit maximum a posteriori GP for testing
FIT_MAP_GP = False

if TRAIN:
    EPOCHS = 5
    LEARNING_RATE = 3e-4
    N_BATCHES = TRAIN_SIZE // BATCH_SIZE
    EVERY_N_BATCHES = N_BATCHES // 10
    # Whether to train to predict the EI rather than predict the I
    # Only used if POLICY_GRADIENT is False
    TRAIN_WITH_EI = False

    if LOAD_SAVED_MODEL_TO_TRAIN:
        # Load the model
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")

        tic(f"Epoch {t+1} train")
        
        train_loop(train_aq_dataloader, model, optimizer,
                   every_n_batches=EVERY_N_BATCHES,
                   policy_gradient=POLICY_GRADIENT,
                   alpha_increment=ALPHA_INCREMENT,
                   train_with_ei=TRAIN_WITH_EI)

        tocl()

        test_loop(small_test_aq_dataloader, model,
                  policy_gradient=POLICY_GRADIENT, fit_map_gp=FIT_MAP_GP,
                  nn_device=device)

    print("Done training!")

    # Save the model
    # torch.save(model.state_dict(), model_path)
else:
    # Load the model
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))



###################### Evaluate performance of model ###########################
model.eval()


test_aq_dataloader = test_aq_dataset.get_dataloader(batch_size=BATCH_SIZE, drop_last=True)
test_loop(test_aq_dataloader, model, policy_gradient=POLICY_GRADIENT,
          fit_map_gp=FIT_MAP_GP, nn_device=device)

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

# decided actually don't want to use this
normal = dist.Normal(0, 1)
def normalize_by_quantile(x, dim=-1):
    indices = torch.argsort(torch.argsort(x, dim=dim), dim=dim)
    max_index = x.size(dim) - 1
    quantiles = indices / max_index
    return normal.icdf(quantiles)

def plot_nn_vs_gp_acquisition_function_1d_grid(
        aq_dataset, n_candidates, nrows, ncols, min_x=0., max_x=1.,
        plot_map=True):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5*ncols, 2.5*nrows),
                            sharex=True, sharey=False)
    
    # Failure case:
    # for x_hist, y_hist, x_cand, improvements, gp_model in test_aq_dataset_big:
    #     if x_hist.size(0) != 2:
    #         continue
    #     if x_hist[0, 0].item() > 0.05 or x_hist[1, 0].item() < 0.95:
    #         continue
    #     break

    it = iter(aq_dataset)
    for row in range(nrows):
        for col in range(ncols):
            x_hist, y_hist, x_cand, improvements, gp_model = next(it)

            x_cand = torch.linspace(0, 1, n_candidates).unsqueeze(1)

            aq_fn = LikelihoodFreeNetworkAcquisitionFunction.from_net(
                model, x_hist, y_hist, exponentiate=not POLICY_GRADIENT, softmax=False)
            ei_nn = aq_fn(x_cand.unsqueeze(1))

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
        plot_map=PLOT_MAP)
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

