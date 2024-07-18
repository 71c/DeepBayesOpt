import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cProfile, pstats, io
from pstats import SortKey
from dataset_with_models import RandomModelSampler
from tictoc import tic, tocl


from botorch.models.transforms.outcome import Power

from gp_acquisition_dataset import create_train_and_test_gp_acquisition_datasets
from acquisition_function_net import (
    AcquisitionFunctionNet, AcquisitionFunctionNetV1and2, AcquisitionFunctionNetV3, AcquisitionFunctionNetV4,
    AcquisitionFunctionNetDense, LikelihoodFreeNetworkAcquisitionFunction)
from predict_EI_simple import calculate_EI_GP
from train_acquisition_function_net import (
    load_configs,
    load_model,
    print_stats,
    save_acquisition_function_net_configs,
    train_acquisition_function_net,
    count_trainable_parameters, count_parameters,
    train_or_test_loop)
from utils import DEVICE, Exp, get_dimension, save_json, load_json, convert_to_json_serializable
from plot_utils import plot_nn_vs_gp_acquisition_function_1d_grid, plot_acquisition_function_net_training_history

import logging
logging.basicConfig(level=logging.WARNING)

##################### Settings for this script #################################
script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(script_dir, "saved_models")

# Whether to train the model.
TRAIN = True
# Whether to load a saved model.
LOAD_SAVED_MODEL = False
# Whether to load the saved dataset config specified in the model info directory
LOAD_SAVED_DATASET_CONFIG = False

# MODEL_AND_INFO_NAME = "model_20240716_010917_a3cf2269d9ef18d89800d5059878d662df8e9bbab2624e6adfd0a6653fcea168"
# MODEL_AND_INFO_NAME = "model_20240716_012546_cea676c3cb0ad3ae82f9463cd125e83ca6569663016c684e4f2113f01f716272"
# MODEL_AND_INFO_NAME = "model_20240716_183601_e5465772fa96e75ee50f68eca55f6da432c90d4d52b5a873137a397722f1e7e8"

# G2
# 6-dim, policy gradient, Exp, 14-50 history points
# outcomes standardized both dataset and history outcomes in forward
# trained for 100 epochs
# MODEL_AND_INFO_NAME = "model_20240716_230558_f182344a0fd9ee8ac324de86db74ccf2f2b60ea2fa07f471cdfb3a9728a64d5d"

# Same as above but 64 instead of 32
MODEL_AND_INFO_NAME = "model_20240717_141109_b7a5a9f189d98493d8f3eefef37e9bd5b4ed023742ddd6f60d4ae91e7c0350e9"

MODEL_AND_INFO_PATH = os.path.join(MODELS_DIR, MODEL_AND_INFO_NAME)

# Whether to fit maximum a posteriori GP for testing
FIT_MAP_GP = False

GET_TRAIN_TRUE_GP_STATS = False
GET_TEST_TRUE_GP_STATS = True

CPROFILE = False
TIME = True
VERBOSE = True

CACHE_DATASETS = True

# The following two are not important.
LAZY_TRAIN = True
LAZY_TEST = True

# Generating the random GP realizations is faster on CPU than on GPU.
# This is likely because the random GP realizations are generated one-by-one
# rather than in batches since the number of points is random so it's difficult
# to batch this. Hence we set device="cpu".
# Also, making the padded batches (the creation of zeros, concatenating, and
# stacking) on CPU rather than on GPU is much faster.
GP_GEN_DEVICE = "cpu"


############################# Settings for training ############################
# Set POLICY_GRADIENT based on the loaded model if loadin a model
if LOAD_SAVED_MODEL:
    POLICY_GRADIENT = load_json(
        os.path.join(MODEL_AND_INFO_PATH, "training_config.json")
    )['policy_gradient']
else:
    POLICY_GRADIENT = True # True for the softmax thing, False for MSE EI

BATCH_SIZE = 64
LEARNING_RATE = 2e-4
EPOCHS = 200
FIX_TRAIN_ACQUISITION_DATASET = False

# Only used if POLICY_GRADIENT is True
INCLUDE_ALPHA = True
# Following 3 are only used if both POLICY_GRADIENT and INCLUDE_ALPHA are True
LEARN_ALPHA = True
INITIAL_ALPHA = 1.0
ALPHA_INCREMENT = None # equivalent to 0.0

EARLY_STOPPING = True
PATIENCE = 10
MIN_DELTA = 0.0
CUMULATIVE_DELTA = False

training_config = dict(
    policy_gradient=POLICY_GRADIENT,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    fix_train_acquisition_dataset=FIX_TRAIN_ACQUISITION_DATASET,
    alpha_increment=ALPHA_INCREMENT,
    early_stopping=EARLY_STOPPING
)
if EARLY_STOPPING:
    training_config = dict(
        **training_config,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        cumulative_delta=CUMULATIVE_DELTA)
################################################################################


########################### Test dataset settings ##############################
test_dataset_config = dict(
    ## How many times bigger the big test dataset is than the train dataset, > 0
    ## test_factor=1 means same size, test_factor=0.5 means half the size, etc
    test_factor=1.0, # 3.0
    ## The proportion of the test dataset that is used for evaluating the model
    ## after each epoch, between 0 and 1
    small_test_proportion_of_test=1.0,
    # The following two should be kept as they are -- ALWAYS want to fix the
    # test. As long as the acqisition dataset is fixed, then whether the
    # function samples dataset is fixed doesn't matter.
    fix_test_samples_dataset=False,
    fix_test_acquisition_dataset=True,
)

if LOAD_SAVED_DATASET_CONFIG:
    gp_realization_config, dataset_size_config, n_points_config, \
        dataset_transform_config, model_sampler = load_configs(MODEL_AND_INFO_PATH)
    DIMENSION = get_dimension(model_sampler.get_model(0))
else:
    # Need the dimension to match if loading a saved model
    if LOAD_SAVED_MODEL:
        _model_sampler = RandomModelSampler.load(
            os.path.join(MODEL_AND_INFO_PATH, "model_sampler"))
        DIMENSION = get_dimension(_model_sampler.get_model(0))
    else:
        DIMENSION = 6

    ###################### GP realization characteristics ##########################
    gp_realization_config = dict(
        # Dimension of the optimization problem
        dimension=DIMENSION,
        # whether to randomize the GP parameters for training data
        randomize_params=False,
        # choose either "uniform" or "normal" (or a custom distribution)
        xvalue_distribution="uniform",
        observation_noise=False,
        models=None,
        model_probabilities=None
    )

    ################## Settings for dataset size and generation ####################
    dataset_size_config = dict(
        # The size of the training acquisition dataset
        train_acquisition_size=10_000,
        # The amount that the dataset is expanded to save compute of GP realizations
        expansion_factor=2,
        # Whether to fix the training dataset function samples
        # (as opposed to generating them randomly with each epoch)
        fix_train_samples_dataset=True
    )

    ########## Set number of history and candidate points generation ###############
    n_points_config = dict(
        # This means whether n history points (or whether the total number of
        # points) is log-uniform
        loguniform=True,
        # Whether to fix the number of candidate points (as opposed to randomized)
        fix_n_candidates=True
    )
    if n_points_config['loguniform']:
        n_points_config['pre_offset'] = 3.0
    if n_points_config['fix_n_candidates']:
        # If fix_n_candidates is True, then the following are used:
        n_points_config = dict(
            # Number of candidate points for training. For MSE EI, could just set to 1.
            train_n_candidates=50,
            # Number of candidate points for testing.
            test_n_candidates=50,
            min_history=14,
            max_history=50,
            **n_points_config
        )
    else:
        # If fix_n_candidates is False, then the following are used:
        n_points_config = dict(
            min_n_candidates=2,
            max_points=30,
            **n_points_config
        )

    dataset_transform_config = dict(
        # Choose an outcome transform. Can be None if no outcome transform
        # TODO (bug): str(Power(2)) = "Power()" but we'd like it to be "Power(2)" so it
        # can be saved uniquely. Maybe use the attributes of the class or something
        # instead. Or alternateively, just don't save the acquisition datasets, or
        # transform the acquisition datasets directly. I think it would be easiest to
        # just not save the acquisition datasets anymore.
        outcome_transform=Exp(),
        standardize_outcomes=True
    )
# Exp technically works, but Power does not
# Make sure to set these appropriately depending on whether the transform
# supports mean transform
# if dataset_transform_config['outcome_transform'] is not None:
#     GET_TRAIN_TRUE_GP_STATS = False
#     GET_TEST_TRUE_GP_STATS = False




####################### Make the train and test datasets #######################
dataset_kwargs = {
    **gp_realization_config,
    **dataset_size_config,
    **n_points_config,
    **dataset_transform_config}

other_kwargs = dict(
        **test_dataset_config,
        
        get_train_true_gp_stats=GET_TRAIN_TRUE_GP_STATS,
        get_test_true_gp_stats=GET_TEST_TRUE_GP_STATS,
        cache_datasets=CACHE_DATASETS,
        lazy_train=LAZY_TRAIN,
        lazy_test=LAZY_TEST,
        gp_gen_device=GP_GEN_DEVICE,
        
        batch_size=BATCH_SIZE,
        fix_train_acquisition_dataset=FIX_TRAIN_ACQUISITION_DATASET)

train_aq_dataset, test_aq_dataset, small_test_aq_dataset = create_train_and_test_gp_acquisition_datasets(
    **dataset_kwargs, **other_kwargs)

# print("Training function samples dataset size:", len(train_dataset))
print("Original training acquisition dataset size parameter:", dataset_size_config['train_acquisition_size'])
print("Training acquisition dataset size:", len(train_aq_dataset),
    "number of batches:", len(train_aq_dataset) // BATCH_SIZE, len(train_aq_dataset) % BATCH_SIZE)

# print("Test function samples dataset size:", len(test_dataset))
print("Test acquisition dataset size:", len(test_aq_dataset),
    "number of batches:", len(test_aq_dataset) // BATCH_SIZE, len(test_aq_dataset) % BATCH_SIZE)
if small_test_aq_dataset != test_aq_dataset:
    print("Small test acquisition dataset size:", len(small_test_aq_dataset),
            "number of batches:", len(small_test_aq_dataset) // BATCH_SIZE, len(small_test_aq_dataset) % BATCH_SIZE)

# for item in train_aq_dataset.base_dataset:
#     print(item.y_values.mean(), item.y_values.std(), item.y_values.shape)
# exit()

# print("Train acquisition dataset:")
# print(train_aq_dataset)
# print("\nTest acquisition dataset:")
# print(test_aq_dataset)
# if small_test_aq_dataset != test_aq_dataset:
#     print("\nSmall test acquisition dataset:")
#     print(small_test_aq_dataset)
# print("\n")
# exit()

################################### Get NN model ###############################

if LOAD_SAVED_MODEL:
    model = load_model(MODEL_AND_INFO_PATH).to(DEVICE)
else:
    model = AcquisitionFunctionNetV1and2(
                DIMENSION,
                pooling="max",
                history_enc_hidden_dims=[64, 64],
                encoded_history_dim=64,
                aq_func_hidden_dims=[64, 64],
                input_xcand_to_local_nn=True,
                input_xcand_to_final_mlp=False,
                include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
                learn_alpha=LEARN_ALPHA,
                initial_alpha=INITIAL_ALPHA,
                activation_at_end_pointnet=True,
                layer_norm_pointnet=False,
                layer_norm_before_end_mlp=False,
                layer_norm_at_end_mlp=False,
                standardize_outcomes=True,
                include_best_y=False,
                activation_pointnet="relu",
                activation_mlp="relu").to(DEVICE)
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
#                                  initial_alpha=INITIAL_ALPHA).to(DEVICE)

# model = AcquisitionFunctionNetV3(DIMENSION,
#                                  history_enc_hidden_dims=[32, 32], pooling="max",
#                  encoded_history_dim=32,
#                  mean_enc_hidden_dims=[32, 32], mean_dim=1,
#                  std_enc_hidden_dims=[32, 32], std_dim=16,
#                  aq_func_hidden_dims=[32, 32], layer_norm=False,
#                  layer_norm_at_end_mlp=False, include_y=True,
#                  include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
#                                  learn_alpha=LEARN_ALPHA,
#                                  initial_alpha=INITIAL_ALPHA).to(DEVICE)

# model = AcquisitionFunctionNetDense(DIMENSION, MAX_HISTORY,
#                                     hidden_dims=[128, 128, 64, 32],
#                                     include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
#                                     learn_alpha=LEARN_ALPHA,
#                                     initial_alpha=INITIAL_ALPHA).to(DEVICE)

print(model)
print("Number of trainable parameters:", count_trainable_parameters(model))
print("Number of parameters:", count_parameters(model))



######################## Train the model #######################################
if TRAIN:
    if CPROFILE:
        pr = cProfile.Profile()
        pr.enable()
    
    if TIME:
        tic("Training!")
    
    # Save the configs for the model and training and datasets
    (model_and_info_folder_name,
     MODEL_AND_INFO_PATH, model_path) = save_acquisition_function_net_configs(
        MODELS_DIR, model, training_config,
        gp_realization_config, dataset_size_config, n_points_config,
        dataset_transform_config, train_aq_dataset.model_sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    training_history_data = train_acquisition_function_net(
        model, train_aq_dataset, optimizer, POLICY_GRADIENT, EPOCHS, BATCH_SIZE,
        DEVICE, ALPHA_INCREMENT, verbose=VERBOSE, n_train_printouts_per_epoch=10,
        test_dataset=test_aq_dataset, small_test_dataset=small_test_aq_dataset,
        get_train_stats_while_training=True,
        get_train_stats_after_training=True,
        ## These both default to reasonable values depending on whether the
        ## acquisition datasets are fixed
        get_train_true_gp_stats=GET_TRAIN_TRUE_GP_STATS,
        get_test_true_gp_stats=GET_TEST_TRUE_GP_STATS,
        save_dir=model_path,
        save_incremental_best_models=False,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        cumulative_delta=CUMULATIVE_DELTA,
    )

    if TIME:
        tocl()

    if CPROFILE:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    print("Done training!")

training_history_path = os.path.join(MODEL_AND_INFO_PATH, 'model', 'training_history_data.json')

if not TRAIN:
    training_history_data = load_json(training_history_path)
    final_test_stats_original = training_history_data['final_test_stats']
    print_stats(final_test_stats_original,
                "Final test stats on the original test dataset")

    test_dataloader = test_aq_dataset.get_dataloader(
                batch_size=BATCH_SIZE, drop_last=False)
    final_test_stats = train_or_test_loop(
                test_dataloader, model, train=False,
                nn_device=DEVICE, policy_gradient=POLICY_GRADIENT,
                verbose=False, desc=f"Compute final test stats",
                get_true_gp_stats=GET_TEST_TRUE_GP_STATS,
                get_map_gp_stats=False,
                get_basic_stats=True)
    print_stats(final_test_stats, "Final test stats on this test dataset")


history_fig = plot_acquisition_function_net_training_history(training_history_data)
history_plot_path = os.path.join(MODEL_AND_INFO_PATH, 'model', 'training_history.pdf')
if not os.path.exists(history_plot_path) or LOAD_SAVED_DATASET_CONFIG:
    history_fig.savefig(history_plot_path, bbox_inches='tight')


######################## Plot performance of model #############################
n_candidates = 2_000
name = "acquisition" if POLICY_GRADIENT else "EI"
PLOT_MAP = False

if DIMENSION == 1:
    nrows, ncols = 5, 5
    fig, axs = plot_nn_vs_gp_acquisition_function_1d_grid(
        test_aq_dataset, model, POLICY_GRADIENT, name,
        n_candidates, nrows, ncols,
        plot_map=PLOT_MAP, nn_device=DEVICE,
        # If POLICY_GRADIENT=False, set this to False if it's hard to see some
        # of the plots
        group_standardization=None 
    )
    fname = f'acqusion_function_net_vs_gp_acquisition_function_1d_grid_{nrows}x{ncols}.pdf'
    path = os.path.join(MODEL_AND_INFO_PATH, fname)
    # Don't want to overwrite the plot if it already exists;
    # it could have been trained on different data from the data we are
    # evaluating it on if TRAIN=False.
    if not os.path.exists(path) or LOAD_SAVED_DATASET_CONFIG:
        fig.savefig(path, bbox_inches='tight')
else:
    it = iter(test_aq_dataset)
    item = next(it)
    x_hist, y_hist, x_cand, improvements, gp_model = item
    x_hist_nn, y_hist_nn, x_cand_nn, improvements_nn = item.to(DEVICE).tuple_no_model
    print(f"Number of history points: {x_hist.size(0)}")

    x_cand = torch.rand(n_candidates, DIMENSION)

    aq_fn = LikelihoodFreeNetworkAcquisitionFunction.from_net(
        model, x_hist_nn, y_hist_nn, exponentiate=not POLICY_GRADIENT, softmax=False)
    ei_nn = aq_fn(x_cand.to(DEVICE).unsqueeze(1))

    ei_true = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, log=False)
    if PLOT_MAP:
        ei_map = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, fit_params=True, log=False)

    # print(f"{name} True:")
    # print(ei_true)
    # print(f"{name} NN:")
    # print(ei_nn)
    # print(f"{name} MAP:")
    # print(ei_map)

    plt.scatter(ei_true.detach().cpu().numpy(), ei_nn.detach().cpu().numpy())
    plt.xlabel(f'{name} True')
    plt.ylabel(f'{name} NN')
    plt.title(f'{name} True vs {name} NN')

    if PLOT_MAP:
        plt.figure()
        plt.scatter(ei_true.detach().cpu().numpy(), ei_map.detach().cpu().numpy())
        plt.xlabel(f'{name} True')
        plt.ylabel(f'{name} MAP')
        plt.title(f'{name} True vs {name} MAP')


plt.show()
