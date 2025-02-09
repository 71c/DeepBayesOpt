import torch
import matplotlib.pyplot as plt
import os
import cProfile, pstats, io
from pstats import SortKey
from dataset_with_models import RandomModelSampler
from tictoc import tic, tocl

from gp_acquisition_dataset import add_gp_acquisition_dataset_args, create_train_and_test_gp_acquisition_datasets_command_helper, get_gp_acquisition_dataset_configs
# AcquisitionFunctionNetV3, AcquisitionFunctionNetV4,
#     AcquisitionFunctionNetDense
from acquisition_function_net import (
    AcquisitionFunctionBodyPointnetV1and2, AcquisitionFunctionNet, AcquisitionFunctionNetFinalMLP, AcquisitionFunctionNetFinalMLPSoftmaxExponentiate, ExpectedImprovementAcquisitionFunctionNet, GittinsAcquisitionFunctionNet, LikelihoodFreeNetworkAcquisitionFunction, TwoPartAcquisitionFunctionNetFixedHistoryOutputDim)
from exact_gp_computations import calculate_EI_GP
from train_acquisition_function_net import (
    METHODS,
    load_configs,
    load_model,
    print_stats,
    save_acquisition_function_net_configs,
    train_acquisition_function_net,
    train_or_test_loop)
from utils import DEVICE, get_dimension, load_json
from plot_utils import plot_nn_vs_gp_acquisition_function_1d_grid, plot_acquisition_function_net_training_history
from nn_utils import count_trainable_parameters, count_parameters

import argparse

import logging
logging.basicConfig(level=logging.WARNING)

##################### Settings for this script #################################
script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(script_dir, "saved_models")

# Run like, e.g.,

# python run_train.py --method gittins --batch_size 32 --lamda 1e-1 --normalize_gi_loss --dimension 6 --layer_width 160 --train_acquisition_size 5000 --test_acquisition_size 1500 --no-save-model --learning_rate 1e-3 --lengthscale 0.1 --expansion_factor 2 --kernel RBF --min_history 1 --max_history 50

# python run_train.py --batch_size 32 --dimension 6 --train_acquisition_size 5000 --test_acquisition_size 1500 --no-save-model --lengthscale 0.1 --expansion_factor 2 --kernel RBF --min_history 1 --max_history 50

# python run_train.py --method gittins --batch_size 32 --lamda 1e-1 --normalize_gi_loss --dimension 6 --layer_width 160 --train_acquisition_size 5000 --test_acquisition_size 1500 --no-save-model
# python run_train.py --method gittins --lamda 1e-1 --normalize_gi_loss --dimension 6 --layer_width 256 --train_acquisition_size 100000 --test_acquisition_size 10000

# python run_train.py --method mse_ei --dimension 6 --layer_width 160 --train_acquisition_size 10000 --test_acquisition_size 1000 --no-save-model

# python run_train.py --method mse_ei --dimension 6 --layer_width 160 --train_acquisition_size 1000 --test_acquisition_size 1000 --no-save-model

# python run_train.py --method gittins --lamda 1e-3 --dimension 6 --layer_width 160 --train_acquisition_size 20000 --test_acquisition_size 1500 --no-save-model
# python run_train.py --method gittins --lamda_min 1e-5 --lamda_max 1e-1 --dimension 6 --layer_width 160 --train_acquisition_size 20000 --test_acquisition_size 1500 --no-save-model


# python run_train.py --method mse_ei --dimension 6 --layer_width 160 --train_acquisition_size 5000 --test_factor 0.3 --no-save-model

# python run_train.py --method mse_ei --dimension 6 --layer_width 32 --train_acquisition_size 1000 --test_factor 0.1

# python run_train.py --method mse_ei --layer_width 32 --train_acquisition_size 10000 --test_acquisition_size 1000 --dimension 6 --learn_tau --initial_tau 0.5
# python run_train.py --method mse_ei --layer_width 128 --train_acquisition_size 50000 --test_acquisition_size 10000 --dimension 6 --learn_tau --initial_tau 0.5 --softplus_batchnorm
# python run_train.py --method mse_ei --layer_width 128 --train_acquisition_size 50000 --test_acquisition_size 10000 --dimension 6 --positive_linear_at_end

parser = argparse.ArgumentParser()

parser.add_argument(
    '--no-train', 
    action='store_false', 
    dest='train', 
    help='If set, do not train the model. Default is to train the model.'
)
parser.add_argument(
    '--no-save-model', 
    action='store_false', 
    dest='save_model', 
    help=('If set, do not save the model. Default is to save the model. '
          'Only applicable if training the model.')
)
parser.add_argument(
    '--load_saved_model', 
    action='store_true', 
    help='Whether to load a saved model. Set this flag to load the saved model.'
)
parser.add_argument(
    '--load_saved_dataset_config', 
    action='store_true', 
    help='Whether to load the saved dataset config specified in the model info directory. Set this flag to load the saved dataset config.'
)
parser.add_argument(
    '--model_and_info_name', 
    type=str, 
    help='Name of the model and info directory. Must be specified if load_saved_model or load_saved_dataset_config'
)

## GP dataset settings
add_gp_acquisition_dataset_args(parser, required=False)

# Which AF training loss function to use
parser.add_argument(
    '--method',
    choices=METHODS
)
# learning_rate = 3e-3 if METHOD == 'gittins' else 3e-4
parser.add_argument(
    '--learning_rate',
    type=float,
    help='Learning rate for training the model'
)
# batch_size = 320 if METHOD == 'mse_ei' else 16
parser.add_argument(
    '--batch_size',
    type=int,
    required=True,
    help='Batch size for training the model'
)

# Layer width
parser.add_argument(
    '--layer_width', 
    type=int,
    help='The width of the NN layers. Required if load_saved_model=False'
)

### Optional settings for NN architecture
parser.add_argument(
    '--standardize_nn_history_outcomes', 
    action='store_true', 
    help=('Whether to standardize the history outcomes when computing the NN '
          'acquisition function. Default is False.')
)

### Options when method=gittins
parser.add_argument(
    '--lamda_min',
    type=float,
    help=('Minimum value of lambda (if using variable lambda). '
          'Only used if method=gittins.')
)
parser.add_argument(
    '--lamda_max',
    type=float,
    help=('Maximum value of lambda (if using variable lambda). '
          'Only used if method=gittins.')
)
parser.add_argument(
    '--lamda',
    type=float,
    help='Value of lambda (if using constant lambda). Only used if method=gittins.'
)
parser.add_argument(
    '--normalize_gi_loss', 
    action='store_true', 
    help=('Whether to normalize the Gittins index loss function. Default is False. '
          'Only used if method=gittins.')
)

### Options for NN when method=mse_ei
parser.add_argument(
    '--learn_tau', 
    action='store_true',
    help=('Set this flag to enable learning of tau=1/beta which is the parameter for softplus'
          ' applied at the end of the MSE acquisition function. Default is False. '
          'Only used if method=mse_ei.')
)
parser.add_argument(
    '--initial_tau',
    type=float,
    help='Initial value of tau. Default is 1.0. Only used if method=mse_ei.'
)
parser.add_argument(
    '--softplus_batchnorm',
    action='store_true',
    help=('Set this flag to apply positive-batchnorm after softplus in the MSE acquisition function. '
          'Default is False. Only used if method=mse_ei.')
)
parser.add_argument(
    '--softplus_batchnorm_momentum',
    type=float,
    default=0.1,
    help=('Momentum for the batchnorm after softplus in the MSE acquisition function. Default is 0.1. '
          'Only used if method=mse_ei.')
)
parser.add_argument(
    '--positive_linear_at_end',
    action='store_true',
    help=('Set this flag to apply positive linear at end technique. Default is False. '
          'Only used if method=mse_ei.')
)
parser.add_argument(
    '--gp_ei_computation',
    action='store_true',
    help=('Set this flag to apply gp_ei_computation at end technique. Default is False. '
          'Only used if method=mse_ei.')
)

args = parser.parse_args()

# Whether to train the model.
TRAIN = args.train
# Whether to load a saved model.
LOAD_SAVED_MODEL = args.load_saved_model
# Whether to load the saved dataset config specified in the model info directory
LOAD_SAVED_DATASET_CONFIG = args.load_saved_dataset_config

# MODEL_AND_INFO_NAME = "model_20240716_010917_a3cf2269d9ef18d89800d5059878d662df8e9bbab2624e6adfd0a6653fcea168"
# MODEL_AND_INFO_NAME = "model_20240716_012546_cea676c3cb0ad3ae82f9463cd125e83ca6569663016c684e4f2113f01f716272"
# MODEL_AND_INFO_NAME = "model_20240716_183601_e5465772fa96e75ee50f68eca55f6da432c90d4d52b5a873137a397722f1e7e8"

# G2
# 6-dim, policy gradient, Exp, 14-50 history points, 32-dim layers
# outcomes standardized both dataset and history outcomes in forward
# trained for 100 epochs, policy gradient, train_acquisition_size=10_000
# Best max EI: 0.18619710690677405
# MODEL_AND_INFO_NAME = "model_20240716_230558_f182344a0fd9ee8ac324de86db74ccf2f2b60ea2fa07f471cdfb3a9728a64d5d"

# Same as above but 64 instead of 32
# Best max EI: 0.27558372714299717
# MODEL_AND_INFO_NAME = "model_20240717_141109_b7a5a9f189d98493d8f3eefef37e9bd5b4ed023742ddd6f60d4ae91e7c0350e9"

# Same as above but trained with MSE
# Best max EI: 0.29747663472631614
# MODEL_AND_INFO_NAME = "model_20240717_210109_c32da74f287bd08d0ef2a3c21c91cb59e5a8672bebf25cdc9638e846ba2a556c"

# Same as above but training dataset size is 50K instead of 10K
# Best max EI: 0.3141892009482172
# MODEL_AND_INFO_NAME = "model_20240717_212151_f3ff357217b4bce861abb578d2b853eebad30b90e2345c4062393bfca2417a3e"

# Same as above but training size is 200K, and layer-width 128 instead of 64
# Best max EI: 0.35370628685185046
# MODEL_AND_INFO_NAME = "model_20240718_004436_e1c8afcd9487924a8dbedbb6675293c42d94e30a63eabcf0c35338d19e3b64f4"

# Same as above but training size is 800K, and layer-width 256 instead of 128
# Best max EI: 0.36729013620012463
# MODEL_AND_INFO_NAME = "model_20240718_030711_3b42b16944fa8b5d8affffdd7c130d4188d4d8f7335a4c99758399fa7efa79ec"

if LOAD_SAVED_MODEL or LOAD_SAVED_DATASET_CONFIG:
    if args.model_and_info_name is None:
        raise ValueError("model_and_info_name should be specified if load_saved_model or load_saved_dataset_config")
    MODEL_AND_INFO_NAME = args.model_and_info_name
    MODEL_AND_INFO_PATH = os.path.join(MODELS_DIR, MODEL_AND_INFO_NAME)
else:
    MODEL_AND_INFO_PATH = None

if (not LOAD_SAVED_MODEL) and (args.layer_width is None):
    raise ValueError("layer_width must be specified if load_saved_model=False")

# Whether to fit maximum a posteriori GP for testing
FIT_MAP_GP = False

CPROFILE = False
TIME = True
VERBOSE = True


############################# Settings for datasets #############################
from gp_acquisition_dataset import (
    GET_TRAIN_TRUE_GP_STATS,
    GET_TEST_TRUE_GP_STATS,
    GP_GEN_DEVICE,
    FIX_TRAIN_ACQUISITION_DATASET
)


############################# Settings for training ############################
# Set METHOD based on the loaded model if loadin a model
if LOAD_SAVED_MODEL:
    METHOD = load_json(
        os.path.join(MODEL_AND_INFO_PATH, "training_config.json")
    )['method']
else:
    if args.method is None:
        raise ValueError("method should be specified if not loading a saved model")
    METHOD = args.method

POLICY_GRADIENT = (METHOD == 'policy_gradient')

if TRAIN:
    if args.learning_rate is None:
        raise ValueError("learning_rate should be specified if training the model")

EPOCHS = 200

# Only used if POLICY_GRADIENT is True
INCLUDE_ALPHA = True
# Following 3 are only used if both POLICY_GRADIENT and INCLUDE_ALPHA are True
LEARN_ALPHA = True
INITIAL_ALPHA = 1.0
ALPHA_INCREMENT = None # equivalent to 0.0

INCLUDE_ALPHA = INCLUDE_ALPHA and POLICY_GRADIENT

EARLY_STOPPING = True
PATIENCE = 20
MIN_DELTA = 0.0
CUMULATIVE_DELTA = False

training_config = dict(
    method=METHOD,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    epochs=EPOCHS,
    fix_train_acquisition_dataset=FIX_TRAIN_ACQUISITION_DATASET,
    early_stopping=EARLY_STOPPING
)
if POLICY_GRADIENT:
    training_config = dict(
        **training_config,
        include_alpha=INCLUDE_ALPHA,
        learn_alpha=LEARN_ALPHA,
        initial_alpha=INITIAL_ALPHA,
        alpha_increment=ALPHA_INCREMENT)

for reason, reason_desc in [(METHOD != 'mse_ei', 'method != mse_ei'),
                            (LOAD_SAVED_MODEL, 'load_saved_model=True')]:
    if reason:
        if args.learn_tau:
            raise ValueError(f"learn_tau should be False if {reason_desc}")
        if args.initial_tau is not None:
            raise ValueError(f"initial_tau should not be specified if {reason_desc}")
        if args.softplus_batchnorm:
            raise ValueError(f"softplus_batchnorm should be False if {reason_desc}")
        if args.positive_linear_at_end:
            raise ValueError(f"positive_linear_at_end should be False if {reason_desc}")
        if args.gp_ei_computation:
            raise ValueError(f"gp_ei_computation should be False if {reason_desc}")

lamda_given = args.lamda is not None
lamda_min_given = args.lamda_min is not None
lamda_max_given = args.lamda_max is not None
if METHOD == 'gittins':
    if lamda_given:
        if lamda_min_given or lamda_max_given:
            raise ValueError(
                "If method=gittins, should specify only either lamda, or both lamda_min and lamda_max")
        args.lamda_min = args.lamda # just to make it easier to pass in as parameter
        VARIABLE_LAMBDA = False
    else:
        if not (lamda_min_given or lamda_max_given):
            # No lamda anything is specified
            raise ValueError(
                "If method=gittins, need to specify either lamda, or both lamda_min and lamda_max")
        if not lamda_min_given:
            # lamda_max is given but lamda_min is not given
            raise ValueError(
                "If method=gittins and lamda_max is specified, then lamda_min must be "
                "specified (or give lamda instead)")
        if not lamda_max_given:
            # lamda_min is given but lamda_max is not given
            raise ValueError(
                "If method=gittins and lamda_min is specified, then lamda_max must be "
                "specified (or give lamda instead)")
        VARIABLE_LAMBDA = True
else: # METHOD = 'mse_ei' or 'policy_gradient'
    if args.initial_tau is None:
        args.initial_tau = 1.0
    if lamda_given or lamda_min_given or lamda_max_given:
        raise ValueError(
            "If method != gittins, then lamda, lamda_min, and lamda_max should not be specified")
    if args.normalize_gi_loss:
        raise ValueError("normalize_gi_loss should be False if method != gittins")


if EARLY_STOPPING:
    training_config = dict(
        **training_config,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        cumulative_delta=CUMULATIVE_DELTA)

################################################################################

if LOAD_SAVED_DATASET_CONFIG:
    gp_realization_config, dataset_size_config, n_points_config, \
        dataset_transform_config, model_sampler = load_configs(MODEL_AND_INFO_PATH)
    args.dimension = get_dimension(model_sampler.get_model(0))
else:
    if args.train_acquisition_size is None:
        raise ValueError("train_acquisition_size should be specified if not loding from dataset config")
    if args.lengthscale is None:
        raise ValueError("lengthscale should be specified if not loding from dataset config")
    if args.min_history is None:
        raise ValueError("min_history should be specified if not loding from dataset config")
    if args.max_history is None:
        raise ValueError("max_history should be specified if not loding from dataset config")

    # Need the dimension to match if loading a saved model
    if LOAD_SAVED_MODEL:
        _model_sampler = RandomModelSampler.load(
            os.path.join(MODEL_AND_INFO_PATH, "model_sampler"))
        args.dimension = get_dimension(_model_sampler.get_model(0))
    elif args.dimension is None:
        raise ValueError("dimension should be specified if not loading a dataset config or saved model")
    
    (gp_realization_config, dataset_size_config,
     n_points_config, dataset_transform_config) = get_gp_acquisition_dataset_configs(
         args, device=GP_GEN_DEVICE)
# Exp technically works, but Power does not
# Make sure to set these appropriately depending on whether the transform
# supports mean transform
# if dataset_transform_config['outcome_transform'] is not None:
#     GET_TRAIN_TRUE_GP_STATS = False
#     GET_TEST_TRUE_GP_STATS = False


####################### Make the train and test datasets #######################
train_aq_dataset, test_aq_dataset, small_test_aq_dataset = create_train_and_test_gp_acquisition_datasets_command_helper(
    args, gp_realization_config, dataset_size_config,
        n_points_config, dataset_transform_config)


################################### Get NN model ###############################

if LOAD_SAVED_MODEL:
    model = load_model(MODEL_AND_INFO_PATH).to(DEVICE)
else:
    af_body_init_params = dict(
        dimension=args.dimension,
        
        history_enc_hidden_dims=[args.layer_width, args.layer_width],
        pooling="max",
        encoded_history_dim=args.layer_width,

        input_xcand_to_local_nn=True,
        input_xcand_to_final_mlp=True,
        
        activation_at_end_pointnet=True,
        layer_norm_pointnet=False,
        dropout_pointnet=None,
        activation_pointnet="relu",

        include_best_y=False,
        n_pointnets=1)
    af_head_init_params = dict(
        hidden_dims=[args.layer_width, args.layer_width],
        activation="relu",
        layer_norm_before_end=False,
        layer_norm_at_end=False,
        dropout=None,
    )

    if METHOD == 'gittins':
        model = GittinsAcquisitionFunctionNet(
            af_class=TwoPartAcquisitionFunctionNetFixedHistoryOutputDim,
            variable_lambda=VARIABLE_LAMBDA,
            costs_in_history=False,
            cost_is_input=False,
            af_body_class=AcquisitionFunctionBodyPointnetV1and2,
            af_head_class=AcquisitionFunctionNetFinalMLP,
            af_body_init_params=af_body_init_params,
            af_head_init_params=af_head_init_params,
            standardize_outcomes=args.standardize_nn_history_outcomes
        ).to(DEVICE)
    elif METHOD == 'policy_gradient' or METHOD == 'mse_ei':
        af_head_init_params = dict(
            **af_head_init_params,
            include_alpha=INCLUDE_ALPHA,
            learn_alpha=LEARN_ALPHA,
            initial_alpha=INITIAL_ALPHA,
            initial_beta=1.0 / args.initial_tau,
            learn_beta=args.learn_tau,
            softplus_batchnorm=args.softplus_batchnorm,
            softplus_batchnorm_momentum=args.softplus_batchnorm_momentum,
            positive_linear_at_end=args.positive_linear_at_end,
            gp_ei_computation=args.gp_ei_computation
        )
        model = ExpectedImprovementAcquisitionFunctionNet(
            af_body_class=AcquisitionFunctionBodyPointnetV1and2,
            af_body_init_params=af_body_init_params,
            af_head_init_params=af_head_init_params,
            standardize_outcomes=args.standardize_nn_history_outcomes
        ).to(DEVICE)
    
    # model = AcquisitionFunctionNetV4(args.dimension,
    #                                 history_enc_hidden_dims=[32, 32], pooling="max",
    #                 include_local_features=True,
    #                 encoded_history_dim=4, include_mean=False,
    #                 mean_enc_hidden_dims=[32, 32], mean_dim=1,
    #                 std_enc_hidden_dims=[32, 32], std_dim=32,
    #                 aq_func_hidden_dims=[32, 32], layer_norm=True,
    #                 layer_norm_at_end_mlp=False,
    #                 include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
    #                                 learn_alpha=LEARN_ALPHA,
    #                                 initial_alpha=INITIAL_ALPHA).to(DEVICE)

    # model = AcquisitionFunctionNetV3(args.dimension, pooling="max",
    #                 history_enc_hidden_dims=[args.layer_width, args.layer_width],
    #                 encoded_history_dim=args.layer_width,
    #                 mean_enc_hidden_dims=[args.layer_width, args.layer_width], mean_dim=1,
    #                 std_enc_hidden_dims=[args.layer_width, args.layer_width], std_dim=16,
    #                 aq_func_hidden_dims=[args.layer_width, args.layer_width], layer_norm=False,
    #                 layer_norm_at_end_mlp=False, include_y=True,
    #                 include_alpha=INCLUDE_ALPHA and POLICY_GRADIENT,
    #                                 learn_alpha=LEARN_ALPHA,
    #                                 initial_alpha=INITIAL_ALPHA).to(DEVICE)

    # model = AcquisitionFunctionNetDense(args.dimension, MAX_HISTORY,
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
    
    if args.save_model:
        # Save the configs for the model and training and datasets
        (model_and_info_folder_name,
        MODEL_AND_INFO_PATH, model_path) = save_acquisition_function_net_configs(
            MODELS_DIR, model, training_config,
            gp_realization_config, dataset_size_config, n_points_config,
            dataset_transform_config, train_aq_dataset.model_sampler)
    else:
        model_path = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                #  weight_decay=1e-2
                                 )
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    training_history_data = train_acquisition_function_net(
        model, train_aq_dataset, optimizer, METHOD, EPOCHS, args.batch_size,
        DEVICE, verbose=VERBOSE, n_train_printouts_per_epoch=10,
        alpha_increment=ALPHA_INCREMENT,
        # lambda_min=args.lamda_min, lambda_max=args.lamda_max,
        normalize_gi_loss=args.normalize_gi_loss,
        test_dataset=test_aq_dataset, small_test_dataset=small_test_aq_dataset,
        get_train_stats_while_training=True,
        get_train_stats_after_training=True,
        ## These both default to reasonable values depending on whether the
        ## acquisition datasets are fixed
        get_train_true_gp_stats=GET_TRAIN_TRUE_GP_STATS,
        get_test_true_gp_stats=GET_TEST_TRUE_GP_STATS,
        save_dir=model_path,
        save_incremental_best_models=True and args.save_model,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        cumulative_delta=CUMULATIVE_DELTA,
    )

    if TIME:
        tocl()

    if CPROFILE:
        pr.disable()
        
        # s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
        # ps.print_stats()
        # print(s.getvalue())

        s = open('stats_output.txt', 'w')
        ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats()

    print("Done training!")


if MODEL_AND_INFO_PATH is not None:
    training_history_path = os.path.join(
        MODEL_AND_INFO_PATH, 'model', 'training_history_data.json')

    if not TRAIN:
        training_history_data = load_json(training_history_path)
        final_test_stats_original = training_history_data['final_test_stats']
        print_stats(final_test_stats_original,
                    "Final test stats on the original test dataset")

        test_dataloader = test_aq_dataset.get_dataloader(
                    batch_size=args.batch_size, drop_last=False)
        final_test_stats = train_or_test_loop(
                    test_dataloader, model, train=False,
                    nn_device=DEVICE, method=METHOD,
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
name = "EI" if METHOD == "mse_ei" else "acquisition"
PLOT_MAP = False

# TODO: Fix the below code to work with Gittins index

if args.dimension == 1:
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

    x_cand = torch.rand(n_candidates, args.dimension)

    aq_fn = LikelihoodFreeNetworkAcquisitionFunction.from_net(
        model, x_hist_nn, y_hist_nn, exponentiate=(METHOD == 'mse_ei'), softmax=False)
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
