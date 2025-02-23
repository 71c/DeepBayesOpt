# Run like, e.g.,
# python run_train.py --dimension 8 --expansion_factor 2 --kernel Matern52 --lengthscale 0.1 --max_history 400 --min_history 1 --test_acquisition_size 10000 --test_n_candidates 1 --train_acquisition_size 2000 --train_n_candidates 1 --batch_size 32 --early_stopping --epochs 200 --lamda_max 1.0 --lamda_min 0.0001 --layer_width 100 --learning_rate 0.003 --method gittins --min_delta 0.0 --normalize_gi_loss --patience 5

# python run_train.py --dimension 8 --expansion_factor 1 --kernel Matern52 --lengthscale 0.1 --max_history 400 --min_history 1 --test_acquisition_size 1000 --test_n_candidates 1 --train_acquisition_size 1000 --train_n_candidates 1 --batch_size 32 --early_stopping --epochs 200 --layer_width 100 --learning_rate 0.003 --method gittins --min_delta 0.0 --normalize_gi_loss --patience 5 --lamda 0.001
import argparse
from functools import cache
import torch
import matplotlib.pyplot as plt
import os
import cProfile, pstats, io
from dataset_with_models import RandomModelSampler
from tictoc import tic, tocl
from datetime import datetime
import argparse

from gp_acquisition_dataset import add_gp_acquisition_dataset_args, add_lamda_args, create_train_test_gp_acq_datasets_helper, get_gp_acquisition_dataset_configs, get_lamda_min_max
# AcquisitionFunctionNetV3, AcquisitionFunctionNetV4,
#     AcquisitionFunctionNetDense
from acquisition_function_net import (
    AcquisitionFunctionBodyPointnetV1and2, AcquisitionFunctionNet, AcquisitionFunctionNetFinalMLP, AcquisitionFunctionNetFinalMLPSoftmaxExponentiate, ExpectedImprovementAcquisitionFunctionNet, GittinsAcquisitionFunctionNet, AcquisitionFunctionNetAcquisitionFunction, TwoPartAcquisitionFunctionNetFixedHistoryOutputDim)
from exact_gp_computations import calculate_EI_GP
from train_acquisition_function_net import (
    METHODS,
    load_model,
    print_stats,
    save_af_net_configs,
    train_acquisition_function_net,
    train_or_test_loop)
from utils import DEVICE, load_json, save_json
from plot_utils import plot_nn_vs_gp_acquisition_function_1d_grid, plot_acquisition_function_net_training_history
from nn_utils import count_trainable_parameters, count_parameters

import logging
logging.basicConfig(level=logging.WARNING)

##################### Settings for this script #################################
# Whether to fit maximum a posteriori GP for testing
FIT_MAP_GP = False

CPROFILE = False
TIME = True
VERBOSE = True

############################# Settings for datasets ############################
from gp_acquisition_dataset import (
    GET_TRAIN_TRUE_GP_STATS,
    GET_TEST_TRUE_GP_STATS,
    GP_GEN_DEVICE,
    FIX_TRAIN_ACQUISITION_DATASET
)


def get_training_config(args: argparse.Namespace):
    training_config = dict(
        method=args.method,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        fix_train_acquisition_dataset=FIX_TRAIN_ACQUISITION_DATASET,
        early_stopping=args.early_stopping,
        use_maxei=args.use_maxei
    )
    if args.method == 'policy_gradient':
        training_config = dict(
            **training_config,
            include_alpha=args.include_alpha,
            learn_alpha=args.learn_alpha,
            initial_alpha=args.initial_alpha,
            alpha_increment=args.alpha_increment)
    elif args.method == 'gittins':
        lamda_min, lamda_max = get_lamda_min_max(args)
        training_config = dict(
            **training_config,
            lamda_min=lamda_min,
            lamda_max=lamda_max,
            normalize_gi_loss=args.normalize_gi_loss)
    elif args.method == 'mse_ei':
        initial_tau = getattr(args, "initial_tau", None)
        initial_tau = 1.0 if initial_tau is None else initial_tau
        training_config = dict(
            **training_config,
            learn_tau=args.learn_tau,
            initial_tau=initial_tau,
            softplus_batchnorm=args.softplus_batchnorm,
            softplus_batchnorm_momentum=args.softplus_batchnorm_momentum,
            positive_linear_at_end=args.positive_linear_at_end,
            gp_ei_computation=args.gp_ei_computation)
    if args.early_stopping:
        training_config = dict(
            **training_config,
            patience=args.patience,
            min_delta=args.min_delta,
            cumulative_delta=args.cumulative_delta)
            
    return training_config


def get_model(args: argparse.Namespace):
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

    if args.method == 'gittins':
        model = GittinsAcquisitionFunctionNet(
            af_class=TwoPartAcquisitionFunctionNetFixedHistoryOutputDim,
            variable_lambda=args.lamda is None,
            costs_in_history=False,
            cost_is_input=False,
            af_body_class=AcquisitionFunctionBodyPointnetV1and2,
            af_head_class=AcquisitionFunctionNetFinalMLP,
            af_body_init_params=af_body_init_params,
            af_head_init_params=af_head_init_params,
            standardize_outcomes=args.standardize_nn_history_outcomes
        )
    elif args.method == 'policy_gradient' or args.method == 'mse_ei':
        initial_tau = getattr(args, "initial_tau", None)
        initial_tau = 1.0 if initial_tau is None else initial_tau
        af_head_init_params = dict(
            **af_head_init_params,
            include_alpha=args.include_alpha,
            learn_alpha=args.learn_alpha,
            initial_alpha=args.initial_alpha,
            initial_beta=1.0 / initial_tau,
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
        )
    
    # model = AcquisitionFunctionNetV4(args.dimension,
    #                                 history_enc_hidden_dims=[32, 32], pooling="max",
    #                 include_local_features=True,
    #                 encoded_history_dim=4, include_mean=False,
    #                 mean_enc_hidden_dims=[32, 32], mean_dim=1,
    #                 std_enc_hidden_dims=[32, 32], std_dim=32,
    #                 aq_func_hidden_dims=[32, 32], layer_norm=True,
    #                 layer_norm_at_end_mlp=False,
    #                 include_alpha=args.include_alpha and policy_gradient_flag,
    #                                 learn_alpha=args.learn_alpha,
    #                                 initial_alpha=args.initial_alpha)

    # model = AcquisitionFunctionNetV3(args.dimension, pooling="max",
    #                 history_enc_hidden_dims=[args.layer_width, args.layer_width],
    #                 encoded_history_dim=args.layer_width,
    #                 mean_enc_hidden_dims=[args.layer_width, args.layer_width], mean_dim=1,
    #                 std_enc_hidden_dims=[args.layer_width, args.layer_width], std_dim=16,
    #                 aq_func_hidden_dims=[args.layer_width, args.layer_width], layer_norm=False,
    #                 layer_norm_at_end_mlp=False, include_y=True,
    #                 include_alpha=args.include_alpha and policy_gradient_flag,
    #                                 learn_alpha=args.learn_alpha,
    #                                 initial_alpha=args.initial_alpha)

    # model = AcquisitionFunctionNetDense(args.dimension, MAX_HISTORY,
    #                                     hidden_dims=[128, 128, 64, 32],
    #                                     include_alpha=args.include_alpha and policy_gradient_flag,
    #                                     learn_alpha=args.learn_alpha,
    #                                     initial_alpha=args.initial_alpha)

    return model


def get_configs_and_model_and_paths(args: argparse.Namespace):
    #### Get AF dataset configs
    gp_af_dataset_configs = get_gp_acquisition_dataset_configs(
        args, device=GP_GEN_DEVICE)

    # Exp technically works, but Power does not
    # Make sure to set these appropriately depending on whether the transform
    # supports mean transform
    # if dataset_transform_config['outcome_transform'] is not None:
    #     GET_TRAIN_TRUE_GP_STATS = False
    #     GET_TEST_TRUE_GP_STATS = False

    ################################### Get NN model ###############################
    #### Get the untrained model
    # This wastes some resources, but need to do it to get the model's init dict to
    # obtain the correct path for saving the model because that is currently how the
    # model is uniquely identified.
    model = get_model(args)

    #### Save the configs for the model and training and datasets
    model_and_info_folder_name, models_path = save_af_net_configs(
        model,
        training_config=get_training_config(args),
        af_dataset_config=gp_af_dataset_configs,
        save=getattr(args, "save_model", False)
    )

    return gp_af_dataset_configs, model, model_and_info_folder_name, models_path


def run_train(args: argparse.Namespace):
    ######################## Check the arguments ###############################
    # Only have include_alpha=True when method=policy_gradient
    policy_gradient_flag = (args.method == 'policy_gradient')
    if not policy_gradient_flag and args.include_alpha:
        raise ValueError("include_alpha should be True only if method=policy_gradient")

    if args.train:
        if args.learning_rate is None:
            raise ValueError("learning_rate should be specified if training the model")
        if args.epochs is None:
            raise ValueError("epochs should be specified if training the model")
    else:
        args.save_model = False
    
    for reason, reason_desc in [(args.method != 'mse_ei', 'method != mse_ei')]:
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
    if args.method == 'gittins':
        if lamda_given:
            if lamda_min_given or lamda_max_given:
                raise ValueError(
                    "If method=gittins, should specify only either lamda, or both lamda_min and lamda_max")
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
    else: # method = 'mse_ei' or 'policy_gradient'
        if args.initial_tau is None:
            args.initial_tau = 1.0
        if lamda_given or lamda_min_given or lamda_max_given:
            raise ValueError(
                "If method != gittins, then lamda, lamda_min, and lamda_max should not be specified")
        if args.normalize_gi_loss:
            raise ValueError("normalize_gi_loss should be False if method != gittins")

    (af_dataset_configs, model,
     model_and_info_folder_name, models_path) = get_configs_and_model_and_paths(args)

    if args.load_saved_model:
        model, model_path = load_model(
            model_and_info_folder_name, return_model_path=True)
    else:
        model_path = None

    model = model.to(DEVICE)

    print(model)
    print("Number of trainable parameters:", count_trainable_parameters(model))
    print("Number of parameters:", count_parameters(model))

    ####################### Make the train and test datasets #######################
    (train_aq_dataset, test_aq_dataset,
     small_test_aq_dataset) = create_train_test_gp_acq_datasets_helper(
         args, af_dataset_configs)

    ######################## Train the model #######################################
    if args.train:
        if args.save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{timestamp}"
            model_path = os.path.join(models_path, model_name)
            print(f"Saving NN to {model_and_info_folder_name}")
        else:
            model_path = None

        if CPROFILE:
            pr = cProfile.Profile()
            pr.enable()
        
        if TIME:
            tic("Training")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                    #  weight_decay=1e-2
                                    )
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        training_history_data = train_acquisition_function_net(
            model, train_aq_dataset, optimizer, args.method, args.epochs, args.batch_size,
            DEVICE, verbose=VERBOSE, n_train_printouts_per_epoch=10,
            alpha_increment=args.alpha_increment,
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
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
            cumulative_delta=args.cumulative_delta,
            use_maxei=args.use_maxei
        )

        if args.save_model:
            latest_model_path = os.path.join(models_path, "latest_model.json")
            save_json({"latest_model": model_name}, latest_model_path)
            print(f"Saved best weights to {model_and_info_folder_name}")

        if TIME:
            tocl()

        if CPROFILE:
            pr.disable()
            
            # s = io.StringIO()
            # ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            # ps.print_stats()
            # print(s.getvalue())

            with open('stats_output.txt', 'w') as s:
                ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
                ps.print_stats()

        if not TIME:
            print("Done training")

    ######################## Evaluate and plot model performance #######################
    if model_path is not None:
        training_history_path = os.path.join(model_path, 'training_history_data.json')

        if not args.train:
            training_history_data = load_json(training_history_path)
            final_test_stats_original = training_history_data['final_test_stats']
            print_stats(final_test_stats_original,
                        "Final test stats on the original test dataset",
                        args.method, args.normalize_gi_loss)

            test_dataloader = test_aq_dataset.get_dataloader(
                        batch_size=args.batch_size, drop_last=False)
            final_test_stats = train_or_test_loop(
                        test_dataloader, model, train=False,
                        nn_device=DEVICE, method=args.method,
                        verbose=False, desc=f"Compute final test stats",
                        get_true_gp_stats=GET_TEST_TRUE_GP_STATS,
                        get_map_gp_stats=False,
                        get_basic_stats=True,
                        alpha_increment=args.alpha_increment,
                        normalize_gi_loss=args.normalize_gi_loss)
            print_stats(final_test_stats,
                        "Final test stats on this test dataset (should be same as above)",
                        args.method, args.normalize_gi_loss)

        history_fig = plot_acquisition_function_net_training_history(training_history_data)
        history_plot_path = os.path.join(model_path, 'training_history.pdf')
        if not os.path.exists(history_plot_path):
            history_fig.savefig(history_plot_path, bbox_inches='tight')

    ######################## Plot performance of model #############################
    ######################## (old useless code)
    # TODO: Fix the below code to work with Gittins index
    plot_stuff = False
    if plot_stuff:
        n_candidates = 2_000
        plot_map = False

        name = "EI" if args.method == "mse_ei" else "acquisition"
        if args.dimension == 1:
            nrows, ncols = 5, 5
            fig, axs = plot_nn_vs_gp_acquisition_function_1d_grid(
                test_aq_dataset, model, policy_gradient_flag, name,
                n_candidates, nrows, ncols,
                plot_map=plot_map, nn_device=DEVICE,
                # If policy_gradient_flag=False, set this to False if it's hard to see some
                # of the plots
                group_standardization=None 
            )
            if model_path is not None:
                fname = f'acqusion_function_net_vs_gp_acquisition_function_1d_grid_{nrows}x{ncols}.pdf'
                path = os.path.join(model_path, fname)
                # Don't want to overwrite the plot if it already exists;
                # it could have been trained on different data from the data we are
                # evaluating it on if args.train=False.
                if not os.path.exists(path):
                    fig.savefig(path, bbox_inches='tight')
        else:
            it = iter(test_aq_dataset)
            item = next(it)
            x_hist, y_hist, x_cand, improvements, gp_model = item
            x_hist_nn, y_hist_nn, x_cand_nn, improvements_nn = item.to(DEVICE).tuple_no_model
            print(f"Number of history points: {x_hist.size(0)}")

            x_cand = torch.rand(n_candidates, args.dimension)

            aq_fn = AcquisitionFunctionNetAcquisitionFunction.from_net(
            model, x_hist_nn, y_hist_nn, exponentiate=(args.method == 'mse_ei'), softmax=False)
            ei_nn = aq_fn(x_cand.to(DEVICE).unsqueeze(1))

            ei_true = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, log=False)
            if plot_map:
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

            if plot_map:
                plt.figure()
                plt.scatter(ei_true.detach().cpu().numpy(), ei_map.detach().cpu().numpy())
                plt.xlabel(f'{name} True')
                plt.ylabel(f'{name} MAP')
                plt.title(f'{name} True vs {name} MAP')


        plt.show()


@cache
def get_run_train_parser():
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

    ################################ Dataset settings ##################################
    dataset_group = parser.add_argument_group("Dataset options")
    add_gp_acquisition_dataset_args(dataset_group)
    
    ############################ NN architecture settings ##############################
    nn_architecture_group = parser.add_argument_group("NN Architecture options")
    nn_architecture_group.add_argument(
        '--layer_width', 
        type=int,
        required=True,
        help='The width of the NN layers.'
    )
    nn_architecture_group.add_argument(
        '--standardize_nn_history_outcomes', 
        action='store_true', 
        help=('Whether to standardize the history outcomes when computing the NN '
            'acquisition function. Default is False.')
    )

    ############################ Training settings #####################################
    training_group = parser.add_argument_group("Training options")
    # Which AF training loss function to use
    training_group.add_argument(
        '--method',
        choices=METHODS,
        required=True,
    )
    training_group.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate for training the model'
    )
    training_group.add_argument(
        '--batch_size',
        type=int,
        required=True,
        help='Batch size for training the model'
    )
    training_group.add_argument(
        '--epochs',
        type=int,
        help='Maximum number of epochs for training the model',
    )
    ### Early stopping
    training_group.add_argument(
        '--early_stopping', 
        action='store_true', 
        help=('Whether to use early stopping. Default is False.')
    )
    training_group.add_argument(
        '--patience',
        type=int,
        help=('Number of epochs with no improvement after which training will be stopped. '
            'Only used if early_stopping=True.')
    )
    training_group.add_argument(
        '--min_delta',
        type=float,
        help=('Minimum change in the monitored quantity to qualify as an improvement. '
            'Only used if early_stopping=True.')
    )
    training_group.add_argument(
        '--cumulative_delta',
        action='store_true',
        help=('Whether to use cumulative delta for early stopping. Default is False. '
            'Only used if early_stopping=True.')
    )
    training_group.add_argument(
        '--use_maxei',
        action='store_true',
        help='Use --use_maxei to use the "max ei" statistic as the evaluation metric '
        'on the test dataset. Otherwise, the evaluation metric is the same as the loss '
        'function used to train the NN.'
    )

    #### Options when method=policy_gradient
    policy_gradient_group = parser.add_argument_group(
        "Training options when method=policy_gradient")
    policy_gradient_group.add_argument(
        '--include_alpha', 
        action='store_true', 
        help='Whether to include alpha. Only used if method=policy_gradient.'
    )
    policy_gradient_group.add_argument(
        '--learn_alpha', 
        action='store_true', 
        help=('Whether to learn alpha. Default is True. Only used if '
            'method=policy_gradient and include_alpha=true.')
    )
    policy_gradient_group.add_argument(
        '--initial_alpha',
        type=float,
        help=('Initial value of alpha. Default is 1.0. Only used if '
              'method=policy_gradient and include_alpha=true.'),
        default=1.0
    )
    policy_gradient_group.add_argument( # default is None, equivalent to 0.0
        '--alpha_increment',
        type=float,
        help=('Increment for alpha. Default is 0.0. Only used if method=policy_gradient'
            ' and include_alpha=true.')
    )

    #### Options when method=gittins
    gittins_group = parser.add_argument_group(
        "Training options when method=gittins")
    gittins_group.add_argument(
        '--normalize_gi_loss', 
        action='store_true', 
        help=('Whether to normalize the Gittins index loss function. Default is False. '
            'Only used if method=gittins.')
    )
    add_lamda_args(gittins_group)

    #### Options for NN when method=mse_ei
    mse_ei_group = parser.add_argument_group(
        "Training options when method=mse_ei")
    mse_ei_group.add_argument(
        '--learn_tau', 
        action='store_true',
        help=('Set this flag to enable learning of tau=1/beta which is the parameter for softplus'
            ' applied at the end of the MSE acquisition function. Default is False. '
            'Only used if method=mse_ei.')
    )
    mse_ei_group.add_argument(
        '--initial_tau',
        type=float,
        help='Initial value of tau. Default is 1.0. Only used if method=mse_ei.'
    )
    mse_ei_group.add_argument(
        '--softplus_batchnorm',
        action='store_true',
        help=('Set this flag to apply positive-batchnorm after softplus in the MSE acquisition function. '
            'Default is False. Only used if method=mse_ei.')
    )
    mse_ei_group.add_argument(
        '--softplus_batchnorm_momentum',
        type=float,
        default=0.1,
        help=('Momentum for the batchnorm after softplus in the MSE acquisition function. Default is 0.1. '
            'Only used if method=mse_ei.')
    )
    mse_ei_group.add_argument(
        '--positive_linear_at_end',
        action='store_true',
        help=('Set this flag to apply positive linear at end technique. Default is False. '
            'Only used if method=mse_ei.')
    )
    mse_ei_group.add_argument(
        '--gp_ei_computation',
        action='store_true',
        help=('Set this flag to apply gp_ei_computation at end technique. Default is False. '
            'Only used if method=mse_ei.')
    )

    return parser


def main():
    parser = get_run_train_parser()
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
