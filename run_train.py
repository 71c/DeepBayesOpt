# Run like, e.g.,
# python run_train.py --dimension 8 --expansion_factor 2 --kernel Matern52 --lengthscale 0.1 --max_history 400 --min_history 1 --test_acquisition_size 3000 --test_n_candidates 1 --train_acquisition_size 2000 --train_n_candidates 1 --batch_size 32 --lamda_max 1.0 --lamda_min 0.0001 --layer_width 100 --learning_rate 0.003 --method gittins --normalize_gi_loss
import torch
import matplotlib.pyplot as plt
import os
import cProfile, pstats, io
from pstats import SortKey
from dataset_with_models import RandomModelSampler
from tictoc import tic, tocl

from gp_acquisition_dataset import add_gp_acquisition_dataset_args, create_train_test_gp_acq_datasets_helper, get_gp_acquisition_dataset_configs
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

import logging
logging.basicConfig(level=logging.WARNING)

##################### Settings for this script #################################
script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(script_dir, "saved_models")

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


def run_train(args):
    ########################## Default settings ################################ 
    args.epochs = 200

    # Only used if policy_gradient_flag is True
    args.include_alpha = True
    # Following 3 are only used if both policy_gradient_flag and args.include_alpha are True
    args.learn_alpha = True
    args.initial_alpha = 1.0
    args.alpha_increment = None # equivalent to 0.0

    args.early_stopping = True
    args.patience = 20
    args.min_delta = 0.0
    args.cumulative_delta = False

    ######################## Check the arguments ###############################
    if args.load_saved_model or args.load_saved_dataset_config:
        if args.model_and_info_name is None:
            raise ValueError("model_and_info_name should be specified if load_saved_model or load_saved_dataset_config")
        model_and_info_path = os.path.join(MODELS_DIR, args.model_and_info_name)
    else:
        model_and_info_path = None

    if (not args.load_saved_model) and (args.layer_width is None):
        raise ValueError("layer_width must be specified if load_saved_model=False")
    
    # Set method based on the loaded model if loading a model
    if args.load_saved_model:
        args.method = load_json(
            os.path.join(model_and_info_path, "training_config.json")
        )['method']
    elif args.method is None:
        raise ValueError("method should be specified if not loading a saved model")

    policy_gradient_flag = (args.method == 'policy_gradient')
    args.include_alpha = args.include_alpha and policy_gradient_flag

    if args.train:
        if args.learning_rate is None:
            raise ValueError("learning_rate should be specified if training the model")

    for reason, reason_desc in [(args.method != 'mse_ei', 'method != mse_ei'),
                                (args.load_saved_model, 'load_saved_model=True')]:
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
            args.lamda_min = args.lamda # just to make it easier to pass in as parameter
            variable_lambda = False
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
            variable_lambda = True
    else: # method = 'mse_ei' or 'policy_gradient'
        if args.initial_tau is None:
            args.initial_tau = 1.0
        if lamda_given or lamda_min_given or lamda_max_given:
            raise ValueError(
                "If method != gittins, then lamda, lamda_min, and lamda_max should not be specified")
        if args.normalize_gi_loss:
            raise ValueError("normalize_gi_loss should be False if method != gittins")

    if args.load_saved_dataset_config:
        gp_realization_config, dataset_size_config, n_points_config, \
            dataset_transform_config, model_sampler = load_configs(model_and_info_path)
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
        if args.load_saved_model:
            _model_sampler = RandomModelSampler.load(
                os.path.join(model_and_info_path, "model_sampler"))
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
    (train_aq_dataset,
     test_aq_dataset,
     small_test_aq_dataset) = create_train_test_gp_acq_datasets_helper(
        args, gp_realization_config, dataset_size_config,
            n_points_config, dataset_transform_config)

    ################################### Get NN model ###############################

    if args.load_saved_model:
        model = load_model(model_and_info_path).to(DEVICE)
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

        if args.method == 'gittins':
            model = GittinsAcquisitionFunctionNet(
                af_class=TwoPartAcquisitionFunctionNetFixedHistoryOutputDim,
                variable_lambda=variable_lambda,
                costs_in_history=False,
                cost_is_input=False,
                af_body_class=AcquisitionFunctionBodyPointnetV1and2,
                af_head_class=AcquisitionFunctionNetFinalMLP,
                af_body_init_params=af_body_init_params,
                af_head_init_params=af_head_init_params,
                standardize_outcomes=args.standardize_nn_history_outcomes
            ).to(DEVICE)
        elif args.method == 'policy_gradient' or args.method == 'mse_ei':
            af_head_init_params = dict(
                **af_head_init_params,
                include_alpha=args.include_alpha,
                learn_alpha=args.learn_alpha,
                initial_alpha=args.initial_alpha,
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
        #                 include_alpha=args.include_alpha and policy_gradient_flag,
        #                                 learn_alpha=args.learn_alpha,
        #                                 initial_alpha=args.initial_alpha).to(DEVICE)

        # model = AcquisitionFunctionNetV3(args.dimension, pooling="max",
        #                 history_enc_hidden_dims=[args.layer_width, args.layer_width],
        #                 encoded_history_dim=args.layer_width,
        #                 mean_enc_hidden_dims=[args.layer_width, args.layer_width], mean_dim=1,
        #                 std_enc_hidden_dims=[args.layer_width, args.layer_width], std_dim=16,
        #                 aq_func_hidden_dims=[args.layer_width, args.layer_width], layer_norm=False,
        #                 layer_norm_at_end_mlp=False, include_y=True,
        #                 include_alpha=args.include_alpha and policy_gradient_flag,
        #                                 learn_alpha=args.learn_alpha,
        #                                 initial_alpha=args.initial_alpha).to(DEVICE)

        # model = AcquisitionFunctionNetDense(args.dimension, MAX_HISTORY,
        #                                     hidden_dims=[128, 128, 64, 32],
        #                                     include_alpha=args.include_alpha and policy_gradient_flag,
        #                                     learn_alpha=args.learn_alpha,
        #                                     initial_alpha=args.initial_alpha).to(DEVICE)

    print(model)
    print("Number of trainable parameters:", count_trainable_parameters(model))
    print("Number of parameters:", count_parameters(model))

    ######################## Train the model #######################################
    #### Settings for training
    training_config = dict(
        method=args.method,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        fix_train_acquisition_dataset=FIX_TRAIN_ACQUISITION_DATASET,
        early_stopping=args.early_stopping
    )
    if policy_gradient_flag:
        training_config = dict(
            **training_config,
            include_alpha=args.include_alpha,
            learn_alpha=args.learn_alpha,
            initial_alpha=args.initial_alpha,
            alpha_increment=args.alpha_increment)
    if args.method == 'gittins':
        training_config = dict(
            **training_config,
            lamda_min=args.lamda_min,
            lamda_max=args.lamda_max,
            normalize_gi_loss=args.normalize_gi_loss)
    if args.early_stopping:
        training_config = dict(
            **training_config,
            patience=args.patience,
            min_delta=args.min_delta,
            cumulative_delta=args.cumulative_delta)

    if args.train:
        if CPROFILE:
            pr = cProfile.Profile()
            pr.enable()
        
        if TIME:
            tic("Training!")
        
        if args.save_model:
            # Save the configs for the model and training and datasets
            (model_and_info_folder_name,
            model_and_info_path, model_path) = save_acquisition_function_net_configs(
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
            model, train_aq_dataset, optimizer, args.method, args.epochs, args.batch_size,
            DEVICE, verbose=VERBOSE, n_train_printouts_per_epoch=10,
            alpha_increment=args.alpha_increment,
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
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
            cumulative_delta=args.cumulative_delta,
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


    if model_and_info_path is not None:
        training_history_path = os.path.join(
            model_and_info_path, 'model', 'training_history_data.json')

        if not args.train:
            training_history_data = load_json(training_history_path)
            final_test_stats_original = training_history_data['final_test_stats']
            print_stats(final_test_stats_original,
                        "Final test stats on the original test dataset")

            test_dataloader = test_aq_dataset.get_dataloader(
                        batch_size=args.batch_size, drop_last=False)
            final_test_stats = train_or_test_loop(
                        test_dataloader, model, train=False,
                        nn_device=DEVICE, method=args.method,
                        verbose=False, desc=f"Compute final test stats",
                        get_true_gp_stats=GET_TEST_TRUE_GP_STATS,
                        get_map_gp_stats=False,
                        get_basic_stats=True)
            print_stats(final_test_stats, "Final test stats on this test dataset")


        history_fig = plot_acquisition_function_net_training_history(training_history_data)
        history_plot_path = os.path.join(model_and_info_path, 'model', 'training_history.pdf')
        if not os.path.exists(history_plot_path) or args.load_saved_dataset_config:
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
            fname = f'acqusion_function_net_vs_gp_acquisition_function_1d_grid_{nrows}x{ncols}.pdf'
            path = os.path.join(model_and_info_path, fname)
            # Don't want to overwrite the plot if it already exists;
            # it could have been trained on different data from the data we are
            # evaluating it on if args.train=False.
            if not os.path.exists(path) or args.load_saved_dataset_config:
                fig.savefig(path, bbox_inches='tight')
        else:
            it = iter(test_aq_dataset)
            item = next(it)
            x_hist, y_hist, x_cand, improvements, gp_model = item
            x_hist_nn, y_hist_nn, x_cand_nn, improvements_nn = item.to(DEVICE).tuple_no_model
            print(f"Number of history points: {x_hist.size(0)}")

            x_cand = torch.rand(n_candidates, args.dimension)

            aq_fn = LikelihoodFreeNetworkAcquisitionFunction.from_net(
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


def main():
    import argparse
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
    # learning_rate = 3e-3 if method == 'gittins' else 3e-4
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate for training the model'
    )
    # batch_size = 320 if method == 'mse_ei' else 16
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
    # d = vars(args)
    # l = argparse.Namespace(**d)

    run_train(args)


if __name__ == "__main__":
    main()
