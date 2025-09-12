# Run like, e.g.,
# python run_train.py --dimension 8 --test_expansion_factor 1 --kernel Matern52 --lengthscale 0.1 --max_history 400 --min_history 1 --test_samples_size 5000 --test_n_candidates 1 --train_samples_size 10000 --train_acquisition_size 2000 --train_n_candidates 1 --batch_size 32 --early_stopping --epochs 200 --layer_width 100 --learning_rate 0.0003 --method gittins --min_delta 0.0 --gi_loss_normalization normal --patience 5 --lamda 0.001 --replacement
from functools import lru_cache
from typing import Optional, Sequence
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cProfile, pstats
from datetime import datetime

from nn_af.acquisition_function_net_save_utils import get_lamda_for_bo_of_nn
from utils.exact_gp_computations import calculate_EI_GP
from utils.utils import DEVICE, load_json, save_json
from utils.plot_utils import (
    plot_acquisition_function_net_training_history_ax,
    plot_nn_vs_gp_acquisition_function_1d_grid,
    plot_acquisition_function_net_training_history)
from utils.nn_utils import count_trainable_parameters, count_parameters
from utils.tictoc import tic, tocl

from dataset_factory import create_train_test_acquisition_datasets_from_args

from nn_af.acquisition_function_net_save_utils import (
    get_nn_af_args_configs_model_paths_from_cmd_args, load_nn_acqf)
from nn_af.acquisition_function_net import AcquisitionFunctionNetAcquisitionFunction
from nn_af.train_acquisition_function_net import (
    print_stats, train_acquisition_function_net, train_or_test_loop)

import logging
logging.basicConfig(level=logging.WARNING)

##################### Settings for this script #################################
# Whether to fit maximum a posteriori GP for testing
FIT_MAP_GP = False
SAVE_INCREMENTAL_BEST_MODELS = False
CPROFILE = False
TIME = True
VERBOSE = True

############################# Settings for datasets ############################
from gp_acquisition_dataset import (
    GET_TRAIN_TRUE_GP_STATS,
    GET_TEST_TRUE_GP_STATS
)


def run_train(cmd_args: Optional[Sequence[str]]=None):
    (args, af_dataset_configs,
     model, model_and_info_folder_name, models_path
    ) = get_nn_af_args_configs_model_paths_from_cmd_args(cmd_args)

    if args.load_saved_model:
        model, model_path = load_nn_acqf(
            model_and_info_folder_name, return_model_path=True)
    else:
        model_path = None

    model = model.to(DEVICE)

    print(model)
    print("Number of trainable parameters:", count_trainable_parameters(model))
    print("Number of parameters:", count_parameters(model))
    print(f"\nSaving model and configs to {model_and_info_folder_name}\n")

    ####################### Make the train and test datasets #######################
    (train_aq_dataset, test_aq_dataset,
     small_test_aq_dataset) = create_train_test_acquisition_datasets_from_args(args)

    ######################## Train the model #######################################
    if args.train:
        if args.save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{timestamp}"
            model_path = os.path.join(models_path, model_name)
        else:
            model_path = None

        if CPROFILE:
            pr = cProfile.Profile()
            pr.enable()
        
        if TIME:
            tic("Training")
        
        print(f"learning rate: {args.learning_rate}, batch size: {args.batch_size}")
        print(f"dimension: {args.dimension}, lengthscale: {args.lengthscale}")
        weight_decay = 0.0 if args.weight_decay is None else args.weight_decay
        c = torch.optim.AdamW if weight_decay > 0 else torch.optim.Adam
        optimizer = c(model.parameters(), lr=args.learning_rate,
                        weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        training_history_data = train_acquisition_function_net(
            model, train_aq_dataset, optimizer, args.method, args.epochs, args.batch_size,
            DEVICE, verbose=VERBOSE, n_train_printouts_per_epoch=10,
            alpha_increment=args.alpha_increment,
            gi_loss_normalization=args.gi_loss_normalization,
            test_dataset=test_aq_dataset, small_test_dataset=small_test_aq_dataset,
            get_train_stats_while_training=True,
            get_train_stats_after_training=True,
            ## These both default to reasonable values depending on whether the
            ## acquisition datasets are fixed
            get_train_true_gp_stats=GET_TRAIN_TRUE_GP_STATS,
            get_test_true_gp_stats=GET_TEST_TRUE_GP_STATS,
            save_dir=model_path,
            save_incremental_best_models=SAVE_INCREMENTAL_BEST_MODELS and args.save_model,
            # early stopping
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
            cumulative_delta=args.cumulative_delta,
            # learning rate scheduler
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_patience=args.lr_scheduler_patience,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_min_lr=args.lr_scheduler_min_lr,
            lr_scheduler_cooldown=args.lr_scheduler_cooldown,
            lr_scheduler_power=args.lr_scheduler_power,
            lr_scheduler_burnin=args.lr_scheduler_burnin,
            # evaluation metric
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
    else:
        training_history_data = None

    ######################## Evaluate and plot model performance #######################
    if not args.train and model_path is not None:
        training_history_path = os.path.join(
            model_path, 'training_history_data.json')
        training_history_data = load_json(training_history_path)
        final_test_stats_original = training_history_data['final_test_stats']
        print_stats(final_test_stats_original,
                    "Final test stats on the original test dataset",
                    args.method, args.gi_loss_normalization)

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
                    gi_loss_normalization=args.gi_loss_normalization)
        print_stats(final_test_stats,
                    "Final test stats on this test dataset (should be same as above)",
                    args.method, args.gi_loss_normalization)
    
    if training_history_data is not None:
        history_fig = plot_acquisition_function_net_training_history(
            training_history_data, plot_log_regret=True)
        if model_path is not None:
            history_plot_path = os.path.join(model_path, 'training_history.pdf')
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
                aq_dataset=test_aq_dataset, nn_model=model, plot_name=name,
                n_candidates=n_candidates, nrows=nrows, ncols=ncols,
                method=args.method,
                min_x=0., max_x=1.,
                lamda=get_lamda_for_bo_of_nn(args.lamda, args.lamda_min, args.lamda_max),
                plot_map=plot_map, nn_device=DEVICE,
                group_standardization=None # May manually set this to True or False
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

            kwargs = dict(exponentiate=True, softmax=False) if args.method == "mse_ei" else dict()
            aq_fn = AcquisitionFunctionNetAcquisitionFunction.from_net(
                model, x_hist_nn, y_hist_nn, **kwargs)
            ei_nn = aq_fn(x_cand.to(DEVICE).unsqueeze(1))

            ei_true = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, log=False)
            if plot_map:
                ei_map = calculate_EI_GP(gp_model, x_hist, y_hist, x_cand, mle=False, log=False)

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


if __name__ == "__main__":
    run_train()
