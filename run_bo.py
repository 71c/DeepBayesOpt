from functools import cache
import math
import os
from typing import Any
import torch
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement
from botorch.utils.sampling import draw_sobol_samples
from botorch.exceptions import UnsupportedError
import argparse
from acquisition_function_net import GittinsAcquisitionFunctionNet
from bayesopt import GPAcquisitionOptimizer, NNAcquisitionOptimizer, OptimizationResultsSingleMethod, get_rff_function_and_name, outcome_transform_function
from dataset_with_models import RandomModelSampler
from gp_acquisition_dataset import GP_GEN_DEVICE, add_gp_args, get_gp_model_from_args_no_outcome_transform, get_outcome_transform
from stable_gittins import StableGittinsIndex
from train_acquisition_function_net import load_configs, load_model, model_is_trained
from utils import add_outcome_transform, dict_to_cmd_args, dict_to_fname_str, dict_to_str


script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(script_dir, 'bayesopt_results')


GP_AF_NAME_PREFIX = "gp_af"
OBJECTIVE_NAME_PREFIX = "objective"


GP_AF_DICT = {
    'LogEI': LogExpectedImprovement,
    'EI': ExpectedImprovement,
    'gittins': StableGittinsIndex
}


def get_arg_names(p) -> list[str]:
    return [action.dest for action in p._group_actions if action.dest != "help"]


def add_bo_loop_args(parser):
    parser.add_argument(
        '--n_iter',
        type=int,
        help='Number of iterations of BO to perform',
        required=True
    )
    parser.add_argument(
        '--n_initial_samples',
        type=int,
        help='Number of initial sobol points to sample at before using the AF',
        required=True
    )


@cache
def get_bo_loop_args_parser():
    parser = argparse.ArgumentParser()
    ################## Objective function (can only be a GP for now) ###################
    objective_function_group = parser.add_argument_group("Objective function")
    objective_function_group.add_argument(
        f'--{OBJECTIVE_NAME_PREFIX}_dimension', 
        type=int, 
        help='Dimension of the objective function',
        required=True
    )
    objective_function_group.add_argument(
        f'--{OBJECTIVE_NAME_PREFIX}_gp_seed',
        type=int, 
        help='Seed for the random GP draw (the objective function)',
        required=True
    )
    add_gp_args(objective_function_group, "objective function",
                name_prefix=OBJECTIVE_NAME_PREFIX,
                required=True, add_randomize_params=True)

    ###################################### BO Policy ###################################
    bo_policy_group = parser.add_argument_group("BO policy")
    add_bo_loop_args(bo_policy_group)
    bo_policy_group.add_argument(
        '--bo_seed',
        type=int,
        help='Seed for the BO loop',
        required=True
    )
    bo_policy_group.add_argument(
        '--lamda',
        type=float,
        help='Value of lambda. Only used if using a Gittins index policy.'
    )
    #### Option 1: policy is to use a NN AF
    bo_policy_group.add_argument(
        '--nn_model_name', 
        type=str, 
        help="Folder name of a NN AF, e.g., "
        "v1/model_5ca3a4cd249655a70da4975316e4da542470e087a6181b4a12532cbfe169ae9d. "
        "Either nn_model_name should be specified or the arguments under 'GP-based AF' "
        "should be specified."
    )
    gp_af_group = parser.add_argument_group("GP-based AF")
    #### Option 2: policy is to use a GP-based AF
    gp_af_group.add_argument(
        f"--{GP_AF_NAME_PREFIX}",
        choices=list(GP_AF_DICT),
        help="If using a GP-based AF, the AF to use"
    )
    gp_af_group.add_argument(
        f"--{GP_AF_NAME_PREFIX}_fit",
        choices=["map", "mle", "exact"],
        help="If using a GP-based AF, the method of fitting the GP. "
            "map for maximum a posteriori, mle for maximum likelihood, "
            "exact for using the true GP parameters. "
            "If unspecified, no GP fitting is used. "
            f"If {GP_AF_NAME_PREFIX}_fit=exact, then the arguments under "
            f"'GP-based AF' besides {GP_AF_NAME_PREFIX} and {GP_AF_NAME_PREFIX}_fit "
            f"should not be specified."
    )
    add_gp_args(gp_af_group, "GP-based AF", name_prefix=GP_AF_NAME_PREFIX,
                required=False, add_randomize_params=False)

    return {
        'parser': parser,
        'objective_function_group': objective_function_group,
        'bo_policy_group': bo_policy_group,
        'gp_af_group': gp_af_group
    }


# Cache the objective function things.
# This greatly speeds up the script bo_experiments_gp.py
# that generates the commands for the BO loops.
# Otherwise, it takes too long to run get_rff_function_and_name many times.
@cache
def _get_gp_objective_things_helper(
    dimension, kernel, lengthscale, randomize_params, gp_seed):
    # Get GP model sampler
    objective_gp_base_model = get_gp_model_from_args_no_outcome_transform(
        dimension=dimension,
        kernel=kernel,
        lengthscale=lengthscale,
        add_priors=randomize_params,
        device=GP_GEN_DEVICE
    )
    objective_gp_sampler = RandomModelSampler(
        [objective_gp_base_model],
        randomize_params=randomize_params
    )
    # Seed
    torch.manual_seed(gp_seed)
    # Get (potentially) random GP parameters
    objective_gp = objective_gp_sampler.sample(deepcopy=True).eval()
    # Get random GP draw
    objective_fn, realization_hash = get_rff_function_and_name(
        objective_gp, dimension=dimension)
    
    return objective_gp, objective_fn, realization_hash


def _get_gp_objective_things(objective_args):
    objective_gp, objective_fn, realization_hash = _get_gp_objective_things_helper(
        dimension=objective_args['dimension'],
        kernel=objective_args['kernel'],
        lengthscale=objective_args['lengthscale'],
        randomize_params=objective_args['randomize_params'],
        gp_seed=objective_args['gp_seed']
    )
    objective_name = f'gp_{realization_hash}'
    # Apply outcome transform to the objective function
    objective_octf, objective_octf_args = get_outcome_transform(
        argparse.Namespace(**objective_args),
        name_prefix=OBJECTIVE_NAME_PREFIX,
        device=GP_GEN_DEVICE)
    if objective_octf is not None:
        octf_str = dict_to_fname_str(objective_octf_args)
        objective_name = f'{objective_name}_{octf_str}'
        objective_fn = outcome_transform_function(objective_fn, objective_octf)
    
    return objective_gp, objective_octf, objective_fn, objective_name


def _get_sobol_samples_and_bounds(bo_seed, n_initial_samples, dimension):
    bounds = torch.stack([torch.zeros(dimension), torch.ones(dimension)])
    torch.manual_seed(bo_seed)
    init_x = draw_sobol_samples(
        bounds=bounds,
        n=1, # Number of BO loops to do
        q=n_initial_samples # Number of sobol points
    )
    return init_x, bounds


def run_bo(objective_args: dict[str, Any],
           bo_policy_args: dict[str, Any],
           gp_af_args: dict[str, Any]):
    (objective_gp, objective_octf,
     objective_fn, objective_name) = _get_gp_objective_things(objective_args)
    dimension = objective_args['dimension']
    ############################# Determine the BO policy ##############################
    nn_model_name = bo_policy_args.get('nn_model_name')
    
    lamda = bo_policy_args['lamda']
    if lamda is not None and lamda <= 0:
        raise ValueError("lamda must be > 0")
    
    results_print_data = {'dimension': dimension}

    if nn_model_name is None: # Using a GP AF
        fit = gp_af_args.get('fit')
        #### Determine the GP model to be used for the AF
        if fit == "exact":
            ## Determine af_gp_model from the GP used for the objective
            for k, v in gp_af_args.items():
                if not (k == "fit" or k == GP_AF_NAME_PREFIX):
                    if v is not None:
                        raise ValueError(f"Cannot specify {GP_AF_NAME_PREFIX}_{k} "
                                         f"if {GP_AF_NAME_PREFIX}_fit=exact")
            af_gp_model = objective_gp
            af_octf = objective_octf
        else:
            ## Determine af_gp_model from the GP args
            af_kernel = gp_af_args['kernel']
            af_lengthscale = gp_af_args['lengthscale']
            if af_kernel is None:
                raise ValueError(
                    f"If not using a NN AF and {GP_AF_NAME_PREFIX}_fit != exact, "
                    f"must specify {GP_AF_NAME_PREFIX}_kernel")
            if af_lengthscale is None:
                raise ValueError(
                    f"If not using a NN AF and {GP_AF_NAME_PREFIX}_fit != exact, "
                    f"must specify {GP_AF_NAME_PREFIX}_lengthscale")
            
            # Add priors if using MAP. If using MLE or no fitting, don't add priors
            add_gp_af_priors_flag = fit == "map"
            
            af_gp_model = get_gp_model_from_args_no_outcome_transform(
                dimension=dimension,
                kernel=af_kernel,
                lengthscale=af_lengthscale,
                add_priors=add_gp_af_priors_flag,
                device=GP_GEN_DEVICE
            )
            af_octf, af_octf_args = get_outcome_transform(
                argparse.Namespace(**gp_af_args),
                name_prefix=GP_AF_NAME_PREFIX,
                device=GP_GEN_DEVICE)
        
        #### Apply outcome transform to the AF GP model
        if af_octf is not None:
            add_outcome_transform(af_gp_model, af_octf)

        gp_af = gp_af_args[GP_AF_NAME_PREFIX]
        if gp_af is None:
            raise ValueError("If not using a NN AF, must specify gp_af")

        af_options = dict(
            optimizer_kwargs_per_function=[{'model': af_gp_model}],
            acquisition_function_class=GP_AF_DICT[gp_af],
            fit_params=fit in {'map', 'mle'}
        )
        
        if gp_af == 'gittins':
            if lamda is None:
                raise ValueError("If using Gittins index, must specify lamda")
            af_options['lmbda'] = lamda
            results_print_data['lambda'] = lamda
        else:
            if lamda is not None:
                raise ValueError("If not using Gittins index, cannot specify lamda")

        optimizer_class = GPAcquisitionOptimizer

        results_print_data = {**results_print_data, 'GP AF': gp_af, 'fit': fit}
    else: # Using a NN AF
        for k, v in gp_af_args.items():
            if v is not None:
                raise ValueError(
                    f"Cannot specify {GP_AF_NAME_PREFIX}_{k} if using a NN AF")
        
        if not model_is_trained(nn_model_name):
            return None
        
        nn_model = load_model(nn_model_name)

        # TODO (maybe): provide exponentiate=False or exponentiate=True here
        # for ExpectedImprovementAcquisitionFunctionNet?
        af_options = dict(
            model=nn_model,
            nn_model_name=nn_model_name
        )

        results_print_data = {**results_print_data, 'NN': nn_model_name}

        if isinstance(nn_model, GittinsAcquisitionFunctionNet):
            if nn_model.costs_in_history:
                raise UnsupportedError("nn_model.costs_in_history=True is currently not"
                                       " supported for Gittins index optimization")
            if nn_model.cost_is_input:
                raise UnsupportedError("nn_model.cost_is_input=True is currently not"
                                       " supported for Gittins index optimization")

            configs = load_configs(nn_model_name)
            train_config = configs['training_config']
            lamda_min, lamda_max = train_config['lamda_min'], train_config['lamda_max']

            if nn_model.variable_lambda:
                if lamda is None:
                    raise ValueError(
                        "If using a Gittins index NN AF architecture "
                        "that has variable_lambda=True, must specify lamda "
                        f"(should be between {lamda_min=} and {lamda_max=})")
                if not (lamda_min <= lamda <= lamda_max):
                    raise ValueError(
                        f"lamda should be between {lamda_min=} and {lamda_max=}")
                af_options['lambda_cand'] = math.log(lamda)
                af_options['is_log'] = True
            else:
                if lamda is not None:
                    if lamda != lamda_min:
                        raise ValueError(
                            f"lamda should be lamda={lamda_min} if using this Gittins "
                            "index NN AF architecture that has variable_lambda=False, "
                            f"but got lamda={lamda}")
                else:
                    lamda = lamda_min
            
            results_print_data['lambda'] = lamda
            results_print_data['variable_lambda'] = nn_model.variable_lambda
        else:
            if lamda is not None:
                raise ValueError(
                    "Cannot specify lamda if not using Gittins index NN AF")

        optimizer_class = NNAcquisitionOptimizer

    results_name = dict_to_str(results_print_data, include_space=True)
    
    bo_seed = bo_policy_args['bo_seed']
    
    init_x, bounds = _get_sobol_samples_and_bounds(
        bo_seed, bo_policy_args['n_initial_samples'], dimension)
    
    # One seed per BO loop. Here, we have n=1 BO loops, so need just 1 seed.
    seeds = [bo_seed]

    return OptimizationResultsSingleMethod(
        objectives=[objective_fn],
        initial_points=init_x,
        n_iter=bo_policy_args['n_iter'],
        seeds=seeds,
        optimizer_class=optimizer_class,
        objective_names=[objective_name],
        save_dir=RESULTS_DIR,
        results_name=results_name, # results_name is only used to print stuff out
        dim=dimension, bounds=bounds, maximize=True,
        **af_options
    )


def bo_loop_dicts_to_cmd_args_list(
        objective_args: dict, bo_policy_args: dict, gp_af_args: dict):
    objective_args_ = {
        f'{OBJECTIVE_NAME_PREFIX}_{k}': v for k, v in objective_args.items()
    }
    gp_af_args_ = {
        (k if k == GP_AF_NAME_PREFIX else f'{GP_AF_NAME_PREFIX}_{k}'): v
        for k, v in gp_af_args.items()
    }
    all_args = {**objective_args_, **bo_policy_args, **gp_af_args_}
    cmd_args_list = dict_to_cmd_args(all_args)
    
    # Just validate that the given args are correctly specified
    # (at least as much as the checks done by the parser)
    parser_info = get_bo_loop_args_parser()
    args = parser_info['parser'].parse_args(cmd_args_list)

    return cmd_args_list


def main():
    parser_info = get_bo_loop_args_parser()
    args = parser_info['parser'].parse_args()

    objective_args = {
        k[len(OBJECTIVE_NAME_PREFIX)+1:]: getattr(args, k)
        for k in get_arg_names(parser_info['objective_function_group'])
    }

    bo_policy_args = {
        k: getattr(args, k)
        for k in get_arg_names(parser_info['bo_policy_group'])
    }

    gp_af_args = {
        (k if k == GP_AF_NAME_PREFIX
         else k[len(GP_AF_NAME_PREFIX)+1:]): getattr(args, k)
        for k in get_arg_names(parser_info['gp_af_group'])
    }

    optimization_results = run_bo(
        objective_args, bo_policy_args, gp_af_args)

    if optimization_results is None:
        raise ValueError("NN model not trained")
    
    # Perform the optimization. Don't do anything with the results here
    for func_name, func_result in optimization_results:
        # There should be just one loop through this since there is just one function
        pass

    # print(f"{objective_args=}")
    # print(f"{bo_policy_args=}")
    # print(f"{gp_af_args=}")
"""
objective_args={'dimension': 8, 'gp_seed': 123, 'kernel': 'Matern52', 'lengthscale': 0.1, 'outcome_transform': None, 'sigma': None, 'randomize_params': False}
bo_policy_args={'n_iter': 30, 'n_initial_samples': 4, 'bo_seed': 99, 'lamda': 0.01, 'nn_model_name': None}
gp_af_args={'gp_af': 'gittins', 'fit': 'exact', 'kernel': None, 'lengthscale': None, 'outcome_transform': None, 'sigma': None}
"""

# LogEI; Objective function is same as GP used for AF, manual specification
# python run_bo.py --objective_dimension 8 --objective_gp_seed 123 --objective_kernel Matern52 --objective_lengthscale 0.1 --n_iter 30 --n_initial_samples 4 --bo_seed 99 --gp_af LogEI --gp_af_kernel Matern52 --gp_af_lengthscale 0.1

# LogEI; Objective function is same as GP used for AF, automatic specification
# python run_bo.py --objective_dimension 8 --objective_gp_seed 123 --objective_kernel Matern52 --objective_lengthscale 0.1 --n_iter 30 --n_initial_samples 4 --bo_seed 99 --gp_af LogEI --gp_af_fit exact

## GP-based Gittins index:
# python run_bo.py --objective_dimension 8 --objective_gp_seed 123 --objective_kernel Matern52 --objective_lengthscale 0.1 --n_iter 30 --n_initial_samples 4 --bo_seed 99 --gp_af gittins --gp_af_fit exact --lamda 0.01


#### EXAMPLE:
## The following is a simple cheap test command for testing mse_ei method:
# python run_train.py --dimension 8 --kernel Matern52  --lengthscale 0.1 --train_samples_size 2000 --train_acquisition_size 2000 --test_samples_size 2000 --train_n_candidates 1 --test_n_candidates 1 --min_history 1 --max_history 60 --layer_width 100 --method mse_ei --learning_rate 0.003 --batch_size 32 --epochs 10
# ====> Saves to v1/model_c9176a1cdf11da57e5d4801812f27622efbb9f182d3d459c7904dba4010cab87
## Next, run the BO on it:
# python run_bo.py --objective_dimension 8 --objective_gp_seed 123 --objective_kernel Matern52 --objective_lengthscale 0.1 --n_iter 30 --n_initial_samples 4 --bo_seed 99 --nn_model_name v1/model_c9176a1cdf11da57e5d4801812f27622efbb9f182d3d459c7904dba4010cab87

# python run_bo.py --objective_dimension 8 --objective_gp_seed 123 --objective_kernel Matern52 --objective_lengthscale 0.1 --n_iter 30 --n_initial_samples 4 --bo_seed 99 --nn_model_name v1/model_6c038ea3a2ae01627595a1ab371a4e0a86411772f33141db9d212bb77d4207d7

## Gittins index, variable lambda:
# python run_train.py --dimension 8 --test_expansion_factor 2 --kernel Matern52 --lengthscale 0.1 --max_history 400 --min_history 1 --test_samples_size 10000 --test_n_candidates 1 --train_samples_size 2000 --train_acquisition_size 2000 --train_n_candidates 1 --batch_size 32 --early_stopping --epochs 200 --layer_width 100 --learning_rate 0.003 --method gittins --min_delta 0.0 --gi_loss_normalization normal --patience 5 --lamda_min 0.001 --lamda_max 1.0
# ===> Saves to v1/model_6388a4fee84e6df0aaea3a018ca0c78caf1667930402fc521e50c77f43d2579d
# python run_bo.py --objective_dimension 8 --objective_gp_seed 123 --objective_kernel Matern52 --objective_lengthscale 0.1 --n_iter 30 --n_initial_samples 4 --bo_seed 99 --nn_model_name v1/model_6388a4fee84e6df0aaea3a018ca0c78caf1667930402fc521e50c77f43d2579d --lamda 0.01


if __name__ == "__main__":
    main()
