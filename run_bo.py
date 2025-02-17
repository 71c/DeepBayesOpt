import torch
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement
import argparse
from bayesopt import GPAcquisitionOptimizer, get_rff_function_and_name, outcome_transform_function
from dataset_with_models import RandomModelSampler
from gp_acquisition_dataset import GP_GEN_DEVICE, add_gp_args, get_gp_model_from_args_no_outcome_transform, get_outcome_transform
from utils import add_outcome_transform, dict_to_fname_str


GP_AF_NAME_PREFIX = "gp_af"
OBJECTIVE_NAME_PREFIX = "objective"


def get_arg_names(p):
    return [action.dest for action in p._group_actions if action.dest != "help"]


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
    bo_policy_group.add_argument(
        '--n_iter',
        type=int,
        help='Number of iterations of BO to perform',
        required=True
    )
    bo_policy_group.add_argument(
        '--n_initial_samples',
        type=int,
        help='Number of initial sobol points to sample at before using the AF',
        required=True
    )
    bo_policy_group.add_argument(
        '--bo_seed',
        type=int,
        help='Seed for the BO loop',
        required=True
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
        choices=["LogEI", "gittins"],
        help="If using a GP-based AF, the AF to use"
    )
    add_gp_args(gp_af_group, "GP-based AF", name_prefix=GP_AF_NAME_PREFIX,
                required=False, add_randomize_params=False)
    gp_af_group.add_argument(
        f"--{GP_AF_NAME_PREFIX}_fit",
        choices=["map", "mle", "exact"],
        help="If using a GP-based AF, the method of fitting the GP. "
            "map for maximum a posteriori, mle for maximum likelihood, "
            "exact for using the true GP parameters. "
            "If unspecified, no GP fitting is used."
    )

    return parser, objective_function_group, bo_policy_group, gp_af_group


def run_bo(objective_args, bo_policy_args, gp_af_args):
    ######################### Determine the objective function #########################
    dimension = objective_args['dimension']
    objective_randomize_params = objective_args['randomize_params']
    # Get GP model sampler
    objective_gp_base_model = get_gp_model_from_args_no_outcome_transform(
        dimension=dimension,
        kernel=objective_args['kernel'],
        lengthscale=objective_args['lengthscale'],
        add_priors=objective_randomize_params,
        device=GP_GEN_DEVICE
    )
    objective_gp_sampler = RandomModelSampler(
        [objective_gp_base_model],
        randomize_params=objective_randomize_params
    )
    # Seed
    objective_gp_seed = objective_args['gp_seed']
    torch.manual_seed(objective_gp_seed)
    # Get (potentially) random GP parameters
    objective_gp = objective_gp_sampler.sample(deepcopy=False).eval()
    # Get random GP draw
    objective_fn, realization_hash = get_rff_function_and_name(objective_gp)
    function_name = f'gp_{realization_hash}'
    # Apply outcome transform to the objective function
    objective_octf, objective_octf_args = get_outcome_transform(
        argparse.Namespace(**objective_args), device=GP_GEN_DEVICE)
    if objective_octf is not None:
        octf_str = dict_to_fname_str(objective_octf_args)
        function_name = f'{function_name}_{octf_str}'
        objective_fn = outcome_transform_function(objective_fn, objective_octf)

    ############################# Determine the BO policy ##############################
    nn_model_name = bo_policy_args['nn_model_name']
    if nn_model_name is None: # Using a GP AF
        gp_af = gp_af_args[GP_AF_NAME_PREFIX]
        fit = gp_af_args['fit']

        if gp_af is None:
            raise ValueError("If not using a NN AF, must specify gp_af")
        
        #### Determine the GP model to be used for the AF
        if fit == "exact":
            ## Determine af_gp_model from the GP used for the objective
            for k, v in gp_af_args.items():
                if not (k == "fit" or k == GP_AF_NAME_PREFIX):
                    if v is not None:
                        raise ValueError(f"Cannot specify {GP_AF_NAME_PREFIX}_{k} "
                                         "if using exact GP model AF")
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
                argparse.Namespace(**gp_af_args), device=GP_GEN_DEVICE)
        
        #### Apply outcome transform to the AF GP model
        if af_octf is not None:
            add_outcome_transform(af_gp_model, af_octf)

        # TODO: Fix this
        af_options = dict(
            optimizer_class=GPAcquisitionOptimizer,
            optimizer_kwargs_per_function=[{'model': af_gp_model}],
            acquisition_function_class=LogExpectedImprovement,
            fit_params=fit in {'map', 'mle'}
        )
    else: # Using a NN AF
        # TODO
        pass
    
    # TODO: Finish this


def main():
    parser, objective_function_group, bo_policy_group, gp_af_group = get_bo_loop_args_parser()
    args = parser.parse_args()

    objective_args = {
        k[len(OBJECTIVE_NAME_PREFIX)+1:]: getattr(args, k)
        for k in get_arg_names(objective_function_group)
    }

    bo_policy_args = {
        k: getattr(args, k)
        for k in get_arg_names(bo_policy_group)
    }

    gp_af_args = {
        (k if k == GP_AF_NAME_PREFIX
         else k[len(GP_AF_NAME_PREFIX)+1:]): getattr(args, k)
        for k in get_arg_names(gp_af_group)
    }

    run_bo(objective_args, bo_policy_args, gp_af_args)

    # print(f"{objective_args=}")
    # print(f"{bo_policy_args=}")
    # print(f"{gp_af_args=}")
#     objective_args={'dimension': 8, 'gp_seed': 103, 'kernel': 'Matern52', 'lengthscale': 0.1, 'outcome_transform': None, 'sigma': None, 'randomize_params': False}
# bo_policy_args={'n_iter': 30, 'n_initial_samples': 1, 'bo_seed': 12, 'nn_model_name': None}
# gp_af_args={'gp_af': None, 'kernel': 'Matern52', 'lengthscale': None, 'outcome_transform': None, 'sigma': None, 'fit': None}

    

if __name__ == "__main__":
    # Example test command:
    # python run_bo.py --objective_dimension 8 --objective_gp_seed 103 --objective_lengthscale 0.1 --n_iter 30 --n_initial_samples 1 --bo_seed 12 --objective_kernel RBF --gp_af LogEI --gp_af_kernel RBF --gp_af_lengthscale 0.1
    main()
