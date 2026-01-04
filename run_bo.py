from collections import defaultdict
import copy
from functools import cache
import itertools
import math
from typing import Any
import argparse
import yaml

from run_train_transfer_bo_baseline import TRANSFER_BO_BASELINE_NAMES, transfer_bo_baseline_is_trained
import torch
from torch import Tensor
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement
from botorch.utils.sampling import draw_sobol_samples
from botorch.exceptions import UnsupportedError
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch.utils.sampling import optimize_posterior_samples

from dataset_factory import add_unified_function_dataset_args, validate_args_for_dataset_type
from transfer_bo_baselines.fsbo.fsbo_optimizer import FSBOOptimizer
from utils.utils import (add_outcome_transform, remove_priors, get_arg_names)
from utils.constants import RESULTS_DIR
from nn_af.acquisition_function_net_save_utils import load_nn_acqf_configs

from nn_af.acquisition_function_net import GittinsAcquisitionFunctionNet
from nn_af.acquisition_function_net_save_utils import load_nn_acqf, nn_acqf_is_trained
from datasets.dataset_with_models import RandomModelSampler
from datasets.hpob_dataset import get_hpob_dataset_dimension, get_hpob_function_min_max, get_hpob_initialization, get_hpob_objective_function
from datasets.cancer_dosage_dataset import CancerDosageObjectiveSampler, get_cancer_dosage_function_min_max
from datasets.gp_acquisition_dataset_manager import (
    GP_GEN_DEVICE, add_gp_args, get_gp_model_from_args_no_outcome_transform,
    get_outcome_transform_from_args as get_outcome_transform)

from bayesopt.bayesopt import (
    GPAcquisitionOptimizer, NNAcquisitionOptimizer, OptimizationResultsSingleMethod,
    RandomSearch, get_rff_function, outcome_transform_function)
from bayesopt.stable_gittins import StableGittinsIndex
from utils_general.utils import dict_to_cmd_args, dict_to_fname_str, dict_to_str


GP_AF_NAME_PREFIX = "gp_af"
OBJECTIVE_NAME_PREFIX = "objective"


GP_AF_DICT = {
    'LogEI': LogExpectedImprovement,
    # 'EI': ExpectedImprovement,
    'gittins': StableGittinsIndex
}




# Load the base configuration
with open("config/bo_config.yml", 'r') as f:
    BO_BASE_CONFIG = yaml.safe_load(f)

GEN_CANDIDATES_CONFIG = {
    d["value"]: d["parameters"]
    for d in BO_BASE_CONFIG['parameters']['optimizer']['parameters']['gen_candidates']['values']
}

GEN_CANDIDATES_NAME_TO_FUNCTION = {
    "torch": {
        "func": gen_candidates_torch
    },
    "L-BFGS-B": {
        "func": gen_candidates_scipy,
        "additional_params": {
            "method": "L-BFGS-B"
        }
    }
}


def _add_bo_loop_args(parser, bo_policy_group, af_opt_group):
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
        required=False # yes required if using GP
    )
    
    af_opt_group.add_argument(
        '--num_restarts',
        type=int,
        help='Number of restarts for the optimizer',
        required=False
    )
    af_opt_group.add_argument(
        '--raw_samples',
        type=int,
        help='Number of random samples for the optimizer',
        required=False
    )
    af_opt_group.add_argument(
        '--gen_candidates',
        choices=list(GEN_CANDIDATES_CONFIG),
        help='Method to generate candidates (see BoTorch optimize_acqf)',
        required=False
    )

    optimize_acqf_arg_names = get_arg_names(af_opt_group)

    sets_dict = {method: set(x.keys()) for method, x in GEN_CANDIDATES_CONFIG.items()}
    sets_list = list(sets_dict.values())
    params_all = sets_list[0].intersection(*sets_list[1:])
    params_defaults = {}
    
    methods_per_param_name = defaultdict(list)
    for method, s in sets_dict.items():
        for k in s:
            methods_per_param_name[k].append(method)
    params_only_one = {k for k, v in methods_per_param_name.items() if len(v) == 1}

    extra_bo_policy_args_set = set()
    extra_bo_policy_args = []

    for gen_candidates_name, gen_candidates_params in GEN_CANDIDATES_CONFIG.items():
        g = parser.add_argument_group(f"optimize_acqf options when gen_candidates={gen_candidates_name}")
        params_defaults[gen_candidates_name] = {}
        for param_name, param_value in gen_candidates_params.items():
            default_value = param_value.get("value", None)
            params_defaults[gen_candidates_name][param_name] = default_value
            if param_name not in params_all:
                if param_name not in params_only_one:
                    raise UnsupportedError(f"Parameter {param_name} is in more than one"
                                           " set of parameters but not in all")
                options = dict(required=False)
                desc = f"Optimizer option when gen_candidates={gen_candidates_name}."
                if default_value is not None:
                    options['type'] = type(default_value)
                    # options['default'] = default_value
                    desc += f" Default value: {default_value}"
                options['help'] = desc
                g.add_argument(f"--{param_name}", **options)
            if param_name not in extra_bo_policy_args_set:
                extra_bo_policy_args_set.add(param_name)
                extra_bo_policy_args.append(param_name)
        
    for param_name in params_all:
        defaults = {
            method: d[param_name]
            for method, d in params_defaults.items()
        }
        types = {type(v) for v in defaults.values()}
        if len(types) > 1:
            raise UnsupportedError(f"Parameter {param_name} has different types: {types}")
        defaults_str = ", ".join(
            f"gen_candidates={k}: {v}" for k, v in defaults.items()
        )
        af_opt_group.add_argument(
            f"--{param_name}",
            type=types.pop(),
            help=f"Optimizer option. Default values: {defaults_str}",
            required=False
        )

    return extra_bo_policy_args, params_all, params_defaults, methods_per_param_name, optimize_acqf_arg_names


@cache
def _get_bo_loop_args_parser():
    parser = argparse.ArgumentParser()
    ################## Objective function ###################

    objective_function_group = parser.add_argument_group("Objective function")

    objective_function_group.add_argument(
        f'--{OBJECTIVE_NAME_PREFIX}_id',
        type=int,
        help=f'If {OBJECTIVE_NAME_PREFIX}_dataset_type=gp, this is the seed for '
        'the random GP draw (the objective function). '
        f'If {OBJECTIVE_NAME_PREFIX}_dataset_type=hpob, this is the dataset ID that '
        'specifies the HPO-B objective function.',
        required=True
    )

    objective_function_groups_arg_names = add_unified_function_dataset_args(
        parser,
        thing_used_for="objective function",
        name_prefix=OBJECTIVE_NAME_PREFIX,
        dataset_group=objective_function_group
    )

    ###################################### BO Policy ##################################
    bo_policy_group = parser.add_argument_group("BO policy and misc. settings")
    af_opt_group = parser.add_argument_group("optimize_acqf options")
    
    (extra_bo_policy_args, params_all, params_defaults,
     methods_per_param_name, optimize_acqf_arg_names) = _add_bo_loop_args(
        parser, bo_policy_group, af_opt_group)
    bo_policy_group.add_argument(
        '--bo_seed',
        type=int,
        help='Seed for the BO loop',
        required=True
    )
    bo_policy_group.add_argument(
        '--hpob_seed',
        choices=[None] + [f"test{i}" for i in range(5)],
        help='Seed for the HPO-B initial points (if using HPO-B)',
        required=False
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
    bo_policy_group.add_argument(
        '--random_search',
        action='store_true',
        help='Whether to use random search instead of BO'
    )
    bo_policy_group.add_argument(
        '--transfer_bo_method',
        type=str,
        choices=TRANSFER_BO_BASELINE_NAMES,
        required=False,
        help='(If running transfer BO baseline) Transfer BO baseline method to use.'
    )
    bo_policy_group.add_argument(
        '--dataset_hash',
        type=str,
        required=False,
        help='(If running transfer BO baseline) Hash of the function samples dataset '
             'used for training the transfer BO baseline.'
    )
    bo_policy_arg_names = get_arg_names(bo_policy_group) + get_arg_names(af_opt_group) + extra_bo_policy_args
    
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
                add_randomize_params=False)

    # Add recompute options
    recompute_group = parser.add_argument_group("Recompute options")
    recompute_group.add_argument(
        '--recompute-result',
        action='store_true',
        help='Recompute/overwrite existing BO result even if already cached'
    )

    return {
        'parser': parser,
        'params_all': params_all,
        'params_defaults': params_defaults,
        'methods_per_param_name': methods_per_param_name,
        'objective_function_arg_names': get_arg_names(objective_function_group),
        'objective_function_groups_arg_names': objective_function_groups_arg_names,
        'bo_policy_arg_names': bo_policy_arg_names,
        'gp_af_arg_names': get_arg_names(gp_af_group),
        'optimize_acqf_arg_names': optimize_acqf_arg_names
    }


def parse_bo_loop_args(cmd_args=None):
    parser_info = {**_get_bo_loop_args_parser()}
    parser = parser_info.pop('parser')
    args = parser.parse_args(args=cmd_args)
    parser_info['args'] = args
    return parser_info


_gp_objective_opt_cache = {}
def _get_gp_function_min_max(
        objective_fn, dimension, objective_name) -> tuple[Tensor, Tensor]:
    if objective_name not in _gp_objective_opt_cache:
        # Need to handle an extra dimension at the beginning because for some reason,
        # in the new BoTorch version where it uses gen_candidates_scipy instead of
        # gen_candidates_torch as the old version did, it adds an extra dimension at the
        # beginning when optimizing.
        # (The original objective function expects 2D inputs of shape (n, d).)
        def fn(x):
            if x.dim() == 3:
                assert x.size(0) == 1
                x = x.squeeze(0)
            return objective_fn(x)
        
        opt_kwargs = dict(
            bounds=_get_bounds(dimension),
            raw_samples=8192, num_restarts=100,
            # return_transformed=False is the default and is what we *would* want,
            # but for some reason, in the new BoTorch version it does
            # f_opt = paths(X_opt.unsqueeze(-2)).squeeze(-2)
            # which gives IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
            # So we will manually un-transform it for the negative one.
            return_transformed=True 
        )
        # It maximizes by default
        try:
            argmax, y_max = optimize_posterior_samples(**opt_kwargs, paths=fn)
        except TypeError as e:
            if e.args == ("gen_candidates_torch() got an unexpected keyword argument 'return_transformed'",):
                # Old version of BoTorch that does not have return_transformed argument
                opt_kwargs.pop('return_transformed')
                argmax, y_max = optimize_posterior_samples(**opt_kwargs, paths=fn)
            else:
                raise

        # To minimize, optimize the negative.
        
        ## Should use new BoTorch to version to minimize as follows.
        ## This is what we would do if it didn't give error
        ## "RuntimeError: Output shape of samples not equal to that of weights"
        # from botorch.acquisition.objective import LinearMCObjective
        # obj = LinearMCObjective(weights=-torch.ones(1))
        # argmin, y_min = optimize_posterior_samples(**opt_kwargs, paths=fn, sample_transform=obj)

        # But since it gives error, we will just optimize the negative function instead
        # (which does the same thing).
        argmin, y_min = optimize_posterior_samples(**opt_kwargs, paths=lambda x: -fn(x))
        y_min = -y_min
        
        print(f"Optimized {objective_name} with {argmin=}, {y_min=}, {argmax=}, {y_max=}")
        _gp_objective_opt_cache[objective_name] = (y_min, y_max)
    return _gp_objective_opt_cache[objective_name]


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
        add_standardize=False,
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
    objective_fn = get_rff_function(objective_gp, dimension=dimension)

    desc_dict = dict(
        dimension=dimension,
        kernel=kernel, lengthscale=lengthscale,
        randomize_params=randomize_params,
        seed=gp_seed
    )
    objective_name = f'gp_{dict_to_fname_str(desc_dict)}'

    return objective_gp, objective_fn, objective_name


def _get_objective_things(objective_args):
    dataset_type = objective_args.get('dataset_type', 'gp')
    if dataset_type == 'gp':
        objective_gp, objective_fn, objective_name = _get_gp_objective_things_helper(
            dimension=objective_args['dimension'],
            kernel=objective_args['kernel'],
            lengthscale=objective_args['lengthscale'],
            randomize_params=objective_args['randomize_params'],
            gp_seed=objective_args['id']
        )
    elif dataset_type == 'hpob':
        search_space_id = objective_args['hpob_search_space_id']
        dataset_id = objective_args['id']
        objective_fn = get_hpob_objective_function(
            search_space_id=search_space_id,
            dataset_id=dataset_id
        )
        objective_name = f"hpob_{search_space_id}_{dataset_id}"
        objective_gp = None
    elif dataset_type == 'cancer_dosage':
        tmp = dict(
            dim_x=objective_args['dimension'],
            dim_features=objective_args['dim_features'],
            nnz_per_row=objective_args['nnz_per_row'],
            scale_intercept=objective_args['scale_intercept'],
            scale_coef=objective_args['scale_coef'],
            noise_std=objective_args['noise_std'],
            is_simplex=objective_args['is_simplex'],
            seed=objective_args['id'],
            matrix_seed=objective_args['matrix_seed']
        )
        objective_fn, objective_fn_info = CancerDosageObjectiveSampler(**tmp).sample()
        objective_name = f'cancer_dosage_{dict_to_fname_str(tmp)}'
        objective_gp = None

    # Apply outcome transform to the objective function
    objective_octf, objective_octf_args = get_outcome_transform(
        argparse.Namespace(**objective_args),
        name_prefix=OBJECTIVE_NAME_PREFIX,
        device=GP_GEN_DEVICE)
    if objective_octf is not None:
        octf_str = dict_to_fname_str(objective_octf_args)
        objective_name = f'{objective_name}_{octf_str}'
        objective_fn = outcome_transform_function(objective_fn, objective_octf)
    
    if dataset_type == 'gp':
        y_min, y_max = _get_gp_function_min_max(
            objective_fn, objective_args['dimension'], objective_name)
    elif dataset_type == 'hpob':
        y_min, y_max = get_hpob_function_min_max(search_space_id, dataset_id)
        y_min, y_max = torch.tensor([y_min]), torch.tensor([y_max])
    elif dataset_type == 'cancer_dosage':
        y_min_val, y_max_val = get_cancer_dosage_function_min_max(
            intercept=objective_fn_info['intercept'],
            coefs=objective_fn_info['coefs'],
            noise_std=objective_args['noise_std'],
            is_simplex=objective_args['is_simplex'],
            objective_name=objective_name
        )
        y_min, y_max = torch.tensor([y_min_val]), torch.tensor([y_max_val])

    return objective_gp, objective_octf, objective_fn, objective_name, y_min, y_max


def _get_bounds(dimension):
    return torch.stack([torch.zeros(dimension), torch.ones(dimension)])


@cache
def _get_sobol_samples(bo_seed, n_initial_samples, dimension):
    torch.manual_seed(bo_seed)
    init_x = draw_sobol_samples(
        bounds=_get_bounds(dimension),
        n=1, # Number of BO loops to do
        q=n_initial_samples # Number of sobol points
    )
    return init_x


def pre_run_bo(objective_args: dict[str, Any],
           bo_policy_args: dict[str, Any],
           gp_af_args: dict[str, Any],
           load_weights: bool=True):
    info = _get_bo_loop_args_parser()

    validate_args_for_dataset_type(
        argparse.Namespace(**{
            f"{OBJECTIVE_NAME_PREFIX}_{k}": v for k, v in objective_args.items()
        }),
        groups_arg_names=info['objective_function_groups_arg_names'],
        check_train_test_size=False,
        prefix=OBJECTIVE_NAME_PREFIX
    )

    hpob_search_space_id = objective_args.get('hpob_search_space_id', None)
    dataset_type = objective_args['dataset_type']

    if dataset_type == 'hpob':
        objective_args['id'] = str(objective_args['id'])

    optimize_acqf_arg_names = info['optimize_acqf_arg_names']

    gen_candidates = bo_policy_args.get('gen_candidates', None)
    
    for method, defaults in info['params_defaults'].items():
        for k, default_val in defaults.items():
            if method == gen_candidates:
                if bo_policy_args[k] is None:
                    bo_policy_args[k] = default_val
            elif bo_policy_args.get(k) is not None:
                param_methods = info['methods_per_param_name'][k]
                if gen_candidates not in param_methods:
                    tmp = ', '.join(param_methods)
                    raise ValueError(f"Cannot specify {k} if {gen_candidates=} "
                                    f"(this is only for gen_candidates={tmp})")
    
    if dataset_type in ['gp', 'cancer_dosage']:
        dimension = objective_args['dimension']
    elif dataset_type == 'hpob':
        dimension = get_hpob_dataset_dimension(hpob_search_space_id)
    
    (objective_gp, objective_octf, objective_fn,
     objective_name, y_min, y_max) = _get_objective_things(objective_args)
    
    ############################# Determine the BO policy ##############################
    lamda = bo_policy_args.get('lamda', None)
    if lamda is not None and lamda <= 0:
        raise ValueError("lamda must be > 0")
    
    results_print_data = {'dimension': dimension}
    if dataset_type == 'hpob':
        results_print_data['search space ID'] = hpob_search_space_id
    
    nn_model_name = bo_policy_args.get('nn_model_name')
    random_search = bo_policy_args.get('random_search', False)
    gp_af = gp_af_args.get(GP_AF_NAME_PREFIX)

    transfer_bo_baseline_method = bo_policy_args.get('transfer_bo_method', None)
    dataset_hash = bo_policy_args.get('dataset_hash', None)
    if transfer_bo_baseline_method is not None:
        if dataset_hash is None:
            raise ValueError("If using transfer_bo_method, must specify dataset_hash")
    else:
        if dataset_hash is not None:
            raise ValueError("Cannot specify dataset_hash if not using transfer_bo_method")

    if gp_af is None:
        for k, v in gp_af_args.items():
            if v is not None:
                raise ValueError(
                    f"Cannot specify {GP_AF_NAME_PREFIX}_{k} if not using a GP AF")
    
    # Just a hack for working with the plotting code
    if gp_af == 'random search':
        gp_af = None

    af_options = {}
    
    if random_search or transfer_bo_baseline_method is not None:
        if random_search and transfer_bo_baseline_method is not None:
            raise ValueError("Cannot use both random_search and transfer_bo_method")
        method_name = "random search" if random_search else transfer_bo_baseline_method
        if nn_model_name is not None:
            raise ValueError(f"Cannot specify nn_model_name if using {method_name}")
        if gp_af is not None:
            raise ValueError(f"Cannot specify {GP_AF_NAME_PREFIX} if using {method_name}")
        for k in optimize_acqf_arg_names:
            if bo_policy_args.get(k) is not None:
                raise ValueError(f"Cannot specify {k} if using {method_name}")
        if lamda is not None:
            raise ValueError(f"If using {method_name}, cannot specify lamda")
        
        if random_search:
            optimizer_class = RandomSearch
        else: # transfer BO baseline
            if not transfer_bo_baseline_is_trained(
                transfer_bo_baseline_method, dataset_hash):
                return None
            if transfer_bo_baseline_method == 'FSBO':
                optimizer_class = FSBOOptimizer
            else:
                pass # In the future, add other transfer BO baselines here
            af_options['dataset_hash'] = dataset_hash
        results_print_data = {**results_print_data, 'method': method_name}
    else:
        # Using BO with optimize_acqf
        # optimize_acqf_arg_names = num_restarts, raw_samples, gen_candidates
        missing_args = []
        for k in optimize_acqf_arg_names:
            v = bo_policy_args.get(k)
            if v is None:
                missing_args.append(k)
                continue
            if k == 'gen_candidates':
                v = GEN_CANDIDATES_NAME_TO_FUNCTION[v]["func"]
            af_options[k] = v
        if missing_args:
            raise ValueError(
                "run_bo.py: Must specify the following missing arguments: "
                f"{', '.join(missing_args)} if not using random search")
        options = {
            k: bo_policy_args[k]
            for k in GEN_CANDIDATES_CONFIG[gen_candidates]
        }
        new_params = GEN_CANDIDATES_NAME_TO_FUNCTION[gen_candidates].get("additional_params", None)
        if new_params is not None:
            options.update(new_params)
        af_options['options'] = options
        
        if gp_af is not None: # Using a GP AF
            if nn_model_name is not None:
                raise ValueError(f"Cannot specify nn_model_name if using {GP_AF_NAME_PREFIX}")
            fit = gp_af_args.get('fit')
            #### Determine the GP model to be used for the AF
            if fit == "exact":
                if objective_gp is None:
                    raise ValueError(f"If using {GP_AF_NAME_PREFIX}_fit=exact, "
                                    f"the objective function must be a GP")
                ## Determine af_gp_model from the GP used for the objective
                for k, v in gp_af_args.items():
                    if not (k == "fit" or k == GP_AF_NAME_PREFIX):
                        if v is not None:
                            raise ValueError(f"Cannot specify {GP_AF_NAME_PREFIX}_{k} "
                                            f"if {GP_AF_NAME_PREFIX}_fit=exact")
                # Remove priors if there are any
                af_gp_model = copy.deepcopy(objective_gp)
                remove_priors(af_gp_model)
                
                af_octf = objective_octf
            else:
                ## Determine af_gp_model from the GP args
                af_kernel = gp_af_args.get('kernel')
                af_lengthscale = gp_af_args.get('lengthscale')
                if af_kernel is None:
                    raise ValueError(
                        f"If using a GP AF and {GP_AF_NAME_PREFIX}_fit != exact, "
                        f"must specify {GP_AF_NAME_PREFIX}_kernel")
                if af_lengthscale is None and fit is None:
                    raise ValueError(
                        f"If using a GP AF and {GP_AF_NAME_PREFIX}_fit is None, "
                        f"must specify {GP_AF_NAME_PREFIX}_lengthscale")
                
                # Add priors if using MAP. If using MLE or no fitting, don't add priors
                add_gp_af_priors_flag = fit == "map"
                
                af_gp_model = get_gp_model_from_args_no_outcome_transform(
                    dimension=dimension,
                    kernel=af_kernel,
                    lengthscale=af_lengthscale, # can be None; then will not be set
                    add_priors=add_gp_af_priors_flag,
                    add_standardize=fit is not None,
                    device=GP_GEN_DEVICE
                )
                af_octf, af_octf_args = get_outcome_transform(
                    argparse.Namespace(**gp_af_args),
                    name_prefix=GP_AF_NAME_PREFIX,
                    device=GP_GEN_DEVICE)
            
            #### Apply outcome transform to the AF GP model
            if af_octf is not None:
                add_outcome_transform(af_gp_model, af_octf)

            af_options = dict(
                **af_options,
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
        elif nn_model_name is not None: # Using a NN AF
            if not nn_acqf_is_trained(nn_model_name):
                return None
            
            nn_model = load_nn_acqf(nn_model_name, load_weights=load_weights)

            # TODO (maybe): provide exponentiate=False or exponentiate=True here
            # for ExpectedImprovementAcquisitionFunctionNet?
            af_options = dict(
                **af_options,
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

                configs = load_nn_acqf_configs(nn_model_name)
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
        else:
            raise ValueError(f"Must either specify {GP_AF_NAME_PREFIX}, specify nn_model_name, or set random_search=True")
    
    results_name = dict_to_str(results_print_data, include_space=True)
    return {
        'dimension': dimension,
        'objective_fn': objective_fn,
        'y_min': y_min,
        'y_max': y_max,
        'optimizer_class': optimizer_class,
        'objective_name': objective_name,
        'results_name': results_name,
        'af_options': af_options,
        'objective_gp': objective_gp,
        'objective_octf': objective_octf
    }


_BO_CACHE = {}

def run_bo(objective_args: dict[str, Any],
           bo_policy_args: dict[str, Any],
           gp_af_args: dict[str, Any],
           load_weights: bool=True,
           recompute_result: bool=False):
    stuff = pre_run_bo(
        objective_args, bo_policy_args, gp_af_args, load_weights=load_weights)
    if stuff is None:
        return None
    
    dimension = stuff['dimension']
    objective_fn = stuff['objective_fn']
    optimizer_class = stuff['optimizer_class']
    objective_name = stuff['objective_name']
    results_name = stuff['results_name']
    af_options = stuff['af_options']
    
    bo_seed = bo_policy_args['bo_seed']
    
    hpob_seed = bo_policy_args.get('hpob_seed', None)
    if hpob_seed is None:
        init_x = _get_sobol_samples(
            bo_seed, bo_policy_args['n_initial_samples'], dimension)
    else:
        search_space_id = objective_args['hpob_search_space_id']
        dataset_id = objective_args['id']
        init_x = get_hpob_initialization(search_space_id, dataset_id, hpob_seed)
        # Add a dimension so that init_x has shape 1 x n_initial_samples x dimension
        init_x = init_x.unsqueeze(0)
    
    # One seed per BO loop. Here, we have n=1 BO loops, so need just 1 seed.
    seeds = [bo_seed]

    def objective_fn_(x):
        out = objective_fn(x).detach() # remove gradients
        out = out.unsqueeze(-1) # get to shape n x m where m=1 (m = number of outputs)
        return out

    y_min, y_max = stuff['y_min'], stuff['y_max']

    return OptimizationResultsSingleMethod(
        objectives=[objective_fn_],
        y_mins=[y_min],
        y_maxs=[y_max],
        initial_points=init_x,
        n_iter=bo_policy_args['n_iter'],
        seeds=seeds,
        optimizer_class=optimizer_class,
        objective_names=[objective_name],
        save_dir=RESULTS_DIR,
        results_name=results_name, # results_name is only used to print stuff out
        dim=dimension,
        bounds=_get_bounds(dimension),
        maximize=True,
        verbose=True,
        result_cache=_BO_CACHE,
        recompute_results=recompute_result,
        **af_options
    )


def bo_loop_dicts_to_cmd_args_list(
        objective_args: dict, bo_policy_args: dict, gp_af_args: dict, validate: bool = True):
    objective_args_ = {
        f'{OBJECTIVE_NAME_PREFIX}_{k}': v for k, v in objective_args.items()
    }
    gp_af_args_ = {
        (k if k == GP_AF_NAME_PREFIX else f'{GP_AF_NAME_PREFIX}_{k}'): v
        for k, v in gp_af_args.items()
    }
    all_args = {**objective_args_, **bo_policy_args, **gp_af_args_}
    cmd_args_list = dict_to_cmd_args(all_args)

    if validate:
        # Just validate that the given args are correctly specified
        # (at least as much as the checks done by the parser)
        info = parse_bo_loop_args(cmd_args_list)

    return cmd_args_list


def main():
    info = parse_bo_loop_args()
    args = info['args']

    objective_arg_names = list(itertools.chain.from_iterable(
        info['objective_function_groups_arg_names'].values()
    )) + info['objective_function_arg_names']

    objective_args = {
        k[len(OBJECTIVE_NAME_PREFIX)+1:]: getattr(args, k)
        for k in objective_arg_names
    }

    bo_policy_args = {
        k: getattr(args, k)
        for k in info['bo_policy_arg_names']
    }

    gp_af_args = {
        (k if k == GP_AF_NAME_PREFIX
         else k[len(GP_AF_NAME_PREFIX)+1:]): getattr(args, k)
        for k in info['gp_af_arg_names']
    }

    optimization_results = run_bo(objective_args, bo_policy_args, gp_af_args,
                                  recompute_result=args.recompute_result)

    if optimization_results is None:
        raise ValueError("NN model not trained")
    
    # Perform the optimization. Don't do anything with the results here
    for func_name, trials_dir, func_result in optimization_results:
        # There should be just one loop through this since there is just one function
        pass
        
        # print(f"Function name: {func_name}")
        # print(f"Trials dir: {trials_dir}")
        # # print(f"Function result: {func_result}")
        
        # x = func_result['x']
        # assert x.shape[0] == 1 # only one BO loop seed
        # x = x[0] # shape (n_iterations, dimension)
        # x_first_component = x[:, 0]
        
        # # Plot a scatter plot
        # plt.scatter(range(len(x_first_component)), x_first_component)
        # plt.xlabel('Iteration Number')
        # plt.ylabel('X Value')
        # plt.title('Scatter Plot of X Value vs Iteration Number')

        # trial_config_str = optimization_results.trial_configs_str[0]
        # plot_path = os.path.join(trials_dir, trial_config_str + ".pdf")
        # plt.savefig(plot_path)
        # plt.close()

    # print(f"{objective_args=}")
    # print(f"{bo_policy_args=}")
    # print(f"{gp_af_args=}")
"""
objective_args={'dimension': 8, 'gp_seed': 123, 'kernel': 'Matern52', 'lengthscale': 0.1, 'outcome_transform': None, 'sigma': None, 'randomize_params': False}
bo_policy_args={'n_iter': 30, 'n_initial_samples': 4, 'bo_seed': 99, 'lamda': 0.01, 'nn_model_name': None}
gp_af_args={'gp_af': 'gittins', 'fit': 'exact', 'kernel': None, 'lengthscale': None, 'outcome_transform': None, 'sigma': None}
"""

if __name__ == "__main__":
    main()
