"""Constants for acquisition function networks."""

POINTNET_ACQF_PARAMS_INPUT_DEFAULT = 'final_only'


POINTNET_ACQF_PARAMS_INPUT_OPTIONS = {
    "local_and_final": dict(
        input_acqf_params_to_local_nn=True,
        input_acqf_params_to_final_mlp=True
    ),
    "local_only": dict(
        input_acqf_params_to_local_nn=True,
        input_acqf_params_to_final_mlp=False
    ),
    "final_only": dict(
        input_acqf_params_to_local_nn=False,
        input_acqf_params_to_final_mlp=True
    )
}


METHODS = ['mse_ei', 'policy_gradient', 'gittins']
