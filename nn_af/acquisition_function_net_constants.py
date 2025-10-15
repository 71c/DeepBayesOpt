"""Constants for acquisition function networks.

This module contains constants that are shared between acquisition_function_net.py
and acquisition_function_net_save_utils.py to avoid circular imports.
"""

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
