from typing import Any


# TODO: In the future, could do this more automatically rather than hard-coding
# everything (and also for get_cmd_options_train_acqf in train_acqf.py)
def get_cmd_options_sample_dataset(options: dict[str, Any]):
    # Extract dataset_type to determine which parameters to include
    dataset_type = options.get('dataset_type', 'gp')

    # Base dataset parameters common to all types
    cmd_opts_sample_dataset = {
        'dataset_type': dataset_type,
        'train_samples_size': options.get('train_samples_size'),
        'test_samples_size': options.get('test_samples_size'),
        'standardize_outcomes': options['standardize_outcomes']
    }

    # Add dataset-specific parameters
    if dataset_type == 'gp':
        cmd_opts_sample_dataset.update({
            k: options.get(k)
            for k in ['dimension', 'kernel', 'lengthscale',
                      'randomize_params', 'outcome_transform', 'sigma']
        })
    elif dataset_type == 'logistic_regression':
        cmd_opts_sample_dataset.update({
            k: options.get(k)
            for k in ['lr_n_samples_range', 'lr_n_features_range', 'lr_bias_range',
                      'lr_coefficient_std', 'lr_noise_range', 'lr_log_lambda_range',
                      'lr_log_uniform_sampling']
        })
    elif dataset_type == 'hpob':
        cmd_opts_sample_dataset.update({
            'hpob_search_space_id': options.get('hpob_search_space_id')
        })
    elif dataset_type == 'cancer_dosage':
        cmd_opts_sample_dataset.update({
            k: options.get(k)
            for k in ['dimension', 'dim_features', 'nnz_per_row',
                      'scale_intercept', 'scale_coef', 'noise_std',
                      'is_simplex', 'matrix_seed']
        })
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return cmd_opts_sample_dataset
