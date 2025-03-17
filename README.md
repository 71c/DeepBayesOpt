# Model-free Offline RL Bayesian Optimization

This repository contains code for research on using offline deep reinforcement learning techniques to learn acquisition functions for Bayesian optimization. Instead of relying on surrogate models like Gaussian Processes, this approach trains neural networks end-to-end on a dataset of objective functions to approximate acquisition functions such as Expected Improvement and the Gittins index.  

The repository includes:
- Scripts for generating synthetic datasets of black-box objective functions, training neural network-based acquisition functions, and running Bayesian optimization loops.
- Fully automated running of experiments using YAML config files and command-line Python scripts to automate the submission of several dependent jobs and job arrays using SLURM.  
- Tools assessing the Bayesian optimization performance through scripts that generate and save plots in a hierarchical, systematic way.

NOTE: I might not update this README as quickly as I change the code.


# Installation

To install the required packages, make a new Python environment, for example using Anaconda:
```bash
conda create --name nn_bo python=3.12.4
```
(I'm running the code on both Python 3.12.4 and 3.9.12, they both work.)

Next, switch to the new environment:
```bash
conda activate nn_bo
```
Then install the required packages using pip:
```bash
pip install -r requirements.txt
```
Next, in `job_array.sub`, update the `conda activate` command to use this correct Python environment: `conda activate nn_bo`. (`job_array.sub` is the SLURM job submission script that is used to submit the jobs to the cluster in `submit_dependent_jobs.py`.)

# Command-line scripts overview
The command-line scripts are organized as follows. See the following sections for more details on each script.
- **BO loops (+ NN training):** `run_bo.py` for running a single BO loop, `bo_experiments_gp.py` for running multiple. `bo_experiments_gp.py` is the most high-level script and the one that is most likely to be used.
- **NN training (+ dataset generation):** `run_train.py` for training a single NN, `train_acqf.py` for training multiple.
- **Dataset generation:** `gp_acquisition_dataset.py` for generating a dataset of black-box objective functions.
- **Making plots:** `bo_experiments_gp_plot.py` for making plots of the BO loops once the results are available.


# BO loops (+ NN training if necessary)
Use `run_bo.py` for running a single BO loop, and use `bo_experiments_gp.py` for running multiple. `bo_experiments_gp.py` is the most high-level script and the one that is most likely to be used.

## Running a single Bayesian optimization loop
To run a BO loop, you can use `run_bo.py`. `run_bo.py --help` will show the description of the arguments. If the NN model has not been trained yet, it will raise an error. In this case, you need to enter the command to train the NN model and then the command to run the BO loop.

Example commands:
- Random Search:
```bash
python run_bo.py --n_initial_samples 1 --n_iter 20 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --random_search --bo_seed 6888556634303915349 --objective_gp_seed 6888556634303915349

python run_bo.py --n_initial_samples 1 --n_iter 20 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --random_search --bo_seed 8643049736318478698 --objective_gp_seed 8643049736318478698
```
- GP-based acquisition function:
```bash
python run_bo.py --n_initial_samples 1 --n_iter 20 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --gp_af LogEI --gp_af_fit exact --num_restarts 160 --raw_samples 3200 --gen_candidates L-BFGS-B --bo_seed 6888556634303915349 --objective_gp_seed 6888556634303915349

python run_bo.py --n_initial_samples 1 --n_iter 20 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --gp_af LogEI --gp_af_fit exact --num_restarts 160 --raw_samples 3200 --gen_candidates L-BFGS-B --bo_seed 8643049736318478698 --objective_gp_seed 8643049736318478698

python run_bo.py --eps 1e-08 --ftol 2.220446049250313e-09 --gen_candidates L-BFGS-B --gp_af gittins --gp_af_fit exact --gtol 1e-05 --lamda 0.01 --maxcor 10 --maxfun 15000 --maxiter 15000 --maxls 20 --n_initial_samples 1 --n_iter 20 --num_restarts 160 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --raw_samples 3200 --bo_seed 6888556634303915349 --objective_gp_seed 6888556634303915349

python run_bo.py --eps 1e-08 --ftol 2.220446049250313e-09 --gen_candidates L-BFGS-B --gp_af gittins --gp_af_fit exact --gtol 1e-05 --lamda 0.01 --maxcor 10 --maxfun 15000 --maxiter 15000 --maxls 20 --n_initial_samples 1 --n_iter 20 --num_restarts 160 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --raw_samples 3200 --bo_seed 8643049736318478698 --objective_gp_seed 8643049736318478698
```
- NN-based acquisition function:
```bash
python run_bo.py --lamda 1e-2 --n_initial_samples 1 --n_iter 20 --nn_model_name v2/model_35f043d1473e2adc8a97027e56c8dc8cefd60ef48a14382cfd07e60e52a55234 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --num_restarts 160 --raw_samples 3200 --gen_candidates L-BFGS-B --bo_seed 6888556634303915349 --objective_gp_seed 6888556634303915349

python run_bo.py --lamda 1e-2 --n_initial_samples 1 --n_iter 20 --nn_model_name v2/model_35f043d1473e2adc8a97027e56c8dc8cefd60ef48a14382cfd07e60e52a55234 --objective_dimension 1 --objective_kernel Matern52 --objective_lengthscale 0.05 --num_restarts 160 --raw_samples 3200 --gen_candidates L-BFGS-B --bo_seed 8643049736318478698 --objective_gp_seed 8643049736318478698
```
See the [section on NN training](#nn-training--dataset-generation-if-necessary) for how to train the NN model and obtain `--nn_model_name`.

For `--gen_candidates` when optimizing AFs, you can currently choose between "L-BFGS-B" and "torch".


## Running multiple Bayesian optimization loops
The following command automatically runs all of the BO loops of both the NNs and the GP-based AFs. Run `python bo_experiments_gp.py --help` to see the description of the arguments. Unlike the command for running a single BO loop, this command will automatically train any NNs that have not been trained yet prior to optimizing with them.

An example command is as follows:
```bash
python bo_experiments_gp.py --nn_base_config config/train_acqf.yml --nn_experiment_config config/train_acqf_experiment_test_simple.yml --bo_base_config config/bo_config.yml --n_gp_draws 8 --seed 8 --sweep_name preliminary-test-small --mail adj53@cornell.edu --gpu_gres gpu:1
```

### Arguments
#### Objective functions and seed
- `--n_gp_draws`: the number of draws of GP objective functions per set of GP params.
- `--seed SEED`: the seed for the random number generator.

#### BO loop
- `--n_iter`: the number of iterations of BO to perform
- `--n_initial_samples`: the number of initial sobol points to sample at before using the AF.

#### NN training experiments
- `--nn_base_config` is the base configuration file, containing the default values and default ranges to search over for all of the hyperparameters.
- `--nn_experiment_config` is the experiment configuration file, containing the specific values and ranges to search over a subset of the hyperparameters for the particular experiment. Replace the value for `--nn_experiment_config` with your desired experiment configuration file.  For example, to investigate the effect of the hyperparameters regarding the dataset, NN architecture, and optimizer settings, we can specify the experiment configuration file to be `config/train_acqf_experiment_training.yml`, which varies `train_samples_size`, `layer_width`, and `learning_rate`, while fixing the dimension to 16 and the method to Gittins index with $\lambda=10^{-4}$.
Alternatively, you can use `config/train_acqf_experiment_test_simple.yml` to just run a single NN training.
- `--always_train`: If this flag is set, train all acquisition function NNs regardless of whether they have already been trained. Default is to only train acquisition function NNs that have not already been trained.

#### SLURM-based job submission
- `--sweep_name` is the name of the "sweep" (in Weights and Biases terminology). In this case, it just corresponds to the name of the directory where the err and out files, and other information about the experiment submission, will be saved.
- `--mail` is the email address to send a notification to when the job is done (optional).
- `--gpu_gres` is the GPU resource to request. In this case, it is requesting one GPU. (Also optional.)

Other arguments like partition and time may be added to the script if necessary.


# NN training (+ dataset generation if necessary)
Although the NN training is automatically done when running the BO loops, you can also just train the all the NNs without doing anything with them just yet, with the following scripts.
Use `run_train.py` for training a single NN, and `train_acqf.py` for training multiple.

## Training a single neural network
`run_train.py` is the script that trains a single neural network. `run_train.py --help` will show the description of the arguments. An example command is as follows:
```bash
python run_train.py --dimension 1 --lengthscale 0.05 --kernel Matern52 --min_history 1 --max_history 20 --replacement --train_n_candidates 1 --test_n_candidates 1 --train_acquisition_size 8192 --train_samples_size 10000 --test_expansion_factor 1 --test_samples_size 5000 --batch_size 512 --early_stopping --min_delta 0.0 --patience 30 --layer_width 200 --learning_rate 3e-4 --lr_scheduler ReduceLROnPlateau --lr_scheduler_patience 15 --lr_scheduler_factor 0.1 --method gittins --lamda 1e-2 --gi_loss_normalization normal --architecture pointnet --epochs 3
```
It will output
```
Saving model and configs to v2/model_35f043d1473e2adc8a97027e56c8dc8cefd60ef48a14382cfd07e60e52a55234
```
This identifies the neural network model, which is uniquely identified by the specific combination of dataset, architecture, training method, optimizer settings, etc. This information, along with the weights of the NN corresponding to the epoch where it performed the best on the test dataset, are saved to this directory.

Example of a long-running command:
```bash
python run_train.py --dimension 1 --kernel Matern52 --lamda 0.01 --lengthscale 0.05 --max_history 20 --min_history 1 --replacement --test_expansion_factor 1 --test_n_candidates 1 --test_samples_size 10000 --train_acquisition_size 30000 --train_n_candidates 1 --train_samples_size 10000 --batch_size 512 --early_stopping --epochs 500 --gi_loss_normalization normal --lamda 0.01 --layer_width 300 --learning_rate 0.001 --lr_scheduler ReduceLROnPlateau --lr_scheduler_cooldown 0 --lr_scheduler_factor 0.1 --lr_scheduler_min_lr 0.0 --lr_scheduler_patience 15 --method gittins --min_delta 0.0 --patience 30 --architecture pointnet
```

### Dataset generation
In order to train the neural network, you need to have a dataset of black-box objective functions. Currently, the dataset is randomly generated using Gaussian process models. Since it takes some time to generate the datasets, they are cached in the `datasets` directory. When running `run_train.py`, if the dataset is not found, it will automatically generate the dataset and save it in the `datasets` directory. But there is also a stand-alone script to generate the dataset manually, because this way the dataset generation can be separated from the neural network training which makes it possible to have finer-grained control of the automated job scheduling.

To generate a synthetic dataset of black-box objective functions, use `gp_acquisition_dataset.py`. Run `python gp_acquisition_dataset.py --help` to see the description of the arguments. An example command to generate a dataset is as follows:
```bash
python gp_acquisition_dataset.py --dimension 16 --kernel Matern52 --lamda_max 1.0 --lamda_min 0.0001 --lengthscale 0.1 --max_history 100 --min_history 1 --replacement --test_expansion_factor 1 --test_n_candidates 1 --test_samples_size 10000 --train_acquisition_size 30000 --train_n_candidates 1 --train_samples_size 10000
```
Note that in the above command, the parameters `lamda_max` and `lamda_min` are used because this means that we are requesting a `CostAwareAcquisitionDataset` which, along with the other attributes like `'x_hist', 'y_hist', 'x_cand', 'y_cand'`, also includes random values of $\lambda$. It is done in this way as "part of the datset" rather than generating the $\lambda$ values in the training loop because this way, the *testing* dataset along with the random $\lambda$ values can be fixed, the stats for true GP model cached, and saved to disk. (In contrast, the train dataset will not have its acquisition function dataset samples fixed.)


## Training multiple neural networks
`python train_acqf.py` is the script that trains multiple neural networks.
Run `python train_acqf.py --help` for the description of the arguments.
An example command is as follows:
```bash
python train_acqf.py --nn_base_config config/train_acqf.yml --nn_experiment_config config/train_acqf_experiment_test_simple.yml --sweep_name preliminary-test-small-train --mail adj53@cornell.edu --gpu_gres gpu:1
```
This will train multiple neural networks with different hyperparameters and save the models.


# Making plots
`bo_experiments_gp_plot.py` will make the plots. You can run `python bo_experiments_gp_plot.py --help` for a description of the arguments. As opposed to either throwing an error or automatically running the prerequisite commands as in the other scripts, this script will only generate plots for the BO loops that have already been run.

An example command is as follows:
```bash
python bo_experiments_gp_plot.py --nn_base_config config/train_acqf.yml --nn_experiment_config config/train_acqf_experiment_1dim_example.yml --bo_base_config config/bo_config.yml --bo_experiment_config config/bo_config_experiment_2_20iter_160.yml --n_gp_draws 2 --seed 8 --use_rows --use_cols --center_stat mean --interval_of_center --plots_group_name test_1dim_maxhistory20_example --plots_name results_20iter
```
This means that at the highest level it will vary the layer width and the training samples size, then the lambda, the GP acquisition function or NN method, and finally the seed for the GP. The GP seed corresponds to individual BO runs that together comprise an error bar that is in the legend of a specific subplot. The higher levels make up subplots within a figure, figures (which correspond to `.pdf` files), and folders containing the figures. 
The script will output something like the following to indicate to the user the structure of the plots:
```
  folder: {'dimension'}
  fname: {'nn.train_samples_size', 'nn.layer_width'}
  line: {'nn.gi_loss_normalization', 'lamda', 'nn.lamda_min', 'nn.learning_rate', 'gp_af', 'nn.lamda_max', 'nn.method', 'nn.lamda'}
  random: {'objective.gp_seed'}
```
To do this, looks at the attributes that were not already specified in `config/plots_config_1.yml` but that do vary in the experiment. Of these, of the ones that are always present across all experiments, it adds them to a new, topmost level. And the rest, it adds to the existing second-to-last level (the "line" level).

On the other hand, if you run the same command above but additionally with the flag `--use_cols` (you can also enable `--use_cols` instead of or in addition to `--use_rows`), then the script will output the following:
```
  fname: {'dimension'}
  col: {'nn.layer_width', 'nn.train_samples_size'}
  line: {'nn.lamda_min', 'nn.method', 'nn.lamda_max', 'lamda', 'nn.lamda', 'gp_af', 'nn.gi_loss_normalization', 'nn.learning_rate'}
  random: {'objective.gp_seed'}
```
This means that each figure (file) corresponds to its own dimension, each subplot with in a figure corresponds to a combination of layer width and training samples size, error bar ("line") corresponds to a different BO policy, and the seed is the source of randomness for the error bar.
Enabling `--use_rows` and/or `use_cols` is more compact since it fits multiple subplots in a single figure so you can view them side by side.

# Overview of the rest of the codebase

## Datasets
- `dataset_with_models.py`: Provides a mechanism for creating a hierarchy of classes that represents datasets and can optionally have a GP model attached to each item in the dataset.
- `function_samples_dataset.py`: Defines classes that represent samples of functions. Uses `dataset_with_models.py` to create the class `FunctionSamplesDataset`, along with `MapFunctionSamplesDataset`, `ListMapFunctionSamplesDataset`, `LazyMapFunctionSamplesDataset`, and `MapFunctionSamplesSubset`. Also defines `TransformedFunctionSamplesIterableDataset`, `TransformedLazyMapFunctionSamplesDataset`, `GaussianProcessRandomDataset`, and `ResizedFunctionSamplesIterableDataset`.
- `acquisition_dataset.py`: Defines classes that represent datasets for training acquisition functions. Uses `function_samples_dataset.py` to create the class `AcquisitionDataset`, along with `MapAcquisitionDataset`, `ListMapAcquisitionDataset`, `LazyMapAcquisitionDataset`, and `MapAcquisitionSubset`. Defines `FunctionSamplesAcquisitionDataset` which is used to create an acquisition dataset from a function samples dataset. Also defines `CostAwareAcquisitionDataset` which simply combines random $\lambda$ values with the acquisition dataset.

## NN AF Architecture: `acquisition_function_net.py`
`acquisition_function_net.py` defines PyTorch modules for acquisition functions in likelihood-free Bayesian optimization. It includes modular classes for structured acquisition function design, such as:
- `AcquisitionFunctionNet`: Base class for acquisition function networks.
- `ParameterizedAcquisitionFunctionNet`: Supports acquisition functions with parameters.
- `TwoPartAcquisitionFunctionNet`: Splits acquisition functions into a feature-extracting body and an MLP head.
- `GittinsAcquisitionFunctionNet`: Implements Gittins index-based acquisition function functionality.
- `ExpectedImprovementAcquisitionFunctionNet`: Implements the architecture and interface used for the neural-network-based expected improvement.
- `AcquisitionFunctionNetModel` and `AcquisitionFunctionNetAcquisitionFunction`: Enable integration with BoTorch to be used in Bayesian optimization loops.

## Code for NN Training: `train_acquisition_function_net.py`
`train_acquisition_function_net.py` contains the code for training neural networks for acquisition functions. In particular, the function `train_acquisition_function_net` is only used in `run_train.py`. Additionally, this script defines functions to be used for loading and saving the NN models.

## Code for running BO loops: `bayesopt.py`
This module implements the core Bayesian optimization loop. It includes:

- **Optimizer Classes:**  
  - **`BayesianOptimizer`**: An abstract base class handling initialization, tracking of the best observed values, and time statistics.  
  - **`RandomSearch`**: A simple random sampling baseline.  
  - **`SimpleAcquisitionOptimizer`**: Uses an acquisition function (optimized via BoTorch) to select new evaluation points.  
  - **`ModelAcquisitionOptimizer`**: Integrates a model (GP or NN) with the acquisition function.  
  - **`NNAcquisitionOptimizer` / `GPAcquisitionOptimizer`**: Concrete implementations for NN-based and GP-based acquisition functions, respectively.

- **Results & Experiment Management:**
  Classes for caching, saving, and validating BO results across functions and trials. This includes `OptimizationResultsSingleMethod` and `OptimizationResultsMultipleMethods`:
  - `OptimizationResultsSingleMethod`: can run several BO loops with the same method with different objective functions and seeds.
  - `OptimizationResultsMultipleMethods`: can run several `OptimizationResultsSingleMethod` instances with different methods.
  
  These classes can only run the BO loops in sequence, not in parallel. For this reason, their full intended usage is now deprecated in favor of `bo_experiments_gp.py`, which submits jobs to repeatedly run `run_bo.py`. `run_bo.py` simply uses `OptimizationResultsSingleMethod` with a single objective function and a single BO seed.

- **Plotting & Utility Functions:**  
  Helper routines for plotting optimization trajectories, generating random GP realizations, and applying outcome transforms. (Much of these are now deprecated in favor of `bo_experiments_gp_plot.py`.)

## Utility Functions
- `utils.py`: provides a comprehensive suite of helper functions and classes to support operations such as outcome transformations, kernel and model setup, JSON serialization, tensor padding, and various utility routines for managing data and configurations in Bayesian optimization experiments.
- `nn_utils.py`: provides utility functions and custom PyTorch modules to build and manage neural network components, including dense layers, learnable positive parameters, custom softplus activations, and PointNet layers with various pooling strategies. It also includes helper routines for tensor dimension checking and masked softmax operations, along with embedded test cases for verification.
- `plot_utils.py`: a few utility functions for plotting things.
