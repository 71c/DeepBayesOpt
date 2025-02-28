# Model-free Offline RL Bayesian Optimization

This repository contains code for research on using offline deep reinforcement learning techniques to learn acquisition functions for Bayesian optimization. Instead of relying on surrogate models like Gaussian Processes, this approach trains neural networks end-to-end on a dataset of objective functions to approximate acquisition functions such as Expected Improvement and the Gittins index.  

The repository includes:
- Scripts for generating synthetic datasets of black-box objective functions, training neural network-based acquisition functions, and running Bayesian optimization loops.
- Fully automated running of experiments using YAML config files and command-line Python scripts to automate the submission of several dependent jobs and job arrays using SLURM.  
- Tools assessing the Bayesian optimization performance through scripts that generate and save plots in a hierarchical, systematic way.


# Installation

To install the required packages, make a new Python environment, e.g. using conda
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


# NN training (+ dataset generation if necessary)

## Training a single neural network
`run_train.py` is the script that trains a single neural network. `run_train.py --help` will show the description of the arguments. An example command is as follows:
```bash
python run_train.py --dimension 16 --kernel Matern52 --lamda_max 1.0 --lamda_min 0.0001 --lengthscale 0.1 --max_history 100 --min_history 1 --replacement --test_expansion_factor 2 --test_n_candidates 1 --test_samples_size 5000 --train_acquisition_size 5000 --train_n_candidates 1 --train_samples_size 10000 --batch_size 32 --early_stopping --epochs 500 --gi_loss_normalization normal --lamda_max 1.0 --lamda_min 0.0001 --layer_width 200 --learning_rate 0.0003 --method gittins --min_delta 0.0 --patience 20
```
It will output
```
Saving NN to v2/model_1798dfc44d64e85c92ab88abd40fb62e97f216968037268b794b92c0a1099b4b
```
This identifies the neural network model -- which is uniquely identified by the specific combination of dataset, architecture, training method, optimizer settings, etc -- and this info and the weights to this directory.

### Dataset generation
In order to train the neural network, you need to have a dataset of black-box objective functions. Currently, the dataset is randomly generated using Gaussian process models. Since it takes some time to generate the datasets, they are cached in the `datasets` directory. When running `run_train.py`, if the dataset is not found, it will automatically generate the dataset and save it in the `datasets` directory. But there is also a stand-alone script to generate the dataset manually, because this way the dataset generation can be separated from the neural network training which makes it possible to have finer-grained control of the automated job scheduling.

To generate a synthetic dataset of black-box objective functions, use `gp_acquisition_dataset.py`. Run `python gp_acquisition_dataset.py --help` to see the description of the arguments. An example command to generate a dataset is as follows:
```bash
python gp_acquisition_dataset.py --dimension 16 --kernel Matern52 --lamda_max 1.0 --lamda_min 0.0001 --lengthscale 0.1 --max_history 100 --min_history 1 --replacement --test_expansion_factor 1 --test_n_candidates 1 --test_samples_size 10000 --train_acquisition_size 30000 --train_n_candidates 1 --train_samples_size 10000
```
This command will generate a dataset with the specified parameters and save it for use in training neural network-based acquisition functions.


## Training multiple neural networks
`python train_acqf.py` is the script that trains multiple neural networks.
Run `python train_acqf.py --help` for the description of the arguments.
An example command is as follows:
```bash
python train_acqf.py --base_config config/train_acqf.yml --experiment_config config/train_acqf_experiment_test_simple.yml --sweep_name preliminary-test-small-train --mail adj53@cornell.edu --gpu_gres gpu:1
```
This will train multiple neural networks with different hyperparameters and save the models.

Here, `--base_config` is the base configuration file, containing the default values and default ranges to search over for all of the hyperparameters. 

`--experiment_config` is the experiment configuration file, containing the specific values and ranges to search over a subset of the hyperparameters for the particular experiment. Replace the value for `--experiment_config` with your desired experiment configuration file.  For example, to investigate the effect of the hyperparameters regarding the dataset, NN architecture, and optimizer settings, we can specify the experiment configuration file to be `config/train_acqf_experiment_training.yml`, which varies `train_samples_size`, `layer_width`, and `learning_rate`, while fixing the dimension to 16 and the method to Gittins index with $\lambda=10^{-4}$.
Alternatively, you can use `config/train_acqf_experiment_test_simple.yml` to just run a single NN training.


# BO loops (+ NN training if necessary)

## Running a single Bayesian optimization loop
To run a BO loop, you can use `run_bo.py`. `run_bo.py --help` will show the description of the arguments.
An example command to use a GP acquisition function is as follows:
```bash
python run_bo.py --bo_seed 6888556634303915349 --gp_af gittins --gp_af_fit exact --lamda 0.01 --n_initial_samples 1 --n_iter 100 --objective_dimension 16 --objective_gp_seed 6888556634303915349 --objective_kernel Matern52 --objective_lengthscale 0.1
```
An example command to use a NN acquisition function is as follows:
```bash
python run_bo.py --bo_seed 6888556634303915349 --lamda 0.01 --n_initial_samples 1 --n_iter 100 --nn_model_name v2/model_1798dfc44d64e85c92ab88abd40fb62e97f216968037268b794b92c0a1099b4b --objective_dimension 16 --objective_gp_seed 6888556634303915349 --objective_kernel Matern52 --objective_lengthscale 0.1
```
If the NN model has not been trained yet, it will raise an error. In this case, you need to enter the command to train the NN model and then the command to run the BO loop.

## Running multiple Bayesian optimization loops
Run the following command to automatically run all of the BO loops of both the NNs and the GP-based AFs:
```bash
python bo_experiments_gp.py --base_config config/train_acqf.yml --experiment_config config/train_acqf_experiment_test_simple.yml --n_gp_draws 8 --seed 8 --n_iter 100 --n_initial_samples 1 --sweep_name preliminary-test-small --mail adj53@cornell.edu --gpu_gres gpu:1
```
Run `python bo_experiments_gp.py --help` to see the description of the arguments. 
Unlike the command for running a single BO loop, this command will automatically train any NNs that have not been trained yet prior to optimizing with them.


# Making plots
`bo_experiments_gp_plot.py` will make the plots. You can run `python bo_experiments_gp_plot.py --help` for a description of the arguments. As opposed to either throwing an error or automatically running the prerequisite commands as in the other scripts, this script will only generate plots for the BO loops that have already been run.

An example command is as follows:
```bash
python bo_experiments_gp_plot.py --base_config config/train_acqf.yml --experiment_config config/train_acqf_experiment_test1.yml --n_gp_draws 8 --seed 8 --n_iter 100 --n_initial_samples 1 --plots_config config/plots_config_1.yml
```
Here, `--plots_config` is the configuration file for the plots.
For example, `config/plots_config_1.yml` is as follows:
```yaml
- ["nn.layer_width", "nn.train_samples_size"]
- ["lamda", "gp_af", "nn.method"]
- ["objective.gp_seed"]
```
This means that at the highest level it will vary the layer width and the training samples size, then the lambda, the GP acquisition function or NN method, and finally the seed for the GP. The GP seed corresponds to individual BO runs that together comprise an error bar that is in the legend of a specific subplot. The higher levels make up subplots within a figure, figures (which correspond to `.pdf` files), and folders containing the figures. 
The script will output the following to indicate to the user the structure of the plots:
```
  folder: {'dimension'}
  fname: {'nn.train_samples_size', 'nn.layer_width'}
  line: {'nn.gi_loss_normalization', 'lamda', 'nn.lamda_min', 'nn.learning_rate', 'gp_af', 'nn.lamda_max', 'nn.method', 'nn.lamda'}
  random: {'objective.gp_seed'}
```
To do this, looks at the attributes that were not already specified in `config/plots_config_1.yml` but that do vary in the experiment. Of these, of the ones that are always present across all experiments, it adds them to a new, topmost level. And the rest, it adds to the existing second-to-last level (the "line" level).

On the other hand, if you run the same command above but additionally with the flag `--use_subplots`, then the script will output the following:
```
  fname: {'dimension'}
  subplot: {'nn.layer_width', 'nn.train_samples_size'}
  line: {'nn.lamda_min', 'nn.method', 'nn.lamda_max', 'lamda', 'nn.lamda', 'gp_af', 'nn.gi_loss_normalization', 'nn.learning_rate'}
  random: {'objective.gp_seed'}
```
This means that each figure (file) corresponds to its own dimension, each subplot with in a figure corresponds to a combination of layer width and training samples size, error bar ("line") corresponds to a different BO policy, and the seed is the source of randomness for the error bar.
Enabling `--use_subplots` is more compact since it fits multiple subplots in a single figure so you can view them side by side.
