import torch
import gpytorch
import pyro
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.gp_regression import SingleTaskGP

from torch.distributions import Uniform, Normal, Independent, Distribution
from typing import List, Union

from torch.utils.data import Dataset, IterableDataset, DataLoader

import os

torch.set_default_dtype(torch.double)


# https://docs.gpytorch.ai/en/stable/_modules/gpytorch/module.html#Module.pyro_sample_from_prior
def _pyro_sample_from_prior(module, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_priors"):
        for prior_name, (prior, closure, setting_closure) in module._priors.items():
            if prior is not None and prior not in memo:
                if setting_closure is None:
                    raise RuntimeError(
                        "Cannot use Pyro for sampling without a setting_closure for each prior,"
                        f" but the following prior had none: {prior_name}, {prior}."
                    )
                memo.add(prior)
                prior = prior.expand(closure(module).shape)
                value = pyro.sample(prefix + ("." if prefix else "") + prior_name, prior)
                setting_closure(module, value)

    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        _pyro_sample_from_prior(module=module_, memo=memo, prefix=submodule_prefix)

    return module


class RandomModelSampler:
    def __init__(self, models: List[SingleTaskGP], model_probabilities=None):
        # Set the model indices as attributes so we can access them for purpose
        # of saving data
        for i, model in enumerate(models):
            model.index = i

        self._models = models
        self._model_initial_parameters = [
            {name: param for name, param in model.named_parameters()}
            for model in models
        ]

        if model_probabilities is None:
            model_probabilities = torch.full([len(models)], 1/len(models))
        else:
            model_probabilities = torch.as_tensor(model_probabilities)
            assert model_probabilities.dim() == 1
            assert len(models) == len(model_probabilities)

        self.model_probabilities = model_probabilities
    
    def sample(self):
        # pick the model
        # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146
        model_index = self.model_probabilities.multinomial(num_samples=1, 
                                                           replacement=True)[0]
        model = self._models[model_index]

        # Randomly set the parameters based on the priors of the model.
        # Instead of doing  `random_model = model.pyro_sample_from_prior()`
        # which does a deep copy which takes long,
        # sample in-place, significantly speeding it up.
        # This also avoids the parameters disappearing.
        _pyro_sample_from_prior(model, memo=None, prefix="")

        return model
    
    def get_model(self, index, model_params=None):
        if model_params is None:
            model_params = self._model_initial_parameters[index]
        model = self._models[index]
        model.initialize(**model_params)
        return model
    
    @property
    def initial_models(self):
        # Set the parameters of each model to be the original ones
        for model, params in zip(self._models, self._model_initial_parameters):
            model.initialize(**params)
        return self._models


class GaussainProcessRandomDataset(IterableDataset):
    def __init__(self, dimension, n_datapoints:int=None,
                 n_datapoints_random_gen=None,
                 xvalue_distribution: Distribution=None,
                 models: List[SingleTaskGP]=None,
                 model_probabilities=None, observation_noise:bool=False,
                 set_random_model_train_data=False, device=None):
        """Create a dataset that generates random Gaussian Process data.

        Usage example:
        dataset = GaussainProcessRandomDataset(dimension=5, n_datapoints=15)
        for x_values, y_values, model in dataset:
            # x_values has shape (n_datapoints, dimension)
            # y_values has shape (n_datapoints,)
            # do something with x_values, y_values, and model

        Args:
            dimension: The dimension of the feature space d
            n_datapoints: number of (x,y) pairs to generate with each sample;
                could be None
            n_datapoints_random_gen: a callable that returns a random natural
                number that is the number of datapoints.
                Note: exactly one of n_datapoints and n_datapoints_random_gen
                should be speicified (not be None).
            xvalue_distribution: a torch.distributions.Distribution object that
                represents the probability distribution for generating each iid
                value $x \in \mathbb{R}^{dimension}$. Default: iid Uniform(0,1)
            models: a list of SingleTaskGP models to choose from randomly,
                with their priors 
                defaults to a single SingleTaskGP model with the default BoTorch
                Matern 5/2 kernel and Gamma priors for the lengthscale,
                outputscale; and noise level also if observation_noise==True.
                It is assumed that each model provided is single-batch.
            model_probabilities: list of probability of choosing each model
            observation_noise: boolean specifying whether to generate the data
                to include the "observation noise" given by the model's
                likelihood (True), or not (False)
            set_random_model_train_data: whether to set the random model train
                data to the random data with each returned values
            device: torch.device, optional -- the desired device to use for
                computations.
        """
        self.dimension = dimension
        self.observation_noise = observation_noise

        # exacly one of them should be specified; verify this by xor
        assert (n_datapoints is None) ^ (n_datapoints_random_gen is None)
        self.n_datapoints = n_datapoints
        self.n_datapoints_random_gen = n_datapoints_random_gen
        self.set_random_model_train_data = set_random_model_train_data
        self.device = device
        
        if xvalue_distribution is None:
            # m = Normal(torch.zeros(dimension), torch.ones(dimension))
            m = Uniform(torch.zeros(dimension, device=device),
                        torch.ones(dimension, device=device))
            xvalue_distribution = Independent(m, 1)
        self.xvalue_distribution = xvalue_distribution

        if models is None: # models is None implies model_probabilities is None
            assert model_probabilities is None

            train_X = torch.zeros(0, dimension, device=device)
            train_Y = torch.zeros(0, 1, device=device)

            # Default: Matern 5/2 kernel with gamma priors on
            # lengthscale and outputscale, and noise level also if
            # observation_noise.
            # If no observation noise is generated, then make the likelihood
            # be fixed noise at almost zero to correspond to what is generated.
            likelihood = None if observation_noise else GaussianLikelihood(
                    noise_prior=None, batch_shape=torch.Size(),
                    noise_constraint=GreaterThan(
                        0.0, transform=None, initial_value=1e-6
                    )
                )
            models = [SingleTaskGP(train_X, train_Y, likelihood=likelihood)]
            model_probabilities = torch.tensor([1.0])
        
        for model in models:
            t = len(model.batch_shape)
            assert t == 0 or t == 1 and model.batch_shape[0] == 1
        
        self.model_sampler = RandomModelSampler(models, model_probabilities)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Get a random model
        model = self.model_sampler.sample()

        # pick the number of data points
        if self.n_datapoints is None: # then it's given by a distribution
            n_datapoints = self.n_datapoints_random_gen()
        else:
            n_datapoints = self.n_datapoints

        # generate the x-values
        x_values = self.xvalue_distribution.sample(torch.Size([n_datapoints]))
        assert x_values.dim() == 2 # should have shape (n_datapoints, dimension)
        
        # make x_values have 1 batch for Botorch
        x_values_botorch = x_values.unsqueeze(0) if len(model.batch_shape) == 1 else x_values

        with gpytorch.settings.prior_mode(True): # sample from prior
            prior = model.posterior(
                x_values_botorch, 
                observation_noise=self.observation_noise)

        # shape (batch_shape, n_datapoints, 1)
        y_values = prior.sample(torch.Size([]))

        if self.set_random_model_train_data:
            # As a hack, need to remove last dimension of y_values because
            # set_train_data isn't really supported in BoTorch
            model.set_train_data(
                x_values_botorch, y_values.squeeze(-1), strict=False)

        return x_values, y_values.squeeze(), model

    def save_realizations(self, n_realizations, dir_name):
        dataset = FunctionSamplesDataset.from_gp_random_dataset(self, n_realizations)
        dataset.save(dir_name)


class FunctionSamplesDataset(Dataset):
    def __init__(self, data: List[dict]):
        self.data = data
    
    @classmethod
    def from_gp_random_dataset(cls, rand_dataset: GaussainProcessRandomDataset, n_realizations: int):
        samples_list = []
        for i in range(n_realizations):
            x_values, y_values, model = next(rand_dataset)
            model_params = {name: param.detach()
                            for name, param in model.named_parameters()}
            samples_list.append({
                'x_values': x_values,
                'y_values': y_values,
                'model_index': model.index,
                'model_params': model_params
            })
        ret = cls(samples_list)
        ret._model_sampler = rand_dataset.model_sampler
        return ret

    @classmethod
    def load(cls, dir_name):
        if not os.path.exists(dir_name): # Error if path doesn't exist
            raise FileNotFoundError(f"Path {dir_name} does not exist")
        if not os.path.isdir(dir_name): # Error if path isn't directory
            raise NotADirectoryError(f"Path {dir_name} is not a directory")

        data = torch.load(os.path.join(dir_name, "data.pt"))
        ret = cls(data)
        
        models_path = os.path.join(dir_name, "models.pt")
        if os.path.exists(models_path):
            models = torch.load(models_path)
            model_sampler = RandomModelSampler(models)
            ret._model_sampler = model_sampler
        
        return ret

    def save(self, dir_name):
        if os.path.exists(dir_name):
            if not os.path.isdir(dir_name):
                raise NotADirectoryError(f"Path {dir_name} is not a directory")
        else:
            os.mkdir(dir_name)

        # Save the models if we have them
        if self.has_models:
            models = self.model_sampler.initial_models
            torch.save(models, os.path.join(dir_name, "models.pt"))

        torch.save(self.data, os.path.join(dir_name, "data.pt"))

    @property
    def has_models(self):
        return hasattr(self, '_model_sampler')

    @property
    def model_sampler(self):
        if not self.has_models:
            raise ValueError(f"This {self.__class__.__name__} does not have models")
        return self._model_sampler
    
    def __getitem__(self, index):
        item = self.data[index]
        if self.has_models:
            model_index = item['model_index']
            model_params = item['model_params']
            model = self.model_sampler.get_model(model_index, model_params)
            return item['x_values'], item['y_values'], model
        else:
            return item['x_values'], item['y_values']
    
    def __len__(self):
        return len(self.data)


class _FirstNIterable:
    """
    Creates an iterable for the first 'n' elements of a given iterable.

    Takes any iterable and an integer 'n', and provides an iterator
    that yields the first 'n' elements of the given iterable. If the original
    iterable contains fewer than 'n' elements, the iterator will yield only the
    available  elements without raising an error.

    Args:
        iterable (iterable): The iterable to wrap.
        n (int): The number of elements to yield from the iterable.

    Example:
        >>> numbers = range(10)  # A range object is an iterable
        >>> first_five = _FirstNIterable(numbers, 5)
        >>> list(first_five)
        [0, 1, 2, 3, 4]

        >>> words = ["apple", "banana", "cherry", "date"]
        >>> first_two = _FirstNIterable(words, 2)
        >>> list(first_two)
        ['apple', 'banana']
    """
    def __init__(self, iterable, n):
        self.iterable = iterable
        self.n = n
    
    def __iter__(self):
        iterator = iter(self.iterable)
        for _ in range(self.n):
            try:
                yield next(iterator)
            except StopIteration:
                break


class TrainAcquisitionFunctionDataset(IterableDataset):
    def __init__(self,
                 dataset: Union[GaussainProcessRandomDataset, FunctionSamplesDataset],
                 n_candidate_points:int=1, dataset_size=None,
                 give_improvements:bool=True):
        if isinstance(dataset, GaussainProcessRandomDataset):
            self.has_models = True
            if dataset_size is None:
                self.data_iterable = dataset
            else:
                self.data_iterable = _FirstNIterable(dataset, dataset_size)
        elif isinstance(dataset, FunctionSamplesDataset):
            self.has_models = dataset.has_models
            if dataset_size is None:
                self.data_iterable = DataLoader(
                    dataset, batch_size=None, shuffle=True)
            else:
                self.data_iterable = DataLoader(
                    dataset, batch_size=None,
                    sampler=torch.utils.data.RandomSampler(
                        dataset, replacement=False, num_samples=dataset_size))
        else:
            raise TypeError(
                "dataset should be of type GaussainProcessRandomDataset or FunctionSamplesDataset")

        if not isinstance(n_candidate_points, int) or n_candidate_points <= 0:
            raise ValueError(f"n_candidate_points should be a positive integer, but got n_candidate_points={n_candidate_points}")
        if not isinstance(give_improvements, bool):
            raise TypeError(f"give_improvements should be a boolean value, but got give_improvements={give_improvements}")

        self.n_candidate_points = n_candidate_points
        self.give_improvements = give_improvements
    
    def __iter__(self):
        n_candidate = self.n_candidate_points
        has_models = self.has_models
        
        # x_values has shape (n_datapoints, dimension)
        # y_values has shape (n_datapoints,)
        for item in self.data_iterable:
            if has_models:
                x_values, y_values, model = item
            else:
                x_values, y_values = item
            
            n_datapoints = x_values.shape[0]

            rand_idx = torch.randperm(n_datapoints)
            candidate_idx = rand_idx[:n_candidate]
            hist_idx = rand_idx[n_candidate:]

            x_hist, y_hist = x_values[hist_idx], y_values[hist_idx]
            x_candidates = x_values[candidate_idx]
            y_candidates = y_values[candidate_idx]

            if self.give_improvements:
                best_f = y_hist.amax(0, keepdim=False) # both T and F work
                improvement_values = torch.nn.functional.relu(
                    y_candidates - best_f, inplace=True)
                ret_y = improvement_values
            else:
                ret_y = y_candidates

            if has_models:
                yield x_hist, y_hist, x_candidates, ret_y, model
            else:
                yield x_hist, y_hist, x_candidates, ret_y
