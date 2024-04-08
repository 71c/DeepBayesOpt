import torch
import gpytorch
import pyro
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.gp_regression import SingleTaskGP

from torch.distributions import Uniform, Normal, Independent, Distribution
from torch.utils.data import Dataset, IterableDataset, DataLoader
from utils import get_uniform_randint_generator

from typing import List, Union
from collections.abc import Sequence

import os
import warnings

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
    """A class that samples a random model with random parameters from its prior
    from a list of models."""
    def __init__(self, models: List[SingleTaskGP], model_probabilities=None):
        """Initializes the RandomModelSampler instance.

        Args:
            models (List[SingleTaskGP]): A list of SingleTaskGP models to choose
                from randomly, with their priors.
            model_probabilities (Tensor, optional): 1D Tensor of probabilities
                OF choosing each model. If None, then set to be uniform.
        """
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
            # Remove the data from the model.
            # Basically equivalent to
            # model.set_train_data(inputs=None, targets=None, strict=False)
            # except that would just do nothing
            model.train_inputs = None
            model.train_targets = None
            model.prediction_strategy = None

            # Set the parameters of the model to be the initial ones
            model.initialize(**params)
        return self._models


class GaussainProcessRandomDataset(IterableDataset):
    """An IterableDataset that generates random Gaussian Process data.

     Usage example:
    ```
    dataset = GaussainProcessRandomDataset(n_datapoints=15, dimension=5)
    for x_values, y_values, model in dataset:
        # x_values has shape (n_datapoints, dimension)
        # y_values has shape (n_datapoints,)
        # do something with x_values, y_values, and model
    ```
    """
    def __init__(self, n_datapoints:int=None, n_datapoints_random_gen=None,
                 observation_noise:bool=False,
                 xvalue_distribution: Distribution=None,
                 models: List[SingleTaskGP]=None,
                 model_probabilities=None,
                 dimension:int=None, device=None,
                 set_random_model_train_data=False):
        """Create a dataset that generates random Gaussian Process data.
        
        Args:
            n_datapoints: number of (x,y) pairs to generate with each sample;
                could be None
            n_datapoints_random_gen: a callable that returns a random natural
                number that is the number of datapoints.
                Note: exactly one of n_datapoints and n_datapoints_random_gen
                should be speicified (not be None).
            observation_noise: boolean specifying whether to generate the data
                to include the "observation noise" given by the model's
                likelihood (True), or not (False)
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
            dimension: int, optional -- The dimension d of the feature space.
                Is only used if xvalue_distribution is None or models is None;
                in this case, it is required. Otherwise, it is ignored.
            device: torch.device, optional -- the desired device to use for
                computations. Is only used if xvalue_distribution is None or
                models is None; otherwise, it is ignored.
            set_random_model_train_data (bool, default: False):
                Whether to set the random model train data to the random data
                with each returned values
        """
        # exacly one of them should be specified; verify this by xor
        if not ((n_datapoints is None) ^ (n_datapoints_random_gen is None)):
            raise ValueError("Exactly one of n_datapoints and n_datapoints_random_gen should be specified.")
        if n_datapoints is not None and (not isinstance(n_datapoints, int) or n_datapoints <= 0):
            raise ValueError("n_datapoints should be a positive integer.")
        self.n_datapoints = n_datapoints
        self.n_datapoints_random_gen = n_datapoints_random_gen
        
        if not isinstance(observation_noise, bool):
            raise TypeError("observation_noise must be a boolean value.")
        self.observation_noise = observation_noise

        if dimension is not None and (not isinstance(dimension, int) or dimension <= 0):
            raise ValueError("dimension should be a positive integer.")

        if not isinstance(set_random_model_train_data, bool):
            raise TypeError("set_random_model_train_data must be a boolean value.")
        self.set_random_model_train_data = set_random_model_train_data
        
        if xvalue_distribution is None:
            if dimension is None:
                raise ValueError("dimension should be specified if xvalue_distribution is None")
            # m = Normal(torch.zeros(dimension), torch.ones(dimension))
            m = Uniform(torch.zeros(dimension, device=device),
                        torch.ones(dimension, device=device))
            xvalue_distribution = Independent(m, 1)
        self.xvalue_distribution = xvalue_distribution

        if models is None: # models is None implies model_probabilities is None
            if model_probabilities is not None:
                raise ValueError("model_probabilities should be None if models is None")
            if dimension is None:
                raise ValueError("dimension should be specified if models is None")

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
        
        # Verify that all the models are single-batch
        for model in models:
            t = len(model.batch_shape)
            assert t == 0 or t == 1 and model.batch_shape[0] == 1
        
        self.model_sampler = RandomModelSampler(models, model_probabilities)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Generate a random Gaussian Process model and sample from it.

        Returns:
            Tuple `(x_values, y_values, model)` where `model` is a
            `SingleTaskGP` instance that was used to generate the data.
            `x_values` has shape `(n_datapoints, dimension)`, and
            `y_values` has shape `(n_datapoints,)`.
        """
        # Get a random model
        model = self.model_sampler.sample()

        # pick the number of data points
        if self.n_datapoints is None: # then it's given by a distribution
            n_datapoints = self.n_datapoints_random_gen()
            if not isinstance(n_datapoints, int) or n_datapoints <= 0:
                raise ValueError("n_datapoints_random_gen should return a positive integer.")
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

    def save_realizations(self, n_realizations:int, dir_name:str):
        """Save the realizations of the dataset to a directory.

        Args:
            n_realizations (int): The number of realizations to save.
            dir_name (str): The directory where the realizations should be saved.
        """
        dataset = FunctionSamplesDataset.from_gp_random_dataset(self, n_realizations)
        dataset.save(dir_name)


class FunctionSamplesDataset(Dataset):
    """A dataset class that holds function samples, as well as optionally
    their associated GP models and parameters.

    Attributes:
        data (List[dict]): The dataset, where each item is a dictionary
            containing 'x_values' and 'y_values', and also 'model_index' and
            'model_params' if models are associated with the dataset.
        model_sampler (RandomModelSampler): The random model sampler, if models
            are associated with the dataset.

    Example:
        ```
        rand_dataset = GaussainProcessRandomDataset(n_datapoints=15, dimension=5)
        function_samples_dataset = FunctionSamplesDataset.from_gp_random_dataset(rand_dataset, 100)
        function_samples_dataset.save('path/to/directory')
        loaded_dataset = FunctionSamplesDataset.load('path/to/directory')
    """
    def __init__(self, data: List[dict], model_sampler:RandomModelSampler=None):
            """
            Initializes an instance of the class with the given data.

            Args:
                data (List[dict]): A list of dictionaries containing the data.
                model_sampler: Optional RandomModelSampler instance to associate
            """
            self.data = data
            self._model_sampler = model_sampler
    
    @classmethod
    def from_gp_random_dataset(cls, dataset: GaussainProcessRandomDataset,
                               n_realizations: int):
        """Creates an instance of FunctionSamplesDataset from a given
        GaussainProcessRandomDataset instance by sampling a specified number of
        data points.

        Args:
            dataset (GaussainProcessRandomDataset):
                The random GP dataset from which to generate samples.
            n_realizations (int, positive):
                The number of function realizations (samples) to generate.

        Returns:
            FunctionSamplesDataset:
            A new instance of FunctionSamplesDataset containing the sampled
            function realizations.

        Example:
            ```
            dataset = GaussainProcessRandomDataset(n_datapoints=15, dimension=5)
            samples_dataset = FunctionSamplesDataset.from_gp_random_dataset(dataset, 100)
        """
        # if not isinstance(dataset, GaussainProcessRandomDataset):
        #     raise TypeError("dataset should be an instance of GaussainProcessRandomDataset")
        if not isinstance(n_realizations, int) or n_realizations <= 0:
            raise ValueError("n_realizations should be a positive integer")
        iterator = iter(dataset)
        samples_list = []
        for i in range(n_realizations):
            x_values, y_values, model = next(iterator)
            # need to copy the data, otherwise everything will be the same
            model_params = {name: param.detach().clone()
                            for name, param in model.named_parameters()}
            samples_list.append({
                'x_values': x_values,
                'y_values': y_values,
                'model_index': model.index,
                'model_params': model_params
            })
        return cls(samples_list, dataset.model_sampler)

    @classmethod
    def load(cls, dir_name: str):
        """
        Loads a dataset from a given directory. The directory must contain a
        saved instance of FunctionSamplesDataset, including the data and
        optionally the models.

        Args:
            dir_name (str): The path to the directory from which the dataset
            should be loaded.

        Returns:
            FunctionSamplesDataset: The loaded dataset instance.
        """
        if not os.path.exists(dir_name): # Error if path doesn't exist
            raise FileNotFoundError(f"Path {dir_name} does not exist")
        if not os.path.isdir(dir_name): # Error if path isn't directory
            raise NotADirectoryError(f"Path {dir_name} is not a directory")

        data = torch.load(os.path.join(dir_name, "data.pt"))
        
        models_path = os.path.join(dir_name, "models.pt")
        if os.path.exists(models_path):
            models = torch.load(models_path)
            model_sampler = RandomModelSampler(models)
            return cls(data, model_sampler)
        
        return cls(data)

    def save(self, dir_name: str):
        """
        Saves the dataset to a specified directory. If the dataset includes
        models, the models are saved as well. If the directory does not
        exist, it will be created.

        Args:
            dir_name (str): The path to the directory where the dataset should
            be saved.
        """
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
        """Boolean variable that is whether the dataset includes model
        information (i.e., GP models and their parameters)."""
        return self._model_sampler is not None

    @property
    def model_sampler(self):
        if not self.has_models:
            raise ValueError(f"This {self.__class__.__name__} does not have models")
        return self._model_sampler
    
    def __getitem__(self, index):
        """Retrieves a single sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: If the dataset includes models, returns a tuple
            (x_values, y_values, model), where 'model' is the GP model that
            generated the sample. Otherwise, returns a tuple(x_values, y_values)
        """
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
    """
    An IterableDataset designed for training a "likelihood-free" DNN acquisition
    function.
    It processes either a GaussainProcessRandomDataset or FunctionSamplesDataset
    instance to generate training data consisting of historical observations and
    candidate points for acquisition function evaluation.
    The data is generated randomly on-the-fly.

    Attributes:
        n_candidate_points (int): The number of candidate points to generate for
            each training example.
        give_improvements (bool): If True, the dataset includes improvement
            values as targets instead of raw y-values of the candidate points.
            Improvement is calculated as the positive difference between the
            candidate point's y-value and the current best observation.

    Yields:
        Tuple containing historical x-values, historical y-values, candidate
        x-values, and either candidate y-values or improvement values, depending
        on the value of `give_improvements`. If `dataset` has model info, then
        each item also includes the associated GP model.

    Example:
        dataset = GaussainProcessRandomDataset(n_datapoints=15, dimension=5)
        
        # Creating the training dataset for acquisition functions
        training_dataset = TrainAcquisitionFunctionDataset(
            dataset=dataset, n_candidate_points=5, give_improvements=True)
        
        # Iterating over the dataset to train an acquisition function
        for x_hist, y_hist, x_cand, improvements, model in training_dataset:
            # Use x_hist, y_hist, x_cand, and improvements for training
            # and model for evaluation of the approximated acquisition function
    """
    def __init__(self,
                 dataset: Union[GaussainProcessRandomDataset, FunctionSamplesDataset],
                 dataset_size:int=None, n_candidate_points:Union[int,str,Sequence[int]]=1,
                 n_samples:str="all", give_improvements:bool=True, min_n_candidates=2):
        """
        Args:
            dataset: The base dataset from which to generate training data for
                acquisition functions.
            dataset_size (Optional[int]): The size of the dataset to generate
                (i.e. number of samples). If None, dataset size is determined by
                the size of the base dataset.
            n_candidate_points (default: 1): The number of candidate points
                to generate for each training example.
                Can be:
                    - A positive integer, in which case the number of candidate
                        points is fixed to that value.
                    - A tuple of two positive integers (min, max), in which case
                        the number of candidate points is chosen uniformly at
                        random from min to max.
                    - A string "uniform", in which case the number of candidate
                        points is chosen uniformly at random from
                        `[min_n_candidates...n_samples-1]`.
                    - A string "binomial", in which case the number of candidate
                        points is chosen from a binomial distribution with
                        parameters n_samples and 0.5, conditioned on being
                        between `min_n_candidates` and n_samples-1.
            n_samples (str; default: "all"): The number of samples to use from
                the dataset each iteration. If "all", all samples are used.
                If "uniform", a uniform random number of samples is used each
                iteration. Specifically,
                    - If n_candidate_points is "uniform" or "binomial", then
                    n_samples is chosen uniformly at random in
                    [min_n_candidates+1...n], where n is the number of samples
                    in the dataset iteration, and then n_candidate_points is
                    chosen based on that.
                    - If n_candidate_points is an integer or a tuple of two
                    integers, then n_candidate_points is first chosen and then
                    n_samples is chosen uniformly in [n_candidate_points+1...n].
            give_improvements (bool): Whether to generate improvement values as
                targets instead of raw y-values of the candidate points.
            min_n_candidates (int): The minimum number of candidate points for
                every iteration. Only used if n_candidate_points is "uniform" or
                "binomial".
            
              min_n_candidates <= n_candidate <= n_samples-1
        ===>  min_n_candidates <= n_samples-1
        ===>  n_samples >= min_n_candidates + 1
        """
        # whether to generate `n_candidates` first or not
        self._gen_n_candidates_first = True
        if isinstance(n_candidate_points, str):
            if not (n_candidate_points == "uniform" or n_candidate_points == "binomial"):
                raise ValueError(f"Invalid value for n_candidate_points: {n_candidate_points}")
            self._gen_n_candidates_first = False
        elif isinstance(n_candidate_points, int):
            if n_candidate_points <= 0:
                raise ValueError(f"n_candidate_points should be positive, but got n_candidate_points={n_candidate_points}")
            self._gen_n_candidates = lambda: n_candidate_points
        else: # n_candidate_points is a tuple (or list) of two integers
            try:
                if not (len(n_candidate_points) == 2 and
                        isinstance(n_candidate_points[0], int) and
                        isinstance(n_candidate_points[1], int) and
                        1 <= n_candidate_points[0] <= n_candidate_points[1]):
                    raise ValueError(f"n_candidate_points should be a positive integer or a tuple of two integers, but got n_candidate_points={n_candidate_points}")
                self._gen_n_candidates = get_uniform_randint_generator(*n_candidate_points)
            except TypeError:
                raise ValueError(f"n_candidate_points should be a string, positive integer, tuple of two integers, but got n_candidate_points={n_candidate_points}")
        self.n_candidate_points = n_candidate_points

        if n_samples == "all":
            self.n_samples = "all"
        elif n_samples == "uniform":
            self.n_samples = "uniform"
        else:
            raise ValueError(f"Invalid value for n_samples: {n_samples}")
        
        if not isinstance(min_n_candidates, int) or min_n_candidates <= 0:
            raise ValueError(f"min_n_candidates should be a positive integer, but got min_n_candidates={min_n_candidates}")
        self.min_n_candidates = min_n_candidates
        
        if not isinstance(give_improvements, bool):
            raise TypeError(f"give_improvements should be a boolean value, but got give_improvements={give_improvements}")
        self.give_improvements = give_improvements

        if dataset_size is not None and (not isinstance(dataset_size, int) or dataset_size <= 0):
            raise ValueError(f"dataset_size should be a positive integer, but got dataset_size={dataset_size}")
        
        if isinstance(dataset, GaussainProcessRandomDataset):
            self.has_models = True
            if dataset_size is None:
                self.data_iterable = dataset
            else:
                self.data_iterable = _FirstNIterable(dataset, dataset_size)
            
            if n_samples == "uniform":
                warnings.warn("n_samples='uniform' for GaussainProcessRandomDataset is supported but wasteful. Consider using n_samples='all' and setting n_datapoints_random_gen in the GaussainProcessRandomDataset instead.")
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
    
    def _pick_random_n_samples_and_n_candidates(self, n_datapoints_original):
        if self._gen_n_candidates_first:
            # generate n_candidates first; either fixed or random
            n_candidates = self._gen_n_candidates()

            # Need to have at least 1 history point
            if not (n_candidates+1 <= n_datapoints_original):
                raise ValueError(f"n_datapoints_original={n_datapoints_original} should be at least n_candidates+1={n_candidates+1}")

            # generate n_samples
            if self.n_samples == "all":
                n_samples = n_datapoints_original
            elif self.n_samples == "uniform":
                n_samples = get_uniform_randint_generator(n_candidates+1, n_datapoints_original)()
        else:
            # n_candidates is "uniform" or "binomial"

            min_n_candidates = self.min_n_candidates

            if not (min_n_candidates+1 <= n_datapoints_original):
                raise ValueError(f"n_datapoints_original={n_datapoints_original} should be at least min_n_candidates+1={min_n_candidates+1}")

            # generate n_samples first; either "all" or "uniform"
            if self.n_samples == "all":
                n_samples = n_datapoints_original
            elif self.n_samples == "uniform":
                n_samples = get_uniform_randint_generator(min_n_candidates+1, n_datapoints_original)()

            # generate n_candidates
            if self.n_candidate_points == "uniform":
                n_candidates = get_uniform_randint_generator(min_n_candidates, n_samples-1)()
            elif self.n_candidate_points == "binomial":
                n_candidates = int(torch.distributions.Binomial(n_samples, 0.5).sample())
                while not (min_n_candidates <= n_candidates <= n_samples-1):
                    n_candidates = int(torch.distributions.Binomial(n_samples, 0.5).sample())
        
        return n_samples, n_candidates

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

            # TODO: randomize n_samples and n_candidates

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
