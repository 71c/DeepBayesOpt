from typing import Callable
import numpy as np
import torch
from torch import Tensor
from run_train_transfer_bo_baseline import get_checkpoint_path_for_transfer_bo_baseline
from bayesopt.bayesopt import BayesianOptimizer
from transfer_bo_baselines.fsbo.fsbo_modules import DeepKernelGP


class FSBOOptimizer(BayesianOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 bounds: Tensor,
                 dataset_hash: str):
        super().__init__(dim, maximize, initial_points, objective, bounds)
        
        checkpoint_path = get_checkpoint_path_for_transfer_bo_baseline(
            'FSBO', dataset_hash)
        
        randomInitializer = np.random.RandomState(0)
        torch_seed = randomInitializer.randint(0,100000)

        self.method = DeepKernelGP(
            dim, torch_seed, log_dir=None, epochs=100, load_model=True,
            checkpoint=checkpoint_path, verbose=True)

    def get_new_point(self) -> Tensor:
        ret = self.method.observe_and_suggest(self.x, self.y)
        return torch.tensor(ret)
