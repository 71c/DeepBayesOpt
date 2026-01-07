from typing import Callable
import numpy as np
import torch
from torch import Tensor
from single_train_baseline import get_relative_checkpoints_path_for_transfer_bo_baseline
from bayesopt.bayesopt import BayesianOptimizer
from transfer_bo_baselines.fsbo.fsbo_modules import DeepKernelGP
from utils.basic_model_save_utils import BASIC_SAVING


class FSBOOptimizer(BayesianOptimizer):
    def __init__(self,
                 dim: int,
                 maximize: bool,
                 initial_points: Tensor,
                 objective: Callable,
                 bounds: Tensor,
                 dataset_hash: str):
        super().__init__(dim, maximize, initial_points, objective, bounds)
        
        relative_checkpoints_path = get_relative_checkpoints_path_for_transfer_bo_baseline(
            'FSBO', dataset_hash)
        checkpoint_path = BASIC_SAVING.get_latest_model_path(relative_checkpoints_path)
        
        randomInitializer = np.random.RandomState(0)
        torch_seed = randomInitializer.randint(0,100000)

        self.method = DeepKernelGP(
            dim, torch_seed, log_dir=None, epochs=100, load_model=True,
            checkpoint=checkpoint_path, verbose=True)

    def get_new_point(self) -> Tensor:
        ret = self.method.observe_and_suggest(self.x, self.y)
        return torch.tensor(ret)
