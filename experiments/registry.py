"""
Experiment Registry Module

Provides functionality to load, validate, and manage experiment configurations
from the centralized registry.
"""

from typing import Any, Dict
from utils.constants import REGISTRY_PATH
from utils_general.experiments.registry import ExperimentRegistryBase


class ExperimentRegistry(ExperimentRegistryBase):
    """Central registry for managing experiment configurations."""
    
    def get_seeds_cfg(self, params: Dict[str, Any]) -> str:
        # Seeds configuration
        seeds_parts = []
        # n_seeds is optional when use_hpob_seeds is True (HPO-B defines its own seeds)
        if 'n_seeds' in params:
            seeds_parts.append(f"--n_seeds {params['n_seeds']}")

        n_objectives = params.get('n_objectives', None)
        if n_objectives is not None:
            seeds_parts.append(f"--n_objectives {n_objectives}")

        use_hpob_seeds = params.get('use_hpob_seeds', False)
        if use_hpob_seeds:
            seeds_parts.append("--use_hpob_seeds")
        
        return " ".join(seeds_parts)


# Global singleton registry instance
REGISTRY = ExperimentRegistry(REGISTRY_PATH)
