"""
Experiment Registry Module

Provides functionality to load, validate, and manage experiment configurations
from the centralized registry.
"""

from abc import ABC, abstractmethod
import yaml
from typing import Dict, List, Any
from pathlib import Path


class ExperimentRegistryBase(ABC):
    """Base class for central registry for managing experiment configurations."""

    @abstractmethod
    def get_seeds_cfg(params: Dict[str, Any]) -> str:
        pass
    
    def __init__(self, registry_path: str):
        """Initialize the registry."""
        if registry_path is None:
            raise ValueError("registry_path should be specified")
        
        self.registry_path = Path(registry_path)
        self._registry_data = None
        
    def load_registry(self) -> Dict[str, Any]:
        """Load the registry from YAML file."""
        if self._registry_data is None:
            if not self.registry_path.exists():
                raise FileNotFoundError(f"Registry file not found: {self.registry_path}")
            
            with open(self.registry_path, 'r') as f:
                self._registry_data = yaml.safe_load(f)
        
        return self._registry_data
    
    def get_experiment(self, name: str) -> Dict[str, Any]:
        """Get experiment configuration by name."""
        registry = self.load_registry()
        experiments = registry.get('experiments', {})
        
        if name not in experiments:
            available = list(experiments.keys())
            raise ValueError(f"Experiment '{name}' not found. Available: {available}")
        
        return experiments[name]
    
    def list_experiments(self) -> List[str]:
        """List all available experiment names."""
        registry = self.load_registry()
        return list(registry.get('experiments', {}).keys())
    
    def get_template(self, name: str) -> Dict[str, Any]:
        """Get template configuration by name."""
        registry = self.load_registry()
        templates = registry.get('templates', {})
        
        if name not in templates:
            available = list(templates.keys())
            raise ValueError(f"Template '{name}' not found. Available: {available}")
        
        return templates[name]
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        registry = self.load_registry()
        return list(registry.get('templates', {}).keys())
    
    def get_experiment_command_args(self, name: str) -> Dict[str, str]:
        """Generate command arguments for an experiment."""
        exp_config = self.get_experiment(name)
        params = exp_config['parameters']
        
        args = {}
        
        # Basic parameters
        args['SEED'] = str(params['seed'])
        args['TRAIN_EXPERIMENT_CFG'] = f'"{params["train_experiment_config"]}"'
        args['RUN_EXPERIMENT_CFG'] = f'"{params["run_experiment_config"]}"'
        args['SWEEP_NAME'] = f'"{params["sweep_name"]}"'
        args['PLOTS_GROUP_NAME'] = f'"{params["plots_group_name"]}"'
        args['RUN_PLOTS_NAME'] = f'"{params["run_plots_name"]}"'

        seeds_cfg = self.get_seeds_cfg(params)
        args['SEEDS_CFG'] = f'"{seeds_cfg}"'
        
        args['TRAIN_CFG'] = f'"--train_base_config config/train_acqf.yml --train_experiment_config {params["train_experiment_config"]}"'
        args['RUN_CFG'] = f'"--run_base_config config/bo_config.yml --run_experiment_config {params["run_experiment_config"]} --seed {params["seed"]} {seeds_cfg}"'
        args['SLURM_CFG'] = f'"--sweep_name {params["sweep_name"]} --mail adj53@cornell.edu --gres gpu:1"'
        args['PLOTS_CFG'] = f'"--plots_group_name {params["plots_group_name"]}"'
        
        return args
    
    def get_plotting_config(self, name: str, plot_type: str, variant: str = 'default') -> Dict[str, Any]:
        """Get plotting configuration for an experiment."""
        exp_config = self.get_experiment(name)
        plotting = exp_config.get('plotting', {})
        
        if plot_type not in plotting:
            raise ValueError(f"Plot type '{plot_type}' not found for experiment '{name}'")
        
        plot_config = plotting[plot_type]
        
        # Handle both old format (direct config) and new format (default + alternatives)
        if 'default' in plot_config:
            # New format with default + alternatives
            if variant == 'default':
                return plot_config['default']
            elif 'alternatives' in plot_config and variant in plot_config['alternatives']:
                return plot_config['alternatives'][variant]
            else:
                available = ['default'] + list(plot_config.get('alternatives', {}).keys())
                raise ValueError(f"Plot variant '{variant}' not found for experiment '{name}' plot type '{plot_type}'. Available: {available}")
        else:
            # Old format (direct config) - treat as default
            if variant != 'default':
                raise ValueError(f"Plot variant '{variant}' not available for experiment '{name}' plot type '{plot_type}'. Only 'default' available.")
            return plot_config
    
    def list_plot_variants(self, name: str, plot_type: str) -> List[str]:
        """List available plot variants for an experiment and plot type."""
        exp_config = self.get_experiment(name)
        plotting = exp_config.get('plotting', {})
        
        if plot_type not in plotting:
            return []
        
        plot_config = plotting[plot_type]
        
        if 'default' in plot_config:
            # New format
            variants = ['default']
            if 'alternatives' in plot_config:
                variants.extend(plot_config['alternatives'].keys())
            return variants
        else:
            # Old format
            return ['default']
