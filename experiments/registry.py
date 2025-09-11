"""
Experiment Registry Module

Provides functionality to load, validate, and manage experiment configurations
from the centralized registry.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path


class ExperimentRegistry:
    """Central registry for managing experiment configurations."""
    
    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the registry."""
        if registry_path is None:
            # Default to registry.yml in the experiments directory
            registry_path = Path(__file__).parent / "registry.yml"
        
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
    
    def validate_experiment(self, exp_config: Dict[str, Any]) -> bool:
        """Validate experiment configuration."""
        required_params = [
            'nn_experiment_config',
            'bo_experiment_config',
            'sweep_name',
            'n_seeds',
            'plots_group_name',
            'bo_plots_name'
        ]
        
        params = exp_config.get('parameters', {})
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Check if config files exist
        for config_key in ['nn_experiment_config', 'bo_experiment_config']:
            config_path = params[config_key]
            if not Path(config_path).exists():
                raise ValueError(f"Config file not found: {config_path}")
        
        return True
    
    def get_experiment_command_args(self, name: str) -> Dict[str, str]:
        """Generate command arguments for an experiment."""
        exp_config = self.get_experiment(name)
        params = exp_config['parameters']
        
        # Build argument strings like the original commands.txt format
        args = {}
        
        # Basic parameters
        args['SEED'] = str(params['seed'])
        args['NN_EXPERIMENT_CFG'] = f'"{params["nn_experiment_config"]}"'
        args['BO_EXPERIMENT_CFG'] = f'"{params["bo_experiment_config"]}"'
        args['SWEEP_NAME'] = f'"{params["sweep_name"]}"'
        args['PLOTS_GROUP_NAME'] = f'"{params["plots_group_name"]}"'
        args['BO_PLOTS_NAME'] = f'"{params["bo_plots_name"]}"'
        
        # Seeds configuration
        seeds_parts = [f"--n_seeds {params['n_seeds']}"]
        if params.get('single_objective', False):
            seeds_parts.append("--single_objective")
        args['SEEDS_CFG'] = f'"{" ".join(seeds_parts)}"'
        
        # Derived configurations (matching original commands.txt)
        args['NN_CFG'] = f'"--nn_base_config config/train_acqf.yml --nn_experiment_config {params["nn_experiment_config"]}"'
        args['BO_CFG'] = f'"--bo_base_config config/bo_config.yml --bo_experiment_config {params["bo_experiment_config"]} --seed {params["seed"]} {" ".join(seeds_parts)}"'
        args['SLURM_CFG'] = f'"--sweep_name {params["sweep_name"]} --mail adj53@cornell.edu --gres gpu:1"'
        args['PLOTS_CFG'] = f'"--plots_group_name {params["plots_group_name"]} --use_rows --use_cols"'
        
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


def get_registry() -> ExperimentRegistry:
    """Get the default experiment registry instance."""
    return ExperimentRegistry()