"""
Experiment Registry Module

Provides functionality to load, validate, and manage experiment configurations
from the centralized registry.
"""

from abc import ABC, abstractmethod
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class ExperimentRegistryBase(ABC):
    """Base class for central registry for managing experiment configurations."""

    @abstractmethod
    def get_seeds_cfg(self, params: Dict[str, Any]) -> str:
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
    
    def _detect_experiment_from_config(self, train_experiment_config: str) -> Optional[str]:
        """
        Detect which experiment is being used based on the train_experiment_config path.
        Returns the experiment name if found, None otherwise.
        """
        experiments = self.list_experiments()
        
        # Normalize the config path for comparison
        config_path = train_experiment_config.strip('"').strip("'")
        
        for exp_name in experiments:
            try:
                exp_config = self.get_experiment(exp_name)
                exp_nn_config = exp_config['parameters']['train_experiment_config']
                
                if exp_nn_config == config_path:
                    return exp_name
            except:
                continue
        
        return None

    def _get_plotting_config_for_experiment(self, experiment_name: str, plot_type: str, variant: str = 'default') -> Optional[Dict[str, Any]]:
        try:
            return self.get_plotting_config(experiment_name, plot_type, variant)
        except:
            return None

    def _auto_configure_plotting(self, train_experiment_config: str, plot_type: str, variant: str = 'default') -> Tuple[List, List, List]:
        """
        Automatically configure plotting parameters based on experiment configuration.
        
        Args:
            train_experiment_config: Path to the NN experiment config file
            plot_type: 'train_plot' or 'run_plot'
            variant: Plot configuration variant (default: 'default')
        
        Returns:
            Tuple of (PRE, ATTR_A, ATTR_B) lists
        """
        # Try to detect the experiment
        experiment_name = self._detect_experiment_from_config(train_experiment_config)
        
        if experiment_name:
            plot_config = self._get_plotting_config_for_experiment(experiment_name, plot_type, variant)
            
            if plot_config:
                pre = plot_config.get('pre', [])
                attr_a = plot_config.get('attr_a', [])
                attr_b = plot_config.get('attr_b', [])
                
                print(f"Auto-detected experiment: {experiment_name}")
                print(f"Using plotting configuration for {plot_type} (variant: {variant}):")
                print(f"  PRE = {pre}")
                print(f"  ATTR_A = {attr_a}")
                print(f"  ATTR_B = {attr_b}")
                
                return pre, attr_a, attr_b
        
        raise ValueError(
            "Could not auto-detect plotting configuration for experiment "
            f"with config: {train_experiment_config}")

    def _apply_auto_plotting(self, train_experiment_config: str, plot_type: str, 
                        globals_dict: Dict[str, Any],
                        variant: str = 'default') -> None:
        """
        Apply automatic plotting configuration to a script's global variables.
        
        This function modifies the global PRE, ATTR_A, ATTR_B variables in place.
        
        Args:
            train_experiment_config: Path to the NN experiment config file
            plot_type: 'train_plot' or 'run_plot'  
            globals_dict: The global namespace dictionary of the calling script
        """
        pre, attr_a, attr_b = self._auto_configure_plotting(train_experiment_config, plot_type, variant)
        
        # Update the global variables
        globals_dict['PRE'] = pre
        globals_dict['ATTR_A'] = attr_a
        globals_dict['ATTR_B'] = attr_b

    def _get_experiment_post_config(self, experiment_name: str, plot_type: str, variant: str = 'default') -> Optional[List]:
        """
        Get the POST configuration for an experiment if it exists.
        
        Args:
            experiment_name: Name of the experiment
            plot_type: 'train_plot' or 'run_plot'
            variant: Plot configuration variant (default: 'default')
        
        Returns:
            POST configuration list or None if not found
        """
        plot_config = self._get_plotting_config_for_experiment(experiment_name, plot_type, variant)
        if plot_config:
            return plot_config.get('post', None)
        return None

    def setup_plotting_from_args(self, args, plot_type: str, globals_dict: Dict[str, Any]) -> None:
        """
        Set up plotting configuration from command line arguments.
        
        This is designed to be called from plotting scripts that use argparse.
        
        Args:
            args: Parsed arguments from argparse
            plot_type: 'train_plot' or 'run_plot'
            globals_dict: The global namespace dictionary of the calling script
        """
        nn_config = getattr(args, 'train_experiment_config', None)
        variant = getattr(args, 'variant', 'default')
        if nn_config:
            self._apply_auto_plotting(nn_config, plot_type, globals_dict, variant=variant)

            # Also try to set POST if available
            experiment_name = self._detect_experiment_from_config(nn_config)
            if experiment_name:
                post_config = self._get_experiment_post_config(experiment_name, plot_type, variant)
                if post_config:
                    globals_dict['POST'] = post_config
                else:
                    globals_dict['POST'] = []  # Default empty if not specified
