#!/usr/bin/env python3
"""
Comprehensive validation script to ensure all experiments in the registry
exactly match their counterparts in commands.txt.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.registry import ExperimentRegistry
from experiments.runner import ExperimentRunner


def parse_commands_txt() -> Dict[str, Dict[str, str]]:
    """Parse commands.txt and extract all experiment configurations."""
    commands_file = project_root / "commands.txt"
    
    with open(commands_file, 'r') as f:
        content = f.read()
    
    # Split by experiment blocks (each starts with a comment and SEED=8)
    experiments = {}
    
    # Find all experiment blocks
    pattern = r'#\s*(.+?)\nSEED=8;\nNN_EXPERIMENT_CFG="(.+?)";\nBO_EXPERIMENT_CFG="(.+?)";\nSWEEP_NAME="(.+?)";\nSEEDS_CFG="(.+?)";\nPLOTS_GROUP_NAME="(.+?)";\nBO_PLOTS_NAME="(.+?)";'
    
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        description, nn_config, bo_config, sweep_name, seeds_cfg, plots_group, bo_plots_name = match
        
        # Clean up description
        description = description.strip()
        if description.endswith(':'):
            description = description[:-1]
        
        # Parse seeds configuration
        single_objective = "--single_objective" in seeds_cfg
        n_seeds_match = re.search(r'--n_seeds (\d+)', seeds_cfg)
        n_seeds = int(n_seeds_match.group(1)) if n_seeds_match else 16
        
        # Create experiment key from config file
        config_key = nn_config
        
        experiments[config_key] = {
            'description': description,
            'nn_experiment_config': nn_config,
            'bo_experiment_config': bo_config,
            'sweep_name': sweep_name,
            'n_seeds': n_seeds,
            'single_objective': single_objective,
            'plots_group_name': plots_group,
            'bo_plots_name': bo_plots_name
        }
    
    return experiments


def validate_experiment_parameters(registry: ExperimentRegistry) -> List[str]:
    """Validate that all experiment parameters match commands.txt."""
    commands_experiments = parse_commands_txt()
    registry_experiments = registry.list_experiments()
    
    errors = []
    
    print(f"Commands.txt has {len(commands_experiments)} experiments")
    print(f"Registry has {len(registry_experiments)} experiments")
    
    # Check that we have the same number of experiments
    if len(commands_experiments) != len(registry_experiments):
        errors.append(f"Count mismatch: commands.txt has {len(commands_experiments)}, registry has {len(registry_experiments)}")
    
    # For each experiment in registry, find matching one in commands.txt and compare
    for exp_name in registry_experiments:
        exp_config = registry.get_experiment(exp_name)
        nn_config = exp_config['parameters']['nn_experiment_config']
        
        if nn_config not in commands_experiments:
            errors.append(f"Registry experiment '{exp_name}' with config '{nn_config}' not found in commands.txt")
            continue
        
        commands_exp = commands_experiments[nn_config]
        registry_params = exp_config['parameters']
        
        # Compare each parameter
        comparisons = [
            ('nn_experiment_config', 'nn_experiment_config'),
            ('bo_experiment_config', 'bo_experiment_config'),
            ('sweep_name', 'sweep_name'),
            ('n_seeds', 'n_seeds'),
            ('single_objective', 'single_objective'),
            ('plots_group_name', 'plots_group_name'),
            ('bo_plots_name', 'bo_plots_name')
        ]
        
        for commands_key, registry_key in comparisons:
            commands_val = commands_exp[commands_key]
            registry_val = registry_params[registry_key]
            
            if commands_val != registry_val:
                errors.append(f"Experiment '{exp_name}': {registry_key} mismatch - commands.txt: '{commands_val}', registry: '{registry_val}'")
    
    return errors


def validate_command_generation(registry: ExperimentRegistry, runner: ExperimentRunner) -> List[str]:
    """Validate that generated commands match the expected format."""
    errors = []
    
    for exp_name in registry.list_experiments():
        try:
            # Generate commands
            args = registry.get_experiment_command_args(exp_name)
            
            # Basic validation of command structure
            required_args = ['SEED', 'NN_EXPERIMENT_CFG', 'BO_EXPERIMENT_CFG', 
                           'SWEEP_NAME', 'SEEDS_CFG', 'PLOTS_GROUP_NAME', 'BO_PLOTS_NAME']
            
            for arg in required_args:
                if arg not in args:
                    errors.append(f"Experiment '{exp_name}': Missing required argument '{arg}'")
            
            # Validate seed is 8
            if args.get('SEED') != '8':
                errors.append(f"Experiment '{exp_name}': SEED should be '8', got '{args.get('SEED')}'")
            
        except Exception as e:
            errors.append(f"Experiment '{exp_name}': Error generating commands - {str(e)}")
    
    return errors


def validate_plot_configurations(registry: ExperimentRegistry) -> List[str]:
    """Validate that plot configurations are properly structured."""
    errors = []
    
    for exp_name in registry.list_experiments():
        try:
            exp_config = registry.get_experiment(exp_name)
            plotting = exp_config.get('plotting', {})
            
            for plot_type in ['train_acqf', 'bo_experiments']:
                if plot_type not in plotting:
                    errors.append(f"Experiment '{exp_name}': Missing plot type '{plot_type}'")
                    continue
                
                plot_config = plotting[plot_type]
                
                # Check if it has default configuration
                if 'default' not in plot_config:
                    errors.append(f"Experiment '{exp_name}': Plot type '{plot_type}' missing 'default' configuration")
                    continue
                
                # Validate default configuration structure
                default_config = plot_config['default']
                required_keys = ['pre', 'attr_a', 'attr_b']
                
                for key in required_keys:
                    if key not in default_config:
                        errors.append(f"Experiment '{exp_name}': Plot type '{plot_type}' default config missing '{key}'")
                
                # If there are alternatives, validate their structure too
                if 'alternatives' in plot_config:
                    for alt_name, alt_config in plot_config['alternatives'].items():
                        for key in required_keys:
                            if key not in alt_config:
                                errors.append(f"Experiment '{exp_name}': Plot type '{plot_type}' alternative '{alt_name}' missing '{key}'")
        
        except Exception as e:
            errors.append(f"Experiment '{exp_name}': Error validating plot configurations - {str(e)}")
    
    return errors


def test_experiment_execution(registry: ExperimentRegistry, runner: ExperimentRunner, 
                            sample_experiments: List[str]) -> List[str]:
    """Test dry-run execution of sample experiments."""
    errors = []
    
    for exp_name in sample_experiments:
        try:
            # Test status check
            returncode, stdout, stderr = runner.run_status_check(exp_name)
            if returncode != 0 and "no such file" not in stderr.lower():
                errors.append(f"Experiment '{exp_name}': Status check failed - {stderr}")
            
            # Test dry-run execution
            returncode, stdout, stderr = runner.run_experiment(exp_name, dry_run=True)
            if returncode != 0:
                errors.append(f"Experiment '{exp_name}': Dry-run execution failed - {stderr}")
            
            # Test plot generation (dry-run)
            returncode, stdout, stderr = runner.generate_plots(exp_name, dry_run=True)
            if returncode != 0:
                errors.append(f"Experiment '{exp_name}': Plot generation dry-run failed - {stderr}")
            
        except Exception as e:
            errors.append(f"Experiment '{exp_name}': Error testing execution - {str(e)}")
    
    return errors


def test_plot_variants(registry: ExperimentRegistry) -> List[str]:
    """Test plot variant functionality."""
    errors = []
    
    # Find experiments with alternatives
    experiments_with_alternatives = []
    for exp_name in registry.list_experiments():
        for plot_type in ['train_acqf', 'bo_experiments']:
            variants = registry.list_plot_variants(exp_name, plot_type)
            if len(variants) > 1:
                experiments_with_alternatives.append((exp_name, plot_type))
                break
    
    print(f"Found {len(experiments_with_alternatives)} experiments with plot alternatives")
    
    for exp_name, plot_type in experiments_with_alternatives:
        try:
            variants = registry.list_plot_variants(exp_name, plot_type)
            
            for variant in variants:
                plot_config = registry.get_plotting_config(exp_name, plot_type, variant)
                
                # Validate structure
                if not isinstance(plot_config, dict):
                    errors.append(f"Experiment '{exp_name}': Plot variant '{variant}' for '{plot_type}' is not a dict")
                    continue
                
                required_keys = ['pre', 'attr_a', 'attr_b']
                for key in required_keys:
                    if key not in plot_config:
                        errors.append(f"Experiment '{exp_name}': Plot variant '{variant}' for '{plot_type}' missing '{key}'")
        
        except Exception as e:
            errors.append(f"Experiment '{exp_name}': Error testing plot variants - {str(e)}")
    
    return errors


def main():
    """Run comprehensive validation tests."""
    print("🔍 Starting comprehensive validation of experiment registry migration")
    print("=" * 70)
    
    registry = ExperimentRegistry()
    runner = ExperimentRunner(registry)
    
    all_errors = []
    
    # Test 1: Validate experiment parameters match commands.txt
    print("\n📋 Test 1: Validating experiment parameters against commands.txt")
    errors = validate_experiment_parameters(registry)
    if errors:
        print(f"❌ Found {len(errors)} parameter validation errors:")
        for error in errors:
            print(f"  - {error}")
        all_errors.extend(errors)
    else:
        print("✅ All experiment parameters match commands.txt")
    
    # Test 2: Validate command generation
    print("\n🚀 Test 2: Validating command generation")
    errors = validate_command_generation(registry, runner)
    if errors:
        print(f"❌ Found {len(errors)} command generation errors:")
        for error in errors:
            print(f"  - {error}")
        all_errors.extend(errors)
    else:
        print("✅ All commands generate correctly")
    
    # Test 3: Validate plot configurations
    print("\n📊 Test 3: Validating plot configurations")
    errors = validate_plot_configurations(registry)
    if errors:
        print(f"❌ Found {len(errors)} plot configuration errors:")
        for error in errors:
            print(f"  - {error}")
        all_errors.extend(errors)
    else:
        print("✅ All plot configurations are valid")
    
    # Test 4: Test plot variants
    print("\n🎨 Test 4: Testing plot variant functionality")
    errors = test_plot_variants(registry)
    if errors:
        print(f"❌ Found {len(errors)} plot variant errors:")
        for error in errors:
            print(f"  - {error}")
        all_errors.extend(errors)
    else:
        print("✅ All plot variants work correctly")
    
    # Test 5: Test sample experiment execution
    print("\n🧪 Test 5: Testing sample experiment execution (dry-run)")
    sample_experiments = [
        'pointnet_max_history_pbgi_1d',
        'compare_3methods_1d', 
        'gittins_8d_big',
        'pointnet_architecture_variations_dataset_size_1d'  # Has alternatives
    ]
    errors = test_experiment_execution(registry, runner, sample_experiments)
    if errors:
        print(f"❌ Found {len(errors)} execution test errors:")
        for error in errors:
            print(f"  - {error}")
        all_errors.extend(errors)
    else:
        print("✅ All sample experiments execute correctly")
    
    # Summary
    print("\n" + "=" * 70)
    if all_errors:
        print(f"❌ VALIDATION FAILED: Found {len(all_errors)} total errors")
        print("\nAll errors:")
        for i, error in enumerate(all_errors, 1):
            print(f"{i:2d}. {error}")
        return 1
    else:
        print("🎉 VALIDATION PASSED: All tests successful!")
        print(f"✅ Successfully validated {len(registry.list_experiments())} experiments")
        print("✅ All experiments match their commands.txt counterparts")
        print("✅ All plot configurations are valid")
        print("✅ Multiple plot variants work correctly")
        print("✅ Experiment execution works correctly")
        return 0


if __name__ == "__main__":
    sys.exit(main())