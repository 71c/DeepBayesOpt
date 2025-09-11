#!/usr/bin/env python3
"""
Validate that plot configurations in registry.yml exactly match those in plotting files.
"""

import re
import yaml
from pathlib import Path

def extract_plot_configs_from_files():
    """Extract plot configurations from train_acqf_plot.py and bo_experiments_gp_plot.py"""
    configs = {}
    
    # Extract from train_acqf_plot.py
    train_acqf_file = Path("train_acqf_plot.py")
    if train_acqf_file.exists():
        content = train_acqf_file.read_text()
        
        # Find all "For experiment_name" blocks followed by PRE/ATTR_A/ATTR_B/POST
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Look for "# For experiment_name" lines
            if line.startswith('# For ') and not line.startswith('# For 1dim_pointnet_model_size_variations-dataset_size'):
                # Extract experiment name
                exp_name = line[6:].strip()  # Remove "# For "
                
                # Look ahead for PRE/ATTR_A/ATTR_B/POST definitions
                config = {}
                j = i + 1
                
                # Read the config block
                while j < len(lines) and (lines[j].startswith('#') or lines[j].strip() == ''):
                    config_line = lines[j].strip()
                    
                    if '# PRE = ' in config_line:
                        # Multi-line PRE parsing
                        pre_content = config_line.split('# PRE = ')[1]
                        if pre_content == '[':
                            # Multi-line list
                            pre_lines = [pre_content]
                            j += 1
                            while j < len(lines) and (lines[j].startswith('#') and (']' not in lines[j] or lines[j].strip().endswith('],'))):
                                pre_lines.append(lines[j].strip()[2:])  # Remove '# '
                                j += 1
                            if j < len(lines) and ']' in lines[j]:
                                pre_lines.append(lines[j].strip()[2:])  # Remove '# '
                            pre_str = ' '.join(pre_lines)
                        else:
                            pre_str = pre_content
                        try:
                            config['pre'] = eval(pre_str)
                        except:
                            pass
                    
                    elif '# ATTR_A = ' in config_line:
                        attr_a_str = config_line.split('# ATTR_A = ')[1]
                        try:
                            config['attr_a'] = eval(attr_a_str)
                        except:
                            pass
                    
                    elif '# ATTR_B = ' in config_line:
                        attr_b_str = config_line.split('# ATTR_B = ')[1]
                        try:
                            config['attr_b'] = eval(attr_b_str)
                        except:
                            pass
                    
                    elif '# POST = ' in config_line:
                        post_content = config_line.split('# POST = ')[1]
                        if post_content == '[':
                            # Multi-line list
                            post_lines = [post_content]
                            j += 1
                            while j < len(lines) and (lines[j].startswith('#') and (']' not in lines[j] or lines[j].strip().endswith('],'))):
                                post_lines.append(lines[j].strip()[2:])  # Remove '# '
                                j += 1
                            if j < len(lines) and ']' in lines[j]:
                                post_lines.append(lines[j].strip()[2:])  # Remove '# '
                            post_str = ' '.join(post_lines)
                        else:
                            post_str = post_content
                        try:
                            config['post'] = eval(post_str)
                        except:
                            pass
                    
                    j += 1
                
                if config:
                    if exp_name not in configs:
                        configs[exp_name] = {}
                    configs[exp_name]['train_acqf'] = config
                
                i = j
            else:
                i += 1
    
    # Extract from bo_experiments_gp_plot.py  
    bo_file = Path("bo_experiments_gp_plot.py")
    if bo_file.exists():
        content = bo_file.read_text()
        
        # Find all "For experiment_name" blocks followed by PRE/ATTR_A/ATTR_B/POST
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Look for "# For experiment_name" lines
            if line.startswith('# For ') and not line.startswith('# For 1dim_pointnet_model_size_variations-dataset_size'):
                # Extract experiment name
                exp_name = line[6:].strip()  # Remove "# For "
                
                # Look ahead for PRE/ATTR_A/ATTR_B/POST definitions
                config = {}
                j = i + 1
                
                # Read the config block
                while j < len(lines) and (lines[j].startswith('#') or lines[j].strip() == ''):
                    config_line = lines[j].strip()
                    
                    if '# PRE = ' in config_line:
                        # Multi-line PRE parsing
                        pre_content = config_line.split('# PRE = ')[1]
                        if pre_content == '[':
                            # Multi-line list
                            pre_lines = [pre_content]
                            j += 1
                            while j < len(lines) and (lines[j].startswith('#') and (']' not in lines[j] or lines[j].strip().endswith('],'))):
                                pre_lines.append(lines[j].strip()[2:])  # Remove '# '
                                j += 1
                            if j < len(lines) and ']' in lines[j]:
                                pre_lines.append(lines[j].strip()[2:])  # Remove '# '
                            pre_str = ' '.join(pre_lines)
                        else:
                            pre_str = pre_content
                        try:
                            config['pre'] = eval(pre_str)
                        except:
                            pass
                    
                    elif '# ATTR_A = ' in config_line:
                        attr_a_str = config_line.split('# ATTR_A = ')[1]
                        try:
                            config['attr_a'] = eval(attr_a_str)
                        except:
                            pass
                    
                    elif '# ATTR_B = ' in config_line:
                        attr_b_str = config_line.split('# ATTR_B = ')[1]
                        try:
                            config['attr_b'] = eval(attr_b_str)
                        except:
                            pass
                    
                    elif '# POST = ' in config_line:
                        post_content = config_line.split('# POST = ')[1]
                        if post_content == '[':
                            # Multi-line list
                            post_lines = [post_content]
                            j += 1
                            while j < len(lines) and (lines[j].startswith('#') and (']' not in lines[j] or lines[j].strip().endswith('],'))):
                                post_lines.append(lines[j].strip()[2:])  # Remove '# '
                                j += 1
                            if j < len(lines) and ']' in lines[j]:
                                post_lines.append(lines[j].strip()[2:])  # Remove '# '
                            post_str = ' '.join(post_lines)
                        else:
                            post_str = post_content
                        try:
                            config['post'] = eval(post_str)
                        except:
                            pass
                    
                    j += 1
                
                if config:
                    if exp_name not in configs:
                        configs[exp_name] = {}
                    configs[exp_name]['bo_experiments'] = config
                
                i = j
            else:
                i += 1
    
    return configs

def load_registry_configs():
    """Load plot configurations from registry.yml"""
    with open("experiments/registry.yml", 'r') as f:
        registry = yaml.safe_load(f)
    
    configs = {}
    plot_group_to_exp = {}  # Map plot group names to experiment names
    
    for exp_name, exp_data in registry['experiments'].items():
        # Build mapping from plot group name to experiment name
        if 'plots_group_name' in exp_data['parameters']:
            plot_group_name = exp_data['parameters']['plots_group_name']
            plot_group_to_exp[plot_group_name] = exp_name
        
        # Extract plotting configs
        if 'plotting' in exp_data:
            configs[exp_name] = {}
            for plot_type, plot_configs in exp_data['plotting'].items():
                # Get default config
                if 'default' in plot_configs:
                    configs[exp_name][plot_type] = plot_configs['default']
                else:
                    # Old format - direct config
                    configs[exp_name][plot_type] = plot_configs
    
    return configs, plot_group_to_exp

def compare_configs(file_configs, registry_configs):
    """Compare configurations between files and registry"""
    all_experiments = set(file_configs.keys()) | set(registry_configs.keys())
    
    mismatches = []
    matches = []
    
    for exp_name in sorted(all_experiments):
        file_config = file_configs.get(exp_name, {})
        registry_config = registry_configs.get(exp_name, {})
        
        # Check each plot type
        all_plot_types = set(file_config.keys()) | set(registry_config.keys())
        
        for plot_type in sorted(all_plot_types):
            file_plot_config = file_config.get(plot_type, {})
            registry_plot_config = registry_config.get(plot_type, {})
            
            if file_plot_config == registry_plot_config:
                if file_plot_config:  # Only report matches if config exists
                    matches.append(f"{exp_name}.{plot_type}")
            else:
                mismatches.append({
                    'experiment': exp_name,
                    'plot_type': plot_type,
                    'file_config': file_plot_config,
                    'registry_config': registry_plot_config
                })
    
    return matches, mismatches

def main():
    print("Validating plot configurations between registry and plotting files...\n")
    
    # Extract configurations
    file_configs = extract_plot_configs_from_files()
    registry_configs, plot_group_to_exp = load_registry_configs()
    
    
    # Map file configs from plot group names to experiment names
    mapped_file_configs = {}
    for plot_group_name, config in file_configs.items():
        if plot_group_name in plot_group_to_exp:
            exp_name = plot_group_to_exp[plot_group_name]
            mapped_file_configs[exp_name] = config
        else:
            print(f"Warning: Plot group '{plot_group_name}' not found in registry")
            mapped_file_configs[plot_group_name] = config
    
    print(f"Found {len(file_configs)} experiments with plot configs in files")
    print(f"Found {len(registry_configs)} experiments with plot configs in registry")
    print(f"Mapped {len(mapped_file_configs)} file configs to experiment names")
    print()
    
    # Compare configurations
    matches, mismatches = compare_configs(mapped_file_configs, registry_configs)
    
    if matches:
        print(f"✅ {len(matches)} plot configurations match exactly:")
        for match in matches:
            print(f"  - {match}")
        print()
    
    if mismatches:
        print(f"❌ {len(mismatches)} plot configuration mismatches found:")
        for mismatch in mismatches:
            print(f"\n  {mismatch['experiment']}.{mismatch['plot_type']}:")
            print(f"    File config:     {mismatch['file_config']}")
            print(f"    Registry config: {mismatch['registry_config']}")
    else:
        print("🎉 All plot configurations match perfectly!")
    
    # Summary
    print(f"\nSummary:")
    print(f"  - Matches: {len(matches)}")
    print(f"  - Mismatches: {len(mismatches)}")
    print(f"  - Total experiments in files: {len(file_configs)}")
    print(f"  - Total experiments in registry: {len(registry_configs)}")
    print(f"  - Mapped file configs: {len(mapped_file_configs)}")
    
    return len(mismatches) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)