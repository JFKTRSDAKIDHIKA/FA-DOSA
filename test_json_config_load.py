#!/usr/bin/env python3
"""
Test script to verify that the saved JSON configuration can be loaded and parsed correctly.
"""

import json
import torch

def test_load_configuration(filepath="final_configuration.json"):
    """
    Load and validate the saved JSON configuration.
    
    Args:
        filepath: Path to the JSON configuration file
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        print("=== Configuration Loading Test ===")
        print(f"Successfully loaded configuration from {filepath}")
        
        # Validate hardware section
        print("\n--- Hardware Configuration ---")
        print(f"Number of PEs: {config['hardware']['num_pes']}")
        print("Buffer Sizes:")
        for buffer_name, size_kb in config['hardware']['buffer_sizes_kb'].items():
            print(f"  {buffer_name}: {size_kb:.2f} KB")
        
        # Validate mapping section
        print("\n--- Mapping Configuration ---")
        for level_name, level_mapping in config['mapping'].items():
            print(f"{level_name}:")
            for dim_name, factors in level_mapping.items():
                temporal = factors['temporal']
                spatial = factors['spatial']
                print(f"  {dim_name}: temporal={temporal:.3f}, spatial={spatial:.3f}")
        
        # Validate fusion section
        print("\n--- Fusion Configuration ---")
        for group_name, fusion_prob in config['fusion'].items():
            print(f"{group_name}: {fusion_prob:.4f}")
        
        print("\n✓ Configuration validation successful!")
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: Configuration file {filepath} not found")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
        return False
    except KeyError as e:
        print(f"❌ Error: Missing required configuration key - {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_load_configuration()