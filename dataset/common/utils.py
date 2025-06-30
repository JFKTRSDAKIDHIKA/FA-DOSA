"""
Backward compatibility layer for the old utils module.

This module provides the same interface as the original utils.py,
but delegates to the new refactored modules for implementation.
"""

# Import all functionality from refactored modules
try:
    from .utils.file_utils import FileHandler
    from .utils.math_utils import MathUtils, RandomUtils
    from .utils.process_utils import ProcessManager
    from .utils.data_structures import ResultCollector, PerformanceTracker
    
    # Create backward-compatible functions by aliasing new functionality
    
    # File operations
    parse_yaml = FileHandler.load_yaml
    parse_json = FileHandler.load_json
    save_yaml = FileHandler.save_yaml
    save_json = FileHandler.save_json
    load_pickle = FileHandler.load_pickle
    save_pickle = FileHandler.save_pickle
    make_tarfile = FileHandler.create_tarfile
    
    # Math operations
    set_random_seed = RandomUtils.set_global_seed
    get_random_seed = RandomUtils.get_current_seed
    get_prime_factors = MathUtils.get_prime_factors
    get_divisors = MathUtils.get_divisors
    calculate_correlation = MathUtils.get_correlation
    
    # Process operations
    run_subprocess = ProcessManager.run_command
    run_subprocess_async = ProcessManager.run_async
    check_subprocess_output = ProcessManager.get_command_output
    
    # Data structures
    OrderedDefaultdict = ResultCollector  # Alias for compatibility
    
    # Legacy helper functions that were commonly used
    def ensure_dir(path):
        """Ensure directory exists (legacy compatibility)."""
        import pathlib
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    def read_file_lines(file_path):
        """Read file lines (legacy compatibility)."""
        with open(file_path, 'r') as f:
            return f.readlines()
    
    def write_file_lines(file_path, lines):
        """Write file lines (legacy compatibility)."""
        with open(file_path, 'w') as f:
            f.writelines(lines)

except ImportError as e:
    # Fallback for environments where refactored modules are not available
    import warnings
    warnings.warn(f"Refactored modules not available: {e}. "
                  "Some functionality may not work correctly.")
    
    # Provide minimal fallback implementations
    def parse_yaml(file_path):
        import yaml
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def parse_json(file_path):
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def set_random_seed(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    def ensure_dir(path):
        import pathlib
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) 