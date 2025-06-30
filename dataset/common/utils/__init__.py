"""
Modern common utilities with backward compatibility.
"""

# Core utilities
from .file_utils import FileHandler
from .process_utils import ProcessManager
from .math_utils import MathUtils, RandomUtils

# Specialized handlers
from .timeloop_interface import TimeloopInterface
from .data_structures import ResultCollector, PerformanceTracker, ConfigDict

# Re-export commonly used items
try:
    from ..logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Backward compatibility functions
parse_yaml = FileHandler.load_yaml
parse_json = FileHandler.load_json
save_yaml = FileHandler.save_yaml
save_json = FileHandler.save_json
load_pickle = FileHandler.load_pickle
save_pickle = FileHandler.save_pickle
make_tarfile = FileHandler.create_tarfile

# Add store_yaml for backward compatibility
def store_yaml(yaml_path, data):
    """Store data as YAML file (backward compatibility function)."""
    return FileHandler.save_yaml(yaml_path, data)

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

# Legacy helper functions
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

__all__ = [
    # Core utilities
    'FileHandler',
    'ProcessManager', 
    'MathUtils',
    'RandomUtils',
    
    # Specialized handlers
    'TimeloopInterface',
    
    # Data structures
    'ResultCollector',
    'PerformanceTracker',
    'ConfigDict',
    
    # Logger
    'logger',
    
    # Backward compatibility functions
    'parse_yaml', 'parse_json', 'save_yaml', 'save_json', 'store_yaml',
    'load_pickle', 'save_pickle', 'make_tarfile',
    'set_random_seed', 'get_random_seed', 'get_prime_factors', 
    'get_divisors', 'calculate_correlation',
    'run_subprocess', 'run_subprocess_async', 'check_subprocess_output',
    'OrderedDefaultdict', 'ensure_dir', 'read_file_lines', 'write_file_lines'
] 