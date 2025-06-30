"""
Refactored Design Space Exploration (DSE) module with improved organization.
"""

# Lazy imports to avoid heavy dependencies on module import

def get_search_engine():
    """Lazy import for SearchEngine to avoid torch dependency on module import."""
    from .search_engine import SearchEngine
    return SearchEngine

# Search strategies - these should be lightweight
from .search_strategies import (
    BayesianOptimizer,
    GradientDescentOptimizer, 
    RandomSearchOptimizer
)

# Lazy imports for components with potential dependencies
def get_mapping_generator():
    from .mapping import MappingGenerator
    return MappingGenerator

def get_mapping_evaluator():
    from .mapping import MappingEvaluator
    return MappingEvaluator

def get_mapping_bounds():
    from .mapping import MappingBounds
    return MappingBounds

def get_hardware_optimizer():
    from .hardware import HardwareOptimizer
    return HardwareOptimizer

def get_hardware_bounds():
    from .hardware import HardwareBounds
    return HardwareBounds

def get_performance_evaluator():
    from .evaluators import PerformanceEvaluator
    return PerformanceEvaluator

def get_baseline_evaluator():
    from .evaluators import BaselineEvaluator
    return BaselineEvaluator

# For backward compatibility, expose these through __getattr__
def __getattr__(name):
    if name == 'SearchEngine':
        return get_search_engine()
    elif name == 'MappingGenerator':
        return get_mapping_generator()
    elif name == 'MappingEvaluator':
        return get_mapping_evaluator()
    elif name == 'MappingBounds':
        return get_mapping_bounds()
    elif name == 'HardwareOptimizer':
        return get_hardware_optimizer()
    elif name == 'HardwareBounds':
        return get_hardware_bounds()
    elif name == 'PerformanceEvaluator':
        return get_performance_evaluator()
    elif name == 'BaselineEvaluator':
        return get_baseline_evaluator()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Core
    'SearchEngine',
    
    # Search strategies
    'BayesianOptimizer',
    'GradientDescentOptimizer',
    'RandomSearchOptimizer',
    
    # Mapping
    'MappingGenerator', 
    'MappingEvaluator',
    'MappingBounds',
    
    # Hardware
    'HardwareOptimizer',
    'HardwareBounds',
    
    # Evaluators
    'PerformanceEvaluator',
    'BaselineEvaluator'
] 