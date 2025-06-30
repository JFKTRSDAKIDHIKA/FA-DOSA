"""
Refactored Design Space Exploration (DSE) module with improved organization.
"""

# Core search engine
from .search_engine import SearchEngine

# Search strategies
from .search_strategies import (
    BayesianOptimizer,
    GradientDescentOptimizer, 
    RandomSearchOptimizer
)

# Mapping utilities
from .mapping import (
    MappingGenerator,
    MappingEvaluator,
    MappingBounds
)

# Hardware utilities
from .hardware import (
    HardwareOptimizer,
    HardwareBounds
)

# Evaluators
from .evaluators import (
    PerformanceEvaluator,
    BaselineEvaluator
)

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