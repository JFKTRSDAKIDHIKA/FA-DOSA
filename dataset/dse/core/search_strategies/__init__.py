"""
Search strategies for design space exploration.

This module provides lightweight access to search strategies without
requiring heavy dependencies like torch or sklearn.
"""

# Direct imports for lightweight access
__all__ = [
    'BayesianOptimizer',
    'GradientDescentOptimizer', 
    'RandomSearchOptimizer',
    'BaseSearchStrategy'
]

def __getattr__(name):
    """Lazy import to avoid heavy dependencies on module import."""
    if name == 'BayesianOptimizer':
        from .bayesian_optimizer import BayesianOptimizer
        return BayesianOptimizer
    elif name == 'GradientDescentOptimizer':
        from .gradient_descent import GradientDescentOptimizer
        return GradientDescentOptimizer
    elif name == 'RandomSearchOptimizer':
        from .random_search import RandomSearchOptimizer
        return RandomSearchOptimizer
    elif name == 'BaseSearchStrategy':
        from .base_strategy import BaseSearchStrategy
        return BaseSearchStrategy
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 