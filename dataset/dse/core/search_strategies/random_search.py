"""Random search strategy."""
from .base_strategy import BaseSearchStrategy

class RandomSearchOptimizer(BaseSearchStrategy):
    """Random search optimization."""
    
    def search(self, n_calls, n_initial_points, **kwargs):
        return {'strategy': 'random', 'status': 'placeholder'}
    
    def _generate_random_point(self):
        return [0.0] * 10 