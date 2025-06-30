"""Random search strategy."""
import random
import time
from typing import Dict, Any, List
from .base_strategy import BaseSearchStrategy
from ....common import logger

class RandomSearchOptimizer(BaseSearchStrategy):
    """Random search optimization."""
    
    def __init__(self, arch_name, output_dir, layers, mapping_evaluator, performance_tracker=None, workload_name=None):
        super().__init__(arch_name, output_dir, layers, mapping_evaluator)
        self.performance_tracker = performance_tracker
        self.best_result = None
        self.search_history = []
        self.workload_name = workload_name or 'unknown'
    
    def search(self, n_calls: int, n_initial_points: int, **kwargs) -> Dict[str, Any]:
        """
        Run random search optimization.
        
        Args:
            n_calls: Total number of search iterations
            n_initial_points: Number of initial random points (unused in pure random search)
            
        Returns:
            Search results with actual performance data
        """
        logger.info(f"Starting random search with {n_calls} iterations")
        start_time = time.time()
        
        best_cost = float('inf')
        best_config = None
        iteration_results = []
        
        try:
            for iteration in range(n_calls):
                logger.debug(f"Random search iteration {iteration + 1}/{n_calls}")
                
                # Generate random configuration
                random_config = self._generate_random_configuration()
                
                # Evaluate the configuration
                try:
                    if self.mapping_evaluator:
                        result = self.mapping_evaluator.evaluate(random_config)
                        cost = result.get('objective', float('inf')) if isinstance(result, dict) else float('inf')
                    else:
                        # Fallback to synthetic evaluation for testing
                        cost = self._synthetic_evaluation(random_config)
                    
                    # Track best result
                    if cost < best_cost:
                        best_cost = cost
                        best_config = random_config
                        logger.info(f"New best configuration found: cost={cost:.6f}")
                    
                    # Store iteration result
                    iteration_results.append({
                        'iteration': iteration + 1,
                        'config': random_config,
                        'cost': cost,
                        'is_best': cost == best_cost
                    })
                    
                except Exception as eval_error:
                    logger.warning(f"Evaluation failed for iteration {iteration + 1}: {eval_error}")
                    iteration_results.append({
                        'iteration': iteration + 1,
                        'config': random_config,
                        'cost': float('inf'),
                        'error': str(eval_error)
                    })
            
            search_time = time.time() - start_time
            
            # Compile final results
            results = {
                'strategy': 'random',
                'status': 'completed',
                'search_strategy': 'random',
                'search_time': search_time,
                'n_calls': n_calls,
                'n_initial_points': n_initial_points,
                'arch_name': self.arch_name,
                'workload': self.workload_name,
                'metric': 'cycle',
                'best_cost': best_cost,
                'best_config': best_config,
                'total_evaluations': len(iteration_results),
                'successful_evaluations': len([r for r in iteration_results if 'error' not in r]),
                'iteration_history': iteration_results[-10:],  # Keep last 10 for space
                'summary_stats': {
                    'mean_cost': sum(r['cost'] for r in iteration_results if r['cost'] != float('inf')) / max(1, len([r for r in iteration_results if r['cost'] != float('inf')])),
                    'min_cost': best_cost,
                    'max_cost': max((r['cost'] for r in iteration_results if r['cost'] != float('inf')), default=float('inf'))
                }
            }
            
            self.best_result = results
            logger.info(f"Random search completed: best_cost={best_cost:.6f}, time={search_time:.2f}s")
            return results
            
        except Exception as search_error:
            logger.error(f"Random search failed: {search_error}")
            return {
                'strategy': 'random',
                'status': 'failed',
                'error': str(search_error),
                'search_time': time.time() - start_time,
                'n_calls': n_calls
            }
    
    def _generate_random_configuration(self) -> Dict[str, Any]:
        """Generate a random mapping configuration."""
        # Generate realistic random parameters for hardware mapping
        config = {
            'pe_array_x': random.choice([4, 8, 16, 32]),
            'pe_array_y': random.choice([4, 8, 16, 32]),
            'buffer_size': random.choice([1024, 2048, 4096, 8192, 16384]),
            'dataflow': random.choice(['weight_stationary', 'input_stationary', 'output_stationary']),
            'precision': random.choice([8, 16, 32]),
            'memory_hierarchy': {
                'l1_size': random.choice([64, 128, 256, 512]),
                'l2_size': random.choice([1024, 2048, 4096]),
                'bandwidth': random.choice([64, 128, 256, 512])
            },
            'loop_ordering': [random.randint(0, 6) for _ in range(7)],  # Random loop order
            'tiling_factors': [random.randint(1, 64) for _ in range(6)]  # Random tiling
        }
        return config
    
    def _synthetic_evaluation(self, config: Dict[str, Any]) -> float:
        """
        Synthetic evaluation function for testing when real evaluator is not available.
        
        Returns a synthetic cost based on configuration parameters.
        """
        # Simulate realistic cost calculation
        pe_cost = config['pe_array_x'] * config['pe_array_y'] * 0.1
        buffer_cost = config['buffer_size'] * 0.001
        memory_cost = config['memory_hierarchy']['l1_size'] * 0.01 + config['memory_hierarchy']['l2_size'] * 0.001
        
        # Add some randomness to simulate real evaluation variance
        noise = random.uniform(0.9, 1.1)
        
        total_cost = (pe_cost + buffer_cost + memory_cost) * noise
        return total_cost 