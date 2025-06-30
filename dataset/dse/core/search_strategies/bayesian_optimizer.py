"""
Bayesian optimization search strategy.
"""

import time
import random
from typing import Dict, Any, List, Tuple
from .base_strategy import BaseSearchStrategy
from ....common import logger

class BayesianOptimizer(BaseSearchStrategy):
    """Bayesian optimization using scikit-optimize."""
    
    def __init__(self, arch_name, output_dir, layers, mapping_evaluator, performance_tracker=None, workload_name=None):
        super().__init__(arch_name, output_dir, layers, mapping_evaluator)
        self.performance_tracker = performance_tracker
        self.search_space = self._define_search_space()
        self.x_iters = []
        self.func_vals = []
        self.best_result = None
        self.workload_name = workload_name or 'unknown'
    
    def search(self, n_calls: int, n_initial_points: int, **kwargs) -> Dict[str, Any]:
        """
        Run Bayesian optimization search.
        
        Args:
            n_calls: Total number of optimization calls
            n_initial_points: Number of initial random exploration points
            
        Returns:
            Optimization results with best configuration found
        """
        logger.info(f"Starting Bayesian optimization with {n_calls} calls, {n_initial_points} initial points")
        start_time = time.time()
        
        try:
            # Try to use scikit-optimize if available
            try:
                from skopt import gp_minimize
                from skopt.space import Real, Integer, Categorical
                from skopt.utils import use_named_args
                
                # Use real Bayesian optimization
                result = self._run_skopt_optimization(n_calls, n_initial_points)
                
            except ImportError:
                logger.warning("scikit-optimize not available, falling back to quasi-random search")
                result = self._run_fallback_optimization(n_calls, n_initial_points)
            
            search_time = time.time() - start_time
            
            # Compile final results
            final_results = {
                'strategy': 'bayesian',
                'status': 'completed',
                'search_strategy': 'bayesian',
                'search_time': search_time,
                'n_calls': n_calls,
                'n_initial_points': n_initial_points,
                'arch_name': self.arch_name,
                'workload': self.workload_name,
                'metric': 'cycle',
                'best_cost': result['fun'],
                'best_config': self._array_to_config(result['x']),
                'total_evaluations': len(result.get('func_vals', [])),
                'optimization_path': result.get('func_vals', [])[-10:],  # Last 10 values
                'convergence_info': {
                    'final_cost': result['fun'],
                    'improvement_over_random': self._calculate_improvement_over_random()
                }
            }
            
            self.best_result = final_results
            logger.info(f"Bayesian optimization completed: best_cost={result['fun']:.6f}, time={search_time:.2f}s")
            return final_results
            
        except Exception as optimization_error:
            logger.error(f"Bayesian optimization failed: {optimization_error}")
            return {
                'strategy': 'bayesian',
                'status': 'failed',
                'error': str(optimization_error),
                'search_time': time.time() - start_time,
                'n_calls': n_calls
            }
    
    def _define_search_space(self) -> List[Tuple[str, Any]]:
        """Define the search space for Bayesian optimization."""
        # Define the optimization dimensions
        space = [
            ('pe_array_x', [4, 8, 16, 32]),        # Categorical choice
            ('pe_array_y', [4, 8, 16, 32]),        # Categorical choice  
            ('buffer_size', (1024, 16384)),        # Integer range
            ('dataflow', ['weight_stationary', 'input_stationary', 'output_stationary']),
            ('precision', [8, 16, 32]),            # Categorical choice
            ('l1_size', (64, 512)),                # Integer range
            ('l2_size', (1024, 4096)),             # Integer range
            ('bandwidth', (64, 512))               # Integer range
        ]
        return space
    
    def _run_skopt_optimization(self, n_calls: int, n_initial_points: int):
        """Run optimization using scikit-optimize."""
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        from skopt.utils import use_named_args
        
        # Define skopt search space
        dimensions = [
            Categorical([4, 8, 16, 32], name='pe_array_x'),
            Categorical([4, 8, 16, 32], name='pe_array_y'),
            Integer(1024, 16384, name='buffer_size'),
            Categorical(['weight_stationary', 'input_stationary', 'output_stationary'], name='dataflow'),
            Categorical([8, 16, 32], name='precision'),
            Integer(64, 512, name='l1_size'),
            Integer(1024, 4096, name='l2_size'),
            Integer(64, 512, name='bandwidth')
        ]
        
        @use_named_args(dimensions)
        def objective(**params):
            """Objective function for optimization."""
            config = self._params_to_config(params)
            cost = self._evaluate_configuration(config)
            self.func_vals.append(cost)
            return cost
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=42,
            verbose=False
        )
        
        return result
    
    def _run_fallback_optimization(self, n_calls: int, n_initial_points: int):
        """Fallback optimization using quasi-random sampling."""
        logger.info("Running fallback quasi-random optimization")
        
        best_cost = float('inf')
        best_x = None
        func_vals = []
        
        for i in range(n_calls):
            # Generate quasi-random point (better than pure random)
            x = self._generate_quasi_random_point(i, n_calls)
            config = self._array_to_config(x)
            cost = self._evaluate_configuration(config)
            
            func_vals.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_x = x
                logger.debug(f"New best at iteration {i}: cost={cost:.6f}")
        
        return {
            'fun': best_cost,
            'x': best_x,
            'func_vals': func_vals,
            'nfev': n_calls
        }
    
    def _generate_quasi_random_point(self, iteration: int, total_iterations: int) -> List[float]:
        """Generate a quasi-random point using low-discrepancy sequence."""
        # Simple quasi-random using Halton-like sequence
        point = []
        for dim in range(8):  # 8 dimensions
            base = 2 + dim
            value = 0
            f = 1.0 / base
            i = iteration + 1
            
            while i > 0:
                value += f * (i % base)
                i //= base
                f /= base
            
            point.append(value)
        
        return point
    
    def _params_to_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert optimization parameters to configuration."""
        config = {
            'pe_array_x': params['pe_array_x'],
            'pe_array_y': params['pe_array_y'],
            'buffer_size': params['buffer_size'],
            'dataflow': params['dataflow'],
            'precision': params['precision'],
            'memory_hierarchy': {
                'l1_size': params['l1_size'],
                'l2_size': params['l2_size'],
                'bandwidth': params['bandwidth']
            },
            'loop_ordering': [random.randint(0, 6) for _ in range(7)],
            'tiling_factors': [random.randint(1, 64) for _ in range(6)]
        }
        return config
    
    def _array_to_config(self, x: List[float]) -> Dict[str, Any]:
        """Convert optimization array to configuration."""
        # Add input validation and error handling
        try:
            # Validate input is a list/array of numbers
            if not isinstance(x, (list, tuple)) or len(x) < 8:
                logger.error(f"Invalid input to _array_to_config: {x}")
                logger.error(f"Input type: {type(x)}, length: {len(x) if hasattr(x, '__len__') else 'N/A'}")
                raise ValueError(f"Expected list/array of 8 numbers, got: {x}")
            
            # Convert any string elements to valid numbers and validate ranges
            validated_x = []
            for i, val in enumerate(x):
                if isinstance(val, str):
                    # If we get a string, it means there's a data corruption issue
                    logger.error(f"String detected in parameter array at index {i}: '{val}'")
                    if 'stationary' in val:
                        # Try to recover from corrupted dataflow strings
                        if 'weight_stationary' in val:
                            recovered_val = 0.33  # Middle of weight_stationary range
                        elif 'input_stationary' in val:
                            recovered_val = 0.66  # Middle of input_stationary range  
                        elif 'output_stationary' in val:
                            recovered_val = 1.0   # Output_stationary range
                        else:
                            recovered_val = 0.33  # Default to weight_stationary
                        logger.warning(f"Recovering from corrupted string at index {i}: '{val}' -> {recovered_val}")
                        validated_x.append(recovered_val)
                    else:
                        # For other strings, try to parse as float or use default
                        try:
                            validated_x.append(float(val))
                        except ValueError:
                            logger.warning(f"Cannot parse string '{val}' as float, using default 0.5")
                            validated_x.append(0.5)
                else:
                    # Ensure it's a valid number in [0, 1] range
                    try:
                        num_val = float(val)
                        num_val = max(0.0, min(1.0, num_val))  # Clamp to [0, 1]
                        validated_x.append(num_val)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid number at index {i}: {val}, using default 0.5")
                        validated_x.append(0.5)
            
            # Map normalized values back to configuration
            pe_choices = [4, 8, 16, 32]
            dataflow_choices = ['weight_stationary', 'input_stationary', 'output_stationary']
            precision_choices = [8, 16, 32]
            
            # Ensure proper index calculation to avoid out-of-bounds errors
            pe_x_idx = min(len(pe_choices) - 1, max(0, int(validated_x[0] * len(pe_choices))))
            pe_y_idx = min(len(pe_choices) - 1, max(0, int(validated_x[1] * len(pe_choices))))
            dataflow_idx = min(len(dataflow_choices) - 1, max(0, int(validated_x[3] * len(dataflow_choices))))
            precision_idx = min(len(precision_choices) - 1, max(0, int(validated_x[4] * len(precision_choices))))
            
            config = {
                'pe_array_x': pe_choices[pe_x_idx],
                'pe_array_y': pe_choices[pe_y_idx],
                'buffer_size': int(1024 + validated_x[2] * (16384 - 1024)),
                'dataflow': dataflow_choices[dataflow_idx],
                'precision': precision_choices[precision_idx],
                'memory_hierarchy': {
                    'l1_size': int(64 + validated_x[5] * (512 - 64)),
                    'l2_size': int(1024 + validated_x[6] * (4096 - 1024)),
                    'bandwidth': int(64 + validated_x[7] * (512 - 64))
                },
                'loop_ordering': [random.randint(0, 6) for _ in range(7)],
                'tiling_factors': [random.randint(1, 64) for _ in range(6)]
            }
            
            logger.debug(f"Successfully converted array to config: dataflow={config['dataflow']}")
            return config
            
        except Exception as e:
            logger.error(f"Error in _array_to_config: {e}")
            logger.error(f"Input was: {x}")
            # Return a safe default configuration
            return {
                'pe_array_x': 8,
                'pe_array_y': 8,
                'buffer_size': 4096,
                'dataflow': 'weight_stationary',
                'precision': 16,
                'memory_hierarchy': {
                    'l1_size': 256,
                    'l2_size': 2048,
                    'bandwidth': 128
                },
                'loop_ordering': [random.randint(0, 6) for _ in range(7)],
                'tiling_factors': [random.randint(1, 64) for _ in range(6)]
            }
    
    def _evaluate_configuration(self, config: Dict[str, Any]) -> float:
        """Evaluate a configuration and return cost."""
        try:
            if self.mapping_evaluator:
                result = self.mapping_evaluator.evaluate(config)
                return result.get('objective', float('inf')) if isinstance(result, dict) else float('inf')
            else:
                return self._synthetic_evaluation(config)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return float('inf')
    
    def _synthetic_evaluation(self, config: Dict[str, Any]) -> float:
        """Synthetic evaluation for testing."""
        # More sophisticated synthetic function with some structure
        pe_cost = config['pe_array_x'] * config['pe_array_y'] * 0.1
        buffer_cost = config['buffer_size'] * 0.001
        memory_cost = (config['memory_hierarchy']['l1_size'] * 0.01 + 
                      config['memory_hierarchy']['l2_size'] * 0.001)
        
        # Add some non-linear interactions
        interaction_cost = (pe_cost * buffer_cost) ** 0.5 * 0.001
        
        # Add structured noise (not pure random)
        noise_factor = 1.0 + 0.1 * (hash(str(config)) % 100 - 50) / 100
        
        total_cost = (pe_cost + buffer_cost + memory_cost + interaction_cost) * noise_factor
        return total_cost
    
    def _calculate_improvement_over_random(self) -> float:
        """Calculate improvement over random search baseline."""
        if not self.func_vals:
            return 0.0
        
        # Compare best result to average of first few (more random) evaluations
        if len(self.func_vals) >= 5:
            random_baseline = sum(self.func_vals[:5]) / 5
            best_result = min(self.func_vals)
            return max(0.0, (random_baseline - best_result) / random_baseline * 100)
        return 0.0 