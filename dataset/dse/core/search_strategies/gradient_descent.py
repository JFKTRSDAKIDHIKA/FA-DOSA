"""
Gradient descent search strategy for design space exploration.
"""

import time
import random
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_strategy import BaseSearchStrategy
from ....common import logger

class GradientDescentOptimizer(BaseSearchStrategy):
    """Gradient descent optimization for discrete design space."""
    
    def __init__(self, arch_name, output_dir, layers, mapping_evaluator, performance_tracker=None, workload_name=None):
        super().__init__(arch_name, output_dir, layers, mapping_evaluator)
        self.performance_tracker = performance_tracker
        self.workload_name = workload_name or 'unknown'
        self.best_result = None
        
        # Gradient descent parameters
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.velocity = None
        
        # Define discrete search space
        self.search_space = self._define_search_space()
        self.param_bounds = self._get_parameter_bounds()
    
    def search(self, n_calls: int, n_initial_points: int, **kwargs) -> Dict[str, Any]:
        """
        Run gradient descent search.
        
        Args:
            n_calls: Total number of optimization steps
            n_initial_points: Number of initial random points for exploration
            
        Returns:
            Optimization results with best configuration found
        """
        logger.info(f"Starting gradient descent optimization with {n_calls} steps, {n_initial_points} initial points")
        start_time = time.time()
        
        try:
            # Initialize with random exploration
            best_cost = float('inf')
            best_config = None
            best_params = None
            iteration_results = []
            
            # Phase 1: Random exploration for initialization
            logger.info(f"Phase 1: Random exploration ({n_initial_points} points)")
            for i in range(min(n_initial_points, n_calls)):
                config = self._generate_random_configuration()
                params = self._config_to_params(config)
                cost = self._evaluate_configuration(config)
                
                if cost < best_cost:
                    best_cost = cost
                    best_config = config
                    best_params = params
                    logger.info(f"New best in exploration: cost={cost:.6f}")
                
                iteration_results.append({
                    'iteration': i + 1,
                    'phase': 'exploration',
                    'config': config,
                    'cost': cost,
                    'is_best': cost == best_cost
                })
            
            # Phase 2: Gradient descent optimization
            if n_calls > n_initial_points and best_params is not None:
                logger.info(f"Phase 2: Gradient descent optimization")
                current_params = best_params.copy()
                self.velocity = np.zeros_like(current_params)
                
                for i in range(n_initial_points, n_calls):
                    # Compute numerical gradient
                    gradient = self._compute_numerical_gradient(current_params)
                    
                    # Update parameters with momentum
                    self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
                    current_params = current_params + self.velocity
                    
                    # Clip to bounds
                    current_params = self._clip_to_bounds(current_params)
                    
                    # Convert to configuration and evaluate
                    config = self._params_to_config(current_params)
                    cost = self._evaluate_configuration(config)
                    
                    # Update best if improved
                    if cost < best_cost:
                        best_cost = cost
                        best_config = config
                        best_params = current_params.copy()
                        logger.info(f"New best in optimization: cost={cost:.6f}")
                    
                    iteration_results.append({
                        'iteration': i + 1,
                        'phase': 'gradient_descent',
                        'config': config,
                        'cost': cost,
                        'is_best': cost == best_cost,
                        'gradient_norm': np.linalg.norm(gradient)
                    })
                    
                    # Adaptive learning rate
                    if i % 10 == 0:
                        self.learning_rate *= 0.95  # Decay learning rate
            
            search_time = time.time() - start_time
            
            # Compile final results
            results = {
                'strategy': 'gradient_descent',
                'status': 'completed',
                'search_strategy': 'gradient_descent',
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
                'optimization_path': [r['cost'] for r in iteration_results],
                'convergence_info': {
                    'final_cost': best_cost,
                    'exploration_phase_length': n_initial_points,
                    'optimization_phase_length': max(0, n_calls - n_initial_points),
                    'final_learning_rate': self.learning_rate
                }
            }
            
            self.best_result = results
            logger.info(f"Gradient descent completed: best_cost={best_cost:.6f}, time={search_time:.2f}s")
            return results
            
        except Exception as optimization_error:
            logger.error(f"Gradient descent optimization failed: {optimization_error}")
            return {
                'strategy': 'gradient_descent',
                'status': 'failed',
                'error': str(optimization_error),
                'search_time': time.time() - start_time,
                'n_calls': n_calls
            }
    
    def _define_search_space(self) -> Dict[str, Any]:
        """Define the search space for gradient descent."""
        return {
            'pe_array_x': [4, 8, 16, 32],
            'pe_array_y': [4, 8, 16, 32],
            'buffer_size': (1024, 16384),
            'dataflow': ['weight_stationary', 'input_stationary', 'output_stationary'],
            'precision': [8, 16, 32],
            'l1_size': (64, 512),
            'l2_size': (1024, 4096),
            'bandwidth': (64, 512)
        }
    
    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get normalized parameter bounds for gradient descent."""
        # All parameters normalized to [0, 1]
        return [(0.0, 1.0) for _ in range(8)]  # 8 parameters
    
    def _config_to_params(self, config: Dict[str, Any]) -> np.ndarray:
        """Convert configuration to normalized parameter vector."""
        params = np.zeros(8)
        
        # Normalize categorical parameters
        pe_choices = self.search_space['pe_array_x']
        dataflow_choices = self.search_space['dataflow']
        precision_choices = self.search_space['precision']
        
        params[0] = pe_choices.index(config['pe_array_x']) / (len(pe_choices) - 1)
        params[1] = pe_choices.index(config['pe_array_y']) / (len(pe_choices) - 1)
        params[2] = (config['buffer_size'] - 1024) / (16384 - 1024)
        params[3] = dataflow_choices.index(config['dataflow']) / (len(dataflow_choices) - 1)
        params[4] = precision_choices.index(config['precision']) / (len(precision_choices) - 1)
        params[5] = (config['memory_hierarchy']['l1_size'] - 64) / (512 - 64)
        params[6] = (config['memory_hierarchy']['l2_size'] - 1024) / (4096 - 1024)
        params[7] = (config['memory_hierarchy']['bandwidth'] - 64) / (512 - 64)
        
        return params
    
    def _params_to_config(self, params: np.ndarray) -> Dict[str, Any]:
        """Convert normalized parameters to configuration."""
        try:
            # Validate input
            if not isinstance(params, np.ndarray) or len(params) < 8:
                logger.error(f"Invalid params input: {params}")
                raise ValueError(f"Expected numpy array of 8 elements, got: {params}")
            
            # Validate and clean parameters
            validated_params = []
            for i, param in enumerate(params):
                if isinstance(param, str):
                    logger.error(f"String detected in params at index {i}: '{param}'")
                    if 'stationary' in param:
                        # Try to recover from corrupted dataflow strings
                        if 'weight_stationary' in param:
                            recovered_val = 0.33
                        elif 'input_stationary' in param:
                            recovered_val = 0.66
                        elif 'output_stationary' in param:
                            recovered_val = 1.0
                        else:
                            recovered_val = 0.33
                        logger.warning(f"Recovering from corrupted string in params at index {i}: '{param}' -> {recovered_val}")
                        validated_params.append(recovered_val)
                    else:
                        try:
                            validated_params.append(float(param))
                        except ValueError:
                            logger.warning(f"Cannot parse param '{param}' as float, using 0.5")
                            validated_params.append(0.5)
                else:
                    # Ensure valid numeric range [0, 1]
                    try:
                        num_val = float(param)
                        num_val = max(0.0, min(1.0, num_val))
                        validated_params.append(num_val)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid number in params at index {i}: {param}, using 0.5")
                        validated_params.append(0.5)
            
            validated_params = np.array(validated_params)
            
            pe_choices = self.search_space['pe_array_x']
            dataflow_choices = self.search_space['dataflow']
            precision_choices = self.search_space['precision']
            
            config = {
                'pe_array_x': pe_choices[int(validated_params[0] * (len(pe_choices) - 1))],
                'pe_array_y': pe_choices[int(validated_params[1] * (len(pe_choices) - 1))],
                'buffer_size': int(1024 + validated_params[2] * (16384 - 1024)),
                'dataflow': dataflow_choices[int(validated_params[3] * (len(dataflow_choices) - 1))],
                'precision': precision_choices[int(validated_params[4] * (len(precision_choices) - 1))],
                'memory_hierarchy': {
                    'l1_size': int(64 + validated_params[5] * (512 - 64)),
                    'l2_size': int(1024 + validated_params[6] * (4096 - 1024)),
                    'bandwidth': int(64 + validated_params[7] * (512 - 64))
                },
                'loop_ordering': [random.randint(0, 6) for _ in range(7)],
                'tiling_factors': [random.randint(1, 64) for _ in range(6)]
            }
            
            logger.debug(f"Successfully converted params to config: dataflow={config['dataflow']}")
            return config
            
        except Exception as e:
            logger.error(f"Error in _params_to_config: {e}")
            logger.error(f"Input params was: {params}")
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
    
    def _compute_numerical_gradient(self, params: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
        """Compute numerical gradient using finite differences."""
        gradient = np.zeros_like(params)
        base_config = self._params_to_config(params)
        base_cost = self._evaluate_configuration(base_config)
        
        for i in range(len(params)):
            # Forward difference
            params_forward = params.copy()
            params_forward[i] = min(1.0, params_forward[i] + epsilon)
            
            config_forward = self._params_to_config(params_forward)
            cost_forward = self._evaluate_configuration(config_forward)
            
            # Compute gradient
            gradient[i] = (cost_forward - base_cost) / epsilon
        
        return gradient
    
    def _clip_to_bounds(self, params: np.ndarray) -> np.ndarray:
        """Clip parameters to valid bounds."""
        return np.clip(params, 0.0, 1.0)
    
    def _generate_random_configuration(self) -> Dict[str, Any]:
        """Generate a random configuration for initialization."""
        config = {
            'pe_array_x': random.choice(self.search_space['pe_array_x']),
            'pe_array_y': random.choice(self.search_space['pe_array_y']),
            'buffer_size': random.randint(1024, 16384),
            'dataflow': random.choice(self.search_space['dataflow']),
            'precision': random.choice(self.search_space['precision']),
            'memory_hierarchy': {
                'l1_size': random.randint(64, 512),
                'l2_size': random.randint(1024, 4096),
                'bandwidth': random.randint(64, 512)
            },
            'loop_ordering': [random.randint(0, 6) for _ in range(7)],
            'tiling_factors': [random.randint(1, 64) for _ in range(6)]
        }
        return config
    
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
        """Synthetic evaluation function for testing."""
        # Realistic cost function with smooth gradients
        pe_cost = config['pe_array_x'] * config['pe_array_y'] * 0.1
        buffer_cost = config['buffer_size'] * 0.001
        memory_cost = (config['memory_hierarchy']['l1_size'] * 0.01 + 
                      config['memory_hierarchy']['l2_size'] * 0.001)
        
        # Add quadratic terms for smoothness
        pe_penalty = (config['pe_array_x'] - 16) ** 2 * 0.01
        buffer_penalty = (config['buffer_size'] - 8192) ** 2 * 1e-8
        
        # Structured noise (deterministic based on config)
        noise_factor = 1.0 + 0.05 * np.sin(hash(str(config)) % 100)
        
        total_cost = (pe_cost + buffer_cost + memory_cost + pe_penalty + buffer_penalty) * noise_factor
        return total_cost 