"""
Base class for search strategies.
"""

from abc import ABC, abstractmethod
import pathlib
from typing import List, Dict, Any, Optional

from ....common.utils import logger
from ....workloads import Prob


class BaseSearchStrategy(ABC):
    """Abstract base class for design space exploration strategies."""
    
    def __init__(self, 
                 arch_name: str,
                 output_dir: pathlib.Path,
                 layers: List[Prob],
                 mapping_evaluator: Any):
        """
        Initialize base search strategy.
        
        Args:
            arch_name: Architecture name
            output_dir: Output directory
            layers: List of problem layers
            mapping_evaluator: Mapping evaluation component
        """
        self.arch_name = arch_name
        self.output_dir = pathlib.Path(output_dir)
        self.layers = layers
        self.mapping_evaluator = mapping_evaluator
        
        # Strategy state
        self.search_history = []
        self.best_result = None
        self.current_iteration = 0
        
        logger.debug(f"Initialized {self.__class__.__name__} strategy")
    
    @abstractmethod
    def search(self, 
               n_calls: int, 
               n_initial_points: int,
               **kwargs) -> Dict[str, Any]:
        """
        Run the search strategy.
        
        Args:
            n_calls: Number of optimization calls
            n_initial_points: Number of initial random points
            **kwargs: Strategy-specific arguments
            
        Returns:
            Search results dictionary
        """
        pass
    
    def initialize_search(self, n_initial_points: int) -> List[Any]:
        """
        Initialize search with random points.
        
        Args:
            n_initial_points: Number of initial points to generate
            
        Returns:
            List of initial points
        """
        logger.debug(f"Generating {n_initial_points} initial points")
        
        # Default implementation - subclasses can override
        initial_points = []
        for i in range(n_initial_points):
            # Generate random point (placeholder implementation)
            point = self._generate_random_point()
            initial_points.append(point)
        
        return initial_points
    
    def _generate_random_point(self) -> Any:
        """
        Generate a random search point.
        
        Returns:
            Random point in search space
        """
        # Placeholder - subclasses must implement
        raise NotImplementedError("Subclasses must implement _generate_random_point")
    
    def evaluate_point(self, point: Any) -> Dict[str, Any]:
        """
        Evaluate a single point in the search space.
        
        Args:
            point: Point to evaluate
            
        Returns:
            Evaluation results
        """
        try:
            result = self.mapping_evaluator.evaluate(point)
            
            # Track history
            self.search_history.append({
                'iteration': self.current_iteration,
                'point': point,
                'result': result
            })
            
            # Update best result
            if self._is_better_result(result):
                self.best_result = result
                logger.debug(f"New best result at iteration {self.current_iteration}")
            
            self.current_iteration += 1
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate point: {e}")
            return {'error': str(e), 'valid': False}
    
    def _is_better_result(self, result: Dict[str, Any]) -> bool:
        """
        Check if a result is better than the current best.
        
        Args:
            result: Result to check
            
        Returns:
            True if result is better
        """
        if self.best_result is None:
            return True
        
        # Default comparison - minimize objective
        # Subclasses can override for different objectives
        if 'objective' in result and 'objective' in self.best_result:
            return result['objective'] < self.best_result['objective']
        
        return False
    
    def get_search_summary(self) -> Dict[str, Any]:
        """
        Get summary of search progress.
        
        Returns:
            Search summary
        """
        return {
            'strategy_name': self.__class__.__name__,
            'arch_name': self.arch_name,
            'num_layers': len(self.layers),
            'iterations_completed': self.current_iteration,
            'total_evaluations': len(self.search_history),
            'best_result': self.best_result,
            'has_valid_results': self.best_result is not None
        }
    
    def reset(self) -> None:
        """Reset search state."""
        self.search_history.clear()
        self.best_result = None
        self.current_iteration = 0
        logger.debug(f"Reset {self.__class__.__name__} strategy state")
    
    def save_checkpoint(self, checkpoint_path: Optional[pathlib.Path] = None) -> pathlib.Path:
        """
        Save search checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            
        Returns:
            Path where checkpoint was saved
        """
        if checkpoint_path is None:
            checkpoint_path = self.output_dir / f"{self.__class__.__name__}_checkpoint.json"
        
        checkpoint_data = {
            'strategy_name': self.__class__.__name__,
            'current_iteration': self.current_iteration,
            'best_result': self.best_result,
            'search_history_length': len(self.search_history)
        }
        
        from ....common.refactored import FileHandler
        FileHandler.save_json(checkpoint_path, checkpoint_data)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: pathlib.Path) -> bool:
        """
        Load search checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if successfully loaded
        """
        try:
            from ....common.refactored import FileHandler
            checkpoint_data = FileHandler.load_json(checkpoint_path)
            
            if checkpoint_data.get('strategy_name') != self.__class__.__name__:
                logger.warning("Checkpoint strategy name mismatch")
                return False
            
            self.current_iteration = checkpoint_data.get('current_iteration', 0)
            self.best_result = checkpoint_data.get('best_result')
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False 