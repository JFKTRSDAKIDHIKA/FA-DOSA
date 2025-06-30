"""
Data structures and utilities for the DOSA project.
"""

import collections
import random
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np

from ..logger import logger


class OrderedDefaultdict(collections.OrderedDict):
    """A defaultdict with OrderedDict as its base class."""

    def __init__(self, default_factory: Optional[Callable] = None, *args, **kwargs):
        """
        Initialize OrderedDefaultdict.
        
        Args:
            default_factory: Factory function for default values
            *args: Positional arguments for OrderedDict
            **kwargs: Keyword arguments for OrderedDict
        """
        if not (default_factory is None or isinstance(default_factory, collections.abc.Callable)):
            raise TypeError('first argument must be callable or None')
        super(OrderedDefaultdict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        """Handle missing keys by creating default values."""
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        """Support for pickle serialization."""
        args = (self.default_factory,) if self.default_factory else tuple()
        return self.__class__, args, None, None, iter(self.items())

    def __repr__(self):
        """String representation."""
        return f'{self.__class__.__name__}({self.default_factory!r}, {list(self.items())!r})'


class DatasetSeed:
    """Global seed management for reproducible datasets."""
    
    def __init__(self, seed: int = 0):
        """
        Initialize with a seed value.
        
        Args:
            seed: Initial seed value
        """
        self.seed = seed
        logger.debug(f"Initialized DatasetSeed with seed {seed}")
    
    def set_seed(self, seed: int) -> None:
        """
        Set the seed value.
        
        Args:
            seed: New seed value
        """
        self.seed = seed
        self._apply_seed()
        logger.info(f"Set global seed to {seed}")
    
    def _apply_seed(self) -> None:
        """Apply the seed to random number generators."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass
    
    def get_seed(self) -> int:
        """
        Get the current seed value.
        
        Returns:
            Current seed value
        """
        return self.seed


class ConfigDict(dict):
    """Enhanced dictionary with dot notation access and validation."""
    
    def __init__(self, *args, **kwargs):
        """Initialize ConfigDict."""
        super().__init__(*args, **kwargs)
        self._make_nested_configdicts()
    
    def _make_nested_configdicts(self):
        """Convert nested dictionaries to ConfigDict instances."""
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)
    
    def __getattr__(self, key: str) -> Any:
        """Allow dot notation access."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        """Allow dot notation setting."""
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self[key] = value
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)
    
    def __delattr__(self, key: str) -> None:
        """Allow dot notation deletion."""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested value using dot-separated key path.
        
        Args:
            key_path: Dot-separated path (e.g., 'arch.memory.size')
            default: Default value if path not found
            
        Returns:
            Value at the key path or default
        """
        try:
            keys = key_path.split('.')
            value = self
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_nested(self, key_path: str, value: Any) -> None:
        """
        Set nested value using dot-separated key path.
        
        Args:
            key_path: Dot-separated path (e.g., 'arch.memory.size')
            value: Value to set
        """
        keys = key_path.split('.')
        current = self
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = ConfigDict()
            elif not isinstance(current[key], dict):
                raise ValueError(f"Cannot set nested key '{key_path}': '{key}' is not a dictionary")
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def flatten(self, prefix: str = '', separator: str = '.') -> Dict[str, Any]:
        """
        Flatten nested dictionary into flat key-value pairs.
        
        Args:
            prefix: Prefix for keys
            separator: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        result = {}
        for key, value in self.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                result.update(ConfigDict(value).flatten(new_key, separator))
            else:
                result[new_key] = value
        
        return result
    
    def update_nested(self, other: Dict[str, Any]) -> None:
        """
        Update this ConfigDict with another dictionary, merging nested structures.
        
        Args:
            other: Dictionary to merge
        """
        for key, value in other.items():
            if key in self and isinstance(self[key], dict) and isinstance(value, dict):
                if not isinstance(self[key], ConfigDict):
                    self[key] = ConfigDict(self[key])
                self[key].update_nested(value)
            else:
                self[key] = ConfigDict(value) if isinstance(value, dict) else value


class ResultCollector:
    """Collect and manage experimental results."""
    
    def __init__(self):
        """Initialize result collector."""
        self.results = []
        self.metadata = {}
        logger.debug("Initialized ResultCollector")
    
    def add_result(self, result: Dict[str, Any], 
                   experiment_id: Optional[str] = None) -> None:
        """
        Add a result to the collection.
        
        Args:
            result: Result dictionary
            experiment_id: Optional experiment identifier
        """
        result_entry = {
            'data': result.copy(),
            'experiment_id': experiment_id,
            'timestamp': np.datetime64('now'),
            'index': len(self.results)
        }
        self.results.append(result_entry)
        logger.debug(f"Added result {len(self.results) - 1}")
    
    def get_results(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get results, optionally filtered by experiment ID.
        
        Args:
            experiment_id: Optional experiment ID filter
            
        Returns:
            List of matching results
        """
        if experiment_id is None:
            return [r['data'] for r in self.results]
        else:
            return [r['data'] for r in self.results if r['experiment_id'] == experiment_id]
    
    def get_best_result(self, metric_key: str, 
                       minimize: bool = True,
                       experiment_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the best result based on a metric.
        
        Args:
            metric_key: Key of the metric to optimize
            minimize: Whether to minimize the metric
            experiment_id: Optional experiment ID filter
            
        Returns:
            Best result dictionary or None if no results
        """
        filtered_results = self.get_results(experiment_id)
        
        if not filtered_results:
            return None
        
        # Filter results that have the metric
        valid_results = [r for r in filtered_results if metric_key in r]
        
        if not valid_results:
            logger.warning(f"No results found with metric '{metric_key}'")
            return None
        
        best_result = min(valid_results, key=lambda x: x[metric_key]) if minimize else \
                     max(valid_results, key=lambda x: x[metric_key])
        
        return best_result
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of collected results.
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {'total_results': 0}
        
        summary = {
            'total_results': len(self.results),
            'experiment_ids': list(set(r['experiment_id'] for r in self.results if r['experiment_id'])),
            'data_keys': set()
        }
        
        # Collect all data keys
        for result in self.results:
            summary['data_keys'].update(result['data'].keys())
        
        summary['data_keys'] = list(summary['data_keys'])
        return summary
    
    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()
        self.metadata.clear()
        logger.debug("Cleared all results")


class PerformanceTracker:
    """Track performance metrics and timing information."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.metrics = OrderedDefaultdict(list)
        self.timings = OrderedDefaultdict(list)
        self.counters = OrderedDefaultdict(int)
        logger.debug("Initialized PerformanceTracker")
    
    def record_metric(self, name: str, value: float) -> None:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name].append(value)
        logger.debug(f"Recorded metric {name}: {value}")
    
    def record_timing(self, name: str, duration: float) -> None:
        """
        Record a timing measurement.
        
        Args:
            name: Timing name
            duration: Duration in seconds
        """
        self.timings[name].append(duration)
        logger.debug(f"Recorded timing {name}: {duration:.4f}s")
    
    def increment_counter(self, name: str, amount: int = 1) -> None:
        """
        Increment a counter.
        
        Args:
            name: Counter name
            amount: Amount to increment
        """
        self.counters[name] += amount
        logger.debug(f"Incremented counter {name} by {amount} (total: {self.counters[name]})")
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with statistics
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = np.array(self.metrics[name])
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for timing measurements.
        
        Args:
            name: Timing name
            
        Returns:
            Dictionary with timing statistics
        """
        if name not in self.timings or not self.timings[name]:
            return {}
        
        values = np.array(self.timings[name])
        return {
            'count': len(values),
            'total_time': float(np.sum(values)),
            'mean_time': float(np.mean(values)),
            'std_time': float(np.std(values)),
            'min_time': float(np.min(values)),
            'max_time': float(np.max(values))
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete performance summary.
        
        Returns:
            Summary of all tracked performance data
        """
        summary = {
            'metrics': {name: self.get_metric_stats(name) for name in self.metrics},
            'timings': {name: self.get_timing_stats(name) for name in self.timings},
            'counters': dict(self.counters)
        }
        return summary
    
    def reset(self) -> None:
        """Reset all tracked performance data."""
        self.metrics.clear()
        self.timings.clear()
        self.counters.clear()
        logger.debug("Reset all performance tracking data") 