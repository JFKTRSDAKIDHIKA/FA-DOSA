"""
DOSA: Mapping-driven hardware design space exploration.

This module provides network-level design space exploration functionality
using the modernized SearchEngine with intelligent device management.
"""

import pathlib
import os
from typing import Dict, Any

from .core import SearchEngine
from ..common import logger


def _setup_device_environment(use_cpu_fallback: bool = True) -> bool:
    """
    Setup device environment and handle CUDA compatibility issues.
    
    Args:
        use_cpu_fallback: Whether to fallback to CPU on CUDA issues
        
    Returns:
        True if using CPU mode, False if using GPU mode
    """
    try:
        import torch
        
        # Check if CUDA is available and compatible
        if torch.cuda.is_available():
            # Test a simple operation to see if CUDA actually works
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor + 1  # Simple operation to trigger any CUDA errors
                logger.info("CUDA is available and working")
                return False  # Using GPU
            except Exception as cuda_error:
                if use_cpu_fallback:
                    logger.warning(f"CUDA error detected: {cuda_error}")
                    logger.info("Automatically falling back to CPU mode")
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    return True  # Using CPU
                else:
                    raise cuda_error
        else:
            logger.info("CUDA not available, using CPU")
            return True  # Using CPU
            
    except ImportError:
        logger.info("PyTorch not available, using CPU")
        return True  # Using CPU


def search_network(arch_name: str,
                   output_dir: str,
                   workload: str,
                   dataset_path: str,
                   predictor: str = "analytical",
                   search_strategy: str = "auto",
                   plot_only: bool = False,
                   ordering: str = "shuffle") -> Dict[str, Any]:
    """
    Run network-level design space exploration.
    
    Args:
        arch_name: Target architecture name (e.g., 'gemmini')
        output_dir: Output directory for results
        workload: Workload name to search
        dataset_path: Path to dataset file
        predictor: Predictor type ('analytical', 'dnn', 'both')
        search_strategy: Search strategy ('auto', 'bayesian', 'gradient_descent', 'random')
        plot_only: Whether to only generate plots
        ordering: Layer ordering strategy
        
    Returns:
        Search results dictionary
    """
    logger.info(f"Starting mapping-driven hardware search")
    logger.info(f"  Architecture: {arch_name}")
    logger.info(f"  Workload: {workload}")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Predictor: {predictor}")
    logger.info(f"  Search Strategy: {search_strategy}")
    logger.info(f"  Output: {output_dir}")
    
    # Convert output_dir to Path
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect and setup device environment
    using_cpu = _setup_device_environment(use_cpu_fallback=True)
    if using_cpu:
        logger.info("🖥️  Running in CPU mode")
    else:
        logger.info("🚀 Running in GPU mode")
    
    try:
        # Create SearchEngine instance with proper GPU settings
        if using_cpu:
            # For CPU mode, we need to modify the SearchEngine to handle CPU-only mode
            search_engine = SearchEngine(
                arch_name=arch_name,
                output_dir=output_path,
                workload=workload,
                metric="cycle",
                gpu_id=None,  # None for CPU mode
                log_times=True
            )
        else:
            search_engine = SearchEngine(
                arch_name=arch_name,
                output_dir=output_path,
                workload=workload,
                metric="cycle",
                gpu_id=0,  # Use GPU 0
                log_times=True
            )
        
        # Run network search
        results = search_engine.search_network(
            dataset_path=dataset_path,
            predictor=predictor,
            search_strategy=search_strategy,
            plot_only=plot_only,
            ordering=ordering
        )
        
        # Add device info to results
        results['device_info'] = {
            'using_cpu': using_cpu,
            'cuda_available': False if using_cpu else True
        }
        
        logger.info("Mapping-driven hardware search completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Mapping-driven hardware search failed: {e}")
        raise


# Legacy function aliases for full backward compatibility
def main(*args, **kwargs):
    """Legacy main function."""
    logger.warning("Using legacy main() function. Consider migrating to search_network()")
    return search_network(*args, **kwargs)


def run_search(*args, **kwargs):
    """Legacy run_search function."""
    logger.warning("Using legacy run_search() function. Consider migrating to search_network()")
    return search_network(*args, **kwargs) 