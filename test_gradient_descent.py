#!/usr/bin/env python3
"""
Test script to verify gradient descent optimization is working correctly.
"""

import pathlib
import sys
import time

# Add the dataset directory to the path
sys.path.insert(0, str(pathlib.Path(__file__).parent / "dataset"))

from dataset.dse.core.search_engine import SearchEngine
from dataset.common import logger

def test_gradient_descent():
    """Test gradient descent optimization with analytical predictor."""
    
    # Setup
    output_dir = pathlib.Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize search engine
    search_engine = SearchEngine(
        arch_name="gemmini",
        output_dir=output_dir,
        workload="resnet50",
        metric="cycle",
        gpu_id=None  # Use CPU for testing
    )
    
    try:
        # Test gradient descent search
        logger.info("Testing gradient descent optimization...")
        
        results = search_engine.search(
            strategy="gradient_descent",
            n_calls=10,  # Small number for quick testing
            n_initial_points=3
        )
        
        # Print results
        logger.info("Gradient descent results:")
        logger.info(f"  Status: {results.get('status', 'unknown')}")
        logger.info(f"  Strategy: {results.get('strategy', 'unknown')}")
        logger.info(f"  Best cost: {results.get('best_cost', 'unknown')}")
        logger.info(f"  Search time: {results.get('search_time', 'unknown'):.2f}s")
        logger.info(f"  Total evaluations: {results.get('total_evaluations', 'unknown')}")
        
        if results.get('best_config'):
            logger.info(f"  Best config: {results['best_config']}")
        
        # Check if gradient descent was actually used
        if results.get('strategy') == 'gradient_descent':
            logger.info("✓ Gradient descent strategy was used successfully!")
            
            # Check for gradient descent specific info
            convergence_info = results.get('convergence_info', {})
            if convergence_info:
                logger.info(f"  Optimization phase length: {convergence_info.get('optimization_phase_length', 'unknown')}")
                logger.info(f"  Final learning rate: {convergence_info.get('final_learning_rate', 'unknown')}")
            
            return True
        else:
            logger.error("✗ Gradient descent strategy was not used!")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False
    finally:
        search_engine.cleanup()

def test_analytical_with_gradient_descent():
    """Test analytical predictor with gradient descent strategy."""
    
    # Setup
    output_dir = pathlib.Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize search engine
    search_engine = SearchEngine(
        arch_name="gemmini",
        output_dir=output_dir,
        workload="resnet50",
        metric="cycle",
        gpu_id=None
    )
    
    try:
        # Test network search with analytical predictor
        logger.info("Testing analytical predictor with gradient descent...")
        
        results = search_engine.search_network(
            dataset_path="./data/timeloop_dataset/dataset.csv",
            predictor="analytical",
            ordering="shuffle"
        )
        
        # Print results
        logger.info("Network search results:")
        logger.info(f"  Status: {results.get('status', 'unknown')}")
        logger.info(f"  Predictor: {results.get('predictor', 'unknown')}")
        
        search_results = results.get('search_results', {})
        if 'analytical' in search_results:
            analytical_results = search_results['analytical']
            logger.info(f"  Analytical strategy: {analytical_results.get('strategy', 'unknown')}")
            logger.info(f"  Analytical best cost: {analytical_results.get('best_cost', 'unknown')}")
            logger.info(f"  Analytical search time: {analytical_results.get('search_time', 'unknown'):.2f}s")
            
            if analytical_results.get('strategy') == 'gradient_descent':
                logger.info("✓ Analytical predictor successfully used gradient descent!")
                return True
            else:
                logger.error("✗ Analytical predictor did not use gradient descent!")
                return False
        else:
            logger.error("✗ No analytical results found!")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False
    finally:
        search_engine.cleanup()

if __name__ == "__main__":
    logger.info("Starting gradient descent tests...")
    
    # Test 1: Direct gradient descent
    success1 = test_gradient_descent()
    
    # Test 2: Analytical predictor with gradient descent
    success2 = test_analytical_with_gradient_descent()
    
    if success1 and success2:
        logger.info("✓ All tests passed! Gradient descent is working correctly.")
    else:
        logger.error("✗ Some tests failed. Please check the implementation.")
        sys.exit(1) 