#!/usr/bin/env python3
"""
DOSA: Differentiable Model-Based One-Loop Search for DNN Accelerators

This is the refactored main entry point that provides a clean, readable interface
for running DOSA experiments while maintaining all original functionality.
"""

import sys
import traceback
from typing import Optional

from config import ArgumentParser, RunConfig
from core import ExperimentRunner
from dataset.common import logger


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for DOSA experiments.
    
    Args:
        args: Command line arguments (None to use sys.argv)
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse and validate arguments
        parser = ArgumentParser()
        parsed_args = parser.parse_args(args)
        
        # Create run configuration
        config = RunConfig.from_args(parsed_args)
        
        # Log the configuration
        logger.info("DOSA experiment started")
        logger.info(f"Configuration: {config}")
        
        # Run experiments
        runner = ExperimentRunner(config)
        runner.run_experiments()
        
        # Log results summary
        summary = runner.get_results_summary()
        logger.info("Experiment Summary:")
        logger.info(f"  Results file: {summary['csv_file']}")
        logger.info(f"  Total rows: {summary['row_count']}")
        logger.info(f"  Architecture: {summary['config']['architecture']}")
        logger.info(f"  Mapper: {summary['config']['mapper']}")
        logger.info(f"  Workloads: {summary['config']['workloads']}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return 1


def run_gemmini_legacy(layers, output_dir, num_mappings, exist):
    """
    Legacy Gemmini-specific experiment runner.
    
    This function maintains the original Gemmini-specific logic for backward compatibility.
    It should be used only when the original behavior is specifically needed.
    
    Args:
        layers: List of layer paths to process
        output_dir: Output directory path
        num_mappings: Number of mappings per configuration
        exist: Whether to process existing data
    """
    import itertools
    from dataset.hw import GemminiConfig
    from core import CSVHandler
    from dataset.common.refactored.file_utils import FileHandler as utils
    
    logger.warning("Using legacy Gemmini runner - consider using the main interface")
    
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration parameters from original implementation
    BUFFER_MULTIPLIERS = [x/2 for x in range(1, 9, 1)]
    PE_MULTIPLIERS = [0.5, 1, 2, 4]
    
    buffer_multiplier_combinations = [
        combo for combo in itertools.product(BUFFER_MULTIPLIERS, repeat=2)
    ]
    
    csv_handler = CSVHandler(output_dir / "dataset.csv")
    
    # Iterate through all configurations
    for pe_multiplier in PE_MULTIPLIERS:
        for sp_mult, acc_mult in buffer_multiplier_combinations:
            hw_config = [
                int(GemminiConfig.BASE_PE * pe_multiplier),
                int(GemminiConfig.BASE_SP_SIZE * sp_mult),
                int(GemminiConfig.BASE_ACC_SIZE * acc_mult),
            ]
            
            try:
                gemmini_config = GemminiConfig(hw_config, logs_dir)
                
                for layer_path in layers:
                    try:
                        results = gemmini_config.run_random_mappings(
                            layer_path, num_mappings, exist
                        )
                        
                        if results:
                            csv_handler.write_results(results)
                            
                    except Exception as e:
                        logger.error(f"Failed to process layer {layer_path}: {e}")
                        logger.debug(traceback.format_exc())
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to create Gemmini config {hw_config}: {e}")
                logger.debug(traceback.format_exc())
                continue
    
    # Create compressed archive
    csv_handler.create_compressed_archive(output_dir / "dataset.csv.tar.gz")
    logger.info("Legacy Gemmini experiments completed")


if __name__ == "__main__":
    sys.exit(main()) 