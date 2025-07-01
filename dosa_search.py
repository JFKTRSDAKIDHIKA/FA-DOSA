#!/usr/bin/env python3
"""
DOSA Search Runner: Design space exploration with mapping-driven hardware search.

This is the refactored search runner that provides a clean interface for
running network-level design space exploration experiments.
"""

import argparse
import pathlib
import sys
import traceback
from typing import Optional
from dataclasses import dataclass

from dataset import DATASET_ROOT_PATH
from dataset.dse import mapping_driven_hw_search
from dataset.common import logger


@dataclass
class SearchConfig:
    """Configuration for DOSA search experiments."""
    
    output_dir: str
    dataset_path: str
    workload: str
    arch_name: str = 'gemmini'
    predictor: str = 'analytical'
    search_strategy: str = 'auto'  # 'auto', 'bayesian', 'gradient_descent', 'random'
    plot_only: bool = False
    ordering: str = 'shuffle'
    use_cpu: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate the search configuration."""
        # Validate predictor
        valid_predictors = ['analytical', 'dnn', 'both']
        if self.predictor not in valid_predictors:
            raise ValueError(f"Predictor '{self.predictor}' not supported. "
                           f"Valid options: {valid_predictors}")
        
        # Validate search strategy
        valid_strategies = ['auto', 'bayesian', 'gradient_descent', 'random']
        if self.search_strategy not in valid_strategies:
            raise ValueError(f"Search strategy '{self.search_strategy}' not supported. "
                           f"Valid options: {valid_strategies}")
        
        # Validate dataset path
        if not pathlib.Path(self.dataset_path).is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        # Validate output directory can be created
        output_path = pathlib.Path(self.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create output directory {self.output_dir}: {e}")
    
    @property
    def output_path(self) -> pathlib.Path:
        """Get the resolved output path."""
        return pathlib.Path(self.output_dir).resolve()


class SearchArgumentParser:
    """Argument parser for DOSA search experiments."""
    
    VALID_PREDICTORS = ['analytical', 'dnn', 'both']
    VALID_ORDERINGS = ['shuffle', 'sequential', 'random']
    VALID_STRATEGIES = ['auto', 'bayesian', 'gradient_descent', 'random']
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description='DOSA: Mapping-driven hardware design space exploration',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            '-o', '--output_dir',
            type=str,
            default='output_dir',
            help='Output directory for search results'
        )
        
        parser.add_argument(
            '--dataset_path',
            type=str,
            required=True,
            help='Path to the dataset file for search'
        )
        
        parser.add_argument(
            '-wl', '--workload',
            type=str,
            required=True,
            help='Name of the workload directory to search'
        )
        
        parser.add_argument(
            '--predictor',
            type=str,
            choices=self.VALID_PREDICTORS,
            default='analytical',
            help='Type of performance predictor to use'
        )
        
        parser.add_argument(
            '--search_strategy',
            type=str,
            choices=self.VALID_STRATEGIES,
            default='auto',
            help='Search strategy to use (auto: automatically choose based on predictor)'
        )
        
        parser.add_argument(
            '--plot_only',
            action='store_true',
            help='Only generate plots from existing results (skip search)'
        )
        
        parser.add_argument(
            '--ordering',
            type=str,
            choices=self.VALID_ORDERINGS,
            default='shuffle',
            help='Ordering strategy for search space exploration'
        )
        
        parser.add_argument(
            '--arch_name',
            type=str,
            default='gemmini',
            help='Target architecture name'
        )
        
        parser.add_argument(
            '--use_cpu',
            action='store_true',
            help='Force CPU-only mode (disable GPU)'
        )
        
        return parser
    
    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse arguments with validation."""
        return self.parser.parse_args(args)


def run_search_experiment(config: SearchConfig) -> None:
    """
    Run design space exploration experiment.
    
    Args:
        config: Search configuration containing all parameters
    """
    logger.info("Starting DOSA search experiment")
    logger.info(f"  Workload: {config.workload}")
    logger.info(f"  Dataset: {config.dataset_path}")
    logger.info(f"  Predictor: {config.predictor}")
    logger.info(f"  Search Strategy: {config.search_strategy}")
    logger.info(f"  Output: {config.output_path}")
    logger.info(f"  Plot only: {config.plot_only}")
    logger.info(f"  Ordering: {config.ordering}")
    logger.info(f"  Use CPU: {config.use_cpu}")
    
    # Set CPU mode if requested
    if config.use_cpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("Forced CPU-only mode")
    
    try:
        # Run the mapping-driven hardware search
        mapping_driven_hw_search.search_network(
            arch_name=config.arch_name,
            output_dir=config.output_dir,
            workload=config.workload,
            dataset_path=config.dataset_path,
            predictor=config.predictor,
            search_strategy=config.search_strategy,
            plot_only=config.plot_only,
            ordering=config.ordering
        )
        
        logger.info("Search experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Search experiment failed: {e}")
        logger.debug(traceback.format_exc())
        raise


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for DOSA search experiments.
    
    Args:
        args: Command line arguments (None to use sys.argv)
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments
        parser = SearchArgumentParser()
        parsed_args = parser.parse_args(args)
        
        # Create configuration
        config = SearchConfig(
            output_dir=parsed_args.output_dir,
            dataset_path=parsed_args.dataset_path,
            workload=parsed_args.workload,
            arch_name=parsed_args.arch_name,
            predictor=parsed_args.predictor,
            search_strategy=parsed_args.search_strategy,
            plot_only=parsed_args.plot_only,
            ordering=parsed_args.ordering,
            use_cpu=parsed_args.use_cpu
        )
        
        # Run search experiment
        run_search_experiment(config)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Search experiment interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Search experiment failed: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 