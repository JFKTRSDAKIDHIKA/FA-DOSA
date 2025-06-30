"""
Argument parser for DOSA project with proper validation and clear structure.
"""

import argparse
import pathlib
from typing import List, Optional

# Try to import DATASET_ROOT_PATH, but provide fallback for testing
try:
    from dataset import DATASET_ROOT_PATH
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    DATASET_ROOT_PATH = "dataset"  # Fallback path


class ArgumentParser:
    """Clean, well-structured argument parser for DOSA experiments."""
    
    # Supported architectures
    SUPPORTED_ARCHITECTURES = ['gemmini', 'simba']
    
    # Supported mappers
    SUPPORTED_MAPPERS = ['random', 'cosa']
    
    # Supported metrics for optimization
    SUPPORTED_METRICS = ['cycle', 'energy', 'edp']
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description='DOSA: Differentiable Model-Based One-Loop Search for DNN Accelerators',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        self._add_general_arguments(parser)
        self._add_architecture_arguments(parser)
        self._add_workload_arguments(parser)
        self._add_mapping_arguments(parser)
        self._add_output_arguments(parser)
        
        return parser
    
    def _add_general_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add general configuration arguments."""
        parser.add_argument(
            '--random_seed',
            type=int,
            default=1,
            help='Random seed for reproducible results'
        )
        
        parser.add_argument(
            '--exist',
            action='store_true',
            help='Process existing data without re-running experiments'
        )
    
    def _add_architecture_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add architecture-related arguments."""
        parser.add_argument(
            '-an', '--arch_name',
            type=str,
            choices=self.SUPPORTED_ARCHITECTURES,
            default='gemmini',
            help='Hardware architecture to use'
        )
        
        parser.add_argument(
            '--arch_file',
            type=str,
            default=None,
            help='Path to specific architecture YAML file (overrides random generation)'
        )
        
        parser.add_argument(
            '--num_arch',
            type=int,
            default=1,
            help='Number of random architectures to evaluate'
        )
    
    def _add_workload_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add workload-related arguments."""
        parser.add_argument(
            '-bwp', '--base_workload_path',
            type=str,
            default=f'{DATASET_ROOT_PATH}/workloads/',
            help='Base directory containing workload definitions'
        )
        
        parser.add_argument(
            '-wl', '--workload',
            action='append',
            required=True,
            help='Workload directory name (use multiple times for multiple workloads)'
        )
        
        parser.add_argument(
            '--layer_idx',
            type=str,
            default='',
            help='Specific DNN layer index to target (optional)'
        )
    
    def _add_mapping_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add mapping-related arguments."""
        parser.add_argument(
            '--mapper',
            type=str,
            choices=self.SUPPORTED_MAPPERS,
            default='random',
            help='Mapping strategy to use'
        )
        
        parser.add_argument(
            '--num_mappings',
            type=int,
            default=1000,
            help='Number of mappings to evaluate per problem/hardware config'
        )
        
        parser.add_argument(
            '--min_metric',
            type=str,
            choices=self.SUPPORTED_METRICS,
            default=None,
            help='Save only the minimum mapping according to this metric'
        )
    
    def _add_output_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add output-related arguments."""
        parser.add_argument(
            '-o', '--output_dir',
            type=str,
            default='output_random',
            help='Output directory for results'
        )
    
    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse arguments with validation."""
        parsed_args = self.parser.parse_args(args)
        if DATASET_AVAILABLE:
            self._validate_arguments(parsed_args)
        return parsed_args
    
    def _validate_arguments(self, args: argparse.Namespace) -> None:
        """Validate parsed arguments for consistency and correctness."""
        # Validate architecture file path if provided
        if args.arch_file and not pathlib.Path(args.arch_file).is_file():
            raise ValueError(f"Architecture file not found: {args.arch_file}")
        
        # Validate base workload path
        workload_path = pathlib.Path(args.base_workload_path)
        if not workload_path.is_dir():
            raise ValueError(f"Workload directory not found: {args.base_workload_path}")
        
        # Validate workload directories exist
        for workload in args.workload:
            wl_path = workload_path / workload
            if not wl_path.is_dir():
                raise ValueError(f"Workload '{workload}' not found in {args.base_workload_path}")
        
        # Validate numeric arguments
        if args.num_arch < 1:
            raise ValueError("Number of architectures must be positive")
        
        if args.num_mappings < 1:
            raise ValueError("Number of mappings must be positive")
        
        if args.random_seed < 0:
            raise ValueError("Random seed must be non-negative")
    
    def print_help(self) -> None:
        """Print help message."""
        self.parser.print_help() 