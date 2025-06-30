"""
Refactored hardware configuration base class with improved modularity.
"""

import pathlib
import random
from typing import Dict, Any, List, Union, Optional, Tuple
from abc import ABC, abstractmethod

from dataset.common import logger
from dataset.workloads import Prob


class HardwareConfigBase(ABC):
    """
    Abstract base class for hardware configurations with modular architecture.
    
    This refactored version separates concerns into specialized components:
    - ArchitectureGenerator: Handles architecture generation and parsing
    - MappingProcessor: Processes mapping conversions and validations  
    - TimeloopRunner: Manages Timeloop execution
    - ResultParser: Parses simulation results
    """
    
    # Subclasses must define these
    NAME: str = "ABSTRACT"
    BASE_ARCH_PATH: pathlib.Path = None
    BASE_MAPSPACE_PATH: pathlib.Path = None
    
    def __init__(self, hw_config: Union[str, List, pathlib.Path], output_dir: pathlib.Path):
        """
        Initialize hardware configuration.
        
        Args:
            hw_config: Hardware configuration (parameters or path)
            output_dir: Output directory for this configuration
        """
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration
        self.hw_config = self._process_hw_config(hw_config)
        self.config_dir = self._create_config_directory()
        self.arch_path = self.config_dir / f"{self.get_config_str()}.yaml"
        
        # Initialize components
        self._init_components()
        
        # Generate architecture if needed
        self._ensure_architecture_exists()
    
    def _process_hw_config(self, hw_config: Union[str, List, pathlib.Path]) -> List:
        """Process and validate hardware configuration input."""
        if isinstance(hw_config, str):
            if hw_config.lower() == "random":
                return self.gen_random_hw_config()
            else:
                # Parse from string format if needed
                try:
                    return eval(hw_config)  # Be careful with this in production
                except:
                    logger.error(f"Cannot parse hw_config string: {hw_config}")
                    raise ValueError(f"Invalid hw_config string: {hw_config}")
        
        elif isinstance(hw_config, pathlib.Path):
            # Parse from existing architecture file
            parsed_config = self.arch_generator.parse_existing_config(hw_config)
            if parsed_config is None:
                raise ValueError(f"Cannot parse architecture file: {hw_config}")
            return parsed_config
        
        elif isinstance(hw_config, (list, tuple)):
            return list(hw_config)
        
        else:
            raise ValueError(f"Unsupported hw_config type: {type(hw_config)}")
    
    def _create_config_directory(self) -> pathlib.Path:
        """Create and return configuration directory for this hardware config."""
        config_str = self.get_config_str()
        config_dir = self.output_dir / f"config_{config_str}"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
    
    def _init_components(self) -> None:
        """Initialize all component modules."""
        from dataset.hw.components import (
            MappingProcessor, TimeloopRunner, 
            ArchitectureGenerator, ResultParser
        )
        
        # Architecture generator
        self.arch_generator = ArchitectureGenerator(
            base_arch_path=self.BASE_ARCH_PATH,
            arch_name=self.NAME
        )
        
        # Mapping processor
        self.mapping_processor = MappingProcessor(
            arch_name=self.NAME,
            mapspace_path=self.BASE_MAPSPACE_PATH
        )
        
        # Timeloop runner
        self.timeloop_runner = TimeloopRunner(
            arch_path=self.arch_path,
            output_dir=self.config_dir
        )
        
        # Result parser
        self.result_parser = ResultParser()
    
    def _ensure_architecture_exists(self) -> None:
        """Generate architecture file if it doesn't exist."""
        if not self.arch_path.exists():
            arch_dict = self.arch_generator.generate_arch_config(
                hw_config=self.hw_config,
                output_path=self.arch_path
            )
            
            if arch_dict is None:
                raise ValueError(f"Failed to generate valid architecture for {self.hw_config}")
            
            logger.debug(f"Generated architecture: {self.arch_path}")
    
    @abstractmethod
    def gen_random_hw_config(self) -> List:
        """Generate random hardware configuration parameters."""
        pass
    
    def get_config_str(self) -> str:
        """Get string representation of hardware configuration."""
        config_parts = [str(round(x, 3) if isinstance(x, float) else x) 
                       for x in self.hw_config]
        return "_".join(config_parts)
    
    def validate_config(self) -> bool:
        """Validate the current hardware configuration."""
        return self.arch_generator.validate_config(self.hw_config)
    
    def run_random_mappings(self, layer_prob: Prob, num_mappings: int, 
                           exist: bool = False, return_min_fn=None) -> List[Dict[str, Any]]:
        """Run random mapping exploration using Timeloop mapper."""
        logger.info(f"Running {num_mappings} random mappings for {layer_prob.config_str()}")
        
        # Prepare output directory
        output_path = self.config_dir / layer_prob.config_str()
        output_path.mkdir(parents=True, exist_ok=True)
        output_log_file = output_path / "random.txt"
        
        if not exist or not output_log_file.exists():
            # Prepare mapspace configuration
            arch_constraints = self._get_arch_specific_constraints()
            mapspace_path = self.timeloop_runner.prepare_mapspace_config(
                mapspace_path=self.BASE_MAPSPACE_PATH,
                num_mappings=num_mappings,
                arch_specific_constraints=arch_constraints
            )
            
            # Run mapper
            success = self.timeloop_runner.run_mapper(
                layer_prob=layer_prob,
                mapspace_path=mapspace_path,
                run_dir=output_path
            )
            
            if not success:
                logger.error(f"Failed to run random mappings for {layer_prob.config_str()}")
                return []
        
        # Parse results
        results = self.result_parser.parse_random_output(output_log_file, layer_prob)
        
        if len(results) == num_mappings:
            logger.info(f"Successfully ran {len(results)} random mappings")
        else:
            logger.warning(f"Ran {len(results)} mappings, expected {num_mappings}")
        
        # Apply minimum selection if requested
        if return_min_fn and results:
            min_result = min(results, key=return_min_fn)
            return [min_result]
        
        return results
    
    def _get_arch_specific_constraints(self) -> Optional[Dict[str, str]]:
        """Get architecture-specific mapping constraints."""
        # Subclasses can override this for specific constraints
        return None 