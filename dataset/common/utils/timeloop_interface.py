"""
Timeloop interface utilities with improved error handling and async support.
"""

import os
import pathlib
from typing import Optional, Union, Tuple

from .process_utils import ProcessManager, CommandBuilder
from ..logger import logger


class TimeloopInterface:
    """Interface for running Timeloop tools with proper error handling."""
    
    # Default paths for Timeloop executables
    TIMELOOP_MODEL_PATH = "timeloop-model"
    TIMELOOP_MAPPER_PATH = "timeloop-mapper"
    COSA_SCRIPT_PATH = "src/cosa.py"
    
    @classmethod
    def check_availability(cls) -> dict:
        """
        Check availability of Timeloop tools.
        
        Returns:
            Dictionary with availability status for each tool
        """
        status = {
            'timeloop-model': ProcessManager.check_executable(cls.TIMELOOP_MODEL_PATH),
            'timeloop-mapper': ProcessManager.check_executable(cls.TIMELOOP_MAPPER_PATH),
            'python': ProcessManager.check_executable('python')
        }
        
        logger.info(f"Timeloop tools availability: {status}")
        return status
    
    @staticmethod
    def run_timeloop_model(arch_path: Union[str, pathlib.Path],
                          mapping_path: Union[str, pathlib.Path], 
                          prob_path: Union[str, pathlib.Path],
                          cwd: Optional[Union[str, pathlib.Path]] = None,
                          timeout: Optional[int] = None) -> bool:
        """
        Run timeloop-model with error handling.
        
        Args:
            arch_path: Architecture configuration file
            mapping_path: Mapping configuration file
            prob_path: Problem specification file
            cwd: Working directory
            timeout: Timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Resolve paths
            arch_path = pathlib.Path(arch_path).resolve()
            mapping_path = pathlib.Path(mapping_path).resolve()
            prob_path = pathlib.Path(prob_path).resolve()
            
            # Validate input files exist
            for path, name in [(arch_path, "architecture"), 
                              (mapping_path, "mapping"), 
                              (prob_path, "problem")]:
                if not path.exists():
                    logger.error(f"{name.capitalize()} file not found: {path}")
                    return False
            
            # Build command
            cmd = CommandBuilder(TimeloopInterface.TIMELOOP_MODEL_PATH) \
                .add_positional(str(arch_path)) \
                .add_positional(str(mapping_path)) \
                .add_positional(str(prob_path))
            
            logger.info(f"Running timeloop-model: {arch_path.name}, {mapping_path.name}, {prob_path.name}")
            
            result = cmd.execute(cwd=cwd, timeout=timeout, capture_output=True)
            
            if result.returncode == 0:
                logger.debug("timeloop-model completed successfully")
                return True
            else:
                logger.error(f"timeloop-model failed with exit code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run timeloop-model: {e}")
            return False
    
    @staticmethod
    def run_timeloop_mapper(arch_path: Union[str, pathlib.Path],
                           mapspace_path: Union[str, pathlib.Path],
                           prob_path: Union[str, pathlib.Path],
                           cwd: Optional[Union[str, pathlib.Path]] = None,
                           timeout: Optional[int] = None,
                           run_async: bool = False):
        """
        Run timeloop-mapper with error handling.
        
        Args:
            arch_path: Architecture configuration file
            mapspace_path: Mapping space configuration file
            prob_path: Problem specification file
            cwd: Working directory
            timeout: Timeout in seconds
            run_async: Whether to run asynchronously
            
        Returns:
            True/Process if successful, False otherwise
        """
        try:
            # Resolve paths
            arch_path = pathlib.Path(arch_path).resolve()
            mapspace_path = pathlib.Path(mapspace_path).resolve()
            prob_path = pathlib.Path(prob_path).resolve()
            
            # Validate input files exist
            for path, name in [(arch_path, "architecture"), 
                              (mapspace_path, "mapspace"), 
                              (prob_path, "problem")]:
                if not path.exists():
                    logger.error(f"{name.capitalize()} file not found: {path}")
                    return False
            
            # Build command
            cmd = CommandBuilder(TimeloopInterface.TIMELOOP_MAPPER_PATH) \
                .add_positional(str(arch_path)) \
                .add_positional(str(mapspace_path)) \
                .add_positional(str(prob_path))
            
            logger.info(f"Running timeloop-mapper: {arch_path.name}, {mapspace_path.name}, {prob_path.name}")
            
            if run_async:
                process = ProcessManager.run_async(cmd.build(), cwd=cwd)
                logger.debug(f"Started timeloop-mapper async with PID {process.pid}")
                return process
            else:
                result = cmd.execute(cwd=cwd, timeout=timeout, capture_output=True)
                
                if result.returncode == 0:
                    logger.debug("timeloop-mapper completed successfully")
                    return True
                else:
                    logger.error(f"timeloop-mapper failed with exit code {result.returncode}")
                    if result.stderr:
                        logger.error(f"Error output: {result.stderr}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to run timeloop-mapper: {e}")
            return False
    
    @staticmethod
    def run_cosa(arch_path: Union[str, pathlib.Path],
                prob_path: Union[str, pathlib.Path],
                mapping_path: Union[str, pathlib.Path],
                output_dir: Union[str, pathlib.Path],
                output_mapper_yaml_path: Union[str, pathlib.Path],
                cwd: Optional[Union[str, pathlib.Path]] = None,
                timeout: Optional[int] = None,
                run_async: bool = False):
        """
        Run CoSA (Co-designed Scheduling and Allocation) tool.
        
        Args:
            arch_path: Architecture configuration file
            prob_path: Problem specification file  
            mapping_path: Mapping configuration file
            output_dir: Output directory
            output_mapper_yaml_path: Output mapper YAML path
            cwd: Working directory
            timeout: Timeout in seconds
            run_async: Whether to run asynchronously
            
        Returns:
            True/Process if successful, False otherwise
        """
        try:
            # Resolve paths
            arch_path = pathlib.Path(arch_path).resolve()
            prob_path = pathlib.Path(prob_path).resolve()
            mapping_path = pathlib.Path(mapping_path).resolve()
            output_dir = pathlib.Path(output_dir)
            output_mapper_yaml_path = pathlib.Path(output_mapper_yaml_path)
            
            # Validate input files exist
            for path, name in [(arch_path, "architecture"), 
                              (prob_path, "problem"),
                              (mapping_path, "mapping")]:
                if not path.exists():
                    logger.error(f"{name.capitalize()} file not found: {path}")
                    return False
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build command
            cmd = CommandBuilder('python') \
                .add_positional(TimeloopInterface.COSA_SCRIPT_PATH) \
                .add_argument('-ap', str(arch_path)) \
                .add_argument('-pp', str(prob_path)) \
                .add_argument('-mp', str(mapping_path)) \
                .add_argument('-o', str(output_dir)) \
                .add_argument('-omap', str(output_mapper_yaml_path))
            
            logger.info(f"Running CoSA: {arch_path.name}, {prob_path.name}, {mapping_path.name}")
            
            if run_async:
                process = ProcessManager.run_async(cmd.build(), cwd=cwd)
                logger.debug(f"Started CoSA async with PID {process.pid}")
                return process
            else:
                result = cmd.execute(cwd=cwd, timeout=timeout, capture_output=True)
                
                if result.returncode == 0:
                    logger.debug("CoSA completed successfully")
                    return True
                else:
                    logger.error(f"CoSA failed with exit code {result.returncode}")
                    if result.stderr:
                        logger.error(f"Error output: {result.stderr}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to run CoSA: {e}")
            return False
    
    @staticmethod
    def run_timeloop_mapper_with_pruning(arch_path: Union[str, pathlib.Path],
                                        prob_path: Union[str, pathlib.Path],
                                        pruning_strategy: str = "optimal",
                                        cwd: Optional[Union[str, pathlib.Path]] = None,
                                        timeout: Optional[int] = None) -> bool:
        """
        Run timeloop-mapper with specific pruning strategies.
        
        Args:
            arch_path: Architecture configuration file (unused, uses hardcoded arch)
            prob_path: Problem specification file
            pruning_strategy: Strategy - "optimal", "linear", or "hybrid"
            cwd: Working directory
            timeout: Timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Map pruning strategies to mapspace files
            mapspace_files = {
                "optimal": "timeloop_configs/mapspace/mapspace_io_optimal.yaml",
                "linear": "timeloop_configs/mapspace/mapspace_io_linear.yaml", 
                "hybrid": "timeloop_configs/mapspace/mapspace_io_hybrid.yaml"
            }
            
            if pruning_strategy not in mapspace_files:
                logger.error(f"Invalid pruning strategy: {pruning_strategy}")
                return False
            
            # Use hardcoded architecture (as in original code)
            fixed_arch_path = pathlib.Path('timeloop_configs/arch/simba_large.yaml').resolve()
            mapspace_path = pathlib.Path(mapspace_files[pruning_strategy]).resolve()
            prob_path = pathlib.Path(prob_path).resolve()
            
            # Validate files exist
            for path, name in [(fixed_arch_path, "architecture"), 
                              (mapspace_path, "mapspace"), 
                              (prob_path, "problem")]:
                if not path.exists():
                    logger.error(f"{name.capitalize()} file not found: {path}")
                    return False
            
            logger.info(f"Running timeloop-mapper with {pruning_strategy.upper()} pruning")
            
            return TimeloopInterface.run_timeloop_mapper(
                fixed_arch_path, mapspace_path, prob_path, cwd, timeout
            )
            
        except Exception as e:
            logger.error(f"Failed to run timeloop-mapper with pruning: {e}")
            return False
    
    @staticmethod
    def parse_timeloop_output(output_dir: Union[str, pathlib.Path]) -> dict:
        """
        Parse Timeloop output files to extract results.
        
        Args:
            output_dir: Directory containing Timeloop output files
            
        Returns:
            Dictionary with parsed results
        """
        try:
            output_dir = pathlib.Path(output_dir)
            
            if not output_dir.exists():
                logger.error(f"Output directory not found: {output_dir}")
                return {}
            
            results = {
                'output_dir': str(output_dir),
                'files_found': [],
                'stats': {}
            }
            
            # Look for common Timeloop output files
            common_files = [
                'timeloop-model.stats.txt',
                'timeloop-model.map.txt', 
                'timeloop-mapper.stats.txt',
                'timeloop-mapper.map.txt'
            ]
            
            for filename in common_files:
                file_path = output_dir / filename
                if file_path.exists():
                    results['files_found'].append(filename)
                    
                    # Basic parsing for stats files
                    if filename.endswith('.stats.txt'):
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                                # Extract basic metrics (this could be expanded)
                                if 'Cycles:' in content:
                                    for line in content.split('\n'):
                                        if 'Cycles:' in line:
                                            results['stats']['cycles'] = line.strip()
                                            break
                        except Exception as e:
                            logger.warning(f"Failed to parse {filename}: {e}")
            
            logger.debug(f"Parsed Timeloop output: {len(results['files_found'])} files found")
            return results
            
        except Exception as e:
            logger.error(f"Failed to parse Timeloop output: {e}")
            return {}
    
    @staticmethod
    def cleanup_temp_files(base_dir: Union[str, pathlib.Path] = '.') -> int:
        """
        Clean up temporary files created by Timeloop.
        
        Args:
            base_dir: Base directory to search for temp files
            
        Returns:
            Number of files cleaned up
        """
        cleanup_patterns = [
            'dramsim.*.log',
            '*.tmp',
            'timeloop-*.tmp',
            'cosa-*.tmp'
        ]
        
        from .file_utils import FileHandler
        return FileHandler.cleanup_files(cleanup_patterns, base_dir) 