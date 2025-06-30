"""
Timeloop runner component for executing Timeloop simulations.
"""

import os
import subprocess
import pathlib
import time
import random
import string
from typing import Dict, Any, Optional, Union, Tuple

from dataset.common import utils, logger
from dataset.workloads import Prob


class TimeloopRunner:
    """Handles execution of Timeloop simulations and mapping operations."""
    
    def __init__(self, arch_path: pathlib.Path, output_dir: pathlib.Path):
        """
        Initialize Timeloop runner.
        
        Args:
            arch_path: Path to architecture configuration file
            output_dir: Base output directory for runs
        """
        self.arch_path = arch_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_run_directory(self, layer_prob: Prob, custom_mapping: bool = False) -> pathlib.Path:
        """
        Create a unique directory for this Timeloop run.
        
        Args:
            layer_prob: Problem specification
            custom_mapping: Whether this is a custom mapping run
            
        Returns:
            Path to the created run directory
        """
        output_path = self.output_dir / layer_prob.config_str()
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create unique subdirectory for this specific run
        timestamp = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())
        random_suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) 
                               for _ in range(16))
        
        run_dir_name = f"timeloop-{timestamp}-{random_suffix}"
        if custom_mapping:
            run_dir_name = f"custom-{run_dir_name}"
        
        run_dir = output_path / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        return run_dir
    
    def run_mapper(self, layer_prob: Prob, mapspace_path: pathlib.Path, 
                   run_dir: Optional[pathlib.Path] = None,
                   run_async: bool = False) -> Union[bool, subprocess.Popen]:
        """
        Run Timeloop mapper for random mapping exploration.
        
        Args:
            layer_prob: Problem specification
            mapspace_path: Path to mapspace configuration
            run_dir: Run directory (created if None)
            run_async: Whether to run asynchronously
            
        Returns:
            True/False for success/failure, or subprocess.Popen if async
        """
        if run_dir is None:
            run_dir = self.create_run_directory(layer_prob)
        
        output_log_file = run_dir / "mapper.txt"
        
        try:
            with open(output_log_file, "w") as log_file:
                result = utils.run_timeloop_mapper(
                    arch=self.arch_path,
                    prob=layer_prob.path,
                    mapspace=mapspace_path,
                    cwd=run_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    run_async=run_async
                )
            
            if run_async:
                logger.debug(f"Started async Timeloop mapper in {run_dir}")
                return result
            else:
                if result:
                    logger.debug(f"Timeloop mapper completed successfully in {run_dir}")
                else:
                    logger.warning(f"Timeloop mapper failed in {run_dir}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to run Timeloop mapper: {e}")
            return False
    
    def run_model(self, layer_prob: Prob, mapping_path: pathlib.Path,
                  run_dir: Optional[pathlib.Path] = None) -> bool:
        """
        Run Timeloop model for a specific mapping.
        
        Args:
            layer_prob: Problem specification
            mapping_path: Path to mapping configuration
            run_dir: Run directory (created if None)
            
        Returns:
            True if successful, False otherwise
        """
        if run_dir is None:
            run_dir = self.create_run_directory(layer_prob, custom_mapping=True)
        
        try:
            success = utils.run_timeloop(
                arch=self.arch_path,
                prob=layer_prob.path,
                mapping=mapping_path,
                cwd=run_dir
            )
            
            if success:
                logger.debug(f"Timeloop model completed successfully in {run_dir}")
            else:
                logger.warning(f"Timeloop model failed in {run_dir}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to run Timeloop model: {e}")
            return False
    
    def run_cosa_mapping(self, layer_prob: Prob, mapspace_path: pathlib.Path,
                        cosa_dir: pathlib.Path, 
                        run_async: bool = False) -> Union[bool, subprocess.Popen]:
        """
        Run CoSA mapper for generating optimized mappings.
        
        Args:
            layer_prob: Problem specification
            mapspace_path: Path to mapspace configuration
            cosa_dir: CoSA installation directory
            run_async: Whether to run asynchronously
            
        Returns:
            True/False for success/failure, or subprocess.Popen if async
        """
        output_path = self.output_dir / layer_prob.config_str()
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare paths
        prob_path = pathlib.Path(layer_prob.path).resolve()
        mapspace_resolved = pathlib.Path(mapspace_path).resolve()
        output_dir_resolved = pathlib.Path(output_path).resolve()
        output_mapper_yaml_path = output_path / "mapspace.yaml"
        
        # Set COSA environment
        os.environ["COSA_DIR"] = str(cosa_dir)
        
        try:
            if run_async:
                result = utils.run_cosa_async(
                    arch_path=self.arch_path,
                    prob_path=prob_path,
                    mapping_path=mapspace_resolved,
                    output_dir=output_dir_resolved,
                    output_mapper_yaml_path=output_mapper_yaml_path.resolve(),
                    cwd=cosa_dir
                )
                logger.debug(f"Started async CoSA mapping for {layer_prob.config_str()}")
                return result
            else:
                success = utils.run_cosa(
                    arch_path=self.arch_path,
                    prob_path=prob_path,
                    mapping_path=mapspace_resolved,
                    output_dir=output_dir_resolved,
                    output_mapper_yaml_path=output_mapper_yaml_path.resolve(),
                    cwd=cosa_dir
                )
                
                if success:
                    logger.debug(f"CoSA mapping completed successfully for {layer_prob.config_str()}")
                else:
                    logger.warning(f"CoSA mapping failed for {layer_prob.config_str()}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to run CoSA mapping: {e}")
            return False
    
    def prepare_mapspace_config(self, mapspace_path: pathlib.Path, 
                               num_mappings: int,
                               arch_specific_constraints: Optional[Dict[str, Any]] = None) -> pathlib.Path:
        """
        Prepare mapspace configuration with specified parameters.
        
        Args:
            mapspace_path: Base mapspace configuration path
            num_mappings: Number of mappings to explore
            arch_specific_constraints: Architecture-specific constraints
            
        Returns:
            Path to the prepared mapspace configuration
        """
        try:
            mapspace_dict = utils.parse_yaml(mapspace_path)
            
            # Configure mapper settings
            mapper_config = mapspace_dict.setdefault("mapper", {})
            mapper_config["num-threads"] = min(num_mappings, 10)
            mapper_config["search-size"] = num_mappings
            mapper_config["timeout"] = max(num_mappings * 10, 1000)
            
            # Apply architecture-specific constraints
            if arch_specific_constraints:
                self._apply_arch_constraints(mapspace_dict, arch_specific_constraints)
            
            # Save modified mapspace
            new_mapspace_path = self.output_dir / "mapspace.yaml"
            utils.store_yaml(new_mapspace_path, mapspace_dict)
            
            return new_mapspace_path
            
        except Exception as e:
            logger.error(f"Failed to prepare mapspace config: {e}")
            return mapspace_path
    
    def _apply_arch_constraints(self, mapspace_dict: Dict[str, Any], 
                               constraints: Dict[str, Any]) -> None:
        """Apply architecture-specific constraints to mapspace."""
        if "mapspace_constraints" not in mapspace_dict:
            return
        
        for entry in mapspace_dict["mapspace_constraints"]:
            target = entry.get("target")
            entry_type = entry.get("type")
            
            if target in constraints and entry_type == "spatial":
                # Apply spatial constraints for this target
                constraint_value = constraints[target]
                if isinstance(constraint_value, str):
                    entry["factors"] = constraint_value
    
    def cleanup_old_runs(self, max_age_hours: int = 24) -> None:
        """
        Clean up old Timeloop run directories.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for item in self.output_dir.iterdir():
                if item.is_dir() and item.name.startswith("timeloop-"):
                    try:
                        # Extract timestamp from directory name
                        timestamp_part = item.name.split("-")[1:4]  # YYYY-MM-DD
                        if len(timestamp_part) >= 3:
                            dir_time = item.stat().st_mtime
                            if current_time - dir_time > max_age_seconds:
                                logger.debug(f"Cleaning up old run directory: {item}")
                                # Could implement actual cleanup here
                                # shutil.rmtree(item)
                    except Exception as e:
                        logger.debug(f"Error checking run directory {item}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def get_run_status(self, run_dir: pathlib.Path) -> Dict[str, Any]:
        """
        Get status information for a Timeloop run.
        
        Args:
            run_dir: Directory containing the run
            
        Returns:
            Dictionary with run status information
        """
        status = {
            "directory": str(run_dir),
            "exists": run_dir.exists(),
            "completed": False,
            "log_files": [],
            "output_files": []
        }
        
        if not run_dir.exists():
            return status
        
        # Check for log files
        for log_file in ["mapper.txt", "cosa.txt", "timeloop.txt"]:
            log_path = run_dir / log_file
            if log_path.exists():
                status["log_files"].append(log_file)
        
        # Check for output files
        for output_file in ["timeloop-model.stats.txt", "timeloop-model.map.txt"]:
            output_path = run_dir / output_file
            if output_path.exists():
                status["output_files"].append(output_file)
        
        # Simple heuristic for completion
        status["completed"] = len(status["output_files"]) > 0
        
        return status 