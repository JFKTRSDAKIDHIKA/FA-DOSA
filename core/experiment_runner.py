"""
Experiment runner for DOSA hardware design space exploration.
"""

import pathlib
import traceback
from typing import List, Dict, Any, Optional

from dataset.hw import GemminiConfig, SimbaConfig
from dataset.workloads import Prob
from dataset.common import logger
from config.run_config import RunConfig
from core.csv_handler import CSVHandler


class ExperimentRunner:
    """Manages the execution of DOSA experiments."""
    
    def __init__(self, config: RunConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Run configuration containing all experiment parameters
        """
        self.config = config
        self.csv_handler = CSVHandler(config.dataset_csv_path)
        
    def run_experiments(self) -> None:
        """Execute all experiments according to the configuration."""
        logger.info("Starting DOSA experiments with configuration:")
        logger.info(f"  Architecture: {self.config.arch_name}")
        logger.info(f"  Mapper: {self.config.mapper}")
        logger.info(f"  Workloads: {self.config.workloads}")
        logger.info(f"  Number of architectures: {self.config.num_arch}")
        logger.info(f"  Number of mappings: {self.config.num_mappings}")
        logger.info(f"  Output directory: {self.config.output_path}")
        
        try:
            self._run_architecture_sweep()
            self._create_compressed_results()
            logger.info("Experiments completed successfully")
            
        except Exception as e:
            logger.error(f"Experiment failed with error: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def _run_architecture_sweep(self) -> None:
        """Run experiments across all architecture configurations."""
        min_metric_fn = self.config.get_min_metric_function()
        
        for arch_idx in range(self.config.num_arch):
            try:
                arch_config = self._create_architecture_config(arch_idx)
                self._run_single_architecture(arch_config, min_metric_fn)
                
                # Log progress
                if arch_idx == self.config.num_arch - 1 or arch_idx % 100 == 0:
                    logger.info(f"Completed {arch_idx + 1} of {self.config.num_arch} architectures")
                    
            except Exception as e:
                logger.error(f"Failed to run architecture {arch_idx}: {e}")
                logger.debug(traceback.format_exc())
                continue
    
    def _create_architecture_config(self, arch_idx: int):
        """Create architecture configuration for the given index."""
        if self.config.arch_file:
            hw_config = pathlib.Path(self.config.arch_file).resolve()
        else:
            hw_config = "random"
        
        if self.config.arch_name == "gemmini":
            return GemminiConfig(hw_config, self.config.logs_dir)
        elif self.config.arch_name == "simba":
            return SimbaConfig(hw_config, self.config.logs_dir)
        else:
            raise ValueError(f"Unsupported architecture: {self.config.arch_name}")
    
    def _run_single_architecture(self, arch_config, min_metric_fn: Optional[callable]) -> None:
        """Run experiments for a single architecture configuration."""
        for layer_path in self.config.workload_layers:
            try:
                layer_prob = Prob(layer_path)
                results = self._run_mapping_experiments(arch_config, layer_prob, min_metric_fn)
                
                if results:
                    self.csv_handler.write_results(results)
                else:
                    logger.warning(f"No results generated for architecture {arch_config.get_config_str()}, "
                                 f"layer {layer_prob.config_str()}")
                    
            except Exception as e:
                logger.error(f"Failed to process layer {layer_path}: {e}")
                logger.debug(traceback.format_exc())
                continue
    
    def _run_mapping_experiments(self, arch_config, layer_prob: Prob, 
                               min_metric_fn: Optional[callable]) -> List[Dict[str, Any]]:
        """Run mapping experiments for a given architecture and layer."""
        try:
            if self.config.mapper == "random":
                return arch_config.run_random_mappings(
                    layer_prob, 
                    self.config.num_mappings, 
                    self.config.exist, 
                    return_min_fn=min_metric_fn
                )
            elif self.config.mapper == "cosa":
                return arch_config.run_cosa(layer_prob, self.config.exist)
            else:
                raise ValueError(f"Unsupported mapper: {self.config.mapper}")
                
        except Exception as e:
            logger.error(f"Mapping experiment failed: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def _create_compressed_results(self) -> None:
        """Create compressed archive of results."""
        if self.csv_handler.exists and not self.csv_handler.is_empty:
            self.csv_handler.create_compressed_archive(self.config.dataset_tarball_path)
        else:
            logger.warning("No results to compress")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment results."""
        return {
            "csv_file": str(self.config.dataset_csv_path),
            "csv_exists": self.csv_handler.exists,
            "row_count": self.csv_handler.get_row_count(),
            "compressed_file": str(self.config.dataset_tarball_path),
            "config": {
                "architecture": self.config.arch_name,
                "mapper": self.config.mapper,
                "workloads": self.config.workloads,
                "num_architectures": self.config.num_arch,
                "num_mappings": self.config.num_mappings,
            }
        } 