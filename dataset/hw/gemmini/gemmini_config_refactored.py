"""
Refactored Gemmini hardware configuration class.
"""

import pathlib
import random
from typing import List, Dict, Any, Optional

from dataset import DATASET_ROOT_PATH
from dataset.hw.hardware_config_refactored import HardwareConfigBase
from dataset.common import logger

# Gemmini-specific constants
GEMMINI_DIR = DATASET_ROOT_PATH / "hw" / "gemmini"


class GemminiConfigRefactored(HardwareConfigBase):
    """
    Refactored Gemmini architecture configuration class.
    
    This version uses the new component-based architecture for better modularity:
    - Cleaner initialization and configuration management
    - Separated architecture generation logic
    - Better error handling and validation
    - Improved mapping processing
    
    Hardware config format: [pe_dim, sp_size_kb, acc_size_kb, ...]
    Optional bandwidth parameters: [sp_read_bw, sp_write_bw, acc_read_bw, acc_write_bw, dram_bw]
    """
    
    NAME = "gemmini"
    BASE_ARCH_PATH = pathlib.Path(f"{GEMMINI_DIR}/arch/arch.yaml").resolve()
    BASE_MAPSPACE_PATH = pathlib.Path(f"{GEMMINI_DIR}/mapspace/mapspace_real_gemmini.yaml").resolve()
    
    # Hardware bounds and defaults
    BASE_PE_DIM = 16
    BASE_SP_SIZE_KB = 128
    BASE_ACC_SIZE_KB = 32
    
    PE_DIM_BOUNDS = (4, 64)          # Min and max PE dimensions
    SP_SIZE_BOUNDS = (16, 1024)      # Min and max scratchpad size in KB
    ACC_SIZE_BOUNDS = (8, 256)       # Min and max accumulator size in KB
    
    # Memory level mapping for Gemmini
    MEM_LEVELS = {
        "registers": 0,
        "accumulator": 1,
        "scratchpad": 2,
        "dram": 3
    }
    
    def __init__(self, hw_config, output_dir: pathlib.Path):
        """
        Initialize Gemmini configuration.
        
        Args:
            hw_config: Hardware configuration parameters or "random"
            output_dir: Output directory for configuration files
        """
        super().__init__(hw_config, output_dir)
        logger.debug(f"Initialized Gemmini config: {self.get_config_str()}")
    
    def gen_random_hw_config(self) -> List[float]:
        """
        Generate random Gemmini hardware configuration.
        
        Returns:
            List of random hardware parameters [pe_dim, sp_size_kb, acc_size_kb]
        """
        # Generate random PE dimension (power of 2 for better hardware efficiency)
        pe_dim_options = [4, 8, 16, 32]
        pe_dim = random.choice(pe_dim_options)
        
        # Generate random memory sizes within reasonable bounds
        sp_size = random.uniform(*self.SP_SIZE_BOUNDS)
        acc_size = random.uniform(*self.ACC_SIZE_BOUNDS)
        
        # Optional: add bandwidth parameters for more detailed modeling
        if random.random() < 0.3:  # 30% chance of including bandwidth params
            sp_read_bw = random.uniform(8, 64)
            sp_write_bw = random.uniform(8, 64)
            acc_read_bw = random.uniform(8, 64)
            acc_write_bw = random.uniform(8, 64)
            dram_bw = random.uniform(4, 32)
            
            return [pe_dim, sp_size, acc_size, sp_read_bw, sp_write_bw, 
                   acc_read_bw, acc_write_bw, dram_bw]
        else:
            return [pe_dim, sp_size, acc_size]
    
    def _get_arch_specific_constraints(self) -> Optional[Dict[str, str]]:
        """
        Get Gemmini-specific architecture constraints for mapping.
        
        Returns:
            Dictionary with target-specific spatial constraints
        """
        if len(self.hw_config) < 1:
            return None
        
        pe_dim = int(self.hw_config[0])
        
        return {
            "Accumulator": f"R=1 S=1 P=1 Q=1 C<={pe_dim} K=1 N=1",
            "Scratchpad": f"R=1 S=1 P=1 Q=1 N=1 C=1 K<={pe_dim}"
        }
    
    def get_pe_dimensions(self) -> int:
        """Get the PE array dimension (assumes square array)."""
        return int(self.hw_config[0]) if self.hw_config else self.BASE_PE_DIM
    
    def get_memory_sizes(self) -> Dict[str, float]:
        """
        Get memory sizes for different levels.
        
        Returns:
            Dictionary with memory sizes in KB
        """
        if len(self.hw_config) < 3:
            return {
                "scratchpad_kb": self.BASE_SP_SIZE_KB,
                "accumulator_kb": self.BASE_ACC_SIZE_KB
            }
        
        return {
            "scratchpad_kb": float(self.hw_config[1]),
            "accumulator_kb": float(self.hw_config[2])
        }
    
    def get_bandwidth_config(self) -> Dict[str, float]:
        """
        Get bandwidth configuration if available.
        
        Returns:
            Dictionary with bandwidth parameters
        """
        default_bw = {
            "sp_read_bw": 16.0,
            "sp_write_bw": 16.0,
            "acc_read_bw": 16.0,
            "acc_write_bw": 16.0,
            "dram_shared_bw": 8.0
        }
        
        if len(self.hw_config) >= 8:
            return {
                "sp_read_bw": float(self.hw_config[3]),
                "sp_write_bw": float(self.hw_config[4]),
                "acc_read_bw": float(self.hw_config[5]),
                "acc_write_bw": float(self.hw_config[6]),
                "dram_shared_bw": float(self.hw_config[7])
            }
        
        return default_bw
    
    def estimate_area(self) -> float:
        """
        Estimate chip area based on configuration.
        
        Returns:
            Estimated area in mm^2
        """
        pe_dim = self.get_pe_dimensions()
        memory_sizes = self.get_memory_sizes()
        
        # Simple area model (these would be calibrated to real hardware)
        pe_area = (pe_dim ** 2) * 0.01  # mm^2 per PE
        sp_area = memory_sizes["scratchpad_kb"] * 0.05  # mm^2 per KB
        acc_area = memory_sizes["accumulator_kb"] * 0.03  # mm^2 per KB
        overhead = 2.0  # mm^2 for interconnect and control
        
        total_area = pe_area + sp_area + acc_area + overhead
        return round(total_area, 2)
    
    def estimate_power(self) -> float:
        """
        Estimate power consumption based on configuration.
        
        Returns:
            Estimated power in Watts
        """
        pe_dim = self.get_pe_dimensions()
        memory_sizes = self.get_memory_sizes()
        
        # Simple power model
        pe_power = (pe_dim ** 2) * 0.1  # W per PE
        sp_power = memory_sizes["scratchpad_kb"] * 0.002  # W per KB
        acc_power = memory_sizes["accumulator_kb"] * 0.001  # W per KB
        static_power = 0.5  # W for static consumption
        
        total_power = pe_power + sp_power + acc_power + static_power
        return round(total_power, 2)
    
    def get_theoretical_throughput(self, layer_shape: Dict[str, int]) -> float:
        """
        Calculate theoretical peak throughput for a given layer.
        
        Args:
            layer_shape: Dictionary with layer dimensions (C, K, R, S, etc.)
            
        Returns:
            Theoretical throughput in operations per cycle
        """
        pe_dim = self.get_pe_dimensions()
        
        # For convolution: each PE can do one MAC per cycle
        # Theoretical max is pe_dim^2 MACs per cycle
        return float(pe_dim ** 2)
    
    def validate_for_layer(self, layer_shape: Dict[str, int]) -> Dict[str, Any]:
        """
        Validate configuration against a specific layer.
        
        Args:
            layer_shape: Layer dimensions
            
        Returns:
            Validation results with feasibility analysis
        """
        pe_dim = self.get_pe_dimensions()
        memory_sizes = self.get_memory_sizes()
        
        # Calculate data requirements
        input_channels = layer_shape.get('C', 1)
        output_channels = layer_shape.get('K', 1)
        filter_size = layer_shape.get('R', 1) * layer_shape.get('S', 1)
        
        # Estimate memory requirements (simplified)
        weight_kb = (input_channels * output_channels * filter_size * 1) / 1024  # 1 byte per weight
        activation_kb = (input_channels * 32 * 32 * 1) / 1024  # Assume 32x32 activation
        
        validation = {
            "feasible": True,
            "issues": [],
            "utilization": {}
        }
        
        # Check if weights fit in scratchpad
        if weight_kb > memory_sizes["scratchpad_kb"]:
            validation["feasible"] = False
            validation["issues"].append(f"Weights ({weight_kb:.1f} KB) exceed scratchpad ({memory_sizes['scratchpad_kb']} KB)")
        
        # Check PE utilization constraints
        if output_channels < pe_dim:
            validation["issues"].append(f"Output channels ({output_channels}) < PE dim ({pe_dim}), underutilization expected")
        
        # Calculate utilization estimates
        validation["utilization"] = {
            "pe_utilization": min(1.0, output_channels / (pe_dim ** 2)),
            "sp_utilization": weight_kb / memory_sizes["scratchpad_kb"],
            "acc_utilization": activation_kb / memory_sizes["accumulator_kb"]
        }
        
        return validation
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive configuration summary.
        
        Returns:
            Dictionary with all configuration details
        """
        summary = super().get_status_summary()
        
        # Add Gemmini-specific information
        summary.update({
            "pe_dimensions": self.get_pe_dimensions(),
            "memory_sizes": self.get_memory_sizes(),
            "bandwidth_config": self.get_bandwidth_config(),
            "estimated_area_mm2": self.estimate_area(),
            "estimated_power_w": self.estimate_power(),
            "architecture_type": "systolic_array",
            "supports_dataflows": ["weight_stationary", "output_stationary"]
        })
        
        return summary
    
    @classmethod
    def create_baseline_config(cls, output_dir: pathlib.Path) -> 'GemminiConfigRefactored':
        """
        Create a baseline Gemmini configuration for comparison.
        
        Args:
            output_dir: Output directory
            
        Returns:
            GemminiConfigRefactored instance with baseline parameters
        """
        baseline_config = [cls.BASE_PE_DIM, cls.BASE_SP_SIZE_KB, cls.BASE_ACC_SIZE_KB]
        return cls(baseline_config, output_dir)
    
    @classmethod
    def create_optimized_config(cls, target_layer: Dict[str, int], 
                              output_dir: pathlib.Path) -> 'GemminiConfigRefactored':
        """
        Create an optimized configuration for a specific target layer.
        
        Args:
            target_layer: Layer specification
            output_dir: Output directory
            
        Returns:
            GemminiConfigRefactored instance optimized for the target layer
        """
        # Simple heuristic optimization
        output_channels = target_layer.get('K', 64)
        input_channels = target_layer.get('C', 64)
        
        # Choose PE dimension based on output channels
        pe_dim = 16  # Default
        if output_channels >= 64:
            pe_dim = 32
        elif output_channels <= 16:
            pe_dim = 8
        
        # Size scratchpad based on weight requirements
        filter_size = target_layer.get('R', 3) * target_layer.get('S', 3)
        weight_kb = (input_channels * output_channels * filter_size) / 1024
        sp_size = max(128, weight_kb * 1.5)  # 50% overhead
        
        # Size accumulator based on output requirements
        acc_size = max(32, output_channels / 4)  # Heuristic
        
        optimized_config = [pe_dim, sp_size, acc_size]
        return cls(optimized_config, output_dir) 