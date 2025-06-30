"""
Architecture generator component for creating and modifying hardware configurations.
"""

import copy
import math
import pathlib
from typing import Dict, Any, List, Optional, Union

from dataset.common import utils, logger


class ArchitectureGenerator:
    """Handles generation and modification of hardware architecture configurations."""
    
    def __init__(self, base_arch_path: pathlib.Path, arch_name: str):
        """
        Initialize architecture generator.
        
        Args:
            base_arch_path: Path to base architecture template
            arch_name: Name of the architecture (e.g., 'gemmini', 'simba')
        """
        self.base_arch_path = base_arch_path
        self.arch_name = arch_name.lower()
        self._base_arch = None
    
    @property
    def base_arch(self) -> Dict[str, Any]:
        """Lazy load base architecture configuration."""
        if self._base_arch is None:
            self._base_arch = utils.parse_yaml(self.base_arch_path)
        return self._base_arch
    
    def generate_arch_config(self, hw_config: Union[List, Dict], 
                           output_path: pathlib.Path) -> Optional[Dict[str, Any]]:
        """
        Generate architecture configuration from hardware parameters.
        
        Args:
            hw_config: Hardware configuration parameters
            output_path: Path to save the generated architecture
            
        Returns:
            Dictionary with architecture parameters, or None if invalid
        """
        try:
            if self.arch_name == "gemmini":
                return self._generate_gemmini_config(hw_config, output_path)
            elif self.arch_name == "simba":
                return self._generate_simba_config(hw_config, output_path)
            else:
                logger.error(f"Unsupported architecture: {self.arch_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate architecture config: {e}")
            return None
    
    def _generate_gemmini_config(self, hw_config: List, 
                                output_path: pathlib.Path) -> Optional[Dict[str, Any]]:
        """Generate Gemmini-specific architecture configuration."""
        if len(hw_config) < 3:
            logger.error(f"Gemmini config requires at least 3 parameters, got {len(hw_config)}")
            return None
        
        # Extract parameters
        pe_dim = int(hw_config[0])  # Number of PEs in each dimension
        sp_size = hw_config[1] * 1024  # Scratchpad size in bytes
        acc_size = hw_config[2] * 1024  # Accumulator size in bytes
        
        # Optional bandwidth parameters
        bandwidth_params = {}
        if len(hw_config) >= 8:
            bandwidth_params = {
                'sp_read_bw': round(hw_config[3], 3),
                'sp_write_bw': round(hw_config[4], 3),
                'acc_read_bw': round(hw_config[5], 3),
                'acc_write_bw': round(hw_config[6], 3),
                'dram_shared_bw': round(hw_config[7], 3)
            }
        else:
            bandwidth_params['dram_shared_bw'] = 8  # Default DRAM bandwidth
        
        return self._build_gemmini_arch(
            pe_dim, sp_size, acc_size, bandwidth_params, output_path
        )
    
    def _build_gemmini_arch(self, pe_dim: int, sp_size: int, acc_size: int,
                           bandwidth_params: Dict[str, float],
                           output_path: pathlib.Path) -> Optional[Dict[str, Any]]:
        """Build complete Gemmini architecture configuration."""
        new_arch = copy.deepcopy(self.base_arch)
        arch_dict = {}
        
        # Set basic parameters
        arch_dict["pe"] = pe_dim * pe_dim
        arch_dict["mac"] = 1
        arch_dict["meshX"] = pe_dim
        
        # Navigate to chip dictionary
        chip_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]
        
        # Configure DRAM bandwidth
        dram_attrs = new_arch["architecture"]["subtree"][0]["local"][0]["attributes"]
        dram_attrs["shared_bandwidth"] = bandwidth_params['dram_shared_bw']
        
        # Configure PE structure
        self._configure_gemmini_pe_structure(chip_dict, pe_dim)
        
        # Configure memory levels
        memory_configs = self._calculate_gemmini_memory_configs(
            pe_dim, sp_size, acc_size, bandwidth_params
        )
        
        # Apply memory configurations
        invalid = self._apply_gemmini_memory_configs(
            chip_dict, memory_configs, arch_dict
        )
        
        if invalid:
            logger.error(f"Invalid Gemmini architecture configuration")
            return None
        
        # Save architecture
        utils.store_yaml(output_path, new_arch)
        logger.debug(f"Generated Gemmini architecture: {output_path}")
        
        return arch_dict
    
    def _configure_gemmini_pe_structure(self, chip_dict: Dict[str, Any], pe_dim: int) -> None:
        """Configure PE array structure for Gemmini."""
        # Configure PE columns
        chip_dict["subtree"][0]["name"] = f"PECols[0..{pe_dim-1}]"
        
        # Configure PE rows
        pe_dict = chip_dict["subtree"][0]["subtree"][0]
        pe_dict["name"] = f"PERows[0..{pe_dim-1}]"
        
        # Configure registers
        new_reg = pe_dict["local"][0]["attributes"]
        new_reg["instances"] = pe_dim * pe_dim
    
    def _calculate_gemmini_memory_configs(self, pe_dim: int, sp_size: int, 
                                        acc_size: int, 
                                        bandwidth_params: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate memory level configurations for Gemmini."""
        # Scratchpad configuration
        sp_word_bits = 8
        sp_width = pe_dim * sp_word_bits
        sp_depth = math.ceil(sp_size / (sp_width // 8))
        
        scratchpad_config = {
            "depth": sp_depth,
            "width": sp_width,
            "word-bits": sp_word_bits,
            "blocksize": pe_dim,
            "instances": 1,
            "shared_bandwidth": pe_dim * 2,
        }
        
        # Accumulator configuration
        acc_word_bits = 32
        acc_width = pe_dim * acc_word_bits
        acc_depth = math.ceil(acc_size / (acc_width // 8))
        
        accumulator_config = {
            "depth": acc_depth,
            "width": acc_width // pe_dim,
            "word-bits": acc_word_bits,
            "blocksize": 1,
            "instances": pe_dim,
            "meshX": pe_dim,
            "shared_bandwidth": 2,
        }
        
        return [scratchpad_config, accumulator_config]
    
    def _apply_gemmini_memory_configs(self, chip_dict: Dict[str, Any],
                                    memory_configs: List[Dict[str, Any]],
                                    arch_dict: Dict[str, Any]) -> bool:
        """Apply memory configurations to Gemmini architecture."""
        # Memory level mapping: 1=Accumulator, 2=Scratchpad
        mem_lvl_to_attrs = {
            1: chip_dict["subtree"][0]["local"][0]["attributes"],
            2: chip_dict["local"][0]["attributes"],
        }
        
        arch_invalid = False
        
        for mem_lvl in range(1, 3):
            # Index mapping: 0=Scratchpad, 1=Accumulator
            config_idx = 2 - mem_lvl
            config = memory_configs[config_idx]
            attrs = mem_lvl_to_attrs[mem_lvl]
            
            # Apply configuration
            attrs["entries"] = config["depth"] * config["blocksize"]
            attrs["depth"] = config["depth"]
            attrs["width"] = config["blocksize"] * config["word-bits"]
            attrs["instances"] = config["instances"]
            attrs["shared_bandwidth"] = config["shared_bandwidth"]
            
            # Handle meshX constraints
            if "meshX" in attrs and "meshX" in config:
                attrs["meshX"] = config["meshX"]
                if attrs["instances"] % attrs["meshX"] != 0:
                    logger.warning(f"Invalid meshX configuration: instances={attrs['instances']}, meshX={attrs['meshX']}")
                    arch_invalid = True
            
            # Store in arch_dict for reference
            for key in attrs:
                arch_dict[f"mem{mem_lvl}_{key}"] = attrs[key]
        
        return arch_invalid
    
    def _generate_simba_config(self, hw_config: List, 
                              output_path: pathlib.Path) -> Optional[Dict[str, Any]]:
        """Generate Simba-specific architecture configuration."""
        # Placeholder for Simba implementation
        logger.warning("Simba architecture generation not yet implemented")
        return None
    
    def parse_existing_config(self, arch_path: pathlib.Path) -> Optional[List]:
        """
        Parse hardware configuration from existing architecture file.
        
        Args:
            arch_path: Path to architecture configuration file
            
        Returns:
            List of hardware configuration parameters
        """
        try:
            if self.arch_name == "gemmini":
                return self._parse_gemmini_config(arch_path)
            elif self.arch_name == "simba":
                return self._parse_simba_config(arch_path)
            else:
                logger.error(f"Unsupported architecture for parsing: {self.arch_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse architecture config: {e}")
            return None
    
    def _parse_gemmini_config(self, arch_path: pathlib.Path) -> Optional[List]:
        """Parse Gemmini architecture configuration."""
        arch_config = utils.parse_yaml(arch_path)
        
        # Navigate to PE configuration
        chip_dict = arch_config["architecture"]["subtree"][0]["subtree"][0]
        
        # Extract PE dimensions
        pe_width_str = chip_dict["subtree"][0]["name"]
        pe_height_str = chip_dict["subtree"][0]["subtree"][0]["name"]
        
        import re
        
        # Parse PE width
        m = re.search(r"PECols\[0..(\d+)\]", pe_width_str)
        if not m:
            logger.error("Couldn't parse PE width from architecture")
            return None
        pe_width = int(m.group(1)) + 1
        
        # Parse PE height
        m = re.search(r"PERows\[0..(\d+)\]", pe_height_str)
        if not m:
            logger.error("Couldn't parse PE height from architecture")
            return None
        pe_height = int(m.group(1)) + 1
        
        num_pe = pe_width * pe_height
        hw_config = [int(num_pe ** 0.5)]
        
        # Extract scratchpad size
        sp_attr = chip_dict["local"][0]["attributes"]
        sp_depth = int(sp_attr["depth"])
        sp_width = int(sp_attr["width"])
        sp_size_kb = (sp_depth * sp_width / 8) / 1024
        hw_config.append(sp_size_kb)
        
        # Extract accumulator size
        acc_attr = chip_dict["subtree"][0]["local"][0]["attributes"]
        acc_depth = int(acc_attr["depth"])
        acc_width = int(acc_attr["width"]) * pe_width
        acc_size_kb = (acc_depth * acc_width / 8) / 1024
        hw_config.append(acc_size_kb)
        
        return hw_config
    
    def _parse_simba_config(self, arch_path: pathlib.Path) -> Optional[List]:
        """Parse Simba architecture configuration."""
        # Placeholder for Simba implementation
        logger.warning("Simba architecture parsing not yet implemented")
        return None
    
    def validate_config(self, hw_config: List) -> bool:
        """
        Validate hardware configuration parameters.
        
        Args:
            hw_config: Hardware configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not hw_config:
            return False
        
        if self.arch_name == "gemmini":
            return self._validate_gemmini_config(hw_config)
        elif self.arch_name == "simba":
            return self._validate_simba_config(hw_config)
        else:
            logger.error(f"Unknown architecture for validation: {self.arch_name}")
            return False
    
    def _validate_gemmini_config(self, hw_config: List) -> bool:
        """Validate Gemmini configuration parameters."""
        if len(hw_config) < 3:
            logger.error("Gemmini config requires at least 3 parameters")
            return False
        
        pe_dim = hw_config[0]
        sp_size = hw_config[1]
        acc_size = hw_config[2]
        
        # Validate PE dimension
        if not isinstance(pe_dim, (int, float)) or pe_dim <= 0:
            logger.error(f"Invalid PE dimension: {pe_dim}")
            return False
        
        # Validate memory sizes
        if not isinstance(sp_size, (int, float)) or sp_size <= 0:
            logger.error(f"Invalid scratchpad size: {sp_size}")
            return False
        
        if not isinstance(acc_size, (int, float)) or acc_size <= 0:
            logger.error(f"Invalid accumulator size: {acc_size}")
            return False
        
        # Optional bandwidth validation
        if len(hw_config) >= 8:
            for i, bw in enumerate(hw_config[3:8]):
                if not isinstance(bw, (int, float)) or bw <= 0:
                    logger.error(f"Invalid bandwidth parameter {i+3}: {bw}")
                    return False
        
        return True
    
    def _validate_simba_config(self, hw_config: List) -> bool:
        """Validate Simba configuration parameters."""
        # Placeholder for Simba validation
        logger.warning("Simba configuration validation not yet implemented")
        return True 