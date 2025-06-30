"""
Mapping processor component for handling mapping-related operations.
"""

import numpy as np
import pathlib
from typing import List, Dict, Any, Union

from dataset.common import utils, logger


class MappingProcessor:
    """Handles mapping processing and conversion operations."""
    
    def __init__(self, arch_name: str, mapspace_path: pathlib.Path):
        """
        Initialize mapping processor.
        
        Args:
            arch_name: Name of the hardware architecture
            mapspace_path: Path to the mapspace configuration file
        """
        self.arch_name = arch_name
        self.mapspace_path = mapspace_path
        self._mapspace_dict = None
    
    @property
    def mapspace_dict(self) -> Dict[str, Any]:
        """Lazy load mapspace dictionary."""
        if self._mapspace_dict is None:
            self._mapspace_dict = utils.parse_yaml(self.mapspace_path)
        return self._mapspace_dict
    
    def flat_mapping_to_dict(self, prob_type: str, flat_mapping: List[int]) -> Dict[str, Any]:
        """
        Convert flat mapping format to Timeloop mapping dictionary.
        
        Args:
            prob_type: Problem type identifier
            flat_mapping: Flat mapping representation
            
        Returns:
            Dictionary containing mapping in Timeloop format
        """
        try:
            return self._convert_flat_mapping(prob_type, flat_mapping)
        except Exception as e:
            logger.error(f"Failed to convert flat mapping: {e}")
            return {}
    
    def _convert_flat_mapping(self, prob_type: str, flat_mapping: List[int]) -> Dict[str, Any]:
        """Internal method to convert flat mapping to dictionary format."""
        flat_mapping = np.array(flat_mapping)
        
        if "mapspace_constraints" not in self.mapspace_dict:
            logger.error(f"Mapspace file {self.mapspace_path} should contain key 'mapspace_constraints'")
            return {}
        
        constraints_lst = self.mapspace_dict.get("mapspace_constraints")
        mapping_lst = []  # final mapping list (under key "mapping")
        targets = []
        
        # Collect names of memory levels
        for constraint in constraints_lst:
            if constraint.get("type") == "bypass":
                mapping_lst.append(constraint)
                targets.append(constraint["target"])
        
        num_mem_lvls = len(targets)
        dims = utils.get_prob_dims(prob_type)
        num_dims = len(dims)
        
        # Validate mapping length
        expected_length = num_mem_lvls * num_dims * 3
        if len(flat_mapping) != expected_length:
            logger.error(f"Mapping length mismatch: got {len(flat_mapping)}, "
                        f"expected {expected_length} for {num_mem_lvls} levels, "
                        f"{num_dims} dims")
            return {}
        
        # Process each memory level
        for mem_lvl in range(num_mem_lvls):
            start_idx = (num_mem_lvls - 1 - mem_lvl) * num_dims * 3
            end_idx = start_idx + num_dims * 3
            mem_lvl_mapping = flat_mapping[start_idx:end_idx]
            
            target = targets[mem_lvl]
            
            # Process spatial factors
            spatial_factors = mem_lvl_mapping[:num_dims]
            spatial_dict = self._create_spatial_mapping(
                target, dims, spatial_factors, mem_lvl
            )
            if spatial_dict:
                mapping_lst.append(spatial_dict)
            
            # Process temporal factors
            temporal_factors = mem_lvl_mapping[num_dims:num_dims*2]
            perms = mem_lvl_mapping[-num_dims:]
            
            temporal_dict = self._create_temporal_mapping(
                target, dims, temporal_factors, perms
            )
            mapping_lst.append(temporal_dict)
        
        return {"mapping": mapping_lst}
    
    def _create_spatial_mapping(self, target: str, dims: List[str], 
                              spatial_factors: np.ndarray, mem_lvl: int) -> Dict[str, Any]:
        """Create spatial mapping dictionary."""
        # Apply architecture-specific constraints
        if self.arch_name == "gemmini":
            spatial_factors = self._apply_gemmini_spatial_constraints(
                spatial_factors, dims, mem_lvl
            )
        
        spatial_factor_str = " ".join([
            f"{dims[i]}={round(spatial_factors[i])}" 
            for i in range(len(dims))
        ])
        
        # Only add spatial factors if there is a dimension > 1
        spatial_add_cond = sum(spatial_factors) > len(spatial_factors)
        
        # Special cases for certain architectures and memory levels
        if self.arch_name == "gemmini" and (mem_lvl == 1 or mem_lvl == 2):
            spatial_add_cond = True
        
        if spatial_add_cond:
            return {
                "target": target,
                "type": "spatial",
                "factors": spatial_factor_str,
            }
        
        return None
    
    def _apply_gemmini_spatial_constraints(self, spatial_factors: np.ndarray, 
                                         dims: List[str], mem_lvl: int) -> np.ndarray:
        """Apply Gemmini-specific spatial constraints."""
        for i in range(len(dims)):
            # Apply specific constraints for different memory levels and dimensions
            if not ((mem_lvl == 1 and dims[i] == "C") or (mem_lvl == 2 and dims[i] == "K")):
                spatial_factors[i] = 1
        return spatial_factors
    
    def _create_temporal_mapping(self, target: str, dims: List[str], 
                               temporal_factors: np.ndarray, 
                               perms: np.ndarray) -> Dict[str, Any]:
        """Create temporal mapping dictionary."""
        temporal_factor_str = " ".join([
            f"{dims[i]}={round(temporal_factors[i])}" 
            for i in range(len(dims))
        ])
        
        # Process permutation
        perm_str = self._create_permutation_string(dims, perms)
        
        return {
            "target": target,
            "type": "temporal",
            "factors": temporal_factor_str,
            "permutation": perm_str,
        }
    
    def _create_permutation_string(self, dims: List[str], perms: np.ndarray) -> str:
        """Create permutation string from permutation array."""
        num_dims = len(dims)
        perm_str_dict = {}
        
        for i in range(num_dims):
            if perms[i] < num_dims:
                perm_str_dict[perms[i]] = dims[i]
        
        # Sort by permutation order and create string
        sorted_dims = [perm_str_dict[k] for k in sorted(perm_str_dict.keys())]
        return "".join(sorted_dims)
    
    def validate_mapping(self, mapping_dict: Dict[str, Any]) -> bool:
        """
        Validate a mapping dictionary for correctness.
        
        Args:
            mapping_dict: Mapping dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if "mapping" not in mapping_dict:
                logger.error("Mapping dictionary missing 'mapping' key")
                return False
            
            mapping_list = mapping_dict["mapping"]
            if not isinstance(mapping_list, list):
                logger.error("Mapping should be a list")
                return False
            
            # Basic validation of mapping entries
            for entry in mapping_list:
                if not isinstance(entry, dict):
                    logger.error("Mapping entry should be a dictionary")
                    return False
                
                if "target" not in entry:
                    logger.error("Mapping entry missing 'target' field")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating mapping: {e}")
            return False 