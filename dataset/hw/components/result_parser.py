"""
Result parser component for processing Timeloop output results.
"""

import re
import pathlib
from typing import List, Dict, Any, Optional

from dataset.common import utils, logger
from dataset.workloads import Prob


class ResultParser:
    """Handles parsing of Timeloop simulation results."""
    
    # Regular expressions for parsing different output formats
    PATTERNS = {
        'cycles': r"Cycles: (\d+)",
        'energy': r"Energy: ([\d.]+) uJ",
        'area': r"Area: ([\d.]+) mm\^2",
        'utilization': r"Utilization: ([\d.]+)%",
        'mapping_valid': r"Mapping is valid: (\w+)",
        'mapping_efficiency': r"Mapping efficiency: ([\d.]+)%"
    }
    
    def __init__(self):
        """Initialize result parser."""
        self.patterns_compiled = {
            key: re.compile(pattern) 
            for key, pattern in self.PATTERNS.items()
        }
    
    def parse_random_output(self, output_log_file: pathlib.Path, 
                          layer_prob: Prob) -> List[Dict[str, Any]]:
        """
        Parse random mapping results from Timeloop output.
        
        Args:
            output_log_file: Path to the Timeloop output log
            layer_prob: Problem specification
            
        Returns:
            List of result dictionaries, one per valid mapping
        """
        try:
            if not output_log_file.exists():
                logger.error(f"Output log file not found: {output_log_file}")
                return []
            
            with open(output_log_file, 'r') as f:
                content = f.read()
            
            # Split content into individual mapping results
            mapping_sections = self._split_mapping_sections(content)
            
            results = []
            for i, section in enumerate(mapping_sections):
                result = self._parse_single_mapping(section, layer_prob, i)
                if result:
                    results.append(result)
            
            logger.debug(f"Parsed {len(results)} valid mappings from {output_log_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error parsing output file {output_log_file}: {e}")
            return []
    
    def _split_mapping_sections(self, content: str) -> List[str]:
        """Split content into individual mapping result sections."""
        # Look for mapping delimiters in Timeloop output
        section_delimiters = [
            "Summary Stats",
            "=== Mapping ===",
            "Found a valid mapping",
            "Mapping found with fitness:"
        ]
        
        sections = []
        current_section = ""
        
        for line in content.split('\n'):
            # Check if this line starts a new section
            is_new_section = any(delimiter in line for delimiter in section_delimiters)
            
            if is_new_section and current_section.strip():
                # Save previous section and start new one
                sections.append(current_section)
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # Add the last section
        if current_section.strip():
            sections.append(current_section)
        
        return sections
    
    def _parse_single_mapping(self, section: str, layer_prob: Prob, 
                            mapping_index: int) -> Optional[Dict[str, Any]]:
        """Parse a single mapping result section."""
        try:
            result = {
                'mapping_index': mapping_index,
                'layer_config': layer_prob.config_str(),
                'valid': False
            }
            
            # Extract basic metrics
            for metric, pattern in self.patterns_compiled.items():
                match = pattern.search(section)
                if match:
                    value = match.group(1)
                    # Convert to appropriate type
                    if metric in ['cycles']:
                        result[f'target.{metric}'] = int(value)
                    elif metric in ['energy', 'area', 'utilization', 'mapping_efficiency']:
                        result[f'target.{metric}'] = float(value)
                    elif metric == 'mapping_valid':
                        result['valid'] = value.lower() == 'true'
                    else:
                        result[metric] = value
            
            # Only return valid mappings
            if not result.get('valid', False):
                return None
            
            # Extract detailed statistics if available
            self._extract_detailed_stats(section, result)
            
            # Extract mapping information if available
            self._extract_mapping_info(section, result)
            
            return result
            
        except Exception as e:
            logger.debug(f"Error parsing mapping section {mapping_index}: {e}")
            return None
    
    def _extract_detailed_stats(self, section: str, result: Dict[str, Any]) -> None:
        """Extract detailed statistics from the section."""
        # Memory access patterns
        memory_patterns = {
            'reads': r"Reads: (\d+)",
            'writes': r"Writes: (\d+)", 
            'updates': r"Updates: (\d+)",
            'fills': r"Fills: (\d+)",
            'address_generations': r"Address generations: (\d+)"
        }
        
        for stat_name, pattern in memory_patterns.items():
            matches = re.findall(pattern, section)
            if matches:
                # Sum all matches (for different memory levels)
                total = sum(int(match) for match in matches)
                result[f'stats.{stat_name}'] = total
        
        # Buffer utilization
        utilization_pattern = r"(\w+) utilization: ([\d.]+)%"
        utilization_matches = re.findall(utilization_pattern, section)
        for buffer_name, util_value in utilization_matches:
            result[f'utilization.{buffer_name.lower()}'] = float(util_value)
    
    def _extract_mapping_info(self, section: str, result: Dict[str, Any]) -> None:
        """Extract mapping configuration information."""
        # Look for mapping specification
        mapping_pattern = r"=== Mapping ===\s*(.*?)(?:===|$)"
        mapping_match = re.search(mapping_pattern, section, re.DOTALL)
        
        if mapping_match:
            mapping_text = mapping_match.group(1).strip()
            result['mapping.raw'] = mapping_text
            
            # Parse specific mapping details
            self._parse_mapping_details(mapping_text, result)
    
    def _parse_mapping_details(self, mapping_text: str, result: Dict[str, Any]) -> None:
        """Parse detailed mapping information."""
        # Extract tiling factors
        tiling_pattern = r"(\w+):\s*\[([^\]]+)\]"
        tiling_matches = re.findall(tiling_pattern, mapping_text)
        
        for dimension, factors in tiling_matches:
            # Parse factor list
            factor_list = [int(f.strip()) for f in factors.split(',') if f.strip().isdigit()]
            if factor_list:
                result[f'mapping.{dimension.lower()}_factors'] = factor_list
        
        # Extract permutation if available
        perm_pattern = r"permutation:\s*([A-Z]+)"
        perm_match = re.search(perm_pattern, mapping_text)
        if perm_match:
            result['mapping.permutation'] = perm_match.group(1)
    
    def parse_single_run_output(self, output_log_file: pathlib.Path) -> Dict[str, Any]:
        """
        Parse output from a single Timeloop run (not random mapping).
        
        Args:
            output_log_file: Path to the output log file
            
        Returns:
            Dictionary with parsed results
        """
        try:
            if not output_log_file.exists():
                logger.error(f"Output log file not found: {output_log_file}")
                return {}
            
            with open(output_log_file, 'r') as f:
                content = f.read()
            
            result = {'valid': False}
            
            # Extract basic metrics
            for metric, pattern in self.patterns_compiled.items():
                match = pattern.search(content)
                if match:
                    value = match.group(1)
                    if metric in ['cycles']:
                        result[f'target.{metric}'] = int(value)
                    elif metric in ['energy', 'area', 'utilization', 'mapping_efficiency']:
                        result[f'target.{metric}'] = float(value)
                    elif metric == 'mapping_valid':
                        result['valid'] = value.lower() == 'true'
                    else:
                        result[metric] = value
            
            # Extract detailed stats and mapping info
            self._extract_detailed_stats(content, result)
            self._extract_mapping_info(content, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing single run output {output_log_file}: {e}")
            return {}
    
    def validate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and filter parsed results.
        
        Args:
            results: List of parsed result dictionaries
            
        Returns:
            List of validated results
        """
        validated = []
        
        for result in results:
            if self._is_valid_result(result):
                validated.append(result)
            else:
                logger.debug(f"Filtered out invalid result: {result.get('mapping_index', 'unknown')}")
        
        logger.debug(f"Validated {len(validated)} out of {len(results)} results")
        return validated
    
    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Check if a result is valid."""
        # Must be marked as valid
        if not result.get('valid', False):
            return False
        
        # Must have basic performance metrics
        required_metrics = ['target.cycles', 'target.energy']
        for metric in required_metrics:
            if metric not in result:
                return False
            
            # Check for reasonable values
            value = result[metric]
            if not isinstance(value, (int, float)) or value <= 0:
                return False
        
        return True
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple results into summary statistics.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with aggregated statistics
        """
        if not results:
            return {}
        
        # Collect metrics for aggregation
        metrics_to_aggregate = ['target.cycles', 'target.energy']
        if 'target.area' in results[0]:
            metrics_to_aggregate.append('target.area')
        
        aggregated = {
            'num_results': len(results),
            'all_valid': all(r.get('valid', False) for r in results)
        }
        
        for metric in metrics_to_aggregate:
            values = [r[metric] for r in results if metric in r]
            if values:
                aggregated[f'{metric}_min'] = min(values)
                aggregated[f'{metric}_max'] = max(values)
                aggregated[f'{metric}_mean'] = sum(values) / len(values)
                aggregated[f'{metric}_count'] = len(values)
        
        return aggregated 