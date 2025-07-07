import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from functools import reduce
from operator import mul

def sanitize_layer_name(layer_name: str) -> str:
    """
    Sanitize layer names to be compatible with PyTorch ModuleDict.
    Replace problematic characters with underscores.
    
    Args:
        layer_name: Original layer name from ONNX
        
    Returns:
        Sanitized layer name safe for PyTorch modules
    """
    # Replace dots, slashes, and other problematic characters with underscores
    sanitized = layer_name.replace('.', '_').replace('/', '_').replace('-', '_')
    # Remove leading/trailing underscores and collapse multiple underscores
    sanitized = '_'.join(filter(None, sanitized.split('_')))
    return sanitized

class Config:
    """
    Global configuration singleton loaded from YAML.
    """
    _instance = None
    
    def __init__(self):
        if Config._instance is not None:
            raise RuntimeError("Config is a singleton - use Config.get_instance()")
        
        with open('config.yaml', 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Extract commonly used values
        self.PROBLEM_DIMS = self._config['dimensions']['loop_dims']
        self.NUM_MEMORY_LEVELS = self._config['hardware']['num_memory_levels']
        self.BYTES_PER_ELEMENT = self._config['hardware']['bytes_per_element']
        
        # Cost model parameters
        self.DRAM_ACCESS_COST = self._config['costs']['dram_access_cost_pj_byte']
        self.SRAM_BASE_COST = self._config['costs']['sram_base_cost_pj_byte']
        self.FUSION_OVERHEAD_COST = self._config['costs']['fusion_overhead_cost_pj']
        self.AREA_COEFFICIENT = self._config['costs']['area_coefficient_mm2_per_kb']
        self.SRAM_ENERGY_SCALING = self._config['costs']['sram_energy_scaling_pj_byte_kb']
        
        # Hardware parameters
        self.DRAM_BANDWIDTH = self._config['hardware']['dram_bandwidth_gb_s']
        self.SRAM_BANDWIDTH = self._config['hardware']['sram_bandwidth_gb_s']
        self.MAC_THROUGHPUT = self._config['hardware']['mac_throughput_gops']
        
        # Loss weights
        self.PENALTY_WEIGHT = self._config['weights']['penalty_weight']
        self.PRODUCT_PENALTY_WEIGHT = self._config['weights']['product_penalty_weight']
    
    @staticmethod
    def get_instance():
        """Get the singleton instance of Config."""
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance

def find_divisors(n: int) -> List[int]:
    """
    Find all positive divisors of n.
    
    Args:
        n: Integer to find divisors for
        
    Returns:
        Sorted list of all positive divisors
    """
    divisors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:  # Avoid duplicates for perfect squares
                divisors.append(n // i)
    return sorted(divisors)

def find_nearest_divisor(n: int, target: float) -> int:
    """
    Find the divisor of n that is closest to target.
    
    Args:
        n: Integer to find divisors for
        target: Target value to approximate
        
    Returns:
        Divisor of n closest to target
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    # Handle edge cases
    if target <= 0:
        return 1
    if target >= n:
        return n
        
    divisors = find_divisors(n)
    return min(divisors, key=lambda x: abs(x - target))

def calculate_penalty_loss(mapping_params: 'MappingParameters') -> torch.Tensor:
    """
    Calculate penalty term for invalid mapping parameters.
    Penalizes tiling factors less than 1.0 as they are invalid in practice.
    
    Args:
        mapping_params: MappingParameters object containing all layer mappings
        
    Returns:
        Total penalty summed across all invalid tiling factors
    """
    total_penalty = torch.tensor(0.0)
    
    # Iterate through all layers and their tiling factors
    for sanitized_name, layer_mapping in mapping_params.mappings.items():
        for factor in layer_mapping.temporal_factors_L2.values():
            # Penalty is max(0, 1 - factor) to penalize factors < 1
            penalty = torch.maximum(1 - factor, torch.tensor(0.0))
            total_penalty = total_penalty + penalty
    
    return total_penalty

def round_tiling_factors(factor_tensor: torch.Tensor, problem_size: int) -> torch.Tensor:
    """
    Round continuous tiling factors to nearest valid divisors.
    This is a non-differentiable operation used only during evaluation.
    
    Args:
        factor_tensor: Continuous tiling factors from optimizer
        problem_size: Problem dimension size to find divisors for
        
    Returns:
        Rounded factors that are valid divisors
    """
    # Convert to numpy for integer operations
    continuous_factor = factor_tensor.detach().cpu().numpy()
    rounded_factor = find_nearest_divisor(problem_size, continuous_factor)
    return torch.tensor(rounded_factor, dtype=torch.float32)

def calculate_macs(dims: Dict[str, int]) -> torch.Tensor:
    """
    Calculate total number of multiply-accumulate operations.
    Handles both convolution and element-wise operations.
    
    Args:
        dims: Dictionary of problem dimensions
        
    Returns:
        Total number of MAC operations
    """
    # For convolution layers (with R, S dimensions)
    if 'R' in dims and 'S' in dims:
        return torch.tensor(
            dims['N'] * dims['K'] * dims['P'] * dims['Q'] * dims['C'] * dims['R'] * dims['S'],
            dtype=torch.float32
        )
    # For element-wise operations (like ReLU)
    else:
        return torch.tensor(
            dims['N'] * dims['K'] * dims['P'] * dims['Q'],
            dtype=torch.float32
        )

class LayerMapping(nn.Module):
    """
    Mapping parameters for a single layer.
    Currently implements single-level tiling with L2 buffer.
    """
    def __init__(self, dims: Dict[str, int]):
        super().__init__()
        
        # Initialize tiling factors for each dimension
        self.temporal_factors_L2 = nn.ParameterDict({
            dim: nn.Parameter(torch.rand(1))
            for dim in Config.get_instance().PROBLEM_DIMS
            if dim in dims
        })
        
        # Store problem dimensions
        self.dims = dims
    
    def get_rounded_factors(self, dims: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Get rounded tiling factors that are valid divisors.
        
        Args:
            dims: Dictionary of problem dimensions
            
        Returns:
            Dictionary of rounded tiling factors
        """
        return {
            dim: round_tiling_factors(self.temporal_factors_L2[dim], dims[dim])
            for dim in self.temporal_factors_L2.keys()
            if dim in dims
        }
    
    def calculate_buffer_requirements(self, dims: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Calculate buffer size requirements for each dimension.
        
        Args:
            dims: Dictionary of problem dimensions
            
        Returns:
            Dictionary of buffer sizes per dimension
        """
        config = Config.get_instance()
        rounded_factors = self.get_rounded_factors(dims)
        
        # Calculate buffer requirements for each dimension
        buffer_reqs = {}
        for dim in rounded_factors:
            # Simple model: buffer size is proportional to tiling factor
            buffer_reqs[dim] = rounded_factors[dim] * config.BYTES_PER_ELEMENT
        
        return buffer_reqs
    
    def calculate_access_counts(self, dims: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Calculate number of memory accesses for each dimension.
        
        Args:
            dims: Dictionary of problem dimensions
            
        Returns:
            Dictionary of access counts per dimension
        """
        rounded_factors = self.get_rounded_factors(dims)
        
        # Calculate access counts for each dimension
        access_counts = {}
        for dim in rounded_factors:
            # Simple model: access count is problem size divided by tiling factor
            access_counts[dim] = torch.tensor(dims[dim], dtype=torch.float32) / rounded_factors[dim]
        
        return access_counts

class ComputationGraph:
    """
    Represents the computation graph of the neural network.
    Tracks layer dimensions and fusion opportunities.
    """
    def __init__(self, layers: Dict[str, Dict] = None, edges: List[Tuple[str, str]] = None, fusion_groups: List[List[str]] = None):
        """
        Initialize computation graph.
        
        Args:
            layers: Optional dictionary of layer configurations
            edges: Optional list of edges between layers
            fusion_groups: Optional list of layer groups that can be fused
        """
        self.layers = layers or {}
        self.edges = edges or []
        self.fusion_groups = fusion_groups or []
    
    def add_layer(self, name: str, dims: Dict[str, int], op_type: str) -> None:
        """
        Add a layer to the graph.
        
        Args:
            name: Layer name
            dims: Dictionary of dimension sizes
            op_type: The operation type of the layer (e.g., 'Conv', 'ReLU')
        """
        self.layers[name] = {'dims': dims, 'type': op_type}
    
    def add_edge(self, src: str, dst: str) -> None:
        """
        Add an edge between layers.
        
        Args:
            src: Source layer name
            dst: Destination layer name
        """
        self.edges.append((src, dst))
    
    def add_fusion_group(self, group: List[str]) -> None:
        """
        Add a group of layers that can be fused.
        
        Args:
            group: List of layer names that can be fused together
        """
        self.fusion_groups.append(group)
    
    def get_layer_names(self) -> List[str]:
        """Get list of all layer names."""
        return list(self.layers.keys())
    
    def get_fusion_group_key(self, group: List[str]) -> str:
        """
        Generate a unique key for a fusion group.
        
        Args:
            group: List of layer names in the fusion group
            
        Returns:
            String key created by joining sorted layer names
        """
        return '__'.join(sorted(group))
    
    def get_node_producers(self, node_name: str) -> List[str]:
        """
        Get the names of all parent (producer) nodes for a given node.
        
        Args:
            node_name: Name of the target node
            
        Returns:
            List of producer node names
        """
        producers = []
        for src, dest in self.edges:
            if dest == node_name:
                producers.append(src)
        return producers
    
    def get_node_consumers(self, node_name: str) -> List[str]:
        """
        Get the names of all child (consumer) nodes for a given node.
        
        Args:
            node_name: Name of the target node
            
        Returns:
            List of consumer node names
        """
        consumers = []
        for src, dest in self.edges:
            if src == node_name:
                consumers.append(dest)
        return consumers
    
    def validate(self) -> None:
        """
        Validate graph structure.
        Ensures all referenced layers exist and fusion groups are valid.
        """
        layer_names = set(self.layers.keys())
        
        # Validate edges
        for src, dst in self.edges:
            if src not in layer_names:
                raise ValueError(f"Edge source layer '{src}' not found")
            if dst not in layer_names:
                raise ValueError(f"Edge destination layer '{dst}' not found")
        
        # Validate fusion groups
        for group in self.fusion_groups:
            for layer in group:
                if layer not in layer_names:
                    raise ValueError(f"Fusion group layer '{layer}' not found")
            
            # Check that fusion group layers are connected
            for i in range(len(group) - 1):
                if (group[i], group[i + 1]) not in self.edges:
                    raise ValueError(
                        f"Fusion group layers '{group[i]}' and '{group[i + 1]}' "
                        "must be connected by an edge"
                    )

class MappingParameters(nn.Module):
    """
    Learnable parameters for layer mappings.
    Currently implements single-level tiling with L2 buffer.
    """
    def __init__(self, graph: ComputationGraph):
        super().__init__()
        
        # Create mapping parameters for each layer
        self.mappings = nn.ModuleDict()
        self.layer_name_mapping = {}  # Map sanitized names back to original names
        
        for layer_name in graph.get_layer_names():
            sanitized_name = sanitize_layer_name(layer_name)
            self.mappings[sanitized_name] = LayerMapping(graph.layers[layer_name]['dims'])
            self.layer_name_mapping[sanitized_name] = layer_name

    def get_mapping_by_original_name(self, original_name: str) -> 'LayerMapping':
        """Get layer mapping using the original layer name."""
        sanitized_name = sanitize_layer_name(original_name)
        return self.mappings[sanitized_name]

    def __str__(self):
        lines = ["Mapping Parameters:"]
        for sanitized_name, mapping in self.mappings.items():
            original_name = self.layer_name_mapping[sanitized_name]
            lines.append(f"  {original_name}:")
            for dim, factor in mapping.temporal_factors_L2.items():
                lines.append(f"    {dim}: {factor.item():.4f}")
        return "\n".join(lines)

def calculate_hardware_config(
    mapping_params: 'MappingParameters',
    graph: 'ComputationGraph'
) -> Dict[str, torch.Tensor]:
    """
    Calculate minimal hardware configuration based on mapping parameters.
    Implements min_hw logic from DOSA paper [3712, 3863].
    
    Args:
        mapping_params: MappingParameters object containing layer mappings
        graph: ComputationGraph object defining the network structure
    
    Returns:
        Dictionary containing hardware configuration parameters:
        - buffer_size: Maximum required buffer size across all layers
        - area_cost: Total hardware area cost
    """
    config = Config.get_instance()
    
    # Calculate buffer requirements for each layer
    max_buffer_size = torch.tensor(0.0)
    
    for layer_name, layer_info in graph.layers.items():
        layer_mapping = mapping_params.get_mapping_by_original_name(layer_name)
        buffer_reqs = layer_mapping.calculate_buffer_requirements(layer_info['dims'])
        
        # Total buffer requirement for this layer
        layer_buffer_size = sum(buffer_reqs.values())
        max_buffer_size = torch.maximum(max_buffer_size, layer_buffer_size)
    
    # Calculate area cost based on maximum buffer size
    area_cost = config.AREA_COEFFICIENT * max_buffer_size
    
    return {
        'buffer_size': max_buffer_size,
        'area_cost': area_cost
    }

def calculate_fused_group_buffer_req(
    group: List[str],
    mapping_params: 'MappingParameters',
    graph: 'ComputationGraph'
) -> torch.Tensor:
    """
    Calculates the total buffer requirement for a (potentially) fused group.
    Uses a simple sum of individual layer requirements.
    
    Args:
        group: A list of layer names in the fusion group.
        mapping_params: The mapping parameters for all layers.
        graph: The computation graph.
        
    Returns:
        The total buffer size required for the group.
    """
    total_buffer_req = torch.tensor(0.0)
    for layer_name in group:
        layer_mapping = mapping_params.get_mapping_by_original_name(layer_name)
        layer_dims = graph.layers[layer_name]['dims']
        buffer_reqs = layer_mapping.calculate_buffer_requirements(layer_dims)
        total_buffer_req += sum(buffer_reqs.values())
    return total_buffer_req

class FusionParameters(nn.Module):
    """
    Learnable parameters for fusion decisions.
    Each potential fusion group has a learnable probability.
    """
    def __init__(self, graph: ComputationGraph):
        super().__init__()
        
        # Create fusion probability parameters for each potential fusion group
        self.fusion_probs = nn.ParameterDict()
        self.group_name_mapping = {}  # Map sanitized group names to original groups
        
        for group in graph.fusion_groups:
            # Create a sanitized group key
            group_key = graph.get_fusion_group_key(group)
            sanitized_key = sanitize_layer_name(group_key)
            
            self.fusion_probs[sanitized_key] = nn.Parameter(torch.rand(1))
            self.group_name_mapping[sanitized_key] = group
    
    def get_fusion_probability(self, group: List[str]) -> torch.Tensor:
        """
        Get the fusion probability for a given group.
        
        Args:
            group: List of layer names in the fusion group
            
        Returns:
            Fusion probability (sigmoid-activated)
        """
        # Create group key and sanitize it
        group_key = '__'.join(sorted(group))
        sanitized_key = sanitize_layer_name(group_key)
        
        if sanitized_key in self.fusion_probs:
            return torch.sigmoid(self.fusion_probs[sanitized_key])
        else:
            # If group not found, return 0 probability
            return torch.tensor(0.0)
    
    def __str__(self):
        lines = ["Fusion Parameters:"]
        for sanitized_key, prob in self.fusion_probs.items():
            original_group = self.group_name_mapping[sanitized_key]
            group_str = ' -> '.join(original_group)
            lines.append(f"  {group_str}: {torch.sigmoid(prob).item():.4f}")
        return "\n".join(lines)

class ConditionalPerformanceModel(nn.Module):
    """
    Models both performance and hardware costs of the neural network based on 
    mapping and fusion decisions. Implements analytical cost models from DOSA paper.
    
    Important Assumption:
    The fusion_groups defined in the graph are assumed to be non-overlapping.
    This means no layer can appear in more than one fusion group.
    """
    def __init__(self):
        super().__init__()
    
    def _calculate_analytical_costs(
        self,
        layer_name: str,
        mapping_params: MappingParameters,
        graph: ComputationGraph,
        hw_config: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate analytical latency and energy costs for a layer.
        Implements formulas from DOSA paper [3809, 3858].
        
        Args:
            layer_name: Name of the layer to analyze
            mapping_params: MappingParameters object
            graph: ComputationGraph object
            hw_config: Dictionary containing hardware configuration
        
        Returns:
            latency: Computed latency based on roofline model
            compute_energy: Energy for computation (MAC operations)
            sram_energy: Energy for SRAM buffer accesses
            dram_energy: Energy for DRAM accesses
        """
        config = Config.get_instance()
        layer_info = graph.layers[layer_name]
        layer_mapping = mapping_params.get_mapping_by_original_name(layer_name)
        
        # Calculate total MACs
        total_macs = calculate_macs(layer_info['dims'])
        
        # Calculate memory access counts
        access_counts = layer_mapping.calculate_access_counts(layer_info['dims'])
        
        # Calculate latencies using roofline model
        compute_latency = total_macs / torch.tensor(config.MAC_THROUGHPUT, dtype=torch.float32)
        memory_latency = sum(access_counts.values()) / torch.tensor(config.DRAM_BANDWIDTH, dtype=torch.float32)
        
        # Final latency is max of compute and memory bound
        latency = torch.maximum(compute_latency, memory_latency)
        
        # Calculate compute energy (simple function of total MACs)
        compute_energy = total_macs * torch.tensor(0.1, dtype=torch.float32)  # 0.1 pJ per MAC operation
        
        # Calculate dynamic SRAM energy based on buffer size
        sram_energy_per_access = (
            torch.tensor(config.SRAM_BASE_COST, dtype=torch.float32) +
            hw_config['buffer_size'] * torch.tensor(config.SRAM_ENERGY_SCALING, dtype=torch.float32)
        )
        
        # Calculate SRAM energy based on buffer requirements
        sram_energy = sum(layer_mapping.calculate_buffer_requirements(layer_info['dims']).values()) * sram_energy_per_access
        
        # Calculate DRAM energy based on access counts
        dram_energy = sum(access_counts.values()) * torch.tensor(config.DRAM_ACCESS_COST, dtype=torch.float32)
        
        return latency, compute_energy, sram_energy, dram_energy

    def forward(
        self,
        mapping_params: MappingParameters,
        fusion_params: FusionParameters,
        graph: ComputationGraph
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate latency, energy, and area costs using analytical models.
        Uses differentiable weighted-average for fusion decisions with physically accurate
        DRAM energy savings modeling.
        
        Args:
            mapping_params: MappingParameters object containing layer mappings
            fusion_params: FusionParameters object containing fusion decisions
            graph: ComputationGraph object defining the network structure
        
        Returns:
            total_latency: Total execution latency across all layers
            total_energy: Total energy consumption across all layers
            area_cost: Hardware cost based on buffer requirements
        """
        config = Config.get_instance()
        
        # Calculate minimal hardware configuration
        hw_config = calculate_hardware_config(mapping_params, graph)
        
        # Initialize total costs
        total_latency = torch.zeros(1, requires_grad=True)
        total_energy = torch.zeros(1, requires_grad=True)
        
        # Create set of layers that are part of any fusion group
        layers_in_fusion_groups = {
            layer 
            for group in graph.fusion_groups 
            for layer in group
        }
        
        # Handle standalone layers (not part of any fusion group)
        for layer_name in graph.get_layer_names():
            if layer_name not in layers_in_fusion_groups:
                # Calculate analytical costs for standalone layer
                latency, compute_energy, sram_energy, dram_energy = self._calculate_analytical_costs(
                    layer_name, mapping_params, graph, hw_config
                )
                total_latency = total_latency + latency
                total_energy = total_energy + compute_energy + sram_energy + dram_energy
        
        # Handle fusion groups using differentiable weighted-average with EDP calculation
        for group in graph.fusion_groups:
            # Get fusion probability for this group
            p_fuse = fusion_params.get_fusion_probability(group)
            
            # === Calculate Non-Fused Costs (cost_no_fuse) ===
            non_fused_latency = torch.tensor(0.0)
            non_fused_energy = torch.tensor(0.0)
            
            for layer_name in group:
                lat, compute_energy, sram_energy, dram_energy = self._calculate_analytical_costs(
                    layer_name, mapping_params, graph, hw_config
                )
                non_fused_latency = non_fused_latency + lat
                non_fused_energy = non_fused_energy + compute_energy + sram_energy + dram_energy
            
            # Energy-Delay Product for non-fused case
            cost_no_fuse = non_fused_latency * non_fused_energy
            
            # === Calculate Fused Costs (cost_fused) ===
            fused_latency = torch.tensor(0.0)
            fused_compute_energy = torch.tensor(0.0)
            fused_sram_energy = torch.tensor(0.0)
            fused_dram_energy = torch.tensor(0.0)
            
            for layer_name in group:
                lat, compute_energy, sram_energy, dram_energy = self._calculate_analytical_costs(
                    layer_name, mapping_params, graph, hw_config
                )
                # Latency is dominated by the slowest path (parallel execution)
                fused_latency = torch.maximum(fused_latency, lat)
                fused_compute_energy = fused_compute_energy + compute_energy
                fused_sram_energy = fused_sram_energy + sram_energy
                fused_dram_energy = fused_dram_energy + dram_energy
            
            # === MODEL THE DRAM SAVING ===
            # For fusion groups, intermediate tensors avoid DRAM write + read
            if len(group) >= 2:
                # Calculate intermediate tensor size from first layer's output
                first_layer = group[0]
                first_layer_dims = graph.layers[first_layer]['dims']
                
                # Simplified proxy for output tensor size (elements * bytes_per_element)
                # Output size = N * K * P * Q for conv layers
                if 'P' in first_layer_dims and 'Q' in first_layer_dims:
                    intermediate_tensor_size = torch.tensor(
                        first_layer_dims['N'] * first_layer_dims.get('K', first_layer_dims.get('C', 1)) * 
                        first_layer_dims['P'] * first_layer_dims['Q'] * config.BYTES_PER_ELEMENT,
                        dtype=torch.float32
                    )
                else:
                    # For other layer types, use a simpler calculation
                    intermediate_tensor_size = torch.tensor(
                        first_layer_dims['N'] * first_layer_dims.get('K', first_layer_dims.get('C', 1)) * config.BYTES_PER_ELEMENT,
                        dtype=torch.float32
                    )
                
                # DRAM energy saved = 2 * intermediate_tensor_size * DRAM_ACCESS_COST
                # (one write to DRAM avoided, one read from DRAM avoided)
                dram_energy_saved = 2 * intermediate_tensor_size * torch.tensor(config.DRAM_ACCESS_COST, dtype=torch.float32)
                
                # Subtract the saving from fused DRAM energy
                fused_dram_energy = fused_dram_energy - dram_energy_saved
                
                # Ensure DRAM energy doesn't go negative
                fused_dram_energy = torch.maximum(fused_dram_energy, torch.tensor(0.0))
            
            # Calculate final fused energy with fusion overhead
            fused_energy = fused_compute_energy + fused_sram_energy + fused_dram_energy + torch.tensor(config.FUSION_OVERHEAD_COST, dtype=torch.float32)
            
            # Energy-Delay Product for fused case
            cost_fused = fused_latency * fused_energy
            
            # === Calculate Weighted-Average EDP for the group ===
            group_edp = p_fuse * cost_fused + (1 - p_fuse) * cost_no_fuse
            
            # Convert back to latency and energy for accumulation
            # For differentiable accumulation, we use the weighted averages
            group_latency = p_fuse * fused_latency + (1 - p_fuse) * non_fused_latency
            group_energy = p_fuse * fused_energy + (1 - p_fuse) * non_fused_energy
            
            total_latency = total_latency + group_latency
            total_energy = total_energy + group_energy
        
        return total_latency, total_energy, hw_config['area_cost']

def calculate_product_constraint_penalty(
    mapping_params: 'MappingParameters',
    graph: 'ComputationGraph'
) -> torch.Tensor:
    """
    Calculate penalty for violation of the product constraint.
    The product of tiling factors for each dimension should equal the problem size.
    Uses log-domain difference for numerical stability.
    
    Args:
        mapping_params: MappingParameters object containing layer mappings
        graph: ComputationGraph object defining the network structure
        
    Returns:
        Total penalty summed across all dimensions and layers
    """
    total_penalty = torch.tensor(0.0)
    
    for layer_name, layer_info in graph.layers.items():
        layer_mapping = mapping_params.get_mapping_by_original_name(layer_name)
        dims = layer_info['dims']
        
        # For each problem dimension
        for dim in Config.get_instance().PROBLEM_DIMS:
            if dim in dims:
                # Get continuous factor (we only have L2 level for now)
                f_continuous = layer_mapping.temporal_factors_L2[dim]
                
                # Calculate log-domain difference
                log_factor = torch.log(f_continuous)
                log_problem_size = torch.log(torch.tensor(dims[dim], dtype=torch.float32))
                
                # Square the difference for the penalty
                penalty = (log_factor - log_problem_size) ** 2
                total_penalty = total_penalty + penalty
    
    return total_penalty

if __name__ == "__main__":
    # This script is now intended to be used as a module.
    # The main experiment runner is in run_experiments.py
    pass 