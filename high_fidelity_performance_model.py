import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Any, Optional
from functools import reduce
from operator import mul

from fa_dosa_demo import ComputationGraph, FusionParameters, HardwareParameters
from mapping_template import LearnableConvReluTemplate

# Define TENSOR_DIM_MAP for data reuse modeling
TENSOR_DIM_MAP = {
    'Input':  ['N', 'C', 'P', 'Q', 'R', 'S'],
    'Weight': ['K', 'C', 'R', 'S'],
    'Output': ['N', 'K', 'P', 'Q']
}


def calculate_macs(dims: Dict[str, int], op_type: str) -> torch.Tensor:
    """
    Calculate the number of MAC operations for a given layer.
    
    Args:
        dims: Dictionary of dimension sizes
        op_type: The operation type of the layer (e.g., 'Conv', 'MatMul')
        
    Returns:
        Number of MAC operations as a tensor
    """
    if op_type == 'Conv':
        # N * K * P * Q * C * R * S
        return torch.tensor(reduce(mul, dims.values(), 1.0))
    elif op_type in ['MatMul', 'Gemm']:
        # For MatMul: M * N * K
        if all(k in dims for k in ['M', 'N', 'K']):
            return torch.tensor(dims['M'] * dims['N'] * dims['K'])
        # For Transformer attention: N * H * S * S * D
        elif all(k in dims for k in ['N', 'H', 'S', 'D']):
            return torch.tensor(dims['N'] * dims['H'] * dims['S'] * dims['S'] * dims['D'])
    elif op_type in ['ReLU', 'GELU', 'Gelu', 'BatchNorm', 'LayerNorm']:
        # Activation functions have much fewer operations
        # Simplified model: 1 operation per element
        elements = 1.0
        for k, v in dims.items():
            if k not in ['R', 'S']:  # Skip filter dimensions for activations
                elements *= v
        return torch.tensor(elements)
    else:
        # Default case: assume minimal computation
        return torch.tensor(1.0)


class HighFidelityPerformanceModel(nn.Module):
    """
    Enhanced performance model that dynamically calculates performance based on fusion decisions.
    This model forms "Fused Groups" based on learnable parameters and calculates the group's
    collective latency, energy, and buffer requirements accordingly.
    """
    def __init__(self, config, graph: Optional[ComputationGraph] = None):
        super().__init__()
        self.config = config
        
        # Initialize learnable templates for Conv+ReLU fusion groups
        self.templates = nn.ModuleDict()
        if graph is not None:
            self._initialize_templates(graph)
    
    def _initialize_templates(self, graph: ComputationGraph):
        """
        Initialize learnable templates for potential Conv+ReLU fusion groups.
        
        Args:
            graph: Computation graph containing fusion groups
        """
        for group in graph.fusion_groups:
            if len(group) >= 2:
                # Check if this is a Conv+ReLU fusion group
                layer_types = [graph.layers[layer_name]['type'] for layer_name in group]
                
                # Look for Conv followed by ReLU pattern
                if len(layer_types) == 2 and layer_types[0] == 'Conv' and layer_types[1] == 'ReLU':
                    # Create group key
                    group_key = '__'.join(group)
                    
                    # Get Conv layer dimensions for template initialization
                    conv_layer = graph.layers[group[0]]
                    problem_dims = conv_layer['dims']
                    
                    # Initialize learnable template
                    template = LearnableConvReluTemplate(problem_dims, self.config)
                    self.templates[group_key] = template
    
    def calculate_per_level_accesses(self, layer_dims: Dict[str, int], mapping_table: Dict) -> Dict[str, torch.Tensor]:
        """
        Refactored method to calculate data movement between adjacent memory levels based on a physically accurate model.
        Accesses = Tile_Size_at_Lower_Level * Num_Reloads
        
        Args:
            layer_dims: Dictionary of layer dimensions
            mapping_table: Dictionary containing mapping factors for each dimension and level
            
        Returns:
            Dictionary mapping interface names to their total access costs in bytes
        """
        accesses = {}
        
        # Get memory levels (excluding compute level)
        memory_levels = [level for level in self.config.MEMORY_HIERARCHY if level['type'] in ['buffer', 'dram']]
        level_names = [level['name'] for level in memory_levels]
        
        # Iterate through interfaces from outside-in (DRAM -> L2 -> L1)
        for i in range(len(memory_levels) - 1):
            upper_level_idx = i + 1  # Outer level (e.g., DRAM)
            lower_level_idx = i      # Inner level (e.g., L2)
            upper_level_name = level_names[upper_level_idx]
            lower_level_name = level_names[lower_level_idx]
            interface_name = f"{upper_level_name}_to_{lower_level_name}"
            
            total_access_bytes_for_interface = torch.tensor(0.0, device=self.config.DEVICE)
            
            # Calculate access for each tensor type
            for tensor_type, relevant_dims in TENSOR_DIM_MAP.items():
                # A. Calculate Tile_Size_at_Lower_Level
                tile_size_at_lower_level = torch.tensor(1.0, device=self.config.DEVICE)
                
                for dim_name in relevant_dims:
                    if dim_name in layer_dims:
                        # Product of all tiling factors (temporal & spatial) up to and including the lower_level
                        for level_idx in range(lower_level_idx + 1):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                tile_size_at_lower_level *= mapping_table[dim_name][level_name]['temporal']
                                tile_size_at_lower_level *= mapping_table[dim_name][level_name]['spatial']
                
                # B. Calculate Num_Reloads
                num_reloads = torch.tensor(1.0, device=self.config.DEVICE)
                
                # Define reuse dimensions for each tensor type based on data reuse patterns
                if tensor_type == 'Input':
                    reuse_dims = ['K']  # Input is reused across output channels
                elif tensor_type == 'Weight':
                    reuse_dims = ['N', 'P', 'Q']  # Weights are reused across batch and spatial dimensions
                elif tensor_type == 'Output':
                    reuse_dims = ['C', 'R', 'S']  # Output is reused across input channels and filter dimensions
                else:
                    reuse_dims = []
                
                for dim_name in reuse_dims:
                    if dim_name in layer_dims:
                        # Product of temporal tiling factors at the upper_level and all levels outside of it
                        for level_idx in range(upper_level_idx, len(memory_levels)):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                num_reloads *= mapping_table[dim_name][level_name]['temporal']
                
                # C. Calculate Tensor_Accesses and Accumulate
                tensor_access_elements = tile_size_at_lower_level * num_reloads
                tensor_access_bytes = tensor_access_elements * self.config.BYTES_PER_ELEMENT
                total_access_bytes_for_interface += tensor_access_bytes
            
            accesses[interface_name] = total_access_bytes_for_interface
        
        return accesses
        
    def forward(self, 
               graph: ComputationGraph, 
               hardware_params: HardwareParameters, 
               fusion_params: FusionParameters, 
               mapping_params) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the performance model with differentiable fusion logic.
        
        Args:
            graph: Computation graph with layers, edges, and fusion groups
            hardware_params: Hardware configuration parameters
            fusion_params: Learnable fusion parameters
            mapping_params: Mapping parameters for each layer
            
        Returns:
            Tuple of (total_latency, total_energy, area_cost)
        """
        # Initialize total costs
        total_latency = torch.tensor(0.0)
        total_energy = torch.tensor(0.0)
        total_buffer_req = torch.tensor(0.0)
        
        # Get hardware area cost
        area_cost = hardware_params.get_area_cost()
        
        # Create cache for single-layer costs to avoid redundant computations
        layer_costs_cache = {}
        
        # Pre-computation pass: calculate costs for all unique layers
        for layer_name in graph.layers.keys():
            if layer_name not in layer_costs_cache:
                layer_latency, layer_energy, layer_buffer = self._calculate_single_layer_cost(
                    layer_name, graph, hardware_params, mapping_params
                )
                layer_costs_cache[layer_name] = (layer_latency, layer_energy, layer_buffer)
        
        # Track processed layers to avoid double counting
        processed_layers = set()
        
        # Iterate through all potential fusion groups
        for group in graph.fusion_groups:
            # Skip if any layer in this group has already been processed
            if any(layer in processed_layers for layer in group):
                continue
            
            if len(group) > 1:
                # Multi-layer group: use differentiable weighted average
                fusion_prob = fusion_params.get_fusion_probability(group)
                
                # Calculate cost if fused
                cost_fused_latency, cost_fused_energy, cost_fused_buffer = self._calculate_group_costs(
                    group, graph, hardware_params, mapping_params, layer_costs_cache
                )
                
                # Calculate cost if not fused (physically correct aggregation)
                unfused_latencies = []
                unfused_energies = []
                unfused_buffers = []
                
                for layer in group:
                    # Retrieve pre-computed costs from cache
                    layer_latency, layer_energy, layer_buffer = layer_costs_cache[layer]
                    unfused_latencies.append(layer_latency)
                    unfused_energies.append(layer_energy)
                    unfused_buffers.append(layer_buffer)
                
                # Physically correct aggregation for unfused execution:
                # - Latency: sum (sequential execution)
                # - Energy: sum (total energy consumption)
                # - Buffer: max (peak on-chip memory demand)
                cost_unfused_latency = torch.sum(torch.stack(unfused_latencies))
                cost_unfused_energy = torch.sum(torch.stack(unfused_energies))
                cost_unfused_buffer = torch.max(torch.stack(unfused_buffers))
                
                # Calculate differentiable weighted average (ensure proper tensor shapes)
                fusion_prob = fusion_prob.squeeze()
                group_latency = fusion_prob * cost_fused_latency.squeeze() + (1 - fusion_prob) * cost_unfused_latency.squeeze()
                group_energy = fusion_prob * cost_fused_energy.squeeze() + (1 - fusion_prob) * cost_unfused_energy.squeeze()
                # For buffer requirement, take the worst-case (max) to ensure capacity constraints
                group_buffer = torch.max(cost_fused_buffer.squeeze(), cost_unfused_buffer.squeeze())
                
            else:
                # Single-layer group: calculate costs directly
                group_latency, group_energy, group_buffer = self._calculate_group_costs(
                    group, graph, hardware_params, mapping_params, layer_costs_cache
                )
            
            # Accumulate costs (ensure proper tensor shapes)
            total_latency = total_latency + group_latency.squeeze()
            total_energy = total_energy + group_energy.squeeze()
            total_buffer_req = total_buffer_req + group_buffer.squeeze()
            
            # Mark layers as processed
            processed_layers.update(group)
        
        # Apply capacity penalty if buffer requirements exceed available resources
        penalty = self._calculate_capacity_penalty(hardware_params, total_buffer_req)
        total_latency = total_latency * (1 + penalty * self.config.PENALTY_WEIGHT)
        total_energy = total_energy * (1 + penalty * self.config.PENALTY_WEIGHT)
        
        # Ensure numerical stability
        min_value = 1e-6
        total_latency = torch.clamp(total_latency, min=min_value)
        total_energy = torch.clamp(total_energy, min=min_value)
        area_cost = torch.clamp(area_cost, min=min_value)
        
        return total_latency, total_energy, area_cost
    

    
    def _calculate_group_costs(self, 
                             group: List[str], 
                             graph: ComputationGraph, 
                             hardware_params: HardwareParameters, 
                             mapping_params, 
                             layer_costs_cache: dict = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate costs for an execution group.
        
        Args:
            group: List of layer names in the execution group
            graph: Computation graph
            hardware_params: Hardware parameters
            mapping_params: Mapping parameters
            layer_costs_cache: Optional cache of pre-computed layer costs
            
        Returns:
            Tuple of (group_latency, group_energy, group_buffer_req)
        """
        is_fused = len(group) > 1
        
        # Check if this is a Conv+ReLU fusion group with a corresponding template
        if is_fused and len(group) == 2:
            layer_types = [graph.layers[layer_name]['type'] for layer_name in group]
            if layer_types[0] == 'Conv' and layer_types[1] == 'ReLU':
                group_key = '__'.join(group)
                if group_key in self.templates:
                    # Use the learnable template for this fused group
                    template = self.templates[group_key]
                    metrics = template.calculate_performance_metrics(hardware_params)
                    return metrics['latency'], metrics['energy'], metrics['buffer_req_kb']
        
        # Fallback to layer-wise cost calculation for non-fused or non-template groups
        return self._calculate_layer_wise_costs(group, graph, hardware_params, mapping_params, layer_costs_cache)
    
    def _calculate_layer_wise_costs(self, 
                                  group: List[str], 
                                  graph: ComputationGraph, 
                                  hardware_params: HardwareParameters, 
                                  mapping_params, 
                                  layer_costs_cache: dict = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate costs using layer-wise approach (fallback method).
        For fused groups, aggregates individual layer costs appropriately.
        
        Args:
            group: List of layer names in the execution group
            graph: Computation graph
            hardware_params: Hardware parameters
            mapping_params: Mapping parameters
            layer_costs_cache: Optional cache of pre-computed layer costs
            
        Returns:
            Tuple of (group_latency, group_energy, group_buffer_req)
        """
        is_fused = len(group) > 1
        
        # Calculate individual layer costs
        layer_latencies = []
        layer_energies = []
        layer_buffer_reqs = []
        
        for layer_name in group:
            if layer_costs_cache and layer_name in layer_costs_cache:
                # Use cached costs if available
                layer_latency, layer_energy, layer_buffer = layer_costs_cache[layer_name]
            else:
                # Fallback to direct calculation
                layer_latency, layer_energy, layer_buffer = self._calculate_single_layer_cost(
                    layer_name, graph, hardware_params, mapping_params
                )
            layer_latencies.append(layer_latency)
            layer_energies.append(layer_energy)
            layer_buffer_reqs.append(layer_buffer)
        
        # Aggregate costs based on fusion status
        if is_fused:
            # For fused groups, latency is the maximum of individual latencies
            stacked_latencies = torch.stack(layer_latencies)
            max_val, _ = torch.max(stacked_latencies, dim=0)
            group_latency = max_val.clone()
            # Add fusion overhead
            fusion_overhead_cycles = 1000  # Default value: 1000 cycles
            fusion_overhead = torch.tensor(fusion_overhead_cycles / 
                                         self.config.CLOCK_FREQUENCY_MHZ / 1e6)
            group_latency += fusion_overhead
            
            # Energy is summed
            group_energy = torch.sum(torch.stack(layer_energies))
            
            # Buffer requirement is summed for fused groups
            group_buffer_req = torch.sum(torch.stack(layer_buffer_reqs))
        else:
            # For non-fused single layer, return the single layer costs
            group_latency = layer_latencies[0]
            group_energy = layer_energies[0]
            group_buffer_req = layer_buffer_reqs[0]
        
        return group_latency, group_energy, group_buffer_req
    
    def _calculate_single_layer_cost(self, 
                                   layer_name: str, 
                                   graph: ComputationGraph, 
                                   hardware_params: HardwareParameters, 
                                   mapping_params) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the performance cost of a single, non-fused layer.
        This method's sole responsibility is to calculate standalone layer performance.
        
        Args:
            layer_name: Name of the layer
            graph: Computation graph
            hardware_params: Hardware parameters
            mapping_params: Mapping parameters
            
        Returns:
            Tuple of (layer_latency, layer_energy, layer_buffer_req)
        """
        layer = graph.layers[layer_name]
        layer_dims = layer['dims']
        layer_type = layer['type']
        
        # Calculate MACs for this layer
        macs = calculate_macs(layer_dims, layer_type)
        
        # Calculate compute latency
        compute_latency = self._calculate_compute_latency(macs, hardware_params)
        
        # Calculate memory access costs
        dram_accesses, buffer_req = self._calculate_memory_requirements(
            layer_dims, layer_type, mapping_params
        )
        
        # Calculate memory latency
        memory_latency = self._calculate_memory_latency(dram_accesses, hardware_params)
        
        # Determine layer latency (roofline model)
        layer_latency = torch.max(compute_latency, memory_latency)
        
        # Calculate energy components
        compute_energy = self._calculate_compute_energy(macs, hardware_params)
        
        # For standalone layers, always use DRAM access
        memory_energy = self._calculate_dram_memory_energy(
            dram_accesses, hardware_params
        )
        
        layer_energy = compute_energy + memory_energy
        
        return layer_latency, layer_energy, buffer_req
    
    def _calculate_compute_latency(self, macs: torch.Tensor, hardware_params: HardwareParameters) -> torch.Tensor:
        """
        Calculate compute latency based on MACs and hardware parameters.
        
        Args:
            macs: Number of MAC operations
            hardware_params: Hardware parameters
            
        Returns:
            Compute latency in seconds
        """
        num_pes = hardware_params.get_num_pes()
        clock_freq = self.config.CLOCK_FREQUENCY_MHZ * 1e6  # Convert to Hz
        
        # Compute latency = MACs / (PEs * frequency)
        return macs / (num_pes * clock_freq + 1e-9)
    
    def _calculate_memory_requirements(self, 
                                     dims: Dict[str, int], 
                                     op_type: str, 
                                     mapping_params) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate memory access requirements.
        
        Args:
            dims: Layer dimensions
            op_type: Operation type
            mapping_params: Mapping parameters
            
        Returns:
            Tuple of (dram_accesses, buffer_requirement)
        """
        # Simplified model: estimate based on dimensions
        if op_type == 'Conv':
            # Input: N * C * H * W
            input_size = dims['N'] * dims['C'] * dims['P'] * dims['Q']
            # Weights: K * C * R * S
            weight_size = dims['K'] * dims['C'] * dims['R'] * dims['S']
            # Output: N * K * P * Q
            output_size = dims['N'] * dims['K'] * dims['P'] * dims['Q']
        elif op_type in ['MatMul', 'Gemm']:
            # Input: M * K
            input_size = dims.get('M', 1) * dims.get('K', 1)
            # Weights: K * N
            weight_size = dims.get('K', 1) * dims.get('N', 1)
            # Output: M * N
            output_size = dims.get('M', 1) * dims.get('N', 1)
        else:
            # For other ops, assume input size equals output size
            elements = 1.0
            for k, v in dims.items():
                if k not in ['R', 'S']:  # Skip filter dimensions
                    elements *= v
            input_size = elements
            weight_size = torch.tensor(0.0)  # No weights for activations
            output_size = elements
        
        # Convert to bytes
        bytes_per_element = self.config.BYTES_PER_ELEMENT
        input_bytes = input_size * bytes_per_element
        weight_bytes = weight_size * bytes_per_element
        output_bytes = output_size * bytes_per_element
        
        # Ensure all values are tensors
        if not isinstance(input_bytes, torch.Tensor):
            input_bytes = torch.tensor(float(input_bytes))
        if not isinstance(weight_bytes, torch.Tensor):
            weight_bytes = torch.tensor(float(weight_bytes))
        if not isinstance(output_bytes, torch.Tensor):
            output_bytes = torch.tensor(float(output_bytes))
        
        # Total DRAM accesses
        dram_accesses = input_bytes + weight_bytes + output_bytes
        
        # Buffer requirement (simplified)
        # Use torch.max instead of torch.maximum to avoid in-place operations
        max_input_weight = torch.max(input_bytes, weight_bytes)
        buffer_req = torch.max(max_input_weight, output_bytes) / 1024.0  # Convert to KB
        
        return dram_accesses, buffer_req
    
    def _calculate_memory_latency(self, 
                                dram_accesses: torch.Tensor, 
                                hardware_params: HardwareParameters) -> torch.Tensor:
        """
        Calculate memory latency based on DRAM accesses.
        
        Args:
            dram_accesses: Number of DRAM accesses in bytes
            hardware_params: Hardware parameters
            
        Returns:
            Memory latency in seconds
        """
        # Find DRAM level in memory hierarchy
        dram_level = next(level for level in self.config.MEMORY_HIERARCHY if level['type'] == 'dram')
        dram_bandwidth = dram_level['bandwidth_gb_s'] * 1e9  # Convert to bytes/s
        
        # Memory latency = accesses / bandwidth
        return dram_accesses / (dram_bandwidth + 1e-9)
    
    def _calculate_compute_energy(self, 
                                macs: torch.Tensor, 
                                hardware_params: HardwareParameters) -> torch.Tensor:
        """
        Calculate compute energy based on MACs.
        
        Args:
            macs: Number of MAC operations
            hardware_params: Hardware parameters
            
        Returns:
            Compute energy in pJ
        """
        # Energy per MAC operation
        return macs * self.config.PE_MAC_EPA_PJ
    
    def _calculate_on_chip_memory_energy(self, 
                                       accesses: torch.Tensor, 
                                       buffer_req: torch.Tensor, 
                                       hardware_params: HardwareParameters) -> torch.Tensor:
        """
        Calculate on-chip memory energy for fused layers.
        
        Args:
            accesses: Number of memory accesses in bytes
            buffer_req: Buffer requirement in KB
            hardware_params: Hardware parameters
            
        Returns:
            Memory energy in pJ
        """
        # For fused layers, use L2_Scratchpad energy model
        accesses_4bytes = accesses / 4.0  # Assuming 4-byte elements
        
        # Get L2 scratchpad size
        l2_size_kb = hardware_params.get_buffer_size_kb('L2_Scratchpad')
        
        # Calculate energy per access based on scratchpad size
        epa = self.config.L2_SPM_BASE_EPA_PJ + \
              self.config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * l2_size_kb
        
        return accesses_4bytes * epa
    
    def _calculate_dram_memory_energy(self, 
                                    accesses: torch.Tensor, 
                                    hardware_params: HardwareParameters) -> torch.Tensor:
        """
        Calculate DRAM memory energy.
        
        Args:
            accesses: Number of memory accesses in bytes
            hardware_params: Hardware parameters
            
        Returns:
            Memory energy in pJ
        """
        # Convert to 4-byte accesses
        accesses_4bytes = accesses / 4.0
        
        # Use DRAM energy per access
        return accesses_4bytes * self.config.L3_DRAM_EPA_PJ
    
    def _calculate_capacity_penalty(self, 
                                   hardware_params: HardwareParameters, 
                                   total_buffer_req: torch.Tensor) -> torch.Tensor:
        """
        Calculate capacity penalty if buffer requirements exceed available resources.
        
        Args:
            hardware_params: Hardware parameters
            total_buffer_req: Total buffer requirement in KB
            
        Returns:
            Capacity penalty factor
        """
        penalty = torch.tensor(0.0)
        
        # Check capacity constraints for each buffer level
        for level in self.config.MEMORY_HIERARCHY:
            if level['type'] == 'buffer':
                available_kb = hardware_params.get_buffer_size_kb(level['name'])
                # Apply penalty if buffer requirement exceeds available capacity
                penalty += torch.relu(total_buffer_req / available_kb - 1.0)
        
        return penalty