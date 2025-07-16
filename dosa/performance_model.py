import torch
import torch.nn as nn
from typing import Tuple
from functools import reduce
from operator import mul

from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping

# Define TENSOR_DIM_MAP
TENSOR_DIM_MAP = {
    'Input':  ['N', 'C', 'P', 'Q', 'R', 'S'],
    'Weight': ['K', 'C', 'R', 'S'],
    'Output': ['N', 'K', 'P', 'Q']
}

class HighFidelityPerformanceModel(nn.Module):
    """
    NEW: 高保真性能模型，能够处理多级存储和细粒度映射。
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def calculate_per_level_accesses(self, layer_dims: dict, mapping_table: dict) -> dict:
        """
        Physically accurate model for data movement between memory levels.
        Accesses = Tile_Size_at_Lower_Level * Num_Reloads
        """
        accesses = {}
        memory_levels = [level for level in self.config.MEMORY_HIERARCHY if level['type'] in ['buffer', 'dram']]
        level_names = [level['name'] for level in memory_levels]
        
        # Define reload dimensions for each tensor type
        reload_dims = {
            'Input': ['N', 'P', 'Q'],  # Reused across K, tiled within C,R,S
            'Weight': ['K'],           # Reused across N, P, Q
            'Output': ['N', 'K', 'P', 'Q']  # Produced once after iterating through C, R, S
        }

        # Iterate through interfaces from outside-in
        for i in range(len(memory_levels) - 1):
            upper_level_idx = i + 1
            lower_level_idx = i
            upper_level_name = level_names[upper_level_idx]
            lower_level_name = level_names[lower_level_idx]
            interface_name = f"{upper_level_name}_to_{lower_level_name}"
            total_access_bytes_for_interface = torch.tensor(0.0, device=self.config.DEVICE)

            for tensor_type, relevant_dims in TENSOR_DIM_MAP.items():
                # Step 1: Calculate Tile_Size_at_Lower_Level
                tile_size_at_lower_level = torch.tensor(1.0, device=self.config.DEVICE)
                
                for dim_name in relevant_dims:
                    if dim_name in layer_dims:
                        # Calculate dim_tile_size: product of all temporal and spatial factors
                        # for this dimension at all levels at and below the lower_level
                        dim_tile_size = torch.tensor(1.0, device=self.config.DEVICE)
                        for level_idx in range(lower_level_idx + 1):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                dim_tile_size *= mapping_table[dim_name][level_name]['temporal']
                                dim_tile_size *= mapping_table[dim_name][level_name]['spatial']
                        tile_size_at_lower_level *= dim_tile_size

                # Step 2: Calculate Num_Reloads
                num_reloads = torch.tensor(1.0, device=self.config.DEVICE)
                
                for dim_name in reload_dims[tensor_type]:
                    if dim_name in layer_dims:
                        # Get total problem size for this dimension
                        total_problem_size = torch.tensor(float(layer_dims[dim_name]), device=self.config.DEVICE)
                        
                        # Calculate total tiled size at lower_level for this dimension
                        dim_tile_size_at_lower = torch.tensor(1.0, device=self.config.DEVICE)
                        for level_idx in range(lower_level_idx + 1):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                dim_tile_size_at_lower *= mapping_table[dim_name][level_name]['temporal']
                                dim_tile_size_at_lower *= mapping_table[dim_name][level_name]['spatial']
                        
                        # Number of reloads for this dimension
                        num_reloads_for_dim = torch.ceil(total_problem_size / (dim_tile_size_at_lower + 1e-9))
                        num_reloads *= num_reloads_for_dim

                # Step 3: Calculate Accesses for the Tensor
                tensor_access_elements = tile_size_at_lower_level * num_reloads
                tensor_access_bytes = tensor_access_elements * self.config.BYTES_PER_ELEMENT
                total_access_bytes_for_interface += tensor_access_bytes

            accesses[interface_name] = total_access_bytes_for_interface

        return accesses

    def forward(self, graph, hw_params: HardwareParameters, fusion_params, mapping: FineGrainedMapping) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_latency = torch.tensor(0.0, device=self.config.DEVICE)
        total_energy = torch.tensor(0.0, device=self.config.DEVICE)
        total_buffer_mismatch_loss = torch.tensor(0.0, device=self.config.DEVICE)
        
        all_factors = mapping.get_all_factors()

        for group in graph.fusion_groups:
            layer_name = group[0]
            layer = graph.layers[layer_name]
            macs = reduce(mul, layer['dims'].values(), 1)
            
            num_pes = hw_params.get_projected_num_pes()
            compute_latency = macs / (num_pes * self.config.CLOCK_FREQUENCY_MHZ * 1e6 + 1e-9)
            
            per_level_accesses = self.calculate_per_level_accesses(layer['dims'], all_factors)
            memory_latencies = []
            for interface, accesses in per_level_accesses.items():
                upper_level_name = interface.split('_to_')[0]
                bandwidth = next(level['bandwidth_gb_s'] for level in self.config.MEMORY_HIERARCHY if level['name'] == upper_level_name)
                memory_latencies.append(accesses / (bandwidth * 1e9 + 1e-9))
            
            if memory_latencies:
                latency = torch.maximum(compute_latency, torch.max(torch.stack(memory_latencies)))
            else:
                latency = compute_latency

            energy = torch.tensor(0.0, device=self.config.DEVICE)
            energy += macs * self.config.PE_MAC_EPA_PJ
            
            for interface, accesses_bytes in per_level_accesses.items():
                lower_level_name = interface.split('_to_')[1]
                accesses_4bytes = accesses_bytes / 4.0

                if lower_level_name == 'L1_Registers':
                    energy += accesses_4bytes * self.config.L1_REG_BASE_EPA_PJ
                elif lower_level_name == 'L2_Scratchpad':
                    size_kb = hw_params.get_buffer_size_kb(lower_level_name)
                    epa = self.config.L2_SPM_BASE_EPA_PJ + self.config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb
                    energy += accesses_4bytes * epa
                elif lower_level_name == 'L3_DRAM':
                    energy += accesses_4bytes * self.config.L3_DRAM_EPA_PJ

            # Calculate buffer mismatch loss for this layer
            for i, level in enumerate(self.config.MEMORY_HIERARCHY):
                if level['type'] == 'buffer':
                    required_kb = self.calculate_buffer_req_kb(layer['dims'], all_factors, i)
                    available_kb = hw_params.get_buffer_size_kb(level['name'])
                    buffer_deficit = torch.relu(required_kb - available_kb)
                    level_mismatch_loss = torch.pow(buffer_deficit, 2)
                    total_buffer_mismatch_loss += level_mismatch_loss

            total_latency += latency
            total_energy += energy

        return total_latency, total_energy, hw_params.get_area_cost(), total_buffer_mismatch_loss

    def calculate_buffer_req_kb(self, dims, factors, level_idx):
        total_buffer_bytes = torch.tensor(0.0)
        level_name = self.config.MEMORY_HIERARCHY[level_idx]['name']

        for tensor_type, tensor_dims in TENSOR_DIM_MAP.items():
            tile_dims = {}
            for dim_name in tensor_dims:
                if dim_name in dims:
                    tile_dims[dim_name] = torch.tensor(1.0)
                    for i in range(level_idx + 1):
                        inner_level_name = self.config.MEMORY_HIERARCHY[i]['name']
                        if inner_level_name in factors[dim_name]:
                            tile_dims[dim_name] = tile_dims[dim_name] * \
                                factors[dim_name][inner_level_name]['temporal'].squeeze() * \
                                factors[dim_name][inner_level_name]['spatial'].squeeze()
            
            tensor_tile_size = reduce(mul, [tile_dims.get(d, torch.tensor(1.0)) for d in tensor_dims if d in dims], torch.tensor(1.0))
            total_buffer_bytes += tensor_tile_size * self.config.BYTES_PER_ELEMENT

        return total_buffer_bytes / 1024.0