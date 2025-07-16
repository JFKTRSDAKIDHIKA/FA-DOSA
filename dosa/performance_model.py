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
        Refactored method to calculate data movement between adjacent memory levels based on a physically accurate model.
        Accesses = Tile_Size_at_Lower_Level * Num_Reloads
        """
        accesses = {}
        memory_levels = [level for level in self.config.MEMORY_HIERARCHY if level['type'] in ['buffer', 'dram']]
        level_names = [level['name'] for level in memory_levels]

        # Iterate through interfaces from outside-in
        for i in range(len(memory_levels) - 1):
            upper_level_idx = i + 1
            lower_level_idx = i
            upper_level_name = level_names[upper_level_idx]
            lower_level_name = level_names[lower_level_idx]
            interface_name = f"{upper_level_name}_to_{lower_level_name}"
            total_access_bytes_for_interface = torch.tensor(0.0, device=self.config.DEVICE)

            for tensor_type, relevant_dims in TENSOR_DIM_MAP.items():
                # A. Calculate Tile_Size_at_Lower_Level
                tile_size_at_lower_level = torch.tensor(1.0, device=self.config.DEVICE)
                for dim_name in relevant_dims:
                    if dim_name in layer_dims:
                        # Product of all tiling factors up to and including the lower_level
                        for level_idx in range(lower_level_idx + 1):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                tile_size_at_lower_level *= mapping_table[dim_name][level_name]['temporal']
                                tile_size_at_lower_level *= mapping_table[dim_name][level_name]['spatial']

                # B. Calculate Num_Reloads
                num_reloads = torch.tensor(1.0, device=self.config.DEVICE)
                # Define reuse dimensions for each tensor type
                if tensor_type == 'Input':
                    reuse_dims = ['K']
                elif tensor_type == 'Weight':
                    reuse_dims = ['N', 'P', 'Q']
                elif tensor_type == 'Output':
                    reuse_dims = ['C', 'R', 'S']
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

    def forward(self, graph, hw_params: HardwareParameters, fusion_params, mapping: FineGrainedMapping) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_latency = torch.tensor(0.0, device=self.config.DEVICE)
        total_energy = torch.tensor(0.0, device=self.config.DEVICE)
        
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

            penalty = torch.tensor(0.0, device=self.config.DEVICE)
            for i, level in enumerate(self.config.MEMORY_HIERARCHY):
                 if level['type'] == 'buffer':
                    required_kb = self.calculate_buffer_req_kb(layer['dims'], all_factors, i)
                    available_kb = hw_params.get_buffer_size_kb(level['name'])
                    penalty += torch.relu(required_kb / available_kb - 1.0)

            total_latency += latency * (1 + penalty * self.config.PENALTY_WEIGHT)
            total_energy += energy * (1 + penalty * self.config.PENALTY_WEIGHT)

        return total_latency, total_energy, hw_params.get_area_cost()

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