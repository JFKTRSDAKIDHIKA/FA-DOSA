import unittest
import torch
from dosa.config import Config
from dosa.performance_model import HighFidelityPerformanceModel, TENSOR_DIM_MAP
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
class TestPerformanceModel(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.perf_model = HighFidelityPerformanceModel(self.config)
        self.hw_params = HardwareParameters()
        layer_dims = {'N': 1, 'C': 64, 'K': 64, 'P': 56, 'Q': 56, 'R': 3, 'S': 3}
        self.mapping = FineGrainedMapping(problem_dims=layer_dims, hierarchy=self.config.MEMORY_HIERARCHY)

    def test_input_tensor_reuse_calculation(self):
        layer_dims = {'N': 1, 'C': 64, 'K': 64, 'P': 56, 'Q': 56, 'R': 3, 'S': 3}

        # Manually construct a clean mapping_table to ensure a controlled test
        dims_to_tile = ['N', 'C', 'K', 'P', 'Q', 'R', 'S']
        memory_levels = [level['name'] for level in self.config.MEMORY_HIERARCHY if level['type'] != 'compute']
        mapping_table = {dim: {level: {'temporal': torch.tensor(1.0), 'spatial': torch.tensor(1.0)} for level in memory_levels} for dim in dims_to_tile}

        # Manually set a non-trivial temporal tile for the 'K' dimension at L2
        # This simulates holding an input tile and reusing it across the 'K' dimension computations
        mapping_table['K']['L2_Scratchpad']['temporal'] = torch.tensor(4.0)
        # The rest of the K dimension is tiled at the DRAM level
        mapping_table['K']['L3_DRAM']['temporal'] = torch.tensor(16.0) # 64 / 4 = 16

        # --- Calculate Ground Truth --- #
        total_expected_access_bytes = torch.tensor(0.0)

        for tensor_type, relevant_dims in TENSOR_DIM_MAP.items():
            # A. Calculate Tile_Size_at_Lower_Level (L2)
            tile_size_at_L2 = torch.tensor(1.0)
            for dim_name in relevant_dims:
                if dim_name in layer_dims:
                    tile_size_at_L2 *= mapping_table[dim_name]['L1_Registers']['temporal'] * mapping_table[dim_name]['L1_Registers']['spatial']
                    tile_size_at_L2 *= mapping_table[dim_name]['L2_Scratchpad']['temporal'] * mapping_table[dim_name]['L2_Scratchpad']['spatial']

            # B. Calculate Num_Reloads from L3
            num_reloads = torch.tensor(1.0)
            if tensor_type == 'Input':
                reload_dims = ['N', 'P', 'Q']  # Input is reloaded when iterating over these dimensions
            elif tensor_type == 'Weight':
                reload_dims = ['K']  # Weight is reloaded when iterating over K
            elif tensor_type == 'Output':
                reload_dims = ['N', 'K', 'P', 'Q']  # Output is produced for each of these dimensions
            else:
                reload_dims = []
            
            for dim_name in reload_dims:
                if dim_name in layer_dims:
                    # Calculate total tiled size at L2 for this dimension
                    dim_tile_size_at_L2 = (mapping_table[dim_name]['L1_Registers']['temporal'] * 
                                          mapping_table[dim_name]['L1_Registers']['spatial'] *
                                          mapping_table[dim_name]['L2_Scratchpad']['temporal'] * 
                                          mapping_table[dim_name]['L2_Scratchpad']['spatial'])
                    # Number of reloads for this dimension
                    num_reloads_for_dim = torch.ceil(torch.tensor(float(layer_dims[dim_name])) / dim_tile_size_at_L2)
                    num_reloads *= num_reloads_for_dim

            tensor_access_elements = tile_size_at_L2 * num_reloads
            total_expected_access_bytes += tensor_access_elements * self.config.BYTES_PER_ELEMENT

        # --- Execution and Assertion --- #
        calculated_accesses = self.perf_model.calculate_per_level_accesses(layer_dims, mapping_table)
        calculated_l3_to_l2_accesses = calculated_accesses['L3_DRAM_to_L2_Scratchpad']

        self.assertAlmostEqual(calculated_l3_to_l2_accesses.item(), total_expected_access_bytes.item(), places=1)

if __name__ == '__main__':
    unittest.main()