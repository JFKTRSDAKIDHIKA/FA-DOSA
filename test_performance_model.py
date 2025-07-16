import unittest
import torch
from dosa.config import Config
from dosa.performance_model import HighFidelityPerformanceModel, TENSOR_DIM_MAP
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
class TestPerformanceModel(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.model = HighFidelityPerformanceModel(self.config)
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

        # We are interested in Accesses(L3_DRAM -> L2_Scratchpad)
        # For the Input tensor, reuse is across the 'K' dimension.
        # The number of refills from L3 is determined by the temporal loops at level >= L2+1=L3.
        # N_refills_from_L3 for Input = mapping_table['K']['L3_DRAM']['temporal'] = 16.0

        # 1. Expected Input Tensor Accesses
        input_tensor_size = layer_dims['N'] * layer_dims['C'] * layer_dims['P'] * layer_dims['Q'] * layer_dims['R'] * layer_dims['S']
        # The reuse factor for the input tensor is determined by the temporal tiling of the 'K' dimension at L3.
        input_reuse_factor = mapping_table['K']['L3_DRAM']['temporal']
        expected_input_accesses = input_tensor_size / input_reuse_factor

        # 2. Expected Weight Tensor Accesses
        # The reuse dimensions for the weight tensor are 'N', 'P', and 'Q'. Their temporal factors at L3 are all 1.0.
        weight_tensor_size = layer_dims['K'] * layer_dims['C'] * layer_dims['R'] * layer_dims['S']
        expected_weight_accesses = weight_tensor_size  # No reuse at this level

        # 3. Expected Output Tensor Accesses
        # The reuse dimensions for the output tensor are 'C', 'R', and 'S'. Their temporal factors at L3 are all 1.0.
        output_tensor_size = layer_dims['N'] * layer_dims['K'] * layer_dims['P'] * layer_dims['Q']
        expected_output_accesses = output_tensor_size  # No reuse at this level

        total_expected_access_elements = expected_input_accesses + expected_weight_accesses + expected_output_accesses
        total_expected_access_bytes = total_expected_access_elements * self.config.BYTES_PER_ELEMENT

        # --- Execution and Assertion --- #
        calculated_accesses = self.model.calculate_per_level_accesses(layer_dims, mapping_table)
        calculated_l3_to_l2_accesses = calculated_accesses['L3_DRAM_to_L2_Scratchpad']

        self.assertAlmostEqual(calculated_l3_to_l2_accesses.item(), total_expected_access_bytes.item(), places=1)

if __name__ == '__main__':
    unittest.main()