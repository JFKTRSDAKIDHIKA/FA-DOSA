import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Memoization cache for divisors
_divisors_cache = {}

def get_divisors(n: int) -> torch.Tensor:
    """
    Get all integer divisors of n as a sorted torch.Tensor.
    Results are memoized to avoid re-computation.
    """
    if n in _divisors_cache:
        return _divisors_cache[n]
    
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    divisors_tensor = torch.tensor(divisors, dtype=torch.float32)
    _divisors_cache[n] = divisors_tensor
    return divisors_tensor

class ComputationGraph:
    def __init__(self):
        self.layers = {}
        self.edges = []
        self.fusion_groups = []
        self.problem_dims = {'N':1, 'C':1, 'K':1, 'P':1, 'Q':1, 'R':1, 'S':1}
    def add_layer(self, name, dims, op_type):
        self.layers[name] = {'dims': dims, 'type': op_type}
        for d, v in dims.items():
            self.problem_dims[d] = max(self.problem_dims[d], v)
    def add_fusion_group(self, group):
        self.fusion_groups.append(group)

class FusionParameters(nn.Module):
    def __init__(self, graph):
        super().__init__()
        num_groups = len(graph.fusion_groups)
        self.fusion_logits = nn.Parameter(torch.randn(num_groups, 1))

    def get_fusion_decisions(self):
        return torch.sigmoid(self.fusion_logits) > 0.5

def calculate_macs(dims):
    return dims.get('N', 1) * dims.get('C', 1) * dims.get('K', 1) * dims.get('P', 1) * dims.get('Q', 1) * dims.get('R', 1) * dims.get('S', 1)

def save_configuration_to_json(hw_params, projected_mapping, file_path="final_configuration.json"):
    config_dict = {
        "num_pes": hw_params.get_projected_num_pes().item(),
        "l1_size_kb": hw_params.get_buffer_size_kb('L1_Registers').item(),
        "l2_size_kb": hw_params.get_buffer_size_kb('L2_Scratchpad').item(),
        "mapping": projected_mapping
    }
    import json
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)