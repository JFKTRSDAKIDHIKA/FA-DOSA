import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple, List, Any
from functools import reduce
from operator import mul

from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping, ProjectToNearestDivisor
from dosa.performance_model import HighFidelityPerformanceModel, TENSOR_DIM_MAP
from dosa.utils import (
    ComputationGraph, FusionParameters, 
    calculate_macs, save_configuration_to_json, get_divisors
)


# --- 核心思想 ---
# 本次重构旨在打破三大简化假设，构建一个更高保真度的FA-DOSA框架：
# 1. 显式的多级存储层次（Memory Hierarchy）：在Config中定义，包括各级Buffer。
# 2. 细粒度的映射参数化（Fine-grained Mapping）：为每个存储层的每个计算维度定义独立的、区分空间/时间的Tiling因子。
# 3. 更精确的性能模型（Fidelity-Enhanced Performance Model）：能够根据细粒度的映射，计算多级存储间的访问成本。


# ... parse_onnx_to_graph 和 calculate_macs 保持不变 ...
def parse_onnx_to_graph(path): # Task 2: Enable True Multi-Layer Fusion Groups
    graph = ComputationGraph()
    for i in range(2): # 简化为2个group以加速
        # Add individual layers to graph.layers
        dims_conv = {'N': 1, 'C': 64*(2**(i//2)), 'K': 64*(2**(i//2)), 'P': 56//(2**(i//2)), 'Q': 56//(2**(i//2)), 'R': 3, 'S': 3}
        dims_relu = dims_conv.copy()  # ReLU has same dimensions as conv output
        
        graph.add_layer(f'conv_{i}', dims_conv, 'Conv')
        graph.add_layer(f'relu_{i}', dims_relu, 'ReLU')
        
        # NEW: Create multi-layer fusion groups (producer-consumer pairs)
        graph.add_fusion_group([f'conv_{i}', f'relu_{i}'])
        
        # NEW: Preserve single-layer (no-fusion) options
        graph.add_fusion_group([f'conv_{i}'])
        graph.add_fusion_group([f'relu_{i}'])
    return graph


# --- 主实验流程 ---

def run_experiment(num_iterations=1000):
    print("--- Running High-Fidelity FA-DOSA Experiment ---")
    config = Config()
    graph = parse_onnx_to_graph("resnet.onnx")

    hw_params = HardwareParameters()
    # 使用一个共享的mapping对象
    mapping = FineGrainedMapping(graph.problem_dims, config.MEMORY_HIERARCHY)
    fusion_params = FusionParameters(graph)
    perf_model = HighFidelityPerformanceModel(config)
    
    all_params = list(hw_params.parameters()) + list(mapping.parameters()) + list(fusion_params.parameters())
    optimizer = optim.Adam(all_params, lr=1e-2)

    for i in range(num_iterations):
        optimizer.zero_grad()
        latency, energy, area = perf_model(graph, hw_params, fusion_params, mapping)
        
        # NEW: Add PE square penalty
        continuous_pes = hw_params.get_num_pes()
        sqrt_pes = torch.sqrt(continuous_pes)
        pe_square_penalty = torch.pow(sqrt_pes - torch.round(sqrt_pes), 2)
        pe_penalty_weight = 0.08

        # Combine all loss terms
        edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
        area_loss = 0.05 * area
        
        loss = edp_loss + area_loss + pe_square_penalty * pe_penalty_weight
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Iter {i}: Loss={loss.item():.4f}, EDP={edp_loss.item():.4f}, Area={area_loss.item():.4f}, PE_Penalty={pe_square_penalty.item():.4f}")
            print(f"         Latency={latency.item():.2e}s, Energy={energy.item():.2e}pJ, Area={area.item():.2f}mm²")
    
    print("\n--- Final Configuration ---")
    print(f"PEs: {hw_params.get_projected_num_pes().item():.0f}")
    for level in config.MEMORY_HIERARCHY:
        if level['type'] == 'buffer':
            print(f"{level['name']} Size: {hw_params.get_buffer_size_kb(level['name']).item():.2f} KB")

    # Get the final projected mapping
    final_mapping = mapping.get_all_factors()

    # Convert tensors to floats for JSON serialization
    for dim_name, dim_factors in final_mapping.items():
        for level_name, level_factors in dim_factors.items():
            final_mapping[dim_name][level_name]['temporal'] = level_factors['temporal'].item()
            final_mapping[dim_name][level_name]['spatial'] = level_factors['spatial'].item()

    # Save the final configuration to JSON
    save_configuration_to_json(hw_params, final_mapping, "final_configuration.json")

if __name__ == "__main__":
    run_experiment(num_iterations=500)