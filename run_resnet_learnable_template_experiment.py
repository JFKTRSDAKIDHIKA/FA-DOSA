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

# --- Task 1: Differentiable Integer Projection Utilities ---

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

class ProjectToNearestDivisor(torch.autograd.Function):
    """
    Straight-Through Estimator for projecting continuous factors to valid integer divisors.
    Forward pass: discrete projection to nearest valid divisor
    Backward pass: pass gradients through unchanged
    """
    @staticmethod
    def forward(ctx, continuous_factor, problem_dim):
        # Get valid divisors for the problem dimension
        valid_divisors = get_divisors(int(problem_dim.item()))
        
        # Find the closest divisor to the continuous factor
        distances = torch.abs(valid_divisors - continuous_factor)
        closest_idx = torch.argmin(distances)
        projected_factor = valid_divisors[closest_idx]
        
        return projected_factor
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient unchanged for continuous_factor
        # Ensure grad_output has the correct shape
        if grad_output.dim() == 0:
            grad_output = grad_output.unsqueeze(0)
        # Return None for problem_dim as it doesn't need gradients
        return grad_output, None

# --- 核心思想 ---
# 本次重构旨在打破三大简化假设，构建一个更高保真度的FA-DOSA框架：
# 1. 显式的多级存储层次（Memory Hierarchy）：在Config中定义，包括各级Buffer。
# 2. 细粒度的映射参数化（Fine-grained Mapping）：为每个存储层的每个计算维度定义独立的、区分空间/时间的Tiling因子。
# 3. 更精确的性能模型（Fidelity-Enhanced Performance Model）：能够根据细粒度的映射，计算多级存储间的访问成本。

class Config:
    """全局配置类，已更新为支持多级存储层次结构。"""
    _instance = None
    def __init__(self):
        self.BYTES_PER_ELEMENT = 4
        self.CLOCK_FREQUENCY_MHZ = 1000
        
        # --- NEW: 定义显式的多级存储层次 ---
        self.MEMORY_HIERARCHY = [
            # level 0: 虚拟层，代表PE内部的计算
            {'name': 'PE', 'type': 'compute'},
            # level 1: 最内层存储，例如Register File
            {'name': 'L1_Registers', 'type': 'buffer', 'size_kb': nn.Parameter(torch.log(torch.tensor(32.0)))},
            # level 2: 中间层存储，例如Scratchpad
            {'name': 'L2_Scratchpad', 'type': 'buffer', 'size_kb': nn.Parameter(torch.log(torch.tensor(256.0)))},
            # level 3: 主存
            {'name': 'L3_DRAM', 'type': 'dram', 'bandwidth_gb_s': 128}
        ]
        
        # 能量模型（单位：pJ）
        self.PE_MAC_EPA_PJ = 0.561 * 1e6
        # 单位能耗（pJ/access），假设一个access为32-bit (4 bytes)
        self.L1_REG_BASE_EPA_PJ = 0.487 * 1e6 
        self.L2_SPM_BASE_EPA_PJ = 0.49 * 1e6
        self.L2_SPM_CAPACITY_COEFF_PJ_PER_KB = 0.025 * 1e6
        self.L3_DRAM_EPA_PJ = 100 * 1e6
        
        # 面积模型参数
        self.AREA_PER_PE_MM2 = 0.015
        self.AREA_PER_KB_L1_MM2 = 0.008 # L1通常更贵
        self.AREA_PER_KB_L2_MM2 = 0.005 # L2相对便宜
        self.AREA_BASE_MM2 = 1.0
        self.PENALTY_WEIGHT = 1e6

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance

class HardwareParameters(nn.Module):
    """硬件参数，现在支持多级Buffer。"""
    def __init__(self, initial_num_pes=128.0, initial_l1_kb=32.0, initial_l2_kb=256.0):
        super().__init__()
        self.log_num_pes = nn.Parameter(torch.log(torch.tensor(float(initial_num_pes))))
        
        # NEW: 为每个可学习的Buffer创建一个参数
        self.log_buffer_sizes_kb = nn.ParameterDict({
            'L1_Registers': nn.Parameter(torch.log(torch.tensor(float(initial_l1_kb)))),
            'L2_Scratchpad': nn.Parameter(torch.log(torch.tensor(float(initial_l2_kb))))
        })
        
    def get_num_pes(self):
        return torch.exp(self.log_num_pes)

    def get_projected_num_pes(self):
        continuous_pes = self.get_num_pes()
        projected_num_pes = torch.round(torch.sqrt(continuous_pes)) ** 2
        return continuous_pes + (projected_num_pes - continuous_pes).detach()
        
    def get_buffer_size_kb(self, level_name: str):
        return torch.exp(self.log_buffer_sizes_kb[level_name])
        
    def get_area_cost(self):
        config = Config.get_instance()
        pe_area = self.get_num_pes() * config.AREA_PER_PE_MM2
        l1_area = self.get_buffer_size_kb('L1_Registers') * config.AREA_PER_KB_L1_MM2
        l2_area = self.get_buffer_size_kb('L2_Scratchpad') * config.AREA_PER_KB_L2_MM2
        return config.AREA_BASE_MM2 + pe_area + l1_area + l2_area

class FineGrainedMapping(nn.Module):
    """
    NEW: 细粒度的映射参数化模块，替代了旧的LearnableConvReluTemplate。
    """
    def __init__(self, problem_dims: Dict[str, int], hierarchy: List[Dict]):
        super().__init__()
        self.dims = problem_dims
        self.hierarchy = hierarchy
        
        # 创建一个嵌套的参数字典来存储所有tiling因子
        # 结构: self.factors[level_name][dim_name]['temporal' or 'spatial']
        self.factors = nn.ModuleDict()

        # 只为片上存储（on-chip buffers）创建可学习的参数
        on_chip_levels = [level['name'] for level in hierarchy if level['type'] == 'buffer']
        
        for level_name in on_chip_levels:
            self.factors[level_name] = nn.ModuleDict()
            for dim_name in self.dims.keys():
                # 使用ParameterDict来正确注册参数
                # 初始化为1（在log空间中为0）
                self.factors[level_name][dim_name] = nn.ParameterDict({
                    'temporal': nn.Parameter(torch.zeros(1)), # log(1) = 0
                    'spatial': nn.Parameter(torch.zeros(1))
                })

    def get_factor(self, level_name, dim_name, factor_type):
        """获取指定level, dim, type的tiling因子。"""
        return torch.clamp(torch.exp(self.factors[level_name][dim_name][factor_type]), min=1.0)

    def get_all_factors(self):
        """
        NEW: Returns physically valid, integer tiling factors using differentiable projection.
        This method replaces get_all_factors() for performance evaluation.
        """
        projected_factors = {}
        on_chip_levels = [level['name'] for level in self.hierarchy if level['type'] == 'buffer']

        for dim_name, total_size in self.dims.items():
            projected_factors[dim_name] = {}
            product_of_on_chip_factors = 1.0
            
            # Project on-chip factors to valid integer divisors
            for level_name in on_chip_levels:
                continuous_temporal = self.get_factor(level_name, dim_name, 'temporal')
                continuous_spatial = self.get_factor(level_name, dim_name, 'spatial')
                
                # Apply differentiable projection
                problem_dim_tensor = torch.tensor(float(total_size))
                projected_temporal = ProjectToNearestDivisor.apply(continuous_temporal, problem_dim_tensor)
                projected_spatial = ProjectToNearestDivisor.apply(continuous_spatial, problem_dim_tensor)
                
                projected_factors[dim_name][level_name] = {
                    'temporal': projected_temporal,
                    'spatial': projected_spatial
                }
                product_of_on_chip_factors *= projected_temporal * projected_spatial

            # 推导DRAM层的temporal因子以满足约束
            dram_level_name = next(level['name'] for level in self.hierarchy if level['type'] == 'dram')
            dram_temporal_factor = total_size / product_of_on_chip_factors
            projected_dram_temporal = ProjectToNearestDivisor.apply(dram_temporal_factor, problem_dim_tensor)

            projected_factors[dim_name][dram_level_name] = {
                'temporal': projected_dram_temporal,
                'spatial': torch.tensor(1.0) # DRAM层没有空间并行
            }
        return projected_factors
        
# --- 其他类保持不变，但它们的调用方式会被新模型改变 ---
class FusionParameters(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.fusion_probs = nn.ParameterDict()
        for group in graph.fusion_groups:
            group_key = '__'.join(sorted(group))
            self.fusion_probs[group_key] = nn.Parameter(torch.randn(1))
    def get_fusion_probability(self, group):
        group_key = '__'.join(sorted(group))
        return torch.sigmoid(self.fusion_probs[group_key]).squeeze()

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

def calculate_macs(dims):
    return reduce(mul, dims.values(), 1.0)


class HighFidelityPerformanceModel(nn.Module):
    """
    NEW: 高保真性能模型，能够处理多级存储和细粒度映射。
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(self, graph: ComputationGraph, hw_params: HardwareParameters, fusion_params: FusionParameters, mapping: FineGrainedMapping) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_latency = torch.tensor(0.0)
        total_energy = torch.tensor(0.0)
        
        # 简化：假设整个网络共享一套mapping参数，实际可扩展为per-layer mapping
        all_factors = mapping.get_all_factors()

        for group in graph.fusion_groups:
            # 此处简化为对单个layer进行评估
            layer_name = group[0]
            layer = graph.layers[layer_name]
            macs = calculate_macs(layer['dims'])
            
            # --- Latency ---
            # Compute Latency
            num_pes = hw_params.get_projected_num_pes()
            compute_latency = macs / (num_pes * self.config.CLOCK_FREQUENCY_MHZ * 1e6 + 1e-9)
            
            # Memory Latency
            mem_latencies = []
            for i, level in enumerate(self.config.MEMORY_HIERARCHY):
                if level['type'] == 'dram':
                    # 计算DRAM访问量和时延
                    dram_accesses_bytes = self.calculate_accesses_bytes(layer['dims'], all_factors, i)
                    mem_latencies.append(dram_accesses_bytes / (level['bandwidth_gb_s'] * 1e9 + 1e-9))
            
            latency = torch.maximum(compute_latency, sum(mem_latencies))

            # --- Energy ---
            energy = torch.tensor(0.0)
            # PE energy
            energy += macs * self.config.PE_MAC_EPA_PJ
            
            # Memory energy
            for i, level in enumerate(self.config.MEMORY_HIERARCHY):
                accesses_bytes = self.calculate_accesses_bytes(layer['dims'], all_factors, i)
                accesses_4bytes = accesses_bytes / 4.0 # 假设一个access是4-byte
                
                if level['name'] == 'L1_Registers':
                    energy += accesses_4bytes * self.config.L1_REG_BASE_EPA_PJ
                elif level['name'] == 'L2_Scratchpad':
                    size_kb = hw_params.get_buffer_size_kb(level['name'])
                    epa = self.config.L2_SPM_BASE_EPA_PJ + self.config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb
                    energy += accesses_4bytes * epa
                elif level['name'] == 'L3_DRAM':
                    energy += accesses_4bytes * self.config.L3_DRAM_EPA_PJ

            # --- Capacity Penalty ---
            penalty = torch.tensor(0.0)
            for i, level in enumerate(self.config.MEMORY_HIERARCHY):
                 if level['type'] == 'buffer':
                    required_kb = self.calculate_buffer_req_kb(layer['dims'], all_factors, i)
                    available_kb = hw_params.get_buffer_size_kb(level['name'])
                    penalty += torch.relu(required_kb / available_kb - 1.0)

            total_latency += latency * (1 + penalty * self.config.PENALTY_WEIGHT)
            total_energy += energy * (1 + penalty * self.config.PENALTY_WEIGHT)

        return total_latency, total_energy, hw_params.get_area_cost()

    def calculate_accesses_bytes(self, dims, factors, level_idx):
        """简化版的访存计算，模拟从上一级存储读写。"""
        # 访问量 = tile大小 * 外层循环次数
        # tile大小在 level_idx-1 确定
        if level_idx == 0: return torch.tensor(0.0) # PE内部无访存
        
        # Tile size at level_idx-1
        tile_size_bytes = self.calculate_buffer_req_bytes(dims, factors, level_idx - 1)

        # Outer loop iterations
        outer_loops = torch.tensor(1.0)
        for i in range(level_idx, len(self.config.MEMORY_HIERARCHY)):
            level_name = self.config.MEMORY_HIERARCHY[i]['name']
            if level_name in factors[next(iter(dims))]:
                for dim_name in dims.keys():
                    outer_loops = outer_loops * factors[dim_name][level_name]['temporal'].squeeze()
        # 简化：假设input, weight, output traffic相同
        return tile_size_bytes * outer_loops * 3

    def calculate_buffer_req_bytes(self, dims, factors, level_idx):
        """计算在level_idx需要的buffer大小，由其内部所有level的tiling决定。"""
        # 简化：只计算input tensor的tile大小
        tile_dims = {}
        for dim_name in dims.keys():
            tile_dims[dim_name] = torch.tensor(1.0)
            for i in range(level_idx + 1):
                level_name = self.config.MEMORY_HIERARCHY[i]['name']
                if level_name in factors[dim_name]:
                    tile_dims[dim_name] = tile_dims[dim_name] * factors[dim_name][level_name]['temporal'].squeeze() * factors[dim_name][level_name]['spatial'].squeeze()
        
        return reduce(mul, tile_dims.values(), torch.tensor(1.0)) * self.config.BYTES_PER_ELEMENT
    
    def calculate_buffer_req_kb(self, dims, factors, level_idx):
        return self.calculate_buffer_req_bytes(dims, factors, level_idx) / 1024.0

def save_configuration_to_json(hw_params: HardwareParameters, 
                              projected_mapping: dict, 
                              fusion_params: FusionParameters, 
                              filepath: str = "final_configuration.json"):
    """
    Save the final optimized configuration to a structured JSON file.
    
    Args:
        hw_params: Hardware parameters object
        projected_mapping: The final projected mapping table
        fusion_params: Fusion parameters object
        filepath: Output JSON file path
    """
    config_dict = {
        "hardware": {},
        "mapping": {},
        "fusion": {}
    }
    
    # Populate hardware section
    config_dict["hardware"]["num_pes"] = int(hw_params.get_projected_num_pes().item())
    config_dict["hardware"]["buffer_sizes_kb"] = {}
    
    for buffer_name, log_size in hw_params.log_buffer_sizes_kb.items():
        config_dict["hardware"]["buffer_sizes_kb"][buffer_name] = float(torch.exp(log_size).item())
    
    # Populate mapping section
    config_dict["mapping"] = projected_mapping
    
    # Populate fusion section
    for group_key, logit_param in fusion_params.fusion_probs.items():
        fusion_prob = torch.sigmoid(logit_param).item()
        config_dict["fusion"][group_key] = float(fusion_prob)
    
    # Write to JSON file
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"Configuration saved to {filepath}")

# --- 主实验流程 ---
def run_experiment(num_iterations=100):
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
        pe_penalty_weight = 1.0

        # Combine all loss terms
        edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
        area_loss = 0.1 * area
        
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
    save_configuration_to_json(hw_params, final_mapping, fusion_params, "final_configuration.json")

if __name__ == "__main__":
    run_experiment(num_iterations=100)