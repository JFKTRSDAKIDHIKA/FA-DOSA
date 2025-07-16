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
    calculate_macs, save_configuration_to_json, get_divisors,
    OptimizationLogger
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

def run_experiment(num_outer_steps=10, num_mapping_steps=200, num_hardware_steps=50, lr_mapping=1e-2, lr_hardware=1e-2):
    # Initialize the optimization logger
    logger = OptimizationLogger("optimization_log.jsonl")
    print("--- Running High-Fidelity FA-DOSA Experiment ---")
    config = Config()
    graph = parse_onnx_to_graph("resnet.onnx")
    
    # Loss function strategy selection
    loss_strategy = 'strategy_A'  # Options: 'strategy_A', 'strategy_B', or 'original'
    edp_weight = 10.0  # For Strategy B: Amplified EDP Benefit

    hw_params = HardwareParameters()
    # 使用一个共享的mapping对象
    mapping = FineGrainedMapping(graph.problem_dims, config.MEMORY_HIERARCHY)
    fusion_params = FusionParameters(graph)
    perf_model = HighFidelityPerformanceModel(config)
    
    # Alternating optimization scheme
    for outer_step in range(num_outer_steps):
        print(f"\n--- Outer Step {outer_step + 1}/{num_outer_steps} ---")
        
        # Phase A: Optimize Mapping & Fusion (freeze hardware)
        print("--- Phase A: Optimizing Mapping & Fusion ---")
        
        # Freeze hardware parameters
        for p in hw_params.parameters():
            p.requires_grad = False
        # Unfreeze mapping and fusion parameters
        for p in list(mapping.parameters()) + list(fusion_params.parameters()):
            p.requires_grad = True
            
        # Create optimizer for mapping and fusion parameters
        map_fus_params = list(mapping.parameters()) + list(fusion_params.parameters())
        optimizer_map = optim.Adam(map_fus_params, lr=lr_mapping)
        
        for i in range(num_mapping_steps):
            optimizer_map.zero_grad()
            latency, energy, area, mismatch_loss = perf_model(graph, hw_params, fusion_params, mapping)
            
            # Calculate loss components
            continuous_pes = hw_params.get_num_pes()
            sqrt_pes = torch.sqrt(continuous_pes)
            pe_square_penalty = torch.pow(sqrt_pes - torch.round(sqrt_pes), 2)
            pe_penalty_weight = 0.08

            # Strategy-based loss calculation
            if loss_strategy == 'strategy_A':
                # Strategy A: Log-space Penalty ("Soft Wall")
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = 0.05 * area
                mismatch_penalty = torch.log(1.0 + mismatch_loss * config.MISMATCH_PENALTY_WEIGHT)
                loss = edp_loss + area_loss + mismatch_penalty + pe_square_penalty * pe_penalty_weight
            elif loss_strategy == 'strategy_B':
                # Strategy B: Amplified EDP Benefit
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = 0.05 * area
                loss = edp_weight * edp_loss + area_loss + mismatch_loss * config.MISMATCH_PENALTY_WEIGHT + pe_square_penalty * pe_penalty_weight
            else:  # Fallback to original
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = 0.05 * area
                loss = edp_loss + area_loss + mismatch_loss * config.MISMATCH_PENALTY_WEIGHT + pe_square_penalty * pe_penalty_weight
            loss.backward()
            optimizer_map.step()
            # Anneal temperature for Gumbel-Softmax after each step
            mapping.anneal_tau()
            
            if i % 10 == 0:
                if loss_strategy == 'strategy_A':
                    print(f"[Map] Iter {i}: Loss={loss.item():.4f}, EDP={edp_loss.item():.4f}, Area={area_loss.item():.4f}, Mismatch_Penalty={mismatch_penalty.item():.4f}, PE_Penalty={pe_square_penalty.item():.4f}")
                else:
                    print(f"[Map] Iter {i}: Loss={loss.item():.4f}, EDP={edp_loss.item():.4f}, Area={area_loss.item():.4f}, Mismatch={mismatch_loss.item():.4f}, PE_Penalty={pe_square_penalty.item():.4f}")
                print(f"         Latency={latency.item():.2e}s, Energy={energy.item():.2e}pJ, Area={area.item():.2f}mm²")
                
                # Log optimization data
                log_data = {
                    'phase': 'A: Mapping',
                    'outer_step': outer_step,
                    'inner_step': i,
                    'loss_total': loss,
                    'loss_components': {
                        'edp': edp_loss,
                        'area': area_loss,
                        'mismatch': mismatch_loss if loss_strategy != 'strategy_A' else mismatch_penalty,
                        'pe_penalty': pe_square_penalty
                    },
                    'performance_metrics': {
                        'latency_sec': latency,
                        'energy_pj': energy,
                        'area_mm2': area
                    },
                    'hardware_params': {
                        'num_pes': hw_params.get_num_pes(),
                        'projected_num_pes': hw_params.get_projected_num_pes(),
                        'l1_size_kb': hw_params.get_buffer_size_kb('L1_Registers'),
                        'l2_size_kb': hw_params.get_buffer_size_kb('L2_Scratchpad')
                    },
                    'mapping_params_snapshot': mapping.get_all_factors(),
                    'gumbel_tau': mapping.projector.tau
                }
                logger.log_step(log_data)
        
        # Phase B: Optimize Hardware (freeze mapping and fusion)
        print("--- Phase B: Optimizing Hardware ---")
        
        # Freeze mapping and fusion parameters
        for p in list(mapping.parameters()) + list(fusion_params.parameters()):
            p.requires_grad = False
        # Unfreeze hardware parameters
        for p in hw_params.parameters():
            p.requires_grad = True
            
        # Create optimizer for hardware parameters
        optimizer_hw = optim.Adam(hw_params.parameters(), lr=lr_hardware)
        
        for i in range(num_hardware_steps):
            optimizer_hw.zero_grad()
            latency, energy, area, mismatch_loss = perf_model(graph, hw_params, fusion_params, mapping)
            
            # Calculate loss components
            continuous_pes = hw_params.get_num_pes()
            sqrt_pes = torch.sqrt(continuous_pes)
            pe_square_penalty = torch.pow(sqrt_pes - torch.round(sqrt_pes), 2)
            pe_penalty_weight = 0.08

            # Strategy-based loss calculation (Phase B: area weight = 0.00)
            if loss_strategy == 'strategy_A':
                # Strategy A: Log-space Penalty ("Soft Wall")
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = 0.00 * area
                mismatch_penalty = torch.log(1.0 + mismatch_loss * config.MISMATCH_PENALTY_WEIGHT)
                loss = edp_loss + area_loss + mismatch_penalty + pe_square_penalty * pe_penalty_weight
            elif loss_strategy == 'strategy_B':
                # Strategy B: Amplified EDP Benefit
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = 0.00 * area
                loss = edp_weight * edp_loss + area_loss + mismatch_loss * config.MISMATCH_PENALTY_WEIGHT + pe_square_penalty * pe_penalty_weight
            else:  # Fallback to original
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = 0.00 * area
                loss = edp_loss + area_loss + mismatch_loss * config.MISMATCH_PENALTY_WEIGHT + pe_square_penalty * pe_penalty_weight
            loss.backward()
            optimizer_hw.step()
            
            if i % 10 == 0:
                if loss_strategy == 'strategy_A':
                    print(f"[HW] Iter {i}: Loss={loss.item():.4f}, EDP={edp_loss.item():.4f}, Area={area_loss.item():.4f}, Mismatch_Penalty={mismatch_penalty.item():.4f}, PE_Penalty={pe_square_penalty.item():.4f}")
                else:
                    print(f"[HW] Iter {i}: Loss={loss.item():.4f}, EDP={edp_loss.item():.4f}, Area={area_loss.item():.4f}, Mismatch={mismatch_loss.item():.4f}, PE_Penalty={pe_square_penalty.item():.4f}")
                print(f"         Latency={latency.item():.2e}s, Energy={energy.item():.2e}pJ, Area={area.item():.2f}mm²")
                
                # Log optimization data
                log_data = {
                    'phase': 'B: Hardware',
                    'outer_step': outer_step,
                    'inner_step': i,
                    'loss_total': loss,
                    'loss_components': {
                        'edp': edp_loss,
                        'area': area_loss,
                        'mismatch': mismatch_loss if loss_strategy != 'strategy_A' else mismatch_penalty,
                        'pe_penalty': pe_square_penalty
                    },
                    'performance_metrics': {
                        'latency_sec': latency,
                        'energy_pj': energy,
                        'area_mm2': area
                    },
                    'hardware_params': {
                        'num_pes': hw_params.get_num_pes(),
                        'projected_num_pes': hw_params.get_projected_num_pes(),
                        'l1_size_kb': hw_params.get_buffer_size_kb('L1_Registers'),
                        'l2_size_kb': hw_params.get_buffer_size_kb('L2_Scratchpad')
                    },
                    'mapping_params_snapshot': mapping.get_all_factors(),
                    'gumbel_tau': mapping.projector.tau
                }
                logger.log_step(log_data)
    
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
    
    # Close the logger
    logger.close()

if __name__ == "__main__":
    run_experiment(num_outer_steps=5, num_mapping_steps=50, num_hardware_steps=50)