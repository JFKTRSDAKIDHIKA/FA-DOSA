"""
FA-DOSA Comprehensive Experiment Runner

This script implements multiple algorithms for hardware-software co-design evaluation:

1. FA-DOSA: Joint optimization of hardware parameters, mapping, and fusion decisions
2. Random Search: Baseline with random parameter sampling 
3. Decoupled SOTA: State-of-the-art decoupled design flow baseline
   - Step 1: Comprehensive discrete hardware DSE with mapping optimization
   - Step 2: Intelligent fusion heuristics on optimal hardware
4. Two-Step Baseline: Simple decoupled optimization (deprecated, kept for comparison)

ENHANCED FOR 2025: Now supports both CNN and Transformer architectures including
BERT, Vision Transformers, and other attention-based models alongside traditional
ResNet, VGG, and MobileNet architectures.

The Decoupled SOTA baseline represents what a skilled hardware architect would achieve
using traditional decoupled design methodologies, making it a strong baseline for
demonstrating FA-DOSA's joint optimization advantages across diverse AI workloads.
"""

import csv
import os
import json
import re
from typing import Dict, List
import torch
import torch.optim as optim
import time
import numpy as np
import random
import torch.nn as nn
import yaml

from fa_dosa_demo import (
    HardwareParameters,
    MappingParameters,
    FusionParameters,
    Config,
    ComputationGraph,
    ConditionalPerformanceModel,
    calculate_total_loss_with_hardware_constraints,
    create_example_optimization_setup,
    calculate_penalty_loss,
    calculate_template_constraint_penalty,
    calculate_fused_group_buffer_req,
)
from onnx_frontend import parse_onnx_to_graph


# ==============================================================================
# UNIFIED FINAL PERFORMANCE EVALUATION FUNCTION
# ==============================================================================

def log_final_configuration(
    algo_name: str,
    workload: str,
    trial_num: int,
    hardware_params: HardwareParameters,
    mapping_params: MappingParameters,
    fusion_params: FusionParameters,
    final_metrics: Dict
) -> None:
    """
    生成详细的最终配置报告，用于后续的外部工具验证（如Timeloop）。
    
    Args:
        algo_name: 算法名称
        workload: 工作负载名称
        trial_num: 试验编号
        hardware_params: 最终硬件参数
        mapping_params: 最终映射参数
        fusion_params: 最终融合参数
        final_metrics: 最终性能指标
    """
    print("\n🔍 Generating configuration report for {}...".format(algo_name))
    print("=" * 80)
    print(f"FINAL CONFIGURATION REPORT")
    print(f"Algorithm: {algo_name}")
    print(f"Workload: {workload}")
    print(f"Trial: {trial_num}")
    print("=" * 80)
    
    # === 硬件配置部分 ===
    print(f"\n📊 HARDWARE CONFIGURATION:")
    print(f"  Processing Elements: {hardware_params.get_num_pes().item():.0f}")
    print(f"  Buffer Size (KB): {hardware_params.get_buffer_size_kb().item():.2f}")
    print(f"  Buffer Size (Bytes): {hardware_params.get_buffer_size_bytes().item():.0f}")
    print(f"  Total Area (mm²): {hardware_params.get_area_cost().item():.4f}")
    
    # === 映射策略配置部分 ===
    print(f"\n🗺️  MAPPING CONFIGURATION:")
    try:
        decision_modules = mapping_params.get_all_decision_modules()
        for group_key, decision_module in decision_modules.items():
            print(f"  Group: {group_key}")
            
            # 显示模板选择概率
            template_probs = decision_module.get_template_probabilities()
            for i, template_name in enumerate(decision_module.template_names):
                prob = template_probs[i].item()
                print(f"    Template {template_name}: {prob:.4f}")
            
            # 显示选中模板的参数
            try:
                selected_template = decision_module.get_selected_template()
                print(f"    Selected Template Parameters:")
                for param_name, param_tensor in selected_template.named_parameters():
                    if 'log_' in param_name:
                        actual_value = torch.exp(param_tensor).item()
                        clean_name = param_name.replace('log_', '')
                        print(f"      {clean_name}: {actual_value:.4f}")
                    else:
                        print(f"      {param_name}: {param_tensor.item():.4f}")
            except Exception as e:
                print(f"    Selected Template Parameters: [Error retrieving: {e}]")
    except Exception as e:
        print(f"  [Error retrieving mapping configuration: {e}]")
    
    # === 融合决策配置部分 ===
    print(f"\n🔀 FUSION CONFIGURATION:")
    try:
        for sanitized_key, prob_tensor in fusion_params.fusion_probs.items():
            # 获取原始组名
            if hasattr(fusion_params, 'group_name_mapping') and sanitized_key in fusion_params.group_name_mapping:
                original_group = fusion_params.group_name_mapping[sanitized_key]
                group_display = ' + '.join(original_group)
            else:
                group_display = sanitized_key
                
            prob_value = torch.sigmoid(prob_tensor).item()
            decision = "FUSE" if prob_value > 0.5 else "NO_FUSE"
            print(f"  {group_display}: {prob_value:.4f} → {decision}")
    except Exception as e:
        print(f"  [Error retrieving fusion configuration: {e}]")
    
    # === 性能指标部分 ===
    print(f"\n⚡ PERFORMANCE METRICS:")
    for metric_name, metric_value in final_metrics.items():
        if isinstance(metric_value, (int, float)):
            if 'edp' in metric_name.lower():
                print(f"  {metric_name}: {metric_value:.4e}")
            elif 'area' in metric_name.lower():
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value:.0f}")
        else:
            print(f"  {metric_name}: {metric_value}")
    
    print("=" * 80)
    print("🎯 Configuration report complete - Ready for Timeloop validation!")
    print("=" * 80)


def save_config_to_json_file(
    algorithm_name: str,
    workload: str,
    trial_num: int,
    hardware_params: HardwareParameters,
    mapping_params: MappingParameters,
    fusion_params: FusionParameters,
    final_metrics: Dict,
    output_dir: str = './final_configs'
) -> str:
    """
    将最终配置参数保存为结构化的JSON文件，用于后续的自动化验证流程。
    
    Args:
        algorithm_name: 算法名称
        workload: 工作负载名称  
        trial_num: 试验编号
        hardware_params: 最终硬件参数
        mapping_params: 最终映射参数
        fusion_params: 最终融合参数
        final_metrics: 最终性能指标
        output_dir: 输出目录路径
    
    Returns:
        str: 保存的JSON文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成清理后的文件名
    def sanitize_filename(name: str) -> str:
        """清理文件名，移除不安全字符"""
        # 移除文件扩展名
        name = re.sub(r'\.(onnx|json)$', '', name, flags=re.IGNORECASE)
        # 替换不安全字符为下划线
        name = re.sub(r'[^\w\-_]', '_', name)
        # 移除多余的下划线
        name = re.sub(r'_+', '_', name)
        # 移除开头和结尾的下划线
        name = name.strip('_')
        return name.lower()
    
    workload_sanitized = sanitize_filename(workload)
    algorithm_sanitized = sanitize_filename(algorithm_name)
    filename = f"{workload_sanitized}_{algorithm_sanitized}_trial_{trial_num}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 构建配置字典
    config_dict = {
        "metadata": {
            "workload": workload,
            "algorithm": algorithm_name,
            "trial": trial_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "hardware_configuration": {
            "processing_elements": int(hardware_params.get_num_pes().item()),
            "buffer_size_kb": round(hardware_params.get_buffer_size_kb().item(), 2),
            "buffer_size_bytes": int(hardware_params.get_buffer_size_bytes().item()),
            "total_area_mm2": round(hardware_params.get_area_cost().item(), 4)
        },
        "fusion_decisions": [],
        "mapping_strategy": {},
        "performance_metrics": {}
    }
    
    # 添加融合决策信息
    try:
        for sanitized_key, prob_tensor in fusion_params.fusion_probs.items():
            # 获取原始组名
            if hasattr(fusion_params, 'group_name_mapping') and sanitized_key in fusion_params.group_name_mapping:
                original_group = fusion_params.group_name_mapping[sanitized_key]
                group_display = original_group  # 保持列表格式
            else:
                group_display = [sanitized_key]  # 转换为列表格式
                
            prob_value = torch.sigmoid(prob_tensor).item()
            fused = prob_value > 0.5
            
            fusion_info = {
                "group": group_display,
                "fused": fused,
                "probability": round(prob_value, 4)
            }
            config_dict["fusion_decisions"].append(fusion_info)
    except Exception as e:
        print(f"  [Warning] Could not extract fusion decisions: {e}")
    
    # 添加映射策略信息
    try:
        decision_modules = mapping_params.get_all_decision_modules()
        for group_key, decision_module in decision_modules.items():
            mapping_info = {
                "selected_template": None,
                "probabilities": {},
                "parameters": {}
            }
            
            # 获取模板选择概率
            template_probs = decision_module.get_template_probabilities()
            max_prob = 0
            selected_template_name = None
            
            for i, template_name in enumerate(decision_module.template_names):
                prob = template_probs[i].item()
                mapping_info["probabilities"][template_name] = round(prob, 4)
                
                if prob > max_prob:
                    max_prob = prob
                    selected_template_name = template_name
            
            mapping_info["selected_template"] = selected_template_name
            
            # 获取选中模板的参数
            try:
                selected_template = decision_module.get_selected_template()
                for param_name, param_tensor in selected_template.named_parameters():
                    if 'log_' in param_name:
                        actual_value = torch.exp(param_tensor).item()
                        clean_name = param_name.replace('log_', '')
                        mapping_info["parameters"][clean_name] = round(actual_value, 4)
                    else:
                        mapping_info["parameters"][param_name] = round(param_tensor.item(), 4)
            except Exception as e:
                print(f"  [Warning] Could not extract parameters for {group_key}: {e}")
            
            config_dict["mapping_strategy"][group_key] = mapping_info
    except Exception as e:
        print(f"  [Warning] Could not extract mapping strategy: {e}")
    
    # 添加性能指标
    for metric_name, metric_value in final_metrics.items():
        if metric_name != '_final_params':  # 跳过参数对象
            if isinstance(metric_value, (int, float)):
                if 'edp' in metric_name.lower():
                    config_dict["performance_metrics"][metric_name] = f"{metric_value:.4e}"
                elif 'area' in metric_name.lower():
                    config_dict["performance_metrics"][metric_name] = round(metric_value, 4)
                elif isinstance(metric_value, float):
                    config_dict["performance_metrics"][metric_name] = round(metric_value, 2)
                else:
                    config_dict["performance_metrics"][metric_name] = metric_value
            else:
                config_dict["performance_metrics"][metric_name] = str(metric_value)
    
    # 保存JSON文件
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        return filepath
    except Exception as e:
        print(f"  [Error] Failed to save JSON file: {e}")
        return None


def evaluate_final_design_point(final_mapping_params, final_fusion_params, hardware_params, graph: ComputationGraph, config):
    """
    Evaluates a final, deterministic design point using a fresh performance model.
    This function acts as the "official referee" to ensure all algorithms are
    compared on a level playing field.

    Args:
        final_mapping_params (MappingParameters): The final mapping parameters.
        final_fusion_params (FusionParameters): The final fusion parameters, which may
                                                contain probabilistic values.
        hardware_params (HardwareParameters): The final hardware parameters.
        graph (ComputationGraph): The computational graph.
        config (Config): The experiment configuration.

    Returns:
        dict: A dictionary of final, scalar performance metrics (.item()'d).
    """
    print("      [EVAL] Running unified final evaluation...")
    
    # 1. Instantiate a new, clean performance model.
    model = ConditionalPerformanceModel()

    # 2. Create deterministic fusion decisions. This is CRITICAL.
    # We round the sigmoid of the fusion logits to get hard 0 or 1 decisions.
    deterministic_fusion_params = FusionParameters(graph)
    with torch.no_grad():
        # --- FIX: Correctly handle ParameterDict iteration ---
        # fusion_probs is a ParameterDict, so we need to iterate over items() to get key-value pairs
        for sanitized_key, prob_logit in final_fusion_params.fusion_probs.items():
            # This simulates the final hardware implementation where a fusion is either
            # definitively performed or not.
            decision = torch.round(torch.sigmoid(prob_logit))
            deterministic_fusion_params.fusion_probs[sanitized_key] = nn.Parameter(decision)

    # 3. Perform the forward pass with the deterministic design point.
    with torch.no_grad():
        latency, energy, area = model(final_mapping_params, deterministic_fusion_params, hardware_params, graph)
        
        # Calculate penalty separately for reporting
        penalty = calculate_penalty_loss(final_mapping_params)
    
    # 4. Calculate final EDP.
    edp = latency * energy

    # --- FIX: Add numerical stability assertions for final evaluation ---
    # These help us determine if problems occur during optimization or final evaluation
    assert not torch.isinf(latency), f"Final latency is infinite! Value: {latency}"
    assert not torch.isnan(latency), f"Final latency is NaN! This indicates evaluation problems."
    assert not torch.isinf(energy), f"Final energy is infinite! Value: {energy}"
    assert not torch.isnan(energy), f"Final energy is NaN! This indicates evaluation problems."
    assert not torch.isinf(area), f"Final area is infinite! Value: {area}"
    assert not torch.isnan(area), f"Final area is NaN! This indicates evaluation problems."
    assert not torch.isinf(edp), f"Final EDP is infinite! Value: {edp}"
    assert not torch.isnan(edp), f"Final EDP is NaN! This indicates evaluation problems."
    
    # Additional safety checks for physically reasonable values
    assert latency > 0, f"Final latency must be positive! Got: {latency}"
    assert energy > 0, f"Final energy must be positive! Got: {energy}"
    assert area > 0, f"Final area must be positive! Got: {area}"

    # 5. Return a dictionary of pure scalar values for clean logging.
    results = {
        'final_edp': edp.item(),
        'final_latency': latency.item(),
        'final_energy': energy.item(),
        'final_area': area.item(),
        'final_penalty': penalty.item()
    }
    print(f"      [EVAL] Unified evaluation complete. Final EDP: {results['final_edp']:.4e}")
    return results


# ==============================================================================
# ALGORITHM IMPLEMENTATIONS
# ==============================================================================

def authoritative_evaluation_model(final_params: dict, graph: 'ComputationGraph', config: 'Config') -> dict:
    mapping_params = final_params['mapping_params']
    fusion_params = final_params['fusion_params']
    hardware_params = final_params['hardware_params']
    performance_model = ConditionalPerformanceModel()
    with torch.no_grad():
        final_fusion_params = FusionParameters(graph)
        for sanitized_key, prob_logit in fusion_params.fusion_probs.items():
            hard_decision = torch.round(torch.sigmoid(prob_logit))
            final_fusion_params.fusion_probs[sanitized_key] = nn.Parameter(hard_decision, requires_grad=False)
        latency, energy, area = performance_model(mapping_params, final_fusion_params, hardware_params, graph)
        edp = latency * energy
    return {
        'final_loss': edp.item(),  # 使用EDP作为最终损失
        'final_edp': edp.item(),
        'final_latency': latency.item(),
        'final_energy': energy.item(),
        'final_area': area.item(),
    }

def run_fadose_experiment(graph: ComputationGraph, config: Config, num_iterations=300, lr=1e-3, workload: str = None, trial_num: int = None) -> dict:
    """
    🚀 FA-DOSA Co-optimization Experiment
    """
    print("🚀 Running FA-DOSA Co-optimization Experiment...")
    mapping_params, fusion_params, hardware_params = create_example_optimization_setup(graph)
    performance_model = ConditionalPerformanceModel()
    optimizer = optim.Adam(
        list(mapping_params.parameters()) + 
        list(fusion_params.parameters()) + 
        list(hardware_params.parameters()), 
        lr=lr
    )
    
    # === STEP 3: Introduce Learning Rate Scheduler for Adaptive Optimization ===
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"   🔄 Starting joint optimization: {num_iterations} iterations...")
    for i in range(num_iterations):
        optimizer.zero_grad()
        total_loss = calculate_total_loss_with_hardware_constraints(performance_model, mapping_params, fusion_params, hardware_params, graph)
        total_loss.backward()
        
        # === STEP 4: Implement Gradient Clipping for Stability ===
        torch.nn.utils.clip_grad_norm_(
            list(mapping_params.parameters()) + 
            list(fusion_params.parameters()) + 
            list(hardware_params.parameters()), 
            max_norm=1.0
        )
        
        optimizer.step()
        
        # === STEP 3: Add scheduler step after optimizer step ===
        scheduler.step(total_loss)
        
        if i % 20 == 0:
            print(f"    Epoch {i}/{num_iterations}, Loss: {total_loss.item():.4f}")
    print("\n   [🏛️ Authoritative Evaluation] Initiating unified final performance assessment...")
    final_params = {'mapping_params': mapping_params, 'fusion_params': fusion_params, 'hardware_params': hardware_params}
    authoritative_results = authoritative_evaluation_model(final_params, graph, config)
    if workload and trial_num is not None:
        save_learned_parameters(workload, trial_num, mapping_params, fusion_params, hardware_params, authoritative_results)
    print("   ✅ FA-DOSA experiment complete.")
    
    # Add parameter objects to results for configuration reporting
    authoritative_results['_final_params'] = {
        'hardware_params': hardware_params,
        'mapping_params': mapping_params,
        'fusion_params': fusion_params
    }
    return authoritative_results

def run_fadose_constrained_experiment(graph: ComputationGraph, config: Config, num_iterations=50, lr=1e-3, workload: str = None, trial_num: int = None) -> dict:
    """
    FA-DOSA with fixed hardware.
    """
    print("\nRunning FA-DOSA (Constrained Hardware) Experiment...")
    mapping_params, fusion_params, hardware_params = create_example_optimization_setup(graph)
    target_area = config._config.get('area_model', {}).get('target_area', 25.0)
    with torch.no_grad():
        initial_area = hardware_params.get_area_cost().item()
        scaling_factor = (target_area / initial_area) ** 0.5 if initial_area > 0 else 1.0
        hardware_params.log_num_pes.data *= scaling_factor
        # Scale both accumulator and scratchpad sizes proportionally
        hardware_params.log_accumulator_size_kb.data *= scaling_factor
        hardware_params.log_scratchpad_size_kb.data *= scaling_factor
    for param in hardware_params.parameters():
        param.requires_grad = False
    print(f"  [CONSTRAINED] Target area: {target_area:.2f} mm²")
    performance_model = ConditionalPerformanceModel()
    optimizer = optim.Adam(
        list(mapping_params.parameters()) + list(fusion_params.parameters()), 
        lr=lr
    )
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = calculate_total_loss_with_hardware_constraints(
            performance_model, mapping_params, fusion_params, hardware_params, graph
        )
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"    Epoch {i}/{num_iterations}, Loss: {loss.item():.4f}")
    final_params = {'mapping_params': mapping_params, 'fusion_params': fusion_params, 'hardware_params': hardware_params}
    authoritative_results = authoritative_evaluation_model(final_params, graph, config)
    print(f"  [SUCCESS] Constrained experiment completed.")
    
    # Add parameter objects to results for configuration reporting
    authoritative_results['_final_params'] = {
        'hardware_params': hardware_params,
        'mapping_params': mapping_params,
        'fusion_params': fusion_params
    }
    return authoritative_results


def run_random_search_experiment(graph: ComputationGraph, config: Config, num_samples=50) -> dict:
    """
    Random search baseline experiment.
    Randomly samples mapping and fusion parameters to find the best configuration.
    
    Args:
        graph: The ComputationGraph for the workload.
        config: The configuration dictionary.
        
    Returns:
        A dictionary containing the best results found.
    """
    print("Running Random Search...")
    
    best_loss = float('inf')
    best_results = None
    best_params = {} # Store the best parameters found
    
    model = ConditionalPerformanceModel()
    
    for sample in range(num_samples):
        # Generate random parameters
        mapping_params = MappingParameters(graph)
        fusion_params = FusionParameters(graph)
        # Split buffer size into accumulator and scratchpad
        total_buffer_size = np.random.randint(64, 4097)
        accumulator_size = total_buffer_size // 4  # 25% for accumulator
        scratchpad_size = total_buffer_size - accumulator_size  # 75% for scratchpad
        
        hardware_params = HardwareParameters(
            initial_num_pes=np.random.randint(16, 1025),
            initial_accumulator_kb=accumulator_size,
            initial_scratchpad_kb=scratchpad_size
        )
        
        # Randomize mapping parameters (template parameters)
        with torch.no_grad():
            for group_key, decision_module in mapping_params.get_all_decision_modules().items():
                # Randomize template selection logits
                decision_module.template_selection_logits.uniform_(-3.0, 3.0)
                
                # Randomize template parameters for each template
                for template_name, template in decision_module.templates.items():
                    for param_name, param_tensor in template.named_parameters():
                        if 'log_' in param_name:  # log-scale parameters
                            # Random log-scale factors (corresponds to factors between ~0.1 and 10.0)
                            param_tensor.uniform_(-2.3, 2.3)  # exp(-2.3) ≈ 0.1, exp(2.3) ≈ 10.0
        
        # Randomize fusion parameters
        with torch.no_grad():
            for sanitized_key, prob in fusion_params.fusion_probs.items():
                # Random logit between -3 and 3 (gives sigmoid range ~0.05 to 0.95)
                prob.fill_(torch.rand(1).item() * 6.0 - 3.0)
        
        try:
            # Evaluate this configuration
            latency, energy, area_cost = model(mapping_params, fusion_params, hardware_params, graph)
            penalty = calculate_penalty_loss(mapping_params)
            template_constraint_penalty = calculate_template_constraint_penalty(mapping_params, hardware_params, graph)
            
            # Calculate loss in log domain (matching FA-DOSA)
            log_latency = torch.log(latency + 1e-9)
            log_energy = torch.log(energy + 1e-9)
            log_area = torch.log(area_cost + 1e-9)
            
            # --- UPDATED: Use template-based constraint penalties ---
            total_loss = (
                config.LATENCY_WEIGHT * log_latency + 
                config.ENERGY_WEIGHT * log_energy + 
                config.AREA_WEIGHT * log_area + 
                config.PENALTY_WEIGHT * penalty + 
                config.PENALTY_WEIGHT * template_constraint_penalty
            )
            
            # Keep track of best configuration
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_params = {
                    'mapping': mapping_params,
                    'fusion': fusion_params,
                    'hardware': hardware_params,
                }
                
            if sample % 10 == 0 and sample > 0:
                print(f"  Sample {sample}/{num_samples}: Best loss so far = {best_loss:.4f}")
                    
        except Exception as e:
            # Skip invalid configurations
            continue
    
    if best_results is None:
        # Fallback if no valid configuration found
        print("  Warning: No valid random configurations found, using fallback values")
        return {
            'final_loss': 99999.0, 'final_edp': 9e9, 'final_latency': 9e9, 
            'final_energy': 9e9, 'final_area': 9e9
        }
    
    print(f"  Random Search completed. Best loss: {best_loss:.4f}")
    
    # --- FIX: Use the unified evaluation function for final physical metrics ---
    # The best parameters found by the random search are now evaluated by the
    # "official referee" to ensure a fair comparison.
    final_metrics = evaluate_final_design_point(
        best_params['mapping'], best_params['fusion'], best_params['hardware'], graph, config
    )
    
    results = {
        'final_loss': best_loss,
        **final_metrics
    }
    
    # Add parameter objects to results for configuration reporting
    results['_final_params'] = {
        'hardware_params': best_params['hardware'],
        'mapping_params': best_params['mapping'],
        'fusion_params': best_params['fusion']
    }
    
    return results

def run_decoupled_sota_experiment(graph: ComputationGraph, config: Config) -> dict:
    """
    Decoupled State-of-the-Art baseline that simulates realistic decoupled design flow.
    This represents what a skilled hardware architect would do using traditional methods.
    
    Step 1: Comprehensive discrete hardware DSE with mapping optimization for each config
    Step 2: Intelligent fusion heuristics on the selected optimal hardware
    
    Args:
        graph: The ComputationGraph for the workload.
        config: The configuration dictionary.
        
    Returns:
        A dictionary containing the final performance metrics.
    """
    print("Running Decoupled SOTA Baseline...")
    
    # === Step 1: Discrete Hardware Design Space Exploration ===
    print("  Step 1: Comprehensive Hardware DSE")
    
    # Define realistic discrete hardware search space (reduced for debugging)
    buffer_sizes_kb = [256, 512]  # KB
    pe_counts = [128, 256]  # Number of processing elements
    
    best_hw_config = None
    best_hw_edp = float('inf')
    best_hw_mapping_params = None
    
    total_hw_configs = len(buffer_sizes_kb) * len(pe_counts)
    current_config = 0
    
    print(f"    Exploring {total_hw_configs} hardware configurations...")
    
    for buffer_size_kb in buffer_sizes_kb:
        for pe_count in pe_counts:
            current_config += 1
            print(f"    [{current_config}/{total_hw_configs}] Testing: {pe_count} PEs, {buffer_size_kb} KB buffer")
            
            # Create a fixed hardware configuration for this iteration
            # Use mock HardwareParameters to represent fixed hardware choice
            # Split buffer size into accumulator and scratchpad
            accumulator_size = buffer_size_kb // 4  # 25% for accumulator
            scratchpad_size = buffer_size_kb - accumulator_size  # 75% for scratchpad
            
            mock_hardware_params = HardwareParameters(
                initial_num_pes=pe_count,
                initial_accumulator_kb=accumulator_size,
                initial_scratchpad_kb=scratchpad_size
            )
            # Fix the parameters to prevent learning (simulate fixed hardware)
            with torch.no_grad():
                mock_hardware_params.log_num_pes.requires_grad_(False)
                mock_hardware_params.log_accumulator_size_kb.requires_grad_(False)
                mock_hardware_params.log_scratchpad_size_kb.requires_grad_(False)
            
            # === Mapping optimization for this fixed hardware ===
            mapping_params = MappingParameters(graph)
            model = ConditionalPerformanceModel()
            
            # Optimizer only for mapping (hardware is fixed)
            optimizer = optim.Adam(mapping_params.parameters(), lr=0.01)
            
            num_epochs_mapping = 20  # Focused optimization for each hardware config (reduced for debugging)
            
            for epoch in range(num_epochs_mapping):
                optimizer.zero_grad()
                
                # Calculate cost assuming no fusion (fusion-agnostic mapping)
                total_latency_nf = torch.tensor(0.0)
                total_energy_nf = torch.tensor(0.0)

                for layer_name in graph.get_layer_names():
                    lat, comp_eng, sram_eng, dram_eng, noc_eng = model._calculate_analytical_costs(
                        layer_name, mapping_params, graph, mock_hardware_params
                    )
                    total_latency_nf += lat
                    total_energy_nf += comp_eng + sram_eng + dram_eng + noc_eng
                    
                # Calculate log values for loss function
                epsilon = 1e-9  # For numerical stability
                log_latency = torch.log(total_latency_nf + epsilon)
                log_energy = torch.log(total_energy_nf + epsilon)
                log_area = torch.log(mock_hardware_params.get_area_cost() + epsilon)
                
                # Calculate penalties
                penalty = calculate_penalty_loss(mapping_params)
                template_constraint_penalty = calculate_template_constraint_penalty(mapping_params, mock_hardware_params, graph)
                
                # Loss function focused on EDP optimization
                total_cost = (
                    config.LATENCY_WEIGHT * log_latency + 
                    config.ENERGY_WEIGHT * log_energy + 
                    config.AREA_WEIGHT * log_area +  # Now uses config weight
                    config.PENALTY_WEIGHT * penalty +
                    config.PENALTY_WEIGHT * template_constraint_penalty
                )
                
                total_cost.backward()
                optimizer.step()
            
            # Evaluate final EDP for this hardware configuration
            final_latency_nf = torch.tensor(0.0)
            final_energy_nf = torch.tensor(0.0)
            
            for layer_name in graph.get_layer_names():
                lat, comp_eng, sram_eng, dram_eng, noc_eng = model._calculate_analytical_costs(
                    layer_name, mapping_params, graph, mock_hardware_params
                )
                final_latency_nf += lat
                final_energy_nf += comp_eng + sram_eng + dram_eng + noc_eng
            
            final_edp = final_latency_nf * final_energy_nf
            
            print(f"      -> EDP: {final_edp.item():.2e}")
            
            # Track best hardware configuration
            if final_edp.item() < best_hw_edp:
                best_hw_edp = final_edp.item()
                best_hw_config = {
                    'buffer_size_kb': buffer_size_kb,
                    'pe_count': pe_count,
                    'hardware_params': mock_hardware_params,
                    'edp': final_edp.item()
                }
                # Deep copy the best mapping parameters
                best_hw_mapping_params = MappingParameters(graph)
                with torch.no_grad():
                    for group_key, curr_decision_module in mapping_params.get_all_decision_modules().items():
                        best_decision_module = best_hw_mapping_params.get_all_decision_modules()[group_key]
                        
                        # Copy template selection logits
                        best_decision_module.template_selection_logits.copy_(curr_decision_module.template_selection_logits)
                        
                        # Copy template parameters
                        for template_name, curr_template in curr_decision_module.templates.items():
                            best_template = best_decision_module.templates[template_name]
                            for (best_param_name, best_param), (curr_param_name, curr_param) in zip(
                                best_template.named_parameters(),
                                curr_template.named_parameters()
                            ):
                                best_param.copy_(curr_param)
    
    print(f"  Step 1 Complete. Best HW: {best_hw_config['pe_count']} PEs, "
          f"{best_hw_config['buffer_size_kb']} KB buffer (EDP: {best_hw_config['edp']:.2e})")
    
    # === Step 2: Intelligent Fusion Strategy on Fixed Optimal Hardware ===
    print("  Step 2: Intelligent Fusion Strategy")
    
    optimal_hw_config = best_hw_config['hardware_params']
    optimal_mapping_params = best_hw_mapping_params
    
    # Calculate non-fused baseline costs for each layer
    layer_costs = {}
    for layer_name in graph.get_layer_names():
        lat, comp_eng, sram_eng, dram_eng, noc_eng = model._calculate_analytical_costs(
            layer_name, optimal_mapping_params, graph, optimal_hw_config
        )
        layer_costs[layer_name] = {
            'latency': lat,
            'energy': comp_eng + sram_eng + dram_eng + noc_eng
        }
    
    # Intelligent fusion decision for each potential fusion group
    selected_fusion_groups = []
    
    for group in graph.fusion_groups:
        print(f"    Evaluating fusion group: {group}")
        
        # Check 1: Buffer constraint
        required_buffer = calculate_fused_group_buffer_req(group, optimal_mapping_params, graph)
        available_buffer = optimal_hw_config.get_buffer_size_bytes()
        
        if required_buffer > available_buffer:
            print(f"      -> REJECT: Buffer constraint violated ({required_buffer:.0f} > {available_buffer:.0f})")
            continue
        
        # Check 2: Latency benefit analysis
        # Non-fused latency: sum of individual layer latencies
        non_fused_latency = sum(layer_costs[layer]['latency'] for layer in group)
        non_fused_energy = sum(layer_costs[layer]['energy'] for layer in group)
        
        # Fused latency: max of individual latencies (parallel execution)
        fused_latency = torch.maximum(*[layer_costs[layer]['latency'] for layer in group])
        fused_energy = non_fused_energy + torch.tensor(config.FUSION_OVERHEAD_COST)
        
        # Apply fusion control overhead (more realistic than free fusion)
        fusion_control_overhead = torch.tensor(50.0, dtype=torch.float32)  # Fixed overhead cycles
        fused_latency = fused_latency + fusion_control_overhead
        
        # Check 3: Overall EDP benefit
        non_fused_edp = non_fused_latency * non_fused_energy
        fused_edp = fused_latency * fused_energy
        
        edp_improvement = (non_fused_edp - fused_edp) / non_fused_edp * 100
        
        if fused_edp < non_fused_edp:
            selected_fusion_groups.append(group)
            print(f"      -> ACCEPT: EDP improvement = {edp_improvement.item():.1f}%")
        else:
            print(f"      -> REJECT: EDP degradation = {-edp_improvement.item():.1f}%")
    
    print(f"  Step 2 Complete. Selected {len(selected_fusion_groups)} fusion groups out of {len(graph.fusion_groups)} candidates")
    
    # --- FIX: Use the unified evaluation function for final physical metrics ---
    
    # 1. Create a deterministic FusionParameters object based on the heuristic's decisions.
    final_fusion_params = FusionParameters(graph)
    with torch.no_grad():
        # Initialize all fusion decisions to 'off' (large negative logit)
        # --- FIX: Correctly iterate over ParameterDict to get key-value pairs ---
        for sanitized_key, prob_tensor in final_fusion_params.fusion_probs.items():
            # Ensure we're working with tensors before calling .fill_()
            assert isinstance(prob_tensor, torch.Tensor), f"ERROR: Fusion prob for key '{sanitized_key}' is not a Tensor! It's a {type(prob_tensor)}."
            prob_tensor.fill_(-1e9)
        
        # Turn 'on' the fusions selected by the heuristic (large positive logit)
        group_map = {tuple(sorted(g)): i for i, g in enumerate(graph.fusion_groups)}
        for group_to_fuse in selected_fusion_groups:
            group_key = tuple(sorted(group_to_fuse))
            if group_key in group_map:
                fusion_idx = group_map[group_key]
                # --- FIX: Use proper indexing to get the tensor from ParameterDict ---
                # Get the key corresponding to this fusion group index
                fusion_keys = list(final_fusion_params.fusion_probs.keys())
                if fusion_idx < len(fusion_keys):
                    target_key = fusion_keys[fusion_idx]
                    target_tensor = final_fusion_params.fusion_probs[target_key]
                    
                    # Ensure we're working with tensors before calling .fill_()
                    assert isinstance(target_tensor, torch.Tensor), f"ERROR: Target fusion tensor for key '{target_key}' is not a Tensor! It's a {type(target_tensor)}."
                    target_tensor.fill_(1e9)

    # 2. Pass the final, complete design to the "official referee" for evaluation.
    # The complex, manual calculation block that was here before has been removed.
    final_metrics = evaluate_final_design_point(
        optimal_mapping_params, final_fusion_params, optimal_hw_config, graph, config
    )

    # For this algorithm, the 'loss' is conceptually the final EDP.
    results = {
        'final_loss': final_metrics['final_edp'],
        **final_metrics
    }
    
    # Add parameter objects to results for configuration reporting
    results['_final_params'] = {
        'hardware_params': optimal_hw_config,
        'mapping_params': optimal_mapping_params,
        'fusion_params': final_fusion_params
    }
    
    return results


def run_twostep_baseline_experiment(graph: ComputationGraph, config: Config) -> dict:
    """
    Simulates a traditional, decoupled design process (HW opt first, then fusion).
    """
    print("Running Two-Step Baseline...")
    
    # === Step 1: Fusion-Agnostic DSE to find Optimal Hardware ===
    
    mapping_params = MappingParameters(graph)
    model = ConditionalPerformanceModel() # Needed for _calculate_analytical_costs
    
    # Create a mock hardware configuration for this baseline (simulates old fixed hardware approach)
    # Split buffer size into accumulator and scratchpad
    total_buffer_size = 512
    accumulator_size = total_buffer_size // 4  # 25% for accumulator
    scratchpad_size = total_buffer_size - accumulator_size  # 75% for scratchpad
    
    mock_hardware_params = HardwareParameters(
        initial_num_pes=256, 
        initial_accumulator_kb=accumulator_size,
        initial_scratchpad_kb=scratchpad_size
    )
    with torch.no_grad():
        mock_hardware_params.log_num_pes.requires_grad_(False)
        mock_hardware_params.log_accumulator_size_kb.requires_grad_(False)
        mock_hardware_params.log_scratchpad_size_kb.requires_grad_(False)
    
    # Optimizer only for mapping parameters (fusion-agnostic)
    optimizer = optim.Adam(mapping_params.parameters(), lr=0.01)
    
    num_epochs_step1 = 30 # Shorter loop for this baseline step (reduced for debugging)
    for epoch in range(num_epochs_step1):
        optimizer.zero_grad()
        
        # Calculate cost assuming no fusion (fusion-agnostic mapping)
        total_latency_nf = torch.tensor(0.0)
        total_energy_nf = torch.tensor(0.0)

        for layer_name in graph.get_layer_names():
            lat, comp_eng, sram_eng, dram_eng, noc_eng = model._calculate_analytical_costs(layer_name, mapping_params, graph, mock_hardware_params)
            total_latency_nf += lat
            total_energy_nf += comp_eng + sram_eng + dram_eng + noc_eng
            
        # Calculate log values for loss function
        epsilon = 1e-9  # For numerical stability
        log_latency = torch.log(total_latency_nf + epsilon)
        log_energy = torch.log(total_energy_nf + epsilon)
        log_area = torch.log(mock_hardware_params.get_area_cost() + epsilon)
        
        # Calculate penalties
        penalty = calculate_penalty_loss(mapping_params)
        template_constraint_penalty = calculate_template_constraint_penalty(mapping_params, mock_hardware_params, graph)
        
        # Loss function for the non-fused case
        total_cost = (
            config.LATENCY_WEIGHT * log_latency + 
            config.ENERGY_WEIGHT * log_energy + 
            config.AREA_WEIGHT * log_area +  # Now uses config weight
            config.PENALTY_WEIGHT * penalty +
            config.PENALTY_WEIGHT * template_constraint_penalty
        )
        
        total_cost.backward()
        optimizer.step()

    print(f"  Step 1 Complete. Mapping optimized for fixed hardware.")

    # === Step 2: Fusion Optimization ===
    print("  Step 2: Optimizing fusion parameters...")
    fusion_params = FusionParameters(graph)
    fusion_optimizer = optim.Adam(fusion_params.parameters(), lr=0.01) # Changed lr_fusion to 0.01
    
    num_epochs_fusion = 20 # Changed num_epochs_fusion to 20 (reduced for debugging)
    for epoch in range(num_epochs_fusion):
        fusion_optimizer.zero_grad()
        
        # We pass the already-optimized mapping_params and the fixed mock_hardware_params
        latency, energy, area = model(mapping_params, fusion_params, mock_hardware_params, graph)
        
        # Calculate penalty separately for this algorithm
        penalty = calculate_penalty_loss(mapping_params)
        
        log_latency = torch.log(latency + 1e-9)
        log_energy = torch.log(energy + 1e-9)
        log_area = torch.log(area + 1e-9)
        
        # --- FIX: Use configurable weights for the loss function ---
        fusion_loss = (
            config.LATENCY_WEIGHT * log_latency + 
            config.ENERGY_WEIGHT * log_energy + 
            config.AREA_WEIGHT * log_area + 
            config.PENALTY_WEIGHT * penalty
        )
        
        fusion_loss.backward()
        fusion_optimizer.step()

        if epoch % 5 == 0:
            print(f"    Fusion Epoch {epoch}/{num_epochs_fusion}, Loss: {fusion_loss.item():.4f}")
    
    print(f"  Two-Step Baseline finished. Final fusion loss: {fusion_loss.item():.4f}")

    # --- FIX: Use the unified evaluation function for final physical metrics ---
    # The final design point is evaluated by the "official referee".
    final_metrics = evaluate_final_design_point(
        mapping_params, fusion_params, mock_hardware_params, graph, config
    )
    
    # The final loss from the fusion optimization stage is used for 'final_loss'.
    results = {
        'final_loss': fusion_loss.item(),
        **final_metrics
    }
    
    # Add parameter objects to results for configuration reporting
    results['_final_params'] = {
        'hardware_params': mock_hardware_params,
        'mapping_params': mapping_params,
        'fusion_params': fusion_params
    }
    
    return results


def save_learned_parameters(
    workload: str,
    trial_num: int,
    mapping_params: MappingParameters,
    fusion_params: FusionParameters,
    hardware_params: HardwareParameters,
    final_metrics: Dict,
    save_dir: str = './saved_parameters'
) -> None:
    """
    Save learned parameters for case study analysis.
    
    Args:
        workload: Name of the workload
        trial_num: Trial number
        mapping_params: Learned mapping parameters
        fusion_params: Learned fusion parameters  
        hardware_params: Learned hardware parameters
        final_metrics: Final performance metrics
        save_dir: Directory to save parameter files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename
    safe_workload = workload.replace('.onnx', '').replace('/', '_')
    timestamp = int(time.time())
    filename = f"{safe_workload}_trial_{trial_num}_{timestamp}.pth"
    filepath = os.path.join(save_dir, filename)
    
    # Extract parameter values
    saved_data = {
        'workload': workload,
        'trial_num': trial_num,
        'timestamp': timestamp,
        'final_metrics': final_metrics,
        'hardware_params': {
            'num_pes': hardware_params.get_num_pes().item(),
            'buffer_size_kb': hardware_params.get_buffer_size_kb().item(),
            'area_mm2': hardware_params.get_area_cost().item(),
        },
        'fusion_params': {},
        'mapping_params': {}
    }
    
    # Extract fusion probabilities
    for sanitized_key, prob_tensor in fusion_params.fusion_probs.items():
        original_group = fusion_params.group_name_mapping[sanitized_key]
        group_key = '__'.join(original_group)
        saved_data['fusion_params'][group_key] = torch.sigmoid(prob_tensor).item()
    
    # Extract mapping parameters (template-based)
    for group_key, decision_module in mapping_params.get_all_decision_modules().items():
        group_layers = mapping_params.group_mapping[group_key]
        
        # Save template selection probabilities
        template_probs = decision_module.get_template_probabilities()
        for i, template_name in enumerate(decision_module.template_names):
            key = f"{group_key}_{template_name}_prob"
            saved_data['mapping_params'][key] = template_probs[i].item()
        
        # Save selected template parameters
        selected_template = decision_module.get_selected_template()
        template_params = {}
        for param_name, param_tensor in selected_template.named_parameters():
            if 'log_' in param_name:
                # Convert log-scale back to actual values
                actual_value = torch.exp(param_tensor).item()
                template_params[param_name.replace('log_', '')] = actual_value
        
        saved_data['mapping_params'][f"{group_key}_selected_params"] = template_params
    
    # Save to file
    torch.save(saved_data, filepath)
    print(f"  ✓ Saved parameters to: {filepath}")

def log_results(results_file: str, results: Dict):
    """
    Appends experiment results to a CSV file.
    
    Args:
        results_file: Path to the CSV file.
        results: A dictionary of results to log.
    """
    file_exists = os.path.isfile(results_file)
    
    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = results.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)

def main():
    """
    Main experiment runner with support for multiple trials and workloads.
    """
    # Load configuration
    config = Config.get_instance()
    
    # Experimental parameters
    NUM_TRIALS = 5  # Number of independent trials for statistical significance
    
    # Define workloads and algorithms
    workloads = [
        # === CNN WORKLOADS ===
        'resnet18.onnx',
        'simple_cnn.onnx', 
        'vgg_small.onnx',
        'mobilenet_v2.onnx',
        'efficientnet_b0.onnx',
        
        # === TRANSFORMER WORKLOADS (2025 ENHANCED) ===
        'bert_base.onnx',           # BERT-Base: 110M parameters, 512 sequence length
        'vit_small.onnx',           # Vision Transformer Small: image classification
        'gpt2_small.onnx',          # GPT-2 Small: autoregressive language model
        'distilbert.onnx',          # DistilBERT: efficient BERT variant
        
        # Note: Ensure these ONNX files are available in the working directory
        # Use export scripts or download from model repositories as needed
    ]
    
    algorithms = {
        'FA-DOSA': run_fadose_experiment,
        'FA-DOSA (Constrained)': run_fadose_constrained_experiment, # <-- NEW
        'Random Search': run_random_search_experiment,
        'Decoupled SOTA': run_decoupled_sota_experiment,
        'Two-Step Baseline': run_twostep_baseline_experiment,
    }
    
    results_file = 'experiment_results.csv'
    
    print(f"Starting comprehensive experiments:")
    print(f"  - Trials per experiment: {NUM_TRIALS}")
    print(f"  - Workloads: {len(workloads)}")
    print(f"  - Algorithms: {len(algorithms)}")
    print(f"  - Total experiments: {NUM_TRIALS * len(workloads) * len(algorithms)}")
    
    # Main experiment loop with multiple trials
    for trial in range(NUM_TRIALS):
        print(f"\n{'='*60}")
        print(f"TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'='*60}")

        # --- FIX: Set seeds for reproducibility for this specific trial ---
        # Using the trial number as the seed ensures each trial is deterministic
        # and the entire experiment can be reproduced exactly.
        seed = trial
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"  [REPRODUCIBILITY] Trial {trial + 1} uses random seed: {seed}")
        # --- End of fix ---
        
        for workload in workloads:
            print(f"\n--- Running experiments for workload: {workload} (Trial {trial + 1}) ---")
            
            try:
                graph = parse_onnx_to_graph(workload)
                print(f"  Loaded graph: {len(graph.layers)} layers, {len(graph.edges)} edges, {len(graph.fusion_groups)} fusion groups")
            except FileNotFoundError:
                print(f"Could not find {workload}. Skipping.")
                continue

            for algo_name, algo_func in algorithms.items():
                print(f"\n  Running Algorithm: {algo_name} (Trial {trial + 1})")
                
                # --- FIX: Enhanced exception handling for robust experiments ---
                # When an algorithm crashes, we log the error and continue with other algorithms
                # instead of stopping the entire experiment suite.
                try:
                    start_time = time.time()
                    
                    # Special handling for FA-DOSA variants to save learned parameters
                    if 'FA-DOSA' in algo_name:
                        results = algo_func(graph, config, workload=workload, trial_num=trial + 1)
                    else:
                        results = algo_func(graph, config)
                    
                    execution_time = time.time() - start_time
                    
                    # 调用配置报告函数 (在写入CSV之前)
                    if '_final_params' in results:
                        final_params = results['_final_params']
                        log_final_configuration(
                            algo_name=algo_name,
                            workload=workload,
                            trial_num=trial + 1,
                            hardware_params=final_params['hardware_params'],
                            mapping_params=final_params['mapping_params'],
                            fusion_params=final_params['fusion_params'],
                            final_metrics=results
                        )
                        
                        # 保存配置到JSON文件
                        json_filepath = save_config_to_json_file(
                            algorithm_name=algo_name,
                            workload=workload,
                            trial_num=trial + 1,
                            hardware_params=final_params['hardware_params'],
                            mapping_params=final_params['mapping_params'],
                            fusion_params=final_params['fusion_params'],
                            final_metrics=results
                        )
                        
                        if json_filepath:
                            print(f"✅ Configuration for '{algo_name}' on '{workload}' (Trial {trial + 1}) saved to: {json_filepath}")
                        else:
                            print(f"❌ Failed to save configuration for '{algo_name}' on '{workload}' (Trial {trial + 1})")
                        
                        # 从结果中移除参数对象，避免CSV序列化问题
                        results_for_csv = {k: v for k, v in results.items() if k != '_final_params'}
                    else:
                        results_for_csv = results
                    
                    # Log results with trial information
                    log_entry = {
                        'trial_num': trial + 1,
                        'workload': workload,
                        'algorithm': algo_name,
                        'execution_time_seconds': execution_time,
                        'status': 'SUCCESS',
                        'error_message': '',
                        **results_for_csv
                    }
                    log_results(results_file, log_entry)
                    
                    # 打印成功信息，但使用安全的字段访问
                    edp_val = results_for_csv.get('final_edp', 'N/A')
                    edp_str = f"{edp_val:.4e}" if isinstance(edp_val, (int, float)) else str(edp_val)
                    
                    latency_val = results_for_csv.get('final_latency', 'N/A')
                    latency_str = f"{latency_val:.0f}" if isinstance(latency_val, (int, float)) else str(latency_val)
                    
                    energy_val = results_for_csv.get('final_energy', 'N/A')
                    energy_str = f"{energy_val:.0f}" if isinstance(energy_val, (int, float)) else str(energy_val)
                    
                    area_val = results_for_csv.get('final_area', 'N/A')
                    area_str = f"{area_val:.4f}" if isinstance(area_val, (int, float)) else str(area_val)
                    
                    print(f"    ✓ SUCCESS in {execution_time:.1f}s: "
                          f"EDP={edp_str}, "
                          f"Latency={latency_str}, "
                          f"Energy={energy_str}, "
                          f"Area={area_str}")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    error_type = type(e).__name__
                    error_message = str(e)
                    
                    print(f"    ✗ ERROR in {algo_name} after {execution_time:.1f}s:")
                    print(f"      Error Type: {error_type}")
                    print(f"      Error Message: {error_message}")
                    
                    # Log detailed error information
                    log_entry = {
                        'trial_num': trial + 1,
                        'workload': workload,
                        'algorithm': algo_name,
                        'execution_time_seconds': execution_time,
                        'status': 'FAILED',
                        'error_message': f"{error_type}: {error_message}",
                        'final_loss': -1,           # Use -1 to indicate failure
                        'final_edp': -1,            # instead of inf which can cause issues
                        'final_latency': -1,
                        'final_energy': -1,
                        'final_area': -1,
                        'final_penalty': -1
                    }
                    log_results(results_file, log_entry)
                    
                    # Continue with next algorithm instead of crashing
                    print(f"      Continuing with next algorithm...")
                    continue

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to {results_file}")
    print(f"Total experiments run: {NUM_TRIALS * len(workloads) * len(algorithms)}")
    
    # Generate summary statistics
    try:
        import pandas as pd
        df = pd.read_csv(results_file)
        print(f"\nSUMMARY STATISTICS:")
        print(f"Results shape: {df.shape}")
        
        for workload in workloads:
            if workload.replace('.onnx', '') in df['workload'].str.replace('.onnx', '', regex=False).values:
                print(f"\n{workload}:")
                workload_data = df[df['workload'] == workload]
                summary = workload_data.groupby('algorithm')[['final_loss', 'final_edp', 'final_latency', 'final_energy', 'final_area']].agg(['mean', 'std', 'min', 'max'])
                print(summary.round(4))
    
    except ImportError:
        print(f"\nInstall pandas for automatic summary statistics: pip install pandas")

if __name__ == "__main__":
    main() 