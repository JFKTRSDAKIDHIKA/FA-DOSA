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
from typing import Dict, List
import torch
import torch.optim as optim
import time
import numpy as np
import random
import torch.nn as nn

from fa_dosa_demo import (
    ConditionalPerformanceModel,
    HardwareParameters,
    MappingParameters,
    FusionParameters,
    Config,
    ComputationGraph,  # <-- FIX: Import the correct graph class
    # --- FIX: Add missing imports for penalty and setup functions ---
    calculate_penalty_loss,
    calculate_product_constraint_penalty,
    calculate_hardware_constraint_penalty,
    create_example_optimization_setup,
    create_joint_optimizer,
    calculate_total_loss_with_hardware_constraints,
    calculate_fused_group_buffer_req
)
from onnx_frontend import parse_onnx_to_graph


# ==============================================================================
# UNIFIED FINAL PERFORMANCE EVALUATION FUNCTION
# ==============================================================================

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

def run_fadose_experiment(graph: ComputationGraph, config: Config, num_iterations=2000, lr=1e-3, workload: str = None, trial_num: int = None) -> dict:
    """
    Runs the full FA-DOSA joint optimization experiment.
    
    Args:
        graph: The ComputationGraph object for the workload.
        config: The configuration dictionary.
        
    Returns:
        A dictionary containing the final performance metrics.
    """
    # Initialize mapping, fusion, and hardware parameters using the new joint approach
    mapping_params, fusion_params, hardware_params = create_example_optimization_setup(graph)
    
    # Initialize performance model
    model = ConditionalPerformanceModel()
    
    # Setup joint optimizer for all parameters
    optimizer = create_joint_optimizer(mapping_params, fusion_params, hardware_params, lr=1e-4)
    
    # Training loop
    num_epochs = 100  # Reduced for testing; use 1000 for final experiments
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Calculate total loss including hardware constraints
        total_loss = calculate_total_loss_with_hardware_constraints(
            model, mapping_params, fusion_params, hardware_params, graph
        )
        
        total_loss.backward()
        optimizer.step()

    # Final evaluation
    final_latency, final_energy, final_area = model(mapping_params, fusion_params, hardware_params, graph)
    final_edp = final_latency * final_energy
    
    # Calculate individual penalty components for reporting
    final_penalty = calculate_penalty_loss(mapping_params)
    final_product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
    final_hardware_penalty = calculate_hardware_constraint_penalty(mapping_params, hardware_params, graph)
    
    # Calculate final loss in log domain (matching training loss)
    log_final_latency = torch.log(final_latency + 1e-9)
    log_final_energy = torch.log(final_energy + 1e-9)
    log_final_area = torch.log(final_area + 1e-9)

    # --- FIX: Use configurable weights for the loss function ---
    # This allows us to guide the optimizer towards more desirable solutions.
    final_total_loss = (
        config.LATENCY_WEIGHT * log_final_latency + 
        config.ENERGY_WEIGHT * log_final_energy + 
        config.AREA_WEIGHT * log_final_area + 
        config.PENALTY_WEIGHT * (final_penalty + final_hardware_penalty) + 
        config.PRODUCT_PENALTY_WEIGHT * final_product_penalty
    )

    print(f"    Final loss: {final_total_loss.item():.4f}")
    
    # --- FIX: Use the unified evaluation function for final physical metrics ---
    # All final performance numbers are now calculated by the "official referee"
    # to ensure fair comparison across all algorithms.
    final_metrics = evaluate_final_design_point(
        mapping_params, fusion_params, hardware_params, graph, config
    )

    # Save learned parameters for case study analysis
    if workload is not None and trial_num is not None:
        save_learned_parameters(
            workload=workload,
            trial_num=trial_num,
            mapping_params=mapping_params,
            fusion_params=fusion_params,
            hardware_params=hardware_params,
            final_metrics=final_metrics
        )
    
    # Return the final loss for debugging, but merge it with the authoritative
    # metrics from the unified evaluation function.
    return {
        'final_loss': final_total_loss.item(),
        **final_metrics
    }


def run_fadose_constrained_experiment(graph: ComputationGraph, config: Config, workload: str = None, trial_num: int = None) -> dict:
    """
    A controlled experiment to test FA-DOSA's software optimization capabilities
    on a FIXED, smaller hardware configuration, identical to a potential SOTA baseline.
    This helps diagnose whether FA-DOSA's issues stem from its hardware search (DSE)
    or its software optimization (mapping/fusion).
    """
    print("Running FA-DOSA (Constrained Hardware) Experiment...")
    
    # --- FIX: Calculate hardware parameters that produce exactly area = 18.92 ---
    # Based on area formula: Area = area_base_mm2 + num_pes * area_per_pe_mm2 + buffer_size_kb * area_per_kb_sram_mm2
    # 18.92 = 1.0 + num_pes * 0.01 + buffer_size_kb * 0.1
    # Let's choose a reasonable balance: 64 PEs and 179.2 KB buffer
    # Verification: 1.0 + 64 * 0.01 + 179.2 * 0.1 = 1.0 + 0.64 + 17.92 = 19.56
    # Let's be more precise: 64 PEs and 175.6 KB buffer
    # Verification: 1.0 + 64 * 0.01 + 175.6 * 0.1 = 1.0 + 0.64 + 17.56 = 19.2
    # Let's try: 64 PEs and 172.8 KB buffer  
    # Verification: 1.0 + 64 * 0.01 + 172.8 * 0.1 = 1.0 + 0.64 + 17.28 = 18.92 ✓
    
    target_area = 18.92
    constrained_pes = 64
    constrained_buffer_kb = 172.8
    
    # Verify the calculation
    calculated_area = 1.0 + constrained_pes * 0.01 + constrained_buffer_kb * 0.1
    print(f"  [CONSTRAINED] Target area: {target_area} mm²")
    print(f"  [CONSTRAINED] Calculated area: {calculated_area:.2f} mm²")
    print(f"  [CONSTRAINED] Hardware fixed to: {constrained_pes} PEs, {constrained_buffer_kb:.1f} KB Buffer")
    
    # Assert that our calculation is correct
    assert abs(calculated_area - target_area) < 0.01, f"Area calculation error: {calculated_area} != {target_area}"

    # --- FIX: Create truly FIXED hardware parameters ---
    hardware_params = HardwareParameters(
        initial_num_pes=constrained_pes,
        initial_buffer_size_kb=constrained_buffer_kb
    )
    
    # --- CRITICAL FIX: Make hardware parameters completely non-trainable ---
    hardware_params.log_num_pes.requires_grad_(False)
    hardware_params.log_buffer_size_kb.requires_grad_(False)
    
    # --- FIX: Add verification that hardware parameters produce the expected area ---
    actual_area = hardware_params.get_area_cost()
    print(f"  [VERIFICATION] Actual hardware area: {actual_area.item():.2f} mm²")
    assert abs(actual_area.item() - target_area) < 0.01, f"Hardware area mismatch: {actual_area.item()} != {target_area}"

    # Initialize mapping and fusion parameters (these are still learnable)
    mapping_params = MappingParameters(graph)
    fusion_params = FusionParameters(graph)
    
    # Initialize performance model
    model = ConditionalPerformanceModel()
    
    # --- FIX: Create an optimizer that EXCLUDES hardware parameters ---
    # Only optimize software parameters: mapping and fusion
    software_params = list(mapping_params.parameters()) + list(fusion_params.parameters())
    optimizer = optim.Adam(software_params, lr=1e-4)
    
    print(f"  [OPTIMIZER] Optimizing {len(software_params)} software parameters (hardware is FIXED)")
    
    # Training loop (same as standard FA-DOSA but with fixed hardware)
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # --- FIX: Add constraint verification during optimization ---
        # Verify hardware parameters haven't changed
        current_area = hardware_params.get_area_cost()
        assert abs(current_area.item() - target_area) < 0.01, f"CONSTRAINT VIOLATION: Hardware area changed to {current_area.item()} at epoch {epoch}"
        
        # We now pass the non-trainable hardware_params to the model
        total_loss = calculate_total_loss_with_hardware_constraints(
            model, mapping_params, fusion_params, hardware_params, graph
        )
        
        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"    Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}, Hardware Area: {current_area.item():.2f} mm²")
            
    # --- FIX: Final verification before evaluation ---
    final_hw_area = hardware_params.get_area_cost()
    print(f"  [FINAL VERIFICATION] Hardware area before evaluation: {final_hw_area.item():.2f} mm²")
    assert abs(final_hw_area.item() - target_area) < 0.01, f"CONSTRAINT VIOLATION: Final hardware area is {final_hw_area.item()}, expected {target_area}"

    # --- FIX: Use the SAME fixed hardware_params object for final evaluation ---
    # DO NOT create new hardware parameters or use any other hardware configuration
    final_metrics = evaluate_final_design_point(
        mapping_params, fusion_params, hardware_params, graph, config
    )
    
    # --- FIX: Add post-evaluation verification ---
    reported_area = final_metrics['final_area']
    print(f"  [POST-EVAL VERIFICATION] Reported area: {reported_area:.2f} mm²")
    assert abs(reported_area - target_area) < 0.01, f"EVALUATION ERROR: Reported area {reported_area} != expected {target_area}"
    
    # --- FIX: Save learned parameters for case study analysis ---
    if workload is not None and trial_num is not None:
        # We pass a modified workload name to distinguish the results
        constrained_workload_name = f"{workload}_constrained"
        save_learned_parameters(
            workload=constrained_workload_name,
            trial_num=trial_num,
            mapping_params=mapping_params,
            fusion_params=fusion_params,
            hardware_params=hardware_params,
            final_metrics=final_metrics
        )

    print(f"  [SUCCESS] Constrained experiment completed. Area constraint maintained: {reported_area:.2f} mm²")
    
    return {
        'final_loss': total_loss.item(),
        **final_metrics
    }


def run_random_search_experiment(graph: ComputationGraph, config: Config, num_samples=1000) -> dict:
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
        hardware_params = HardwareParameters(
            initial_num_pes=np.random.randint(16, 1025),
            initial_buffer_size_kb=np.random.randint(64, 4097)
        )
        
        # Randomize mapping parameters
        with torch.no_grad():
            for sanitized_name, layer_mapping in mapping_params.mappings.items():
                for dim, factor in layer_mapping.temporal_factors_L2.items():
                    # Random factor between 0.1 and 10.0
                    factor.fill_(torch.rand(1).item() * 9.9 + 0.1)
        
        # Randomize fusion parameters
        with torch.no_grad():
            for sanitized_key, prob in fusion_params.fusion_probs.items():
                # Random logit between -3 and 3 (gives sigmoid range ~0.05 to 0.95)
                prob.fill_(torch.rand(1).item() * 6.0 - 3.0)
        
        try:
            # Evaluate this configuration
            latency, energy, area_cost = model(mapping_params, fusion_params, hardware_params, graph)
            penalty = calculate_penalty_loss(mapping_params)
            product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
            
            # Calculate loss in log domain (matching FA-DOSA)
            log_latency = torch.log(latency + 1e-9)
            log_energy = torch.log(energy + 1e-9)
            log_area = torch.log(area_cost + 1e-9)
            
            # --- FIX: Use configurable weights for the loss function ---
            total_loss = (
                config.LATENCY_WEIGHT * log_latency + 
                config.ENERGY_WEIGHT * log_energy + 
                config.AREA_WEIGHT * log_area + 
                config.PENALTY_WEIGHT * penalty + 
                config.PRODUCT_PENALTY_WEIGHT * product_penalty
            )
            
            # Keep track of best configuration
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_params = {
                    'mapping': mapping_params,
                    'fusion': fusion_params,
                    'hardware': hardware_params,
                }
                
            if sample % 200 == 0 and sample > 0:
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
    
    return {
        'final_loss': best_loss,
        **final_metrics
    }

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
    
    # Define realistic discrete hardware search space
    buffer_sizes_kb = [128, 256, 512, 1024]  # KB
    pe_counts = [64, 128, 256, 512]  # Number of processing elements
    
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
            mock_hardware_params = HardwareParameters(
                initial_num_pes=pe_count,
                initial_buffer_size_kb=buffer_size_kb
            )
            # Fix the parameters to prevent learning (simulate fixed hardware)
            with torch.no_grad():
                mock_hardware_params.log_num_pes.requires_grad_(False)
                mock_hardware_params.log_buffer_size_kb.requires_grad_(False)
            
            # === Mapping optimization for this fixed hardware ===
            mapping_params = MappingParameters(graph)
            model = ConditionalPerformanceModel()
            
            # Optimizer only for mapping (hardware is fixed)
            optimizer = optim.Adam(mapping_params.parameters(), lr=0.01)
            
            num_epochs_mapping = 100  # Focused optimization for each hardware config
            
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
                    
                edp_nf = total_latency_nf * total_energy_nf
                penalty = calculate_penalty_loss(mapping_params)
                product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
                
                # Loss function focused on EDP optimization
                area_cost = mock_hardware_params.get_area_cost()
                total_cost = (
                    edp_nf + area_cost * 0.001 +  # Small area weight
                    config.PENALTY_WEIGHT * penalty +
                    config.PRODUCT_PENALTY_WEIGHT * product_penalty
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
                    for (best_name, best_layer), (curr_name, curr_layer) in zip(
                        best_hw_mapping_params.mappings.items(), 
                        mapping_params.mappings.items()
                    ):
                        for (best_dim, best_factor), (curr_dim, curr_factor) in zip(
                            best_layer.temporal_factors_L2.items(),
                            curr_layer.temporal_factors_L2.items()
                        ):
                            best_factor.copy_(curr_factor)
    
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
    return {
        'final_loss': final_metrics['final_edp'],
        **final_metrics
    }


def run_twostep_baseline_experiment(graph: ComputationGraph, config: Config) -> dict:
    """
    Simulates a traditional, decoupled design process (HW opt first, then fusion).
    """
    print("Running Two-Step Baseline...")
    
    # === Step 1: Fusion-Agnostic DSE to find Optimal Hardware ===
    
    mapping_params = MappingParameters(graph)
    model = ConditionalPerformanceModel() # Needed for _calculate_analytical_costs
    
    # Create a mock hardware configuration for this baseline (simulates old fixed hardware approach)
    mock_hardware_params = HardwareParameters(initial_num_pes=256, initial_buffer_size_kb=512)
    with torch.no_grad():
        mock_hardware_params.log_num_pes.requires_grad_(False)
        mock_hardware_params.log_buffer_size_kb.requires_grad_(False)
    
    # Optimizer only for mapping parameters (fusion-agnostic)
    optimizer = optim.Adam(mapping_params.parameters(), lr=0.01)
    
    num_epochs_step1 = 150 # Shorter loop for this baseline step
    for epoch in range(num_epochs_step1):
        optimizer.zero_grad()
        
        # In this step, we assume no fusion occurs (p_fuse=0)
        # We must calculate the cost manually without the full model forward pass
        total_latency_nf = torch.tensor(0.0)
        total_energy_nf = torch.tensor(0.0)

        for layer_name in graph.get_layer_names():
            lat, comp_eng, sram_eng, dram_eng, noc_eng = model._calculate_analytical_costs(layer_name, mapping_params, graph, mock_hardware_params)
            total_latency_nf += lat
            total_energy_nf += comp_eng + sram_eng + dram_eng + noc_eng
            
        edp_nf = total_latency_nf * total_energy_nf
        penalty = calculate_penalty_loss(mapping_params)
        product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
        
        # Loss function for the non-fused case
        area_cost = mock_hardware_params.get_area_cost()
        total_cost = (
            edp_nf + area_cost * edp_nf +
            config.PENALTY_WEIGHT * penalty * edp_nf +
            config.PRODUCT_PENALTY_WEIGHT * product_penalty * edp_nf
        )
        
        total_cost.backward()
        optimizer.step()

    print(f"  Step 1 Complete. Mapping optimized for fixed hardware.")

    # === Step 2: Fusion Optimization ===
    print("  Step 2: Optimizing fusion parameters...")
    fusion_params = FusionParameters(graph)
    fusion_optimizer = optim.Adam(fusion_params.parameters(), lr=0.01) # Changed lr_fusion to 0.01
    
    num_epochs_fusion = 100 # Changed num_epochs_fusion to 100
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

        if epoch % 200 == 0:
            print(f"    Fusion Epoch {epoch}/{num_epochs_fusion}, Loss: {fusion_loss.item():.4f}")
    
    print(f"  Two-Step Baseline finished. Final fusion loss: {fusion_loss.item():.4f}")

    # --- FIX: Use the unified evaluation function for final physical metrics ---
    # The final design point is evaluated by the "official referee".
    final_metrics = evaluate_final_design_point(
        mapping_params, fusion_params, mock_hardware_params, graph, config
    )
    
    # The final loss from the fusion optimization stage is used for 'final_loss'.
    return {
        'final_loss': fusion_loss.item(),
        **final_metrics
    }


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
    
    # Extract mapping parameters
    for sanitized_name, layer_mapping in mapping_params.mappings.items():
        original_name = mapping_params.layer_name_mapping[sanitized_name]
        saved_data['mapping_params'][original_name] = {}
        for dim, factor_tensor in layer_mapping.temporal_factors_L2.items():
            saved_data['mapping_params'][original_name][dim] = factor_tensor.item()
    
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
                    
                    # Log results with trial information
                    log_entry = {
                        'trial_num': trial + 1,
                        'workload': workload,
                        'algorithm': algo_name,
                        'execution_time_seconds': execution_time,
                        'status': 'SUCCESS',
                        'error_message': '',
                        **results
                    }
                    log_results(results_file, log_entry)
                    
                    print(f"    ✓ SUCCESS in {execution_time:.1f}s: Loss={results['final_loss']:.4f}, "
                          f"EDP={results['final_edp']:.2e}, "
                          f"Latency={results['final_latency']:.0f}, "
                          f"Energy={results['final_energy']:.0f}, "
                          f"Area={results['final_area']:.6f}")
                    
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