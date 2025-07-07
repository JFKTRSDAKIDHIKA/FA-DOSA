import csv
import os
from typing import Dict, List
from fa_dosa_demo import (
    ComputationGraph,
    MappingParameters,
    FusionParameters,
    ConditionalPerformanceModel,
    calculate_penalty_loss,
    calculate_product_constraint_penalty,
    calculate_hardware_config,
    calculate_fused_group_buffer_req,
    Config
)
from onnx_frontend import parse_onnx_to_graph
import torch
import torch.optim as optim

def run_fadose_experiment(graph: ComputationGraph, config: Config) -> dict:
    """
    Runs the full FA-DOSA optimization experiment for a given graph.
    
    Args:
        graph: The ComputationGraph object for the workload.
        config: The configuration dictionary.
        
    Returns:
        A dictionary containing the final performance metrics.
    """
    # Initialize mapping and fusion parameters
    mapping_params = MappingParameters(graph)
    fusion_params = FusionParameters(graph)
    
    # Initialize performance model
    model = ConditionalPerformanceModel()
    
    # Setup optimizer
    optimizer = optim.Adam([
        {'params': mapping_params.parameters()},
        {'params': fusion_params.parameters()}
    ], lr=0.001)
    
    # Training loop
    num_epochs = 200
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        latency, energy, area_cost = model(mapping_params, fusion_params, graph)
        edp = latency * energy
        penalty = calculate_penalty_loss(mapping_params)
        product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
        
        total_cost = (
            edp +
            area_cost * edp +
            config.PENALTY_WEIGHT * penalty * edp +
            config.PRODUCT_PENALTY_WEIGHT * product_penalty * edp
        )
        
        total_cost.backward()
        optimizer.step()

    # Final evaluation
    final_latency, final_energy, final_area = model(mapping_params, fusion_params, graph)
    final_edp = final_latency * final_energy
    final_penalty = calculate_penalty_loss(mapping_params)
    final_product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
    final_total_cost = (
        final_edp + 
        final_area * final_edp + 
        config.PENALTY_WEIGHT * final_penalty * final_edp + 
        config.PRODUCT_PENALTY_WEIGHT * final_product_penalty * final_edp
    )

    return {
        'final_loss': final_total_cost.item(),
        'final_edp': final_edp.item(),
        'final_latency': final_latency.item(),
        'final_energy': final_energy.item(),
        'final_area': final_area.item(),
    }

def run_random_search_experiment(graph: ComputationGraph, config: Config) -> dict:
    """
    Placeholder for a random search baseline experiment.
    
    Args:
        graph: The ComputationGraph for the workload.
        config: The configuration dictionary.
        
    Returns:
        A dictionary with dummy results.
    """
    print("Running Random Search (dummy)...")
    return {
        'final_loss': 99999.0,
        'final_edp': 99999.0,
        'final_latency': 999.0,
        'final_energy': 999.0,
        'final_area': 99.0,
    }

def run_twostep_baseline_experiment(graph: ComputationGraph, config: Config) -> dict:
    """
    Simulates a traditional, decoupled design process (HW opt first, then fusion).
    """
    print("Running Two-Step Baseline...")
    
    # === Step 1: Fusion-Agnostic DSE to find Optimal Hardware ===
    
    mapping_params = MappingParameters(graph)
    model = ConditionalPerformanceModel() # Needed for _calculate_analytical_costs
    
    # Optimizer only for mapping parameters (fusion-agnostic)
    optimizer = optim.Adam(mapping_params.parameters(), lr=0.01)
    
    num_epochs_step1 = 150 # Shorter loop for this baseline step
    for epoch in range(num_epochs_step1):
        optimizer.zero_grad()
        
        # In this step, we assume no fusion occurs (p_fuse=0)
        # We must calculate the cost manually without the full model forward pass
        hw_config_iter = calculate_hardware_config(mapping_params, graph)
        total_latency_nf = torch.tensor(0.0)
        total_energy_nf = torch.tensor(0.0)

        for layer_name in graph.get_layer_names():
            lat, eng = model._calculate_analytical_costs(layer_name, mapping_params, graph, hw_config_iter)
            total_latency_nf += lat
            total_energy_nf += eng
            
        edp_nf = total_latency_nf * total_energy_nf
        penalty = calculate_penalty_loss(mapping_params)
        product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
        
        # Loss function for the non-fused case
        total_cost = (
            edp_nf + hw_config_iter['area_cost'] * edp_nf +
            config.PENALTY_WEIGHT * penalty * edp_nf +
            config.PRODUCT_PENALTY_WEIGHT * product_penalty * edp_nf
        )
        
        total_cost.backward()
        optimizer.step()

    # After optimization, fix the hardware config based on the final mapping params
    hw_config_decoupled = calculate_hardware_config(mapping_params, graph)
    final_area_cost = hw_config_decoupled['area_cost']
    print(f"  Step 1 Complete. Decoupled HW Buffer Size: {hw_config_decoupled['buffer_size'].item():.2f}, Area: {final_area_cost.item():.6f}")

    # === Step 2: Apply Heuristic Fusion on the Fixed Hardware ===

    final_fused_groups = set()
    for group in graph.fusion_groups:
        required_buffer = calculate_fused_group_buffer_req(group, mapping_params, graph)
        if required_buffer <= hw_config_decoupled['buffer_size']:
            final_fused_groups.add(tuple(sorted(group)))
            print(f"  Heuristically fusing group: {group}")

    # === Final Evaluation with the fixed HW and heuristic fusion plan ===
    
    total_latency = torch.tensor(0.0)
    total_energy = torch.tensor(0.0)
    processed_layers = set()

    # Calculate cost for heuristically fused groups
    for group_tuple in final_fused_groups:
        group = list(group_tuple)
        fused_latency = torch.tensor(0.0)
        fused_energy = torch.tensor(0.0)
        for layer_name in group:
            lat, eng = model._calculate_analytical_costs(layer_name, mapping_params, graph, hw_config_decoupled)
            fused_latency = torch.maximum(fused_latency, lat)
            fused_energy += eng
        
        total_latency += fused_latency
        total_energy += fused_energy + torch.tensor(config.FUSION_OVERHEAD_COST)
        processed_layers.update(group)

    # Calculate cost for standalone layers
    for layer_name in graph.get_layer_names():
        if layer_name not in processed_layers:
            lat, eng = model._calculate_analytical_costs(layer_name, mapping_params, graph, hw_config_decoupled)
            total_latency += lat
            total_energy += eng
            
    final_edp = total_latency * total_energy
    # Note: Penalties are not recalculated here, as they relate to the mapping, which is already fixed.
    final_total_cost = final_edp + final_area_cost * final_edp

    return {
        'final_loss': final_total_cost.item(),
        'final_edp': final_edp.item(),
        'final_latency': total_latency.item(),
        'final_energy': total_energy.item(),
        'final_area': final_area_cost.item(),
    }

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
    Main experiment runner.
    """
    # Load configuration
    config = Config.get_instance()
    
    # Define workloads and algorithms
    workloads = ['resnet18.onnx'] # Add more models like 'resnet18.onnx'
    algorithms = {
        'FA-DOSA': run_fadose_experiment,
        'Random Search': run_random_search_experiment,
        'Two-Step Baseline': run_twostep_baseline_experiment,
    }
    
    results_file = 'experiment_results.csv'
    
    # Main experiment loop
    for workload in workloads:
        print(f"\n--- Running experiments for workload: {workload} ---")
        
        try:
            graph = parse_onnx_to_graph(workload)
        except FileNotFoundError:
            print(f"Could not find {workload}. Skipping.")
            continue

        for algo_name, algo_func in algorithms.items():
            print(f"\nRunning Algorithm: {algo_name}")
            
            results = algo_func(graph, config)
            
            # Log results
            log_entry = {
                'workload': workload,
                'algorithm': algo_name,
                **results
            }
            log_results(results_file, log_entry)
            
            print(f"Results for {algo_name}: {results}")

    print(f"\nAll experiments complete. Results saved to {results_file}")

if __name__ == "__main__":
    main() 