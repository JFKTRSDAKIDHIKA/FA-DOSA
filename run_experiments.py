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
    ], lr=1e-5)
    
    # Training loop
    num_epochs = 100  # Reduced for testing; use 1000 for final experiments
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        latency, energy, area_cost = model(mapping_params, fusion_params, graph)
        penalty = calculate_penalty_loss(mapping_params)
        product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
        
        # Apply log transformation to stabilize optimization
        # Add small epsilon to prevent log(0)
        log_latency = torch.log(latency + 1e-9)
        log_energy = torch.log(energy + 1e-9)
        log_area = torch.log(area_cost + 1e-9)
        
        # New loss as weighted sum of logs (converts multiplicative to additive)
        # This replaces: total_cost = (latency * energy) * area_cost + penalties
        # With: total_loss = log(latency) + log(energy) + log(area) + penalties
        total_loss = (
            log_latency + log_energy + log_area + 
            config.PENALTY_WEIGHT * penalty + 
            config.PRODUCT_PENALTY_WEIGHT * product_penalty
        )
        
        total_loss.backward()
        optimizer.step()

    # Final evaluation
    final_latency, final_energy, final_area = model(mapping_params, fusion_params, graph)
    final_edp = final_latency * final_energy
    final_penalty = calculate_penalty_loss(mapping_params)
    final_product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
    
    # Calculate final loss in log domain (matching training loss)
    log_final_latency = torch.log(final_latency + 1e-9)
    log_final_energy = torch.log(final_energy + 1e-9)
    log_final_area = torch.log(final_area + 1e-9)
    final_total_loss = (
        log_final_latency + log_final_energy + log_final_area + 
        config.PENALTY_WEIGHT * final_penalty + 
        config.PRODUCT_PENALTY_WEIGHT * final_product_penalty
    )

    return {
        'final_loss': final_total_loss.item(),
        'final_edp': final_edp.item(),
        'final_latency': final_latency.item(),
        'final_energy': final_energy.item(),
        'final_area': final_area.item(),
    }

def run_random_search_experiment(graph: ComputationGraph, config: Config) -> dict:
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
    
    num_samples = 200  # Number of random configurations to try (reduced for testing)
    best_loss = float('inf')
    best_results = None
    
    model = ConditionalPerformanceModel()
    
    for sample in range(num_samples):
        # Generate random mapping parameters
        mapping_params = MappingParameters(graph)
        fusion_params = FusionParameters(graph)
        
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
            latency, energy, area_cost = model(mapping_params, fusion_params, graph)
            penalty = calculate_penalty_loss(mapping_params)
            product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
            
            # Calculate loss in log domain (matching FA-DOSA)
            log_latency = torch.log(latency + 1e-9)
            log_energy = torch.log(energy + 1e-9)
            log_area = torch.log(area_cost + 1e-9)
            
            total_loss = (
                log_latency + log_energy + log_area + 
                config.PENALTY_WEIGHT * penalty + 
                config.PRODUCT_PENALTY_WEIGHT * product_penalty
            )
            
            # Keep track of best configuration
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                final_edp = latency * energy
                best_results = {
                    'final_loss': total_loss.item(),
                    'final_edp': final_edp.item(),
                    'final_latency': latency.item(),
                    'final_energy': energy.item(),
                    'final_area': area_cost.item(),
                }
                
                if sample % 100 == 0:
                    print(f"  Sample {sample}: Best loss so far = {best_loss:.4f}")
                    
        except Exception as e:
            # Skip invalid configurations
            continue
    
    if best_results is None:
        # Fallback if no valid configuration found
        print("  Warning: No valid random configurations found, using fallback values")
        best_results = {
            'final_loss': 99999.0,
            'final_edp': 99999.0,
            'final_latency': 999.0,
            'final_energy': 999.0,
            'final_area': 99.0,
        }
    
    print(f"  Random Search completed. Best loss: {best_results['final_loss']:.4f}")
    return best_results

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
            lat, comp_eng, sram_eng, dram_eng = model._calculate_analytical_costs(layer_name, mapping_params, graph, hw_config_iter)
            total_latency_nf += lat
            total_energy_nf += comp_eng + sram_eng + dram_eng
            
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
            lat, comp_eng, sram_eng, dram_eng = model._calculate_analytical_costs(layer_name, mapping_params, graph, hw_config_decoupled)
            fused_latency = torch.maximum(fused_latency, lat)
            fused_energy += comp_eng + sram_eng + dram_eng
        
        total_latency += fused_latency
        total_energy += fused_energy + torch.tensor(config.FUSION_OVERHEAD_COST)
        processed_layers.update(group)

    # Calculate cost for standalone layers
    for layer_name in graph.get_layer_names():
        if layer_name not in processed_layers:
            lat, comp_eng, sram_eng, dram_eng = model._calculate_analytical_costs(layer_name, mapping_params, graph, hw_config_decoupled)
            total_latency += lat
            total_energy += comp_eng + sram_eng + dram_eng
            
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
    Main experiment runner with support for multiple trials and workloads.
    """
    # Load configuration
    config = Config.get_instance()
    
    # Experimental parameters
    NUM_TRIALS = 2  # Number of independent trials for statistical significance
    
    # Define workloads and algorithms
    workloads = [
        'resnet18.onnx',
        'simple_cnn.onnx',
        'vgg_small.onnx',
        # Enable additional models for comprehensive testing:
        # 'mobilenet_v2.onnx',
        # 'efficientnet_b0.onnx'
    ]
    
    algorithms = {
        'FA-DOSA': run_fadose_experiment,
        'Random Search': run_random_search_experiment,
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
                
                try:
                    results = algo_func(graph, config)
                    
                    # Log results with trial information
                    log_entry = {
                        'trial_num': trial + 1,
                        'workload': workload,
                        'algorithm': algo_name,
                        **results
                    }
                    log_results(results_file, log_entry)
                    
                    print(f"    Results: Loss={results['final_loss']:.4f}, "
                          f"EDP={results['final_edp']:.2e}, "
                          f"Latency={results['final_latency']:.0f}, "
                          f"Energy={results['final_energy']:.0f}, "
                          f"Area={results['final_area']:.6f}")
                    
                except Exception as e:
                    print(f"    Error in {algo_name}: {e}")
                    # Log error case
                    log_entry = {
                        'trial_num': trial + 1,
                        'workload': workload,
                        'algorithm': algo_name,
                        'final_loss': float('inf'),
                        'final_edp': float('inf'),
                        'final_latency': 0,
                        'final_energy': 0,
                        'final_area': 0,
                    }
                    log_results(results_file, log_entry)

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
            if workload.replace('.onnx', '') in df['workload'].str.replace('.onnx', '').values:
                print(f"\n{workload}:")
                workload_data = df[df['workload'] == workload]
                summary = workload_data.groupby('algorithm')[['final_loss', 'final_edp', 'final_latency', 'final_energy', 'final_area']].agg(['mean', 'std', 'min', 'max'])
                print(summary.round(4))
    
    except ImportError:
        print(f"\nInstall pandas for automatic summary statistics: pip install pandas")

if __name__ == "__main__":
    main() 