import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Enable anomaly detection to help identify in-place operation issues
torch.autograd.set_detect_anomaly(True)

from fa_dosa_demo import ComputationGraph, FusionParameters
from high_fidelity_performance_model import HighFidelityPerformanceModel
from run_resnet_learnable_template_experiment import Config, HardwareParameters, FineGrainedMapping


def create_test_graph() -> ComputationGraph:
    """
    Create a test computation graph with fusion opportunities.
    
    Returns:
        ComputationGraph with test layers and edges
    """
    graph = ComputationGraph()
    
    # Add Conv -> BatchNorm -> ReLU pattern
    dims_conv1 = {'N': 1, 'C': 64, 'K': 64, 'P': 56, 'Q': 56, 'R': 3, 'S': 3}
    graph.add_layer('conv1', dims_conv1, 'Conv')
    dims_bn1 = {'N': 1, 'C': 64, 'K': 64, 'P': 56, 'Q': 56}
    graph.add_layer('bn1', dims_bn1, 'BatchNorm')
    dims_relu1 = {'N': 1, 'C': 64, 'K': 64, 'P': 56, 'Q': 56}
    graph.add_layer('relu1', dims_relu1, 'ReLU')
    
    # Add edges for Conv -> BatchNorm -> ReLU
    graph.add_edge('conv1', 'bn1')
    graph.add_edge('bn1', 'relu1')
    
    # Add Conv -> ReLU pattern
    dims_conv2 = {'N': 1, 'C': 64, 'K': 128, 'P': 28, 'Q': 28, 'R': 3, 'S': 3}
    graph.add_layer('conv2', dims_conv2, 'Conv')
    dims_relu2 = {'N': 1, 'C': 128, 'K': 128, 'P': 28, 'Q': 28}
    graph.add_layer('relu2', dims_relu2, 'ReLU')
    
    # Add edge for Conv -> ReLU
    graph.add_edge('conv2', 'relu2')
    
    # Add MatMul -> Add pattern
    dims_matmul1 = {'M': 512, 'N': 512, 'K': 512}
    graph.add_layer('matmul1', dims_matmul1, 'MatMul')
    dims_add1 = {'M': 512, 'N': 512}
    graph.add_layer('add1', dims_add1, 'Add')
    
    # Add edge for MatMul -> Add
    graph.add_edge('matmul1', 'add1')
    
    # Add MatMul -> GELU pattern
    dims_matmul2 = {'M': 512, 'N': 2048, 'K': 512}
    graph.add_layer('matmul2', dims_matmul2, 'MatMul')
    dims_gelu1 = {'M': 512, 'N': 2048}
    graph.add_layer('gelu1', dims_gelu1, 'GELU')
    
    # Add edge for MatMul -> GELU
    graph.add_edge('matmul2', 'gelu1')
    
    # Add LayerNorm -> MatMul pattern
    dims_ln1 = {'M': 512, 'N': 512}
    graph.add_layer('ln1', dims_ln1, 'LayerNorm')
    dims_matmul3 = {'M': 512, 'N': 512, 'K': 512}
    graph.add_layer('matmul3', dims_matmul3, 'MatMul')
    
    # Add edge for LayerNorm -> MatMul
    graph.add_edge('ln1', 'matmul3')
    
    # Add a standalone layer
    dims_conv3 = {'N': 1, 'C': 128, 'K': 256, 'P': 14, 'Q': 14, 'R': 3, 'S': 3}
    graph.add_layer('conv3', dims_conv3, 'Conv')
    
    # Identify and register fusion groups
    graph.identify_and_register_fusion_groups()
    
    return graph


def run_fusion_experiment(num_iterations: int = 100) -> Tuple[List[float], List[float], List[float]]:
    """
    Run an experiment to optimize fusion decisions.
    
    Args:
        num_iterations: Number of optimization iterations
        
    Returns:
        Tuple of (latency_history, energy_history, fusion_prob_history)
    """
    # Create config and graph
    config = Config.get_instance()
    graph = create_test_graph()
    
    # Initialize parameters
    hw_params = HardwareParameters()
    mapping = FineGrainedMapping(graph.layers['conv1']['dims'], config.MEMORY_HIERARCHY)
    fusion_params = FusionParameters(graph)
    perf_model = HighFidelityPerformanceModel(config, graph)
    
    # Collect all parameters for optimization
    all_params = list(hw_params.parameters()) + list(mapping.parameters()) + list(fusion_params.parameters())
    optimizer = optim.Adam(all_params, lr=1e-2)
    
    # Track metrics
    latency_history = []
    energy_history = []
    fusion_prob_history = []
    
    # Run optimization
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        latency, energy, area = perf_model(graph, hw_params, fusion_params, mapping)
        
        # Calculate loss
        loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9) + 0.1 * area
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record metrics
        latency_history.append(latency.item())
        energy_history.append(energy.item())
        
        # Record fusion probabilities
        fusion_probs = {}
        for group in graph.fusion_groups:
            if len(group) > 1:  # Only multi-layer groups
                group_key = '__'.join(sorted(group))
                prob = fusion_params.get_fusion_probability(group).item()
                fusion_probs[group_key] = prob
        fusion_prob_history.append(fusion_probs)
        
        # Print progress
        if i % 10 == 0 or i == num_iterations - 1:
            print(f"Iteration {i}: Latency = {latency.item():.4e}, Energy = {energy.item():.4e}")
            print("Fusion Probabilities:")
            for group_key, prob in fusion_probs.items():
                print(f"  {group_key}: {prob:.4f}")
            print()
    
    return latency_history, energy_history, fusion_prob_history


def plot_results(latency_history: List[float], energy_history: List[float], fusion_prob_history: List[Dict[str, float]]):
    """
    Plot the results of the fusion experiment.
    
    Args:
        latency_history: History of latency values
        energy_history: History of energy values
        fusion_prob_history: History of fusion probabilities
    """
    plt.figure(figsize=(15, 10))
    
    # Plot latency
    plt.subplot(3, 1, 1)
    plt.plot(latency_history)
    plt.title('Latency over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Latency (s)')
    plt.grid(True)
    
    # Plot energy
    plt.subplot(3, 1, 2)
    plt.plot(energy_history)
    plt.title('Energy over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (pJ)')
    plt.grid(True)
    
    # Plot fusion probabilities
    plt.subplot(3, 1, 3)
    for group_key in fusion_prob_history[0].keys():
        probs = [history.get(group_key, 0) for history in fusion_prob_history]
        plt.plot(probs, label=group_key)
    
    plt.title('Fusion Probabilities over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fusion_experiment_results.png')
    plt.close()


def main():
    """
    Main function to run the fusion experiment and plot results.
    """
    print("=== Running Layer Fusion Experiment ===")
    latency_history, energy_history, fusion_prob_history = run_fusion_experiment(num_iterations=100)
    
    print("\n=== Final Results ===")
    print(f"Final Latency: {latency_history[-1]:.4e}")
    print(f"Final Energy: {energy_history[-1]:.4e}")
    print("Final Fusion Probabilities:")
    for group_key, prob in fusion_prob_history[-1].items():
        print(f"  {group_key}: {prob:.4f}")
    
    print("\nPlotting results...")
    plot_results(latency_history, energy_history, fusion_prob_history)
    print("Results saved to 'fusion_experiment_results.png'")


if __name__ == "__main__":
    main()