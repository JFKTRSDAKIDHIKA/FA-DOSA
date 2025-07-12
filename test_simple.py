"""
Simple test script for FA-DOSA framework
"""
import torch
import torch.optim as optim
import time

from fa_dosa_demo import (
    HardwareParameters,
    MappingParameters,
    FusionParameters,
    Config,
    ComputationGraph,
    ConditionalPerformanceModel,
    create_example_optimization_setup,
    calculate_total_loss_with_hardware_constraints,
)
from onnx_frontend import parse_onnx_to_graph


def simple_authoritative_evaluation(final_params: dict, graph: ComputationGraph, config: Config) -> dict:
    """Simple authoritative evaluation function"""
    mapping_params = final_params['mapping_params']
    fusion_params = final_params['fusion_params']
    hardware_params = final_params['hardware_params']
    performance_model = ConditionalPerformanceModel()
    
    with torch.no_grad():
        # Get deterministic fusion decisions
        final_fusion_params = FusionParameters(graph)
        for sanitized_key, prob_logit in fusion_params.fusion_probs.items():
            hard_decision = torch.round(torch.sigmoid(prob_logit))
            final_fusion_params.fusion_probs[sanitized_key] = torch.nn.Parameter(hard_decision, requires_grad=False)
        
        latency, energy, area = performance_model(mapping_params, final_fusion_params, hardware_params, graph)
        edp = latency * energy
    
    return {
        'final_loss': edp.item(),
        'final_edp': edp.item(),
        'final_latency': latency.item(),
        'final_energy': energy.item(),
        'final_area': area.item(),
    }


def simple_fa_dosa_test(graph: ComputationGraph, config: Config, num_iterations=100) -> dict:
    """Simple FA-DOSA test with minimal iterations"""
    print("🚀 Running Simple FA-DOSA Test...")
    
    # Create parameters
    mapping_params, fusion_params, hardware_params = create_example_optimization_setup(graph)
    performance_model = ConditionalPerformanceModel()
    
    # Create optimizer
    optimizer = optim.Adam(
        list(mapping_params.parameters()) + 
        list(fusion_params.parameters()) + 
        list(hardware_params.parameters()), 
        lr=1e-3
    )
    
    print(f"   🔄 Starting optimization: {num_iterations} iterations...")
    start_time = time.time()
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        total_loss = calculate_total_loss_with_hardware_constraints(
            performance_model, mapping_params, fusion_params, hardware_params, graph
        )
        total_loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"    Epoch {i}/{num_iterations}, Loss: {total_loss.item():.4f}")
    
    optimization_time = time.time() - start_time
    print(f"   ✅ Optimization finished in {optimization_time:.2f} seconds")
    
    # Final evaluation
    print("   [🏛️ Evaluation] Computing final metrics...")
    final_params = {
        'mapping_params': mapping_params, 
        'fusion_params': fusion_params, 
        'hardware_params': hardware_params
    }
    results = simple_authoritative_evaluation(final_params, graph, config)
    
    print("   ✅ Test complete!")
    return results


def main():
    """Main test function"""
    print("=" * 60)
    print("FA-DOSA SIMPLE TEST")
    print("=" * 60)
    
    # Load config and model
    config = Config.get_instance()
    
    try:
        # Use a simple model file - you can change this path
        model_path = 'resnet18.onnx'
        print(f"Loading model: {model_path}")
        graph = parse_onnx_to_graph(model_path)
        print(f"Loaded graph: {len(graph.layers)} layers, {len(graph.fusion_groups)} fusion groups")
        
        # Run test
        results = simple_fa_dosa_test(graph, config, num_iterations=50)
        
        # Print results
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4e}")
            else:
                print(f"{key}: {value}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 