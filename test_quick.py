"""
Quick test for format fix
"""
import torch
import torch.optim as optim
from fa_dosa_demo import *
from onnx_frontend import parse_onnx_to_graph

def quick_test():
    print("🔧 Quick format test...")
    config = Config.get_instance()
    graph = parse_onnx_to_graph('resnet18.onnx')
    
    # Create minimal setup
    mapping_params, fusion_params, hardware_params = create_example_optimization_setup(graph)
    performance_model = ConditionalPerformanceModel()
    
    # Run just a few iterations
    optimizer = optim.Adam(list(mapping_params.parameters()) + list(fusion_params.parameters()) + list(hardware_params.parameters()), lr=1e-3)
    
    for i in range(3):  # Very few iterations
        optimizer.zero_grad()
        loss = calculate_total_loss_with_hardware_constraints(performance_model, mapping_params, fusion_params, hardware_params, graph)
        loss.backward()
        optimizer.step()
        print(f"    Iteration {i}: Loss = {loss.item():.4f}")
    
    # Test the evaluation and formatting
    final_params = {'mapping_params': mapping_params, 'fusion_params': fusion_params, 'hardware_params': hardware_params}
    
    # Simulate authoritative evaluation
    with torch.no_grad():
        final_fusion_params = FusionParameters(graph)
        for sanitized_key, prob_logit in fusion_params.fusion_probs.items():
            hard_decision = torch.round(torch.sigmoid(prob_logit))
            final_fusion_params.fusion_probs[sanitized_key] = torch.nn.Parameter(hard_decision, requires_grad=False)
        latency, energy, area = performance_model(mapping_params, final_fusion_params, hardware_params, graph)
        edp = latency * energy
    
    results = {
        'final_loss': edp.item(),
        'final_edp': edp.item(),
        'final_latency': latency.item(),
        'final_energy': energy.item(),
        'final_area': area.item(),
    }
    
    # Test the formatting logic that was causing problems
    edp_val = results.get('final_edp', 'N/A')
    edp_str = f"{edp_val:.4e}" if isinstance(edp_val, (int, float)) else str(edp_val)
    
    latency_val = results.get('final_latency', 'N/A')
    latency_str = f"{latency_val:.0f}" if isinstance(latency_val, (int, float)) else str(latency_val)
    
    energy_val = results.get('final_energy', 'N/A')
    energy_str = f"{energy_val:.0f}" if isinstance(energy_val, (int, float)) else str(energy_val)
    
    area_val = results.get('final_area', 'N/A')
    area_str = f"{area_val:.4f}" if isinstance(area_val, (int, float)) else str(area_val)
    
    print(f"✅ SUCCESS: EDP={edp_str}, Latency={latency_str}, Energy={energy_str}, Area={area_str}")
    print("🎉 Format test passed!")

if __name__ == '__main__':
    quick_test() 