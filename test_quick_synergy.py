"""
快速测试新的协同优化功能与实验框架的兼容性
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
    calculate_total_loss_with_hardware_constraints,
    create_example_optimization_setup,
)
from onnx_frontend import parse_onnx_to_graph


def quick_synergy_test():
    """快速测试协同优化功能"""
    print("🚀 快速协同优化兼容性测试...")
    
    # 加载配置
    config = Config.get_instance()
    
    # 使用现有的ONNX文件
    try:
        graph = parse_onnx_to_graph('simple_cnn.onnx')
        print(f"  ✓ 加载图成功: {len(graph.layers)} 层, {len(graph.fusion_groups)} 个融合组")
    except FileNotFoundError:
        print("  ✗ 未找到 simple_cnn.onnx")
        return
    
    # 使用标准的设置函数
    mapping_params, fusion_params, hardware_params = create_example_optimization_setup(graph)
    performance_model = ConditionalPerformanceModel()
    
    print("  📊 初始评估...")
    with torch.no_grad():
        initial_latency, initial_energy, initial_area = performance_model(
            mapping_params, fusion_params, hardware_params, graph
        )
        print(f"    初始 EDP: {(initial_latency * initial_energy).item():.4e}")
    
    # 创建优化器（模拟实验脚本中的设置）
    optimizer = optim.Adam(
        list(mapping_params.parameters()) + 
        list(fusion_params.parameters()) + 
        list(hardware_params.parameters()), 
        lr=1e-3
    )
    
    print("  🔄 开始优化（50轮）...")
    start_time = time.time()
    
    for i in range(50):
        optimizer.zero_grad()
        total_loss = calculate_total_loss_with_hardware_constraints(
            performance_model, mapping_params, fusion_params, hardware_params, graph
        )
        total_loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"    轮次 {i}: 损失 = {total_loss.item():.4f}")
    
    elapsed = time.time() - start_time
    print(f"  ⏱️  优化完成，耗时: {elapsed:.2f}s")
    
    # 最终评估
    with torch.no_grad():
        final_latency, final_energy, final_area = performance_model(
            mapping_params, fusion_params, hardware_params, graph
        )
        final_edp = final_latency * final_energy
        print(f"    最终 EDP: {final_edp.item():.4e}")
        
        improvement = (initial_latency * initial_energy - final_edp) / (initial_latency * initial_energy) * 100
        print(f"    性能提升: {improvement.item():.2f}%")
    
    # 检查融合决策
    print("  🔀 最终融合决策:")
    for sanitized_key, prob_tensor in fusion_params.fusion_probs.items():
        prob_value = torch.sigmoid(prob_tensor).item()
        decision = "FUSE" if prob_value > 0.5 else "NO_FUSE"
        print(f"    {sanitized_key}: {prob_value:.3f} → {decision}")
    
    print("  ✅ 兼容性测试完成!")
    
    return {
        'initial_edp': (initial_latency * initial_energy).item(),
        'final_edp': final_edp.item(),
        'improvement_percent': improvement.item(),
        'execution_time': elapsed
    }


if __name__ == "__main__":
    results = quick_synergy_test()
    if results:
        print(f"\n📈 测试结果:")
        print(f"  初始 EDP: {results['initial_edp']:.4e}")
        print(f"  最终 EDP: {results['final_edp']:.4e}")
        print(f"  性能提升: {results['improvement_percent']:.2f}%")
        print(f"  执行时间: {results['execution_time']:.2f}s") 