"""
测试映射-融合协同优化功能
"""
import torch
import torch.optim as optim

from fa_dosa_demo import (
    HardwareParameters,
    MappingParameters,
    FusionParameters,
    Config,
    ComputationGraph,
    ConditionalPerformanceModel,
    calculate_total_loss_with_hardware_constraints,
)
from onnx_frontend import parse_onnx_to_graph


def test_mapping_fusion_synergy():
    """测试映射-融合协同优化功能"""
    print("🔄 测试映射-融合协同优化功能...")
    
    # 创建测试图
    try:
        graph = parse_onnx_to_graph('simple_cnn.onnx')
        print(f"  ✓ 成功加载图: {len(graph.layers)} 层, {len(graph.fusion_groups)} 个融合组")
    except FileNotFoundError:
        print("  ⚠️  未找到 simple_cnn.onnx，创建模拟图...")
        # 创建简单的测试图
        graph = ComputationGraph()
        
        # 添加两个卷积层
        graph.add_layer('conv1', {'N': 1, 'C': 3, 'K': 64, 'P': 32, 'Q': 32, 'R': 3, 'S': 3}, 'Conv')
        graph.add_layer('conv2', {'N': 1, 'C': 64, 'K': 128, 'P': 30, 'Q': 30, 'R': 3, 'S': 3}, 'Conv')
        
        # 添加一个融合组
        graph.add_fusion_group(['conv1', 'conv2'])
        
        print(f"  ✓ 创建模拟图: {len(graph.layers)} 层, {len(graph.fusion_groups)} 个融合组")
    
    # 创建参数
    mapping_params = MappingParameters(graph)
    fusion_params = FusionParameters(graph)
    hardware_params = HardwareParameters(initial_num_pes=64, initial_accumulator_kb=64.0, initial_scratchpad_kb=192.0)
    
    # 创建性能模型
    model = ConditionalPerformanceModel()
    
    print("\n📊 初始性能评估...")
    
    # 初始前向传播
    with torch.no_grad():
        latency, energy, area = model(mapping_params, fusion_params, hardware_params, graph)
        print(f"  初始延迟: {latency.item():.6f}")
        print(f"  初始能耗: {energy.item():.6f}")
        print(f"  初始面积: {area.item():.6f}")
    
    # 打印融合概率
    print("\n🔀 初始融合概率:")
    for sanitized_key, prob_tensor in fusion_params.fusion_probs.items():
        if hasattr(fusion_params, 'group_name_mapping') and sanitized_key in fusion_params.group_name_mapping:
            original_group = fusion_params.group_name_mapping[sanitized_key]
            group_display = ' + '.join(original_group)
        else:
            group_display = sanitized_key
        prob_value = torch.sigmoid(prob_tensor).item()
        print(f"  {group_display}: {prob_value:.4f}")
    
    # 打印映射策略
    print("\n🗺️  初始映射策略:")
    decision_modules = mapping_params.get_all_decision_modules()
    for group_key, decision_module in decision_modules.items():
        print(f"  组 {group_key}:")
        template_probs = decision_module.get_template_probabilities()
        for i, template_name in enumerate(decision_module.template_names):
            prob = template_probs[i].item()
            print(f"    {template_name}: {prob:.4f}")
    
    print("\n🚀 开始协同优化训练...")
    
    # 创建优化器
    optimizer = optim.Adam(
        list(mapping_params.parameters()) + 
        list(fusion_params.parameters()) + 
        list(hardware_params.parameters()), 
        lr=0.01
    )
    
    # 短期训练
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 计算损失
        total_loss = calculate_total_loss_with_hardware_constraints(
            model, mapping_params, fusion_params, hardware_params, graph
        )
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"  轮次 {epoch}: 损失 = {total_loss.item():.6f}")
    
    print("\n📊 优化后性能评估...")
    
    # 最终前向传播
    with torch.no_grad():
        final_latency, final_energy, final_area = model(mapping_params, fusion_params, hardware_params, graph)
        print(f"  最终延迟: {final_latency.item():.6f}")
        print(f"  最终能耗: {final_energy.item():.6f}")
        print(f"  最终面积: {final_area.item():.6f}")
    
    # 打印最终融合概率
    print("\n🔀 最终融合概率:")
    for sanitized_key, prob_tensor in fusion_params.fusion_probs.items():
        if hasattr(fusion_params, 'group_name_mapping') and sanitized_key in fusion_params.group_name_mapping:
            original_group = fusion_params.group_name_mapping[sanitized_key]
            group_display = ' + '.join(original_group)
        else:
            group_display = sanitized_key
        prob_value = torch.sigmoid(prob_tensor).item()
        decision = "FUSE" if prob_value > 0.5 else "NO_FUSE"
        print(f"  {group_display}: {prob_value:.4f} → {decision}")
    
    # 打印最终映射策略
    print("\n🗺️  最终映射策略:")
    for group_key, decision_module in decision_modules.items():
        print(f"  组 {group_key}:")
        template_probs = decision_module.get_template_probabilities()
        for i, template_name in enumerate(decision_module.template_names):
            prob = template_probs[i].item()
            print(f"    {template_name}: {prob:.4f}")
    
    print("\n✅ 映射-融合协同优化测试完成!")


def test_dynamic_fusion_overhead():
    """测试动态融合开销功能"""
    print("\n🔧 测试动态融合开销功能...")
    
    # 创建简单的测试图
    graph = ComputationGraph()
    graph.add_layer('conv1', {'N': 1, 'C': 64, 'K': 128, 'P': 32, 'Q': 32, 'R': 3, 'S': 3}, 'Conv')
    graph.add_layer('conv2', {'N': 1, 'C': 128, 'K': 256, 'P': 30, 'Q': 30, 'R': 3, 'S': 3}, 'Conv')
    graph.add_fusion_group(['conv1', 'conv2'])
    
    mapping_params = MappingParameters(graph)
    hardware_params = HardwareParameters(initial_num_pes=64, initial_accumulator_kb=64.0, initial_scratchpad_kb=192.0)
    model = ConditionalPerformanceModel()
    
    # 获取融合组的决策模块
    fusion_group = ['conv1', 'conv2']
    decision_module = mapping_params.get_decision_module_for_fusion_group(fusion_group)
    
    print("  🧪 测试不同映射模板的融合开销...")
    
    # 测试每个模板的融合开销
    for template_name in decision_module.template_names:
        template_instance = decision_module.get_template_by_name(template_name)
        
        # 计算非融合成本
        non_fusion_latency, non_fusion_energy = model.calculate_group_costs(
            fusion_group, template_instance, hardware_params, graph, 
            is_fusion_calculation=False
        )
        
        # 计算融合成本（动态开销）
        fusion_latency, fusion_energy = model.calculate_group_costs(
            fusion_group, template_instance, hardware_params, graph, 
            is_fusion_calculation=True
        )
        
        # 计算开销差异
        latency_overhead = fusion_latency - non_fusion_latency
        energy_overhead = fusion_energy - non_fusion_energy
        
        print(f"    {template_name}:")
        print(f"      延迟开销: {latency_overhead.item():.6f}")
        print(f"      能耗开销: {energy_overhead.item():.6f}")
        
        # 提取 tiling 因子（如果可用）
        if hasattr(template_instance, 'get_M0'):
            try:
                M0 = template_instance.get_M0().item()
                K0 = template_instance.get_K0().item() if hasattr(template_instance, 'get_K0') else 1.0
                N0 = template_instance.get_N0().item() if hasattr(template_instance, 'get_N0') else 1.0
                print(f"      Tiling因子: M0={M0:.2f}, K0={K0:.2f}, N0={N0:.2f}")
            except:
                print(f"      Tiling因子: 无法获取")
    
    print("  ✅ 动态融合开销测试完成!")


if __name__ == "__main__":
    test_mapping_fusion_synergy()
    test_dynamic_fusion_overhead() 