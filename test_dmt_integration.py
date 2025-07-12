#!/usr/bin/env python3
"""
测试脚本：验证基于模板的 DMT 系统是否正确集成

这个脚本测试：
1. MappingParameters 是否正确初始化模板决策模块
2. ConditionalPerformanceModel 是否能使用新的模板系统
3. 约束函数是否正确工作
4. 优化循环是否能运行
"""

import torch
import torch.nn as nn
from fa_dosa_demo import (
    ConditionalPerformanceModel,
    HardwareParameters,
    MappingParameters,
    FusionParameters,
    Config,
    ComputationGraph,
    calculate_penalty_loss,
    calculate_template_constraint_penalty,
    create_example_optimization_setup,
    create_joint_optimizer,
    calculate_total_loss_with_hardware_constraints
)

def create_test_graph():
    """创建一个简单的测试计算图"""
    graph = ComputationGraph()
    
    # 添加两个简单的层
    graph.add_layer("conv1", {
        'N': 1, 'C': 3, 'K': 64, 'P': 224, 'Q': 224, 'R': 3, 'S': 3
    }, "Conv")
    
    graph.add_layer("relu1", {
        'N': 1, 'K': 64, 'P': 224, 'Q': 224
    }, "Relu")
    
    # 添加边和融合组
    graph.add_edge("conv1", "relu1")
    graph.add_fusion_group(["conv1", "relu1"])
    
    return graph

def test_mapping_parameters():
    """测试 MappingParameters 的模板决策模块"""
    print("=== 测试 MappingParameters ===")
    
    graph = create_test_graph()
    mapping_params = MappingParameters(graph)
    
    # 检查是否正确创建了决策模块
    all_modules = mapping_params.get_all_decision_modules()
    print(f"创建了 {len(all_modules)} 个决策模块")
    
    # 检查每个决策模块是否包含四种模板
    for group_key, module in all_modules.items():
        print(f"组 '{group_key}' 包含的模板: {list(module.templates.keys())}")
        
        # 检查模板概率
        probs = module.get_template_probabilities()
        print(f"  模板概率: {probs.detach().numpy()}")
        
        # 检查选中的模板
        selected_template = module.get_selected_template()
        print(f"  选中的模板: {selected_template.template_name}")
    
    print("✅ MappingParameters 测试通过\n")

def test_template_constraints():
    """测试模板约束函数"""
    print("=== 测试模板约束函数 ===")
    
    graph = create_test_graph()
    mapping_params = MappingParameters(graph)
    hardware_params = HardwareParameters(initial_num_pes=64, initial_buffer_size_kb=256.0)
    
    # 测试 tiling 因子惩罚
    penalty = calculate_penalty_loss(mapping_params)
    print(f"Tiling 因子惩罚: {penalty.item():.6f}")
    
    # 测试模板约束惩罚
    template_penalty = calculate_template_constraint_penalty(mapping_params, hardware_params, graph)
    print(f"模板约束惩罚: {template_penalty.item():.6f}")
    
    print("✅ 约束函数测试通过\n")

def test_performance_model():
    """测试重构后的 ConditionalPerformanceModel"""
    print("=== 测试 ConditionalPerformanceModel ===")
    
    graph = create_test_graph()
    mapping_params = MappingParameters(graph)
    fusion_params = FusionParameters(graph)
    hardware_params = HardwareParameters(initial_num_pes=64, initial_buffer_size_kb=256.0)
    
    model = ConditionalPerformanceModel()
    
    # 前向传播 (新的6值返回签名)
    latency, compute_energy, sram_energy, dram_energy, noc_energy, area = model(mapping_params, fusion_params, hardware_params, graph)
    
    # 计算总能耗以保持兼容性
    total_energy = compute_energy + sram_energy + dram_energy + noc_energy
    
    print(f"延迟: {latency.item():.6f}")
    print(f"总能耗: {total_energy.item():.6f}")
    print(f"  计算能耗: {compute_energy.item():.6f}")
    print(f"  SRAM能耗: {sram_energy.item():.6f}")
    print(f"  DRAM能耗: {dram_energy.item():.6f}")
    print(f"  NoC能耗: {noc_energy.item():.6f}")
    print(f"面积: {area.item():.6f}")
    
    # 检查梯度
    print("检查梯度计算...")
    total_loss = calculate_total_loss_with_hardware_constraints(
        model, mapping_params, fusion_params, hardware_params, graph
    )
    
    print(f"总损失: {total_loss.item():.6f}")
    
    # 反向传播测试
    total_loss.backward()
    print("✅ 梯度计算成功")
    
    print("✅ ConditionalPerformanceModel 测试通过\n")

def test_optimization_loop():
    """测试优化循环"""
    print("=== 测试优化循环 ===")
    
    graph = create_test_graph()
    mapping_params, fusion_params, hardware_params = create_example_optimization_setup(graph)
    
    model = ConditionalPerformanceModel()
    optimizer = create_joint_optimizer(mapping_params, fusion_params, hardware_params, lr=0.01)
    
    print("运行 10 步优化...")
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        total_loss = calculate_total_loss_with_hardware_constraints(
            model, mapping_params, fusion_params, hardware_params, graph
        )
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  步骤 {epoch}: 损失 = {total_loss.item():.6f}")
    
    print("✅ 优化循环测试通过\n")

def test_template_parameters():
    """测试模板参数的可学习性"""
    print("=== 测试模板参数 ===")
    
    graph = create_test_graph()
    mapping_params = MappingParameters(graph)
    
    # 获取一个决策模块
    all_modules = mapping_params.get_all_decision_modules()
    first_module = list(all_modules.values())[0]
    
    # 获取一个模板实例
    full_template = first_module.get_template_by_name('full')
    
    print(f"Full 模板参数:")
    print(f"  M0: {full_template.get_M0().item():.4f}")
    print(f"  K0: {full_template.get_K0().item():.4f}")
    print(f"  N0: {full_template.get_N0().item():.4f}")
    
    # 获取 TiledKN 模板
    tiled_kn_template = first_module.get_template_by_name('tiled_kn')
    
    print(f"TiledKN 模板参数:")
    print(f"  M0: {tiled_kn_template.get_M0().item():.4f}")
    print(f"  K0: {tiled_kn_template.get_K0().item():.4f}")
    print(f"  N0: {tiled_kn_template.get_N0().item():.4f}")
    
    # 测试缓冲区需求计算
    buffer_reqs = tiled_kn_template.get_buffer_requirements()
    print(f"TiledKN 缓冲区需求: {buffer_reqs}")
    
    # 测试访存次数计算
    access_counts = tiled_kn_template.get_access_counts()
    print(f"TiledKN 访存次数: {access_counts}")
    
    print("✅ 模板参数测试通过\n")

def main():
    """运行所有测试"""
    print("🚀 开始 DMT 集成测试\n")
    
    try:
        test_mapping_parameters()
        test_template_constraints()
        test_performance_model()
        test_optimization_loop()
        test_template_parameters()
        
        print("🎉 所有测试通过！基于模板的 DMT 系统已成功集成。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 