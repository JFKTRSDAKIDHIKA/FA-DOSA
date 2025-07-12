#!/usr/bin/env python3
"""
测试JSON配置保存功能

这个脚本验证新的save_config_to_json_file函数是否正确工作，
包括文件创建、JSON结构、数据完整性等。
"""

import json
import os
import shutil
from run_experiments import save_config_to_json_file
from fa_dosa_demo import create_example_optimization_setup
from onnx_frontend import parse_onnx_to_graph


def test_json_config_save():
    """测试JSON配置保存功能"""
    print("🧪 Testing JSON Configuration Save Functionality")
    print("=" * 60)
    
    # 清理测试目录
    test_output_dir = './test_configs'
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    
    try:
        # 1. 创建测试数据
        print("📋 Step 1: Creating test data...")
        graph = parse_onnx_to_graph('resnet18.onnx')
        hardware_params, mapping_params, fusion_params, config = create_example_optimization_setup(graph)
        
        # 模拟最终性能指标
        final_metrics = {
            'final_loss': 123.45,
            'final_edp': 1.234e6,
            'final_latency': 1000.0,
            'final_energy': 1234.0,
            'final_area': 9.876,
            'final_penalty': 0.1,
            'execution_time_seconds': 45.67,
            'status': 'SUCCESS'
        }
        
        print(f"  ✓ Graph loaded: {len(graph.layers)} layers, {len(graph.fusion_groups)} fusion groups")
        print(f"  ✓ Test metrics prepared: {len(final_metrics)} metrics")
        
        # 2. 测试JSON保存功能
        print("\n💾 Step 2: Testing JSON save functionality...")
        json_filepath = save_config_to_json_file(
            algorithm_name='FA-DOSA',
            workload='resnet18.onnx',
            trial_num=1,
            hardware_params=hardware_params,
            mapping_params=mapping_params,
            fusion_params=fusion_params,
            final_metrics=final_metrics,
            output_dir=test_output_dir
        )
        
        if json_filepath:
            print(f"  ✓ JSON file created: {json_filepath}")
        else:
            print("  ❌ Failed to create JSON file")
            return False
        
        # 3. 验证文件存在
        print("\n📁 Step 3: Verifying file creation...")
        if os.path.exists(json_filepath):
            file_size = os.path.getsize(json_filepath)
            print(f"  ✓ File exists: {json_filepath}")
            print(f"  ✓ File size: {file_size} bytes")
        else:
            print(f"  ❌ File not found: {json_filepath}")
            return False
        
        # 4. 验证JSON结构
        print("\n🔍 Step 4: Verifying JSON structure...")
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 检查必需的顶级键
            required_keys = ['metadata', 'hardware_configuration', 'fusion_decisions', 'mapping_strategy', 'performance_metrics']
            for key in required_keys:
                if key in config_data:
                    print(f"  ✓ Found section: {key}")
                else:
                    print(f"  ❌ Missing section: {key}")
                    return False
            
            # 检查metadata内容
            metadata = config_data['metadata']
            expected_metadata = ['workload', 'algorithm', 'trial', 'timestamp']
            for key in expected_metadata:
                if key in metadata:
                    print(f"  ✓ Metadata contains: {key} = {metadata[key]}")
                else:
                    print(f"  ❌ Missing metadata: {key}")
                    return False
            
            # 检查硬件配置
            hw_config = config_data['hardware_configuration']
            expected_hw = ['processing_elements', 'buffer_size_kb', 'buffer_size_bytes', 'total_area_mm2']
            for key in expected_hw:
                if key in hw_config:
                    print(f"  ✓ Hardware config contains: {key} = {hw_config[key]}")
                else:
                    print(f"  ❌ Missing hardware config: {key}")
                    return False
            
            # 检查融合决策
            fusion_decisions = config_data['fusion_decisions']
            print(f"  ✓ Fusion decisions count: {len(fusion_decisions)}")
            if len(fusion_decisions) > 0:
                first_fusion = fusion_decisions[0]
                fusion_keys = ['group', 'fused', 'probability']
                for key in fusion_keys:
                    if key in first_fusion:
                        print(f"  ✓ Fusion decision contains: {key}")
                    else:
                        print(f"  ❌ Missing fusion decision key: {key}")
                        return False
            
            # 检查映射策略
            mapping_strategy = config_data['mapping_strategy']
            print(f"  ✓ Mapping strategy groups: {len(mapping_strategy)}")
            if len(mapping_strategy) > 0:
                first_group = list(mapping_strategy.keys())[0]
                mapping_info = mapping_strategy[first_group]
                mapping_keys = ['selected_template', 'probabilities', 'parameters']
                for key in mapping_keys:
                    if key in mapping_info:
                        print(f"  ✓ Mapping info contains: {key}")
                    else:
                        print(f"  ❌ Missing mapping info key: {key}")
                        return False
            
            # 检查性能指标
            perf_metrics = config_data['performance_metrics']
            print(f"  ✓ Performance metrics count: {len(perf_metrics)}")
            expected_metrics = ['final_loss', 'final_edp', 'final_latency', 'final_energy', 'final_area']
            for key in expected_metrics:
                if key in perf_metrics:
                    print(f"  ✓ Performance metric: {key} = {perf_metrics[key]}")
                else:
                    print(f"  ❌ Missing performance metric: {key}")
                    return False
            
        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Error reading JSON: {e}")
            return False
        
        # 5. 测试多个文件保存
        print("\n🔄 Step 5: Testing multiple file saves...")
        test_cases = [
            ('Random Search', 'vgg_small.onnx', 2),
            ('Decoupled SOTA', 'mobilenet_v2.onnx', 3),
            ('FA-DOSA (Constrained)', 'resnet18.onnx', 4)
        ]
        
        for algo_name, workload, trial_num in test_cases:
            json_filepath = save_config_to_json_file(
                algorithm_name=algo_name,
                workload=workload,
                trial_num=trial_num,
                hardware_params=hardware_params,
                mapping_params=mapping_params,
                fusion_params=fusion_params,
                final_metrics=final_metrics,
                output_dir=test_output_dir
            )
            
            if json_filepath and os.path.exists(json_filepath):
                filename = os.path.basename(json_filepath)
                print(f"  ✓ Created: {filename}")
            else:
                print(f"  ❌ Failed to create file for {algo_name}")
                return False
        
        # 6. 验证文件名清理功能
        print("\n🧹 Step 6: Testing filename sanitization...")
        test_names = [
            ('FA-DOSA (Special)', 'bert@base.onnx', 1),
            ('Random/Search', 'model with spaces.onnx', 2),
            ('Test\\Algorithm', 'file:name.onnx', 3)
        ]
        
        for algo_name, workload, trial_num in test_names:
            json_filepath = save_config_to_json_file(
                algorithm_name=algo_name,
                workload=workload,
                trial_num=trial_num,
                hardware_params=hardware_params,
                mapping_params=mapping_params,
                fusion_params=fusion_params,
                final_metrics=final_metrics,
                output_dir=test_output_dir
            )
            
            if json_filepath and os.path.exists(json_filepath):
                filename = os.path.basename(json_filepath)
                print(f"  ✓ Sanitized filename: {filename}")
                # 验证文件名只包含安全字符
                if all(c.isalnum() or c in '_-.' for c in filename):
                    print(f"    ✓ Filename is safe")
                else:
                    print(f"    ❌ Filename contains unsafe characters")
                    return False
            else:
                print(f"  ❌ Failed to create file with special characters")
                return False
        
        print("\n🎉 All tests passed!")
        print(f"📁 Test files created in: {test_output_dir}")
        
        # 显示创建的文件列表
        print("\n📋 Created files:")
        for root, dirs, files in os.walk(test_output_dir):
            for file in files:
                if file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    size = os.path.getsize(filepath)
                    print(f"  - {file} ({size} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_json_config_save()
    if success:
        print("\n✅ JSON configuration save functionality is working correctly!")
    else:
        print("\n❌ JSON configuration save functionality has issues!")
        exit(1) 