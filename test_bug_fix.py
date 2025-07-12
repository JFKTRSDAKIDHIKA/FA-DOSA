"""
测试bug修复后的实验框架
"""
import torch
import time
from fa_dosa_demo import (
    Config,
    ComputationGraph,
    create_example_optimization_setup,
)
from run_experiments import (
    run_fadose_experiment,
    run_decoupled_sota_experiment,
    run_twostep_baseline_experiment,
    run_random_search_experiment,
)


def test_all_algorithms():
    """测试所有算法是否能正常运行"""
    print("🧪 测试所有算法...")
    
    # 创建测试图
    graph = ComputationGraph()
    graph.add_layer('conv1', {'N': 1, 'C': 3, 'K': 32, 'P': 32, 'Q': 32, 'R': 3, 'S': 3}, 'Conv')
    graph.add_layer('relu1', {'N': 1, 'C': 32, 'K': 32, 'P': 32, 'Q': 32, 'R': 1, 'S': 1}, 'Relu')
    graph.add_layer('conv2', {'N': 1, 'C': 32, 'K': 64, 'P': 30, 'Q': 30, 'R': 3, 'S': 3}, 'Conv')
    
    graph.add_fusion_group(['conv1', 'relu1'])
    
    config = Config.get_instance()
    
    algorithms = {
        'FA-DOSA': lambda: run_fadose_experiment(graph, config, num_iterations=50),
        'Random Search': lambda: run_random_search_experiment(graph, config, num_samples=100),
        'Decoupled SOTA': lambda: run_decoupled_sota_experiment(graph, config),
        'Two-Step Baseline': lambda: run_twostep_baseline_experiment(graph, config),
    }
    
    results = {}
    
    for algo_name, algo_func in algorithms.items():
        print(f"\n  🚀 测试 {algo_name}...")
        start_time = time.time()
        
        try:
            result = algo_func()
            execution_time = time.time() - start_time
            
            # 检查结果完整性
            required_keys = ['final_edp', 'final_latency', 'final_energy', 'final_area']
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                print(f"    ⚠️  警告: 缺少结果键: {missing_keys}")
            else:
                print(f"    ✅ 成功! EDP: {result['final_edp']:.2e}, 耗时: {execution_time:.2f}s")
                
            # 检查是否有参数对象（用于配置报告）
            if '_final_params' in result:
                params = result['_final_params']
                param_types = [type(params[key]).__name__ for key in params]
                print(f"    📊 参数对象: {param_types}")
            else:
                print(f"    ⚠️  缺少参数对象（无法生成配置报告）")
                
            results[algo_name] = {
                'success': True,
                'execution_time': execution_time,
                'final_edp': result.get('final_edp', -1)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"    ❌ 失败: {str(e)}")
            results[algo_name] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
    
    # 汇总结果
    print("\n📈 测试结果汇总:")
    successful = 0
    for algo_name, result in results.items():
        if result['success']:
            successful += 1
            print(f"  ✅ {algo_name}: EDP={result['final_edp']:.2e}")
        else:
            print(f"  ❌ {algo_name}: {result['error']}")
    
    print(f"\n🎯 成功率: {successful}/{len(algorithms)} ({successful/len(algorithms)*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    test_all_algorithms() 