"""
最终验证脚本：测试完整的实验流程，包括配置报告功能
"""
import os
import time
from fa_dosa_demo import Config, ComputationGraph
from run_experiments import *
from onnx_frontend import parse_onnx_to_graph


def run_mini_experiment():
    """运行一个简化的实验，验证所有功能"""
    print("🚀 运行最终验证实验...")
    
    # 配置
    config = Config.get_instance()
    NUM_TRIALS = 1  # 简化为1次试验
    
    # 工作负载（使用现有的ONNX文件或创建测试图）
    try:
        graph = parse_onnx_to_graph('simple_cnn.onnx')
        workload = 'simple_cnn.onnx'
        print(f"  ✓ 加载工作负载: {workload}")
    except FileNotFoundError:
        # 创建测试图
        graph = ComputationGraph()
        graph.add_layer('conv1', {'N': 1, 'C': 3, 'K': 32, 'P': 32, 'Q': 32, 'R': 3, 'S': 3}, 'Conv')
        graph.add_layer('relu1', {'N': 1, 'C': 32, 'K': 32, 'P': 32, 'Q': 32, 'R': 1, 'S': 1}, 'Relu')
        graph.add_fusion_group(['conv1', 'relu1'])
        workload = 'test_workload'
        print(f"  ✓ 创建测试工作负载: {workload}")
    
    # 算法列表（缩短运行时间）
    algorithms = {
        'FA-DOSA': lambda: run_fadose_experiment(graph, config, num_iterations=100, workload=workload, trial_num=1),
        'Decoupled SOTA': lambda: run_decoupled_sota_experiment(graph, config),
    }
    
    results_file = 'test_experiment_results.csv'
    
    print(f"  📊 开始实验循环...")
    
    for trial in range(NUM_TRIALS):
        print(f"\n--- Trial {trial + 1}/{NUM_TRIALS} ---")
        
        for algo_name, algo_func in algorithms.items():
            print(f"\n  🔧 运行算法: {algo_name}")
            
            try:
                start_time = time.time()
                results = algo_func()
                execution_time = time.time() - start_time
                
                # 测试配置报告功能
                if '_final_params' in results:
                    print(f"    📋 生成配置报告...")
                    final_params = results['_final_params']
                    log_final_configuration(
                        algo_name=algo_name,
                        workload=workload,
                        trial_num=trial + 1,
                        hardware_params=final_params['hardware_params'],
                        mapping_params=final_params['mapping_params'],
                        fusion_params=final_params['fusion_params'],
                        final_metrics=results
                    )
                    
                    # 从结果中移除参数对象，避免CSV序列化问题
                    results_for_csv = {k: v for k, v in results.items() if k != '_final_params'}
                else:
                    results_for_csv = results
                
                # 记录结果
                log_entry = {
                    'trial_num': trial + 1,
                    'workload': workload,
                    'algorithm': algo_name,
                    'execution_time_seconds': execution_time,
                    'status': 'SUCCESS',
                    'error_message': '',
                    **results_for_csv
                }
                log_results(results_file, log_entry)
                
                # 打印成功信息
                edp_val = results_for_csv.get('final_edp', 'N/A')
                edp_str = f"{edp_val:.4e}" if isinstance(edp_val, (int, float)) else str(edp_val)
                print(f"    ✅ SUCCESS in {execution_time:.1f}s: EDP={edp_str}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_type = type(e).__name__
                error_message = str(e)
                
                print(f"    ❌ ERROR in {algo_name} after {execution_time:.1f}s:")
                print(f"      Error Type: {error_type}")
                print(f"      Error Message: {error_message}")
                
                # 记录错误
                log_entry = {
                    'trial_num': trial + 1,
                    'workload': workload,
                    'algorithm': algo_name,
                    'execution_time_seconds': execution_time,
                    'status': 'FAILED',
                    'error_message': f"{error_type}: {error_message}",
                    'final_loss': -1,
                    'final_edp': -1,
                    'final_latency': -1,
                    'final_energy': -1,
                    'final_area': -1,
                    'final_penalty': -1
                }
                log_results(results_file, log_entry)
    
    print(f"\n🎯 实验完成！结果保存在: {results_file}")
    
    # 读取并显示结果
    if os.path.exists(results_file):
        try:
            import pandas as pd
            df = pd.read_csv(results_file)
            print(f"\n📈 实验结果总结:")
            success_count = len(df[df['status'] == 'SUCCESS'])
            total_count = len(df)
            print(f"  成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
            
            # 显示成功的结果
            successful = df[df['status'] == 'SUCCESS']
            if len(successful) > 0:
                print(f"\n  成功的实验:")
                for _, row in successful.iterrows():
                    print(f"    {row['algorithm']}: EDP={row['final_edp']:.2e}")
            
            # 显示失败的结果
            failed = df[df['status'] == 'FAILED']
            if len(failed) > 0:
                print(f"\n  失败的实验:")
                for _, row in failed.iterrows():
                    print(f"    {row['algorithm']}: {row['error_message']}")
                    
        except ImportError:
            print(f"  安装pandas查看详细统计: pip install pandas")
    
    return results_file


if __name__ == "__main__":
    results_file = run_mini_experiment()
    
    # 清理测试文件
    if os.path.exists(results_file):
        try:
            os.remove(results_file)
            print(f"\n🧹 清理测试文件: {results_file}")
        except:
            pass 