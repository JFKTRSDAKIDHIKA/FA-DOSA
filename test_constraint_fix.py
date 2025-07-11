#!/usr/bin/env python3
"""
Test script to verify that the run_fadose_constrained_experiment fix works correctly.
"""

import torch
from onnx_frontend import parse_onnx_to_graph  
from run_experiments import run_fadose_constrained_experiment
from fa_dosa_demo import Config

def test_constrained_experiment():
    print('=== Testing FA-DOSA Constrained Experiment Fix ===')
    print()

    # Create test graph using available model
    graph = parse_onnx_to_graph('./resnet18.onnx')
    config = Config.get_instance()

    print('Running constrained experiment...')
    try:
        results = run_fadose_constrained_experiment(graph, config, workload='test', trial_num=1)
        
        print()
        print('=== Results ===')
        print(f'Final area: {results["final_area"]:.2f} mm²')
        print(f'Final latency: {results["final_latency"]:.2e}')
        print(f'Final energy: {results["final_energy"]:.2e}')
        print(f'Final EDP: {results["final_edp"]:.2e}')
        
        # Verify constraint
        expected_area = 18.92
        actual_area = results['final_area']
        area_diff = abs(actual_area - expected_area)
        
        print()
        print('=== Constraint Verification ===')
        print(f'Expected area: {expected_area} mm²')
        print(f'Actual area: {actual_area:.2f} mm²')
        print(f'Difference: {area_diff:.4f} mm²')
        
        if area_diff < 0.01:
            print()
            print('✅ CONSTRAINT VERIFICATION SUCCESS!')
            print(f'Area correctly constrained to {expected_area} mm²')
            return True
        else:
            print()
            print('❌ CONSTRAINT VERIFICATION FAILED!')
            print(f'Expected: {expected_area} mm², Actual: {actual_area} mm²')
            return False
            
    except Exception as e:
        print(f'❌ Experiment failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_constrained_experiment()
    if success:
        print("\n🎉 All tests passed! The constraint fix is working correctly.")
    else:
        print("\n💥 Test failed! There are still issues with the constraint implementation.") 