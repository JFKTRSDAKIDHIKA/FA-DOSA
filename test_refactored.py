#!/usr/bin/env python3
"""
Simple test script to verify the refactored DOSA code works correctly.
"""

import sys
import pathlib

# Add current directory to path to import our modules
current_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_argument_parser():
    """Test the argument parser functionality."""
    print("Testing ArgumentParser...")
    
    try:
        from config import ArgumentParser
        
        parser = ArgumentParser()
        
        # Test with minimal valid arguments
        test_args = ['-wl', 'conv', '--output_dir', 'test_output']
        args = parser.parse_args(test_args)
        
        assert args.workload == ['conv']
        assert args.output_dir == 'test_output'
        assert args.arch_name == 'gemmini'  # default value
        
        print("✓ ArgumentParser test passed")
        return True
        
    except Exception as e:
        print(f"✗ ArgumentParser test failed: {e}")
        return False

def test_run_config():
    """Test the run configuration functionality."""
    print("Testing RunConfig...")
    
    try:
        from config import RunConfig
        
        # Test configuration creation
        config = RunConfig(
            arch_name='gemmini',
            workloads=['conv'],
            base_workload_path='dataset/workloads',
            output_dir='test_output'
        )
        
        # Check that basic properties work
        assert config.arch_name == 'gemmini'
        assert config.workloads == ['conv']
        assert config.output_path is not None
        
        print("✓ RunConfig test passed")
        return True
        
    except Exception as e:
        print(f"✗ RunConfig test failed: {e}")
        return False

def test_csv_handler():
    """Test the CSV handler functionality."""
    print("Testing CSVHandler...")
    
    try:
        from core import CSVHandler
        import tempfile
        import os
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv_path = pathlib.Path(f.name)
        
        try:
            handler = CSVHandler(temp_csv_path)
            
            # Test writing some dummy data
            test_data = [
                {'name': 'test1', 'value': 100},
                {'name': 'test2', 'value': 200}
            ]
            
            handler.write_results(test_data)
            
            # Check that file exists and has content
            assert handler.exists
            assert not handler.is_empty
            assert handler.get_row_count() == 2
            
            print("✓ CSVHandler test passed")
            return True
            
        finally:
            # Clean up temporary file
            if temp_csv_path.exists():
                os.unlink(temp_csv_path)
        
    except Exception as e:
        print(f"✗ CSVHandler test failed: {e}")
        return False

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        # Test config module imports
        from config import ArgumentParser, RunConfig
        
        # Test core module imports  
        from core import CSVHandler, ExperimentRunner
        
        print("✓ All imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running DOSA refactored code tests...\n")
    
    tests = [
        test_imports,
        test_argument_parser,
        test_run_config,
        test_csv_handler,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored code is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the code.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 