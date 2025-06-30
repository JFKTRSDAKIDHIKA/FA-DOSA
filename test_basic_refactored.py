#!/usr/bin/env python3
"""
Basic test script to verify the refactored DOSA code structure without heavy dependencies.
"""

import sys
import pathlib

# Add current directory to path to import our modules
current_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_config_module_structure():
    """Test that the config module has the expected structure."""
    print("Testing config module structure...")
    
    try:
        # Test basic imports without triggering dataset imports
        import config.argument_parser
        import config.run_config
        
        # Check that classes exist
        assert hasattr(config.argument_parser, 'ArgumentParser')
        assert hasattr(config.run_config, 'RunConfig')
        
        print("✓ Config module structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ Config module structure test failed: {e}")
        return False

def test_core_module_structure():
    """Test that the core module has the expected structure."""
    print("Testing core module structure...")
    
    try:
        # Test basic imports
        import core.csv_handler
        
        # Check that classes exist
        assert hasattr(core.csv_handler, 'CSVHandler')
        
        print("✓ Core module structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ Core module structure test failed: {e}")
        return False

def test_argument_parser_basic():
    """Test the argument parser basic functionality."""
    print("Testing ArgumentParser basic functionality...")
    
    try:
        from config.argument_parser import ArgumentParser
        
        parser = ArgumentParser()
        
        # Test with minimal valid arguments (no validation since dataset not available)
        test_args = ['-wl', 'conv', '--output_dir', 'test_output']
        args = parser.parse_args(test_args)
        
        assert args.workload == ['conv']
        assert args.output_dir == 'test_output'
        assert args.arch_name == 'gemmini'  # default value
        
        print("✓ ArgumentParser basic test passed")
        return True
        
    except Exception as e:
        print(f"✗ ArgumentParser basic test failed: {e}")
        return False

def test_run_config_basic():
    """Test the run configuration basic functionality."""
    print("Testing RunConfig basic functionality...")
    
    try:
        from config.run_config import RunConfig
        
        # Test configuration creation with minimal setup
        config = RunConfig(
            arch_name='gemmini',
            workloads=['conv'],
            base_workload_path='test_workloads',  # Won't validate since utils not available
            output_dir='test_output'
        )
        
        # Check that basic properties work
        assert config.arch_name == 'gemmini'
        assert config.workloads == ['conv']
        assert config.output_path is not None
        
        print("✓ RunConfig basic test passed")
        return True
        
    except Exception as e:
        print(f"✗ RunConfig basic test failed: {e}")
        return False

def test_csv_handler_basic():
    """Test the CSV handler basic functionality."""
    print("Testing CSVHandler basic functionality...")
    
    try:
        from core.csv_handler import CSVHandler
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
            
            print("✓ CSVHandler basic test passed")
            return True
            
        finally:
            # Clean up temporary file
            if temp_csv_path.exists():
                os.unlink(temp_csv_path)
        
    except Exception as e:
        print(f"✗ CSVHandler basic test failed: {e}")
        return False

def test_refactored_files_exist():
    """Test that all refactored files exist."""
    print("Testing refactored files exist...")
    
    try:
        files_to_check = [
            'config/__init__.py',
            'config/argument_parser.py', 
            'config/run_config.py',
            'core/__init__.py',
            'core/csv_handler.py',
            'core/experiment_runner.py',
            'run_refactored.py',
            'run_search_refactored.py',
            'REFACTORING_README.md'
        ]
        
        for file_path in files_to_check:
            assert pathlib.Path(file_path).exists(), f"File {file_path} not found"
        
        print("✓ All refactored files exist")
        return True
        
    except Exception as e:
        print(f"✗ Refactored files existence test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("Running DOSA basic refactored code tests...\n")
    
    tests = [
        test_refactored_files_exist,
        test_config_module_structure,
        test_core_module_structure,
        test_argument_parser_basic,
        test_run_config_basic,
        test_csv_handler_basic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The refactored code structure is working correctly.")
        print("Note: Full functionality tests require torch and other dependencies to be installed.")
        return 0
    else:
        print("❌ Some tests failed. Please check the code.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 