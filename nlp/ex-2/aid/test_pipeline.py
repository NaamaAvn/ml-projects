#!/usr/bin/env python3
"""
Test script to verify the language modeling pipeline structure.
This script checks that all required files exist and can be imported.
"""

import os
import sys
import importlib.util

def test_imports():
    """Test that all pipeline modules can be imported."""
    print("Testing module imports...")
    
    modules_to_test = [
        ('data/01_load_and_split_data.py', '01_load_and_split_data'),
        ('data/02_analyze_data.py', '02_analyze_data'),
        ('data/03_create_vocabulary.py', '03_create_vocabulary'),
        ('data/04_lm_create_sequences.py', '04_lm_create_sequences'),
        ('model/05_train_language_model.py', '05_train_language_model'),
        ('model/06_test_model.py', '06_test_model')
    ]
    
    for file_path, module_name in modules_to_test:
        if os.path.exists(file_path):
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"✓ {file_path} - Import successful")
            except Exception as e:
                print(f"✗ {file_path} - Import failed: {e}")
        else:
            print(f"✗ {file_path} - File not found")

def test_directory_structure():
    """Test that the directory structure is correct."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data',
        'model',
        'data/processed_lm_data'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/ - Directory exists")
        else:
            print(f"✗ {dir_path}/ - Directory missing")

def test_main_script():
    """Test that the main script can be executed."""
    print("\nTesting main script...")
    
    if os.path.exists('main.py'):
        try:
            # Test argument parsing
            import subprocess
            result = subprocess.run([sys.executable, 'main.py', '--status'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✓ main.py - Status check successful")
            else:
                print(f"✗ main.py - Status check failed: {result.stderr}")
        except Exception as e:
            print(f"✗ main.py - Execution failed: {e}")
    else:
        print("✗ main.py - File not found")

def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    dependencies = [
        'torch',
        'torchtext',
        'matplotlib',
        'seaborn',
        'numpy',
        'tqdm',
        'argparse'
    ]
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✓ {dep} - Available")
        except ImportError:
            print(f"✗ {dep} - Not available")

def main():
    """Run all tests."""
    print("Language Modeling Pipeline - Structure Test")
    print("=" * 50)
    
    test_dependencies()
    test_directory_structure()
    test_imports()
    test_main_script()
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == '__main__':
    main() 