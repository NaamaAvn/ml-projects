#!/usr/bin/env python3
"""
Setup script for first run of the Language Model.

This script ensures all necessary directories exist and provides guidance
for the first training run.
"""

import os
import sys

def setup_directories():
    """Create all necessary directories."""
    directories = [
        './data/processed_lm_data/',
        './data/processed_lm_data/plots/',
        './models/',
        './logs/'
    ]
    
    print("Creating necessary directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}")
    
    print("\nAll directories created successfully!")

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchtext', 
        'matplotlib',
        'seaborn',
        'tqdm',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\nAll dependencies are installed!")
        return True

def show_first_run_instructions():
    """Show instructions for the first run."""
    print("\n" + "="*60)
    print("FIRST RUN INSTRUCTIONS")
    print("="*60)
    print("1. The first run will take longer because it needs to:")
    print("   - Download the IMDB dataset")
    print("   - Process and tokenize all text data")
    print("   - Build vocabulary from training data")
    print("   - Create input-target pairs for language modeling")
    print("   - Save all processed data for future runs")
    print("\n2. Recommended first run:")
    print("   python quick_train.py ultra")
    print("\n3. This will:")
    print("   - Use only 5% of the data")
    print("   - Train for 3 epochs")
    print("   - Take ~2-3 minutes")
    print("   - Save all processed data for faster future runs")
    print("\n4. After the first run, you can use:")
    print("   - python quick_train.py fast    (10% data, 5 epochs)")
    print("   - python quick_train.py medium  (25% data, 10 epochs)")
    print("   - python quick_train.py full    (100% data, 15 epochs)")
    print("\n5. Or use the interactive menu:")
    print("   python quick_train.py")
    print("="*60)

def main():
    """Main setup function."""
    print("🤖 Language Model Setup")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\nPlease install missing dependencies and run this script again.")
        return
    
    # Show instructions
    show_first_run_instructions()
    
    print("\nSetup completed! You're ready to start training.")
    print("\nQuick start:")
    print("python quick_train.py ultra")

if __name__ == '__main__':
    main() 