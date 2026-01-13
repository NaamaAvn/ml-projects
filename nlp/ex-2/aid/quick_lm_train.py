#!/usr/bin/env python3
"""
Quick training script for the Language Model.

This script provides easy access to different training modes:
1. Ultra-fast demo (5% data, 3 epochs)
2. Fast training (10% data, 5 epochs) 
3. Medium training (25% data, 10 epochs)
4. Full training (100% data, 15 epochs)
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process_imdb_lm import main, train_fast_demo, train_full_model

def ultra_fast_demo():
    """Ultra-fast demo for testing - 5% data, 3 epochs"""
    print("🚀 ULTRA-FAST DEMO (5% data, 3 epochs)")
    print("="*50)
    main(
        use_existing_data=True,
        fast_training=True,
        data_subset_ratio=0.05,  # 5% of data
        test_subset_ratio=0.05,  # 5% of test data
        max_epochs=3
    )

def fast_training():
    """Fast training - 10% data, 5 epochs"""
    print("⚡ FAST TRAINING (10% data, 5 epochs)")
    print("="*50)
    main(
        use_existing_data=True,
        fast_training=True,
        data_subset_ratio=0.1,  # 10% of data
        test_subset_ratio=0.1,  # 10% of test data
        max_epochs=5
    )

def medium_training():
    """Medium training - 25% data, 10 epochs"""
    print("🏃 MEDIUM TRAINING (25% data, 10 epochs)")
    print("="*50)
    main(
        use_existing_data=True,
        fast_training=True,
        data_subset_ratio=0.25,  # 25% of data
        test_subset_ratio=0.25,  # 25% of test data
        max_epochs=10
    )

def full_training():
    """Full training - 100% data, 15 epochs"""
    print("🏋️ FULL TRAINING (100% data, 15 epochs)")
    print("="*50)
    main(
        use_existing_data=True,
        fast_training=False,
        data_subset_ratio=1.0,  # 100% of data
        test_subset_ratio=1.0,  # 100% of test data
        max_epochs=15
    )

def show_menu():
    """Show training options menu"""
    print("\n🎯 LANGUAGE MODEL TRAINING OPTIONS")
    print("="*50)
    print("1. Ultra-fast demo (5% data, 3 epochs) - ~2-3 minutes")
    print("2. Fast training (10% data, 5 epochs) - ~5-10 minutes")
    print("3. Medium training (25% data, 10 epochs) - ~15-25 minutes")
    print("4. Full training (100% data, 15 epochs) - ~1-2 hours")
    print("5. Exit")
    print("="*50)

def main_menu():
    """Interactive menu for training options"""
    while True:
        show_menu()
        try:
            choice = input("\nSelect training option (1-5): ").strip()
            
            if choice == '1':
                ultra_fast_demo()
            elif choice == '2':
                fast_training()
            elif choice == '3':
                medium_training()
            elif choice == '4':
                full_training()
            elif choice == '5':
                print("Goodbye! 👋")
                break
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == '__main__':
    print("🤖 Language Model Quick Training")
    print("="*50)
    
    # Check if we have command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'ultra':
            ultra_fast_demo()
        elif mode == 'fast':
            fast_training()
        elif mode == 'medium':
            medium_training()
        elif mode == 'full':
            full_training()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: ultra, fast, medium, full")
    else:
        # Interactive mode
        main_menu() 