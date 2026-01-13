#!/usr/bin/env python3
"""
Main script for the Language Modeling pipeline on IMDB dataset.

This script orchestrates the entire pipeline:
1. Load and split data
2. Analyze data and create EDA plots
3. Create vocabulary
4. Create sequences and dataloaders
5. Train language model
6. Test the model

Each step can be run independently or as part of the full pipeline.
"""

import os
import sys
import subprocess
import argparse
import json
from typing import Dict, Any

def run_step(step_name: str, script_path: str, args: list = None) -> bool:
    """
    Run a single step of the pipeline.
    
    Args:
        step_name: Name of the step for logging
        script_path: Path to the script to run
        args: Additional arguments to pass to the script
        
    Returns:
        True if step completed successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running {step_name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n{step_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {step_name}: {e}")
        return False

def check_step_prerequisites(step: int, data_dir: str = './data/', 
                           model_dir: str = './model/') -> bool:
    """
    Check if prerequisites are met for a given step.
    
    Args:
        step: Step number (1-6)
        data_dir: Data directory
        model_dir: Model directory
        
    Returns:
        True if prerequisites are met, False otherwise
    """
    if step == 1:
        # Step 1 has no prerequisites
        return True
    
    elif step == 2:
        # Step 2 requires data_splits.json
        return os.path.exists(f'{data_dir}/data_splits.json')
    
    elif step == 3:
        # Step 3 requires data_splits.json
        return os.path.exists(f'{data_dir}/data_splits.json')
    
    elif step == 4:
        # Step 4 requires data_splits.json and vocab.json
        return (os.path.exists(f'{data_dir}/data_splits.json') and 
                os.path.exists(f'{data_dir}/vocab.json'))
    
    elif step == 5:
        # Step 5 requires processed sequences and vocabulary
        return (os.path.exists(f'{data_dir}/vocab.json') and 
                os.path.exists(f'{data_dir}/processed_lm_data/train.json') and
                os.path.exists(f'{data_dir}/processed_lm_data/val.json'))
    
    elif step == 6:
        # Step 6 requires trained model and test data
        return (os.path.exists(f'{model_dir}/language_model.pth') and 
                os.path.exists(f'{data_dir}/processed_lm_data/test.json'))
    
    return False

def run_full_pipeline(data_dir: str = './data/',
                     model_dir: str = './model/',
                     plots_dir: str = './plots/',
                     results_dir: str = './results/',
                     fast_mode: bool = False,
                     epochs: int = 10,
                     batch_size: int = 32,
                     data_ratio: float = 1.0,
                     test_ratio: float = 1.0,
                     use_async: bool = False) -> bool:
    """
    Run the complete pipeline from start to finish.
    
    Args:
        data_dir: Directory for data processing
        model_dir: Directory for model files
        plots_dir: Directory to save plots
        results_dir: Directory to save results
        fast_mode: Whether to use fast training parameters
        epochs: Number of training epochs
        batch_size: Batch size for training
        data_ratio: Ratio of data to use for training/validation
        test_ratio: Ratio of data to use for testing
        use_async: Whether to use async training for better latency
        
    Returns:
        True if pipeline completed successfully, False otherwise
    """
    print("Starting Language Modeling Pipeline")
    print("=" * 60)
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load and split data
    if not run_step("Step 1: Load and Split Data",
                   './scripts/01_load_and_split_data.py',
                   ['--output-dir', data_dir]):
        return False
    
    # Step 2: Analyze data and create EDA plots
    if not run_step("Step 2: Analyze Data and Create EDA Plots",
                   './scripts/02_analyze_data.py',
                   ['--data-dir', data_dir, '--output-dir', data_dir]):
        return False
    
    # Step 3: Create vocabulary
    if not run_step("Step 3: Create Vocabulary",
                   './scripts/03_create_vocabulary.py',
                   ['--data-dir', data_dir, '--output-dir', data_dir]):
        return False
    
    # Step 4: Create sequences and dataloaders
    if not run_step("Step 4: Create Sequences and DataLoaders",
                   './scripts/04_lm_create_sequences.py',
                   ['--data-dir', data_dir, '--output-dir', data_dir + '/processed_lm_data/',
                    '--batch-size', str(batch_size),
                    '--data-ratio', str(data_ratio),
                    '--test-ratio', str(test_ratio)]):
        return False
    
    # Step 5: Train language model
    train_args = [
        '--data-dir', data_dir,
        '--model-dir', model_dir,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size)
    ]
    
    if fast_mode:
        train_args.extend([
            '--embedding-dim', '64',
            '--hidden-dim', '128',
            '--num-layers', '1',
            '--dropout', '0.2',
            '--learning-rate', '0.01'
        ])
    
    if use_async:
        train_args.append('--use-async')
    
    if not run_step("Step 5: Train Language Model",
                   './train.py',
                   train_args):
        return False
    
    # Step 6: Evaluate the model
    eval_args = [
        '--data-dir', data_dir,
        '--model-path', f'{model_dir}/language_model.pth',
        '--vocab-path', f'{data_dir}/vocab.json',
        '--batch-size', str(batch_size),
        '--results-dir', results_dir
    ]
    
    if not run_step("Step 6: Evaluate Model", './evaluate.py', eval_args):
        return False
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    return True

def run_single_step(step: int, data_dir: str = './data/',
                   model_dir: str = './model/',
                   plots_dir: str = './plots/',
                   results_dir: str = './results/',
                   data_ratio: float = 1.0,
                   test_ratio: float = 1.0,
                   **kwargs) -> bool:
    """
    Run a single step of the pipeline.
    
    Args:
        step: Step number (1-6)
        data_dir: Data directory
        model_dir: Model directory
        plots_dir: Directory to save plots
        results_dir: Directory to save results
        data_ratio: Ratio of data to use for training/validation (0.1 = 10%%, 1.0 = 100%%)
        test_ratio: Ratio of data to use for testing (0.1 = 10%%, 1.0 = 100%%)
        **kwargs: Additional arguments for the step
        
    Returns:
        True if step completed successfully, False otherwise
    """
    if not check_step_prerequisites(step, data_dir, model_dir):
        print(f"Prerequisites not met for step {step}")
        print("Please run previous steps first.")
        return False
    
    step_configs = {
        1: {
            'name': 'Load and Split Data',
            'script': './scripts/01_load_and_split_data.py',
            'args': ['--output-dir', data_dir]
        },
        2: {
            'name': 'Analyze Data and Create EDA Plots',
            'script': './scripts/02_analyze_data.py',
            'args': ['--data-dir', data_dir, '--output-dir', data_dir]
        },
        3: {
            'name': 'Create Vocabulary',
            'script': './scripts/03_create_vocabulary.py',
            'args': ['--data-dir', data_dir, '--output-dir', data_dir]
        },
        4: {
            'name': 'Create Sequences and DataLoaders',
            'script': './scripts/04_lm_create_sequences.py',
            'args': ['--data-dir', data_dir, '--output-dir', data_dir + '/processed_lm_data/',
                    '--batch-size', str(kwargs.get('batch_size', 32)),
                    '--data-ratio', str(data_ratio),
                    '--test-ratio', str(test_ratio)]
        },
        5: {
            'name': 'Train Language Model',
            'script': './train.py',
            'args': ['--data-dir', data_dir, '--model-dir', model_dir,
                    '--epochs', str(kwargs.get('epochs', 10)),
                    '--batch-size', str(kwargs.get('batch_size', 32))]
        },
        6: {
            'name': 'Evaluate Model',
            'script': './evaluate.py',
            'args': ['--data-dir', data_dir, '--model-path', f'{model_dir}/language_model.pth',
                     '--vocab-path', f'{data_dir}/vocab.json', '--batch-size', str(kwargs.get('batch_size', 32)),
                     '--results-dir', results_dir]
        }
    }
    
    if step not in step_configs:
        print(f"Invalid step number: {step}")
        return False
    
    config = step_configs[step]
    
    # Add fast mode parameters for step 5 if requested
    if step == 5 and kwargs.get('fast_mode', False):
        config['args'].extend([
            '--embedding-dim', '64',
            '--hidden-dim', '128',
            '--num-layers', '1',
            '--dropout', '0.2',
            '--learning-rate', '0.01'
        ])
    
    # Add async training parameter for step 5 if requested
    if step == 5 and kwargs.get('use_async', False):
        config['args'].append('--use-async')
    
    return run_step(config['name'], config['script'], config['args'])

def print_pipeline_status(data_dir: str = './data/',
                         model_dir: str = './model/') -> None:
    """
    Print the status of the pipeline (which steps have been completed).
    
    Args:
        data_dir: Data directory
        model_dir: Model directory
    """
    print("\nPipeline Status:")
    print("=" * 40)
    
    steps = [
        (1, "Load and Split Data", [f'{data_dir}/data_splits.json']),
        (2, "Analyze Data", [f'{data_dir}/dataset_stats.json', f'{data_dir}/eda_plots/']),
        (3, "Create Vocabulary", [f'{data_dir}/vocab.json']),
        (4, "Create Sequences", [f'{data_dir}/processed_lm_data/train.json', f'{data_dir}/processed_lm_data/val.json', f'{data_dir}/processed_lm_data/test.json']),
        (5, "Train Model", [f'{model_dir}/language_model.pth']),
        (6, "Test Model", [f'{model_dir}/language_model.pth', f'{data_dir}/processed_lm_data/test.json'])
    ]
    
    for step_num, step_name, required_files in steps:
        completed = all(os.path.exists(file) for file in required_files)
        status = "✓ Completed" if completed else "✗ Not completed"
        print(f"Step {step_num}: {step_name} - {status}")

def main():
    """Main function to handle command line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description='Language Modeling Pipeline on IMDB Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --full
  
  # Run full pipeline in fast mode
  python main.py --full --fast
  
  # Run single step
  python main.py --step 3
  
  # Run step 6 (evaluation)
  python main.py --step 6
  
  # Check pipeline status
  python main.py --status
  
  # Run step 5 with custom parameters
  python main.py --step 5 --epochs 15 --batch-size 64
        """
    )
    
    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--full', action='store_true',
                             help='Run the complete pipeline')
    action_group.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5, 6],
                             help='Run a specific step (1-6)')
    action_group.add_argument('--status', action='store_true',
                             help='Check pipeline status')
    
    # General arguments
    parser.add_argument('--data-dir', type=str, default='./data/',
                       help='Directory for data processing')
    parser.add_argument('--model-dir', type=str, default='./model/',
                       help='Directory for model files')
    parser.add_argument('--plots-dir', type=str, default='./plots/',
                       help='Directory to save plots')
    parser.add_argument('--results-dir', type=str, default='./results/',
                       help='Directory to save results')
    
    # Training arguments
    parser.add_argument('--fast', action='store_true',
                       help='Use fast training parameters (smaller model, faster training)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training/testing')
    parser.add_argument('--use-async', action='store_true',
                       help='Use async training for better latency')
    
    # Data subset arguments
    parser.add_argument('--data-ratio', type=float, default=1.0,
                       help='Ratio of data to use for training/validation (0.1 = 10%%, 1.0 = 100%%)')
    parser.add_argument('--test-ratio', type=float, default=1.0,
                       help='Ratio of data to use for testing (0.1 = 10%%, 1.0 = 100%%)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.status:
        print_pipeline_status(args.data_dir, args.model_dir)
        return
    
    elif args.full:
        success = run_full_pipeline(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            plots_dir=args.plots_dir,
            results_dir=args.results_dir,
            fast_mode=args.fast,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_ratio=args.data_ratio,
            test_ratio=args.test_ratio,
            use_async=args.use_async
        )
        if not success:
            sys.exit(1)
    
    elif args.step:
        success = run_single_step(
            step=args.step,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            plots_dir=args.plots_dir,
            results_dir=args.results_dir,
            data_ratio=args.data_ratio,
            test_ratio=args.test_ratio,
            fast_mode=args.fast,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_async=args.use_async
        )
        if not success:
            sys.exit(1)

if __name__ == '__main__':
    main()
