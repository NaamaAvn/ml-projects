"""
Configuration file for the Language Modeling Pipeline.
Centralizes all parameters and settings for easy modification.
"""

import os

# Directory Configuration
DATA_DIR = './data/processed_lm_data/'
MODEL_DIR = './model/'
EDA_PLOTS_DIR = os.path.join(DATA_DIR, 'eda_plots')

# Data Processing Configuration
SEQUENCE_LENGTH = 50
BATCH_SIZE = 32
MIN_FREQ = 2  # Minimum word frequency for vocabulary
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']

# Model Configuration
DEFAULT_MODEL_CONFIG = {
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.001
}

FAST_MODEL_CONFIG = {
    'embedding_dim': 64,
    'hidden_dim': 128,
    'num_layers': 1,
    'dropout': 0.2,
    'learning_rate': 0.01
}

# Training Configuration
DEFAULT_TRAINING_CONFIG = {
    'epochs': 10,
    'batch_size': 32,
    'gradient_clip': 1.0,
    'save_best': True,
    'early_stopping_patience': 5
}

# Text Generation Configuration
GENERATION_CONFIG = {
    'max_length': 50,
    'temperature': 0.8,
    'top_k': 50,
    'top_p': 0.9
}

# EDA Configuration
EDA_CONFIG = {
    'sequence_length_bins': 50,
    'word_length_bins': 30,
    'top_words_count': 20,
    'figure_size': (10, 6)
}

# File Paths
FILE_PATHS = {
    'data_splits': os.path.join(DATA_DIR, 'data_splits.json'),
    'vocab': os.path.join(DATA_DIR, 'vocab.json'),
    'tokenizer_info': os.path.join(DATA_DIR, 'tokenizer_info.json'),
    'train_sequences': os.path.join(DATA_DIR, 'train.json'),
    'val_sequences': os.path.join(DATA_DIR, 'val.json'),
    'test_sequences': os.path.join(DATA_DIR, 'test.json'),
    'dataset_stats': os.path.join(DATA_DIR, 'dataset_stats.json'),
    'dataloader_configs': os.path.join(DATA_DIR, 'dataloader_configs.json'),
    'model_checkpoint': os.path.join(MODEL_DIR, 'language_model.pth'),
    'training_history': os.path.join(MODEL_DIR, 'training_history.png'),
    'test_results': os.path.join(MODEL_DIR, 'test_results.json')
}

# Validation Configuration
VALIDATION_CONFIG = {
    'train_val_split': 0.9,  # 90% train, 10% validation
    'random_seed': 42
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_logs': True,
    'log_file': os.path.join(MODEL_DIR, 'pipeline.log')
}

def get_model_config(fast_mode=False):
    """Get model configuration based on mode."""
    return FAST_MODEL_CONFIG if fast_mode else DEFAULT_MODEL_CONFIG

def get_training_config(fast_mode=False, **kwargs):
    """Get training configuration with optional overrides."""
    config = DEFAULT_TRAINING_CONFIG.copy()
    
    if fast_mode:
        config['epochs'] = 5  # Fewer epochs for fast mode
    
    # Override with any provided kwargs
    config.update(kwargs)
    
    return config

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [DATA_DIR, MODEL_DIR, EDA_PLOTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_file_path(file_key):
    """Get file path by key."""
    return FILE_PATHS.get(file_key, None)

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check for valid sequence length
    if SEQUENCE_LENGTH <= 0:
        errors.append("SEQUENCE_LENGTH must be positive")
    
    # Check for valid batch size
    if BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")
    
    # Check for valid model parameters
    for config_name, config in [('DEFAULT', DEFAULT_MODEL_CONFIG), ('FAST', FAST_MODEL_CONFIG)]:
        if config['embedding_dim'] <= 0:
            errors.append(f"{config_name}_MODEL_CONFIG embedding_dim must be positive")
        if config['hidden_dim'] <= 0:
            errors.append(f"{config_name}_MODEL_CONFIG hidden_dim must be positive")
        if config['num_layers'] <= 0:
            errors.append(f"{config_name}_MODEL_CONFIG num_layers must be positive")
        if not 0 <= config['dropout'] <= 1:
            errors.append(f"{config_name}_MODEL_CONFIG dropout must be between 0 and 1")
        if config['learning_rate'] <= 0:
            errors.append(f"{config_name}_MODEL_CONFIG learning_rate must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True 