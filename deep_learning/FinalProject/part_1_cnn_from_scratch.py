"""
Part 1: CNN from Scratch - Winter vs Summer Classification

This module implements and compares multiple CNN architectures for binary image classification
(winter vs summer). It includes:
- Two model architectures: SimpleCNN (baseline) and AdvancedCNN (deeper with regularization)
- Multiple experiments comparing different hyperparameters and training strategies
- Comprehensive training/validation tracking and visualization

The experiments compare:
- Different optimizers (Adam vs SGD with momentum)
- Batch normalization effects
- Dropout rates
- Weight decay regularization
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- 1. Global Setup ---
def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_DIR = '../Exercise1/data/organized/' 

# --- 2. Define Architectures ---

class SimpleCNN(nn.Module):
    """
    A simple CNN architecture with 2 convolutional layers.
    
    Architecture:
        - Conv1: 3 -> 16 channels (3x3 kernel)
        - MaxPool (2x2)
        - Conv2: 16 -> 32 channels (3x3 kernel)
        - MaxPool (2x2)
        - FC1: flattened -> 128 units
        - FC2: 128 -> 2 classes (winter/summer)
    
    Input size: 256x256x3
    Output size: 2 (logits for binary classification)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 256 -> 128 -> 64
        self.flatten_size = 32 * 64 * 64 
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 256, 256)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_size) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AdvancedCNN(nn.Module):
    """
    A deeper CNN architecture with 4 convolutional blocks and regularization.
    
    Architecture:
        - 4 convolutional blocks, each with:
            - Conv layer (with channel progression: 32 -> 64 -> 128 -> 256)
            - Optional Batch Normalization
            - ReLU activation
            - MaxPool (2x2)
        - FC1: flattened -> 512 units
        - Dropout (configurable rate)
        - FC2: 512 -> 2 classes
    
    Args:
        use_bn (bool): Whether to use batch normalization. Default is True.
        dropout_rate (float): Dropout probability after first FC layer. Default is 0.5.
    
    Input size: 256x256x3
    Output size: 2 (logits for binary classification)
    """
    def __init__(self, use_bn=True, dropout_rate=0.5):
        super(AdvancedCNN, self).__init__()
        self.use_bn = use_bn
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 256x256 -> 4 pools -> 16x16
        self.flatten_size = 256 * 16 * 16
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 256, 256)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        def block(x, conv, bn):
            x = conv(x)
            if self.use_bn: x = bn(x)
            return self.pool(F.relu(x))

        x = block(x, self.conv1, self.bn1)
        x = block(x, self.conv2, self.bn2)
        x = block(x, self.conv3, self.bn3)
        x = block(x, self.conv4, self.bn4)
        
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 3. Helper Functions ---

def get_data_loaders(data_dir, batch_size, use_augmentation=True):
    """
    Create train and validation data loaders with appropriate transformations.
    
    Args:
        data_dir (str): Path to the directory containing 'train' and 'validation' subdirectories
        batch_size (int): Number of samples per batch
        use_augmentation (bool): Whether to apply data augmentation to training set. Default is True.
    
    Returns:
        tuple: (train_loader, val_loader) - DataLoader objects for training and validation
        
    Note:
        Training augmentation includes: random horizontal flip, random rotation (±15°),
        and color jitter. Validation uses only resize and normalization.
    """
    if use_augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform=val_transforms)
    
    workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    return train_loader, val_loader

def get_optimizer(model, opt_name, lr, weight_decay=0):
    """
    Create an optimizer for the given model.
    
    Args:
        model (nn.Module): The neural network model
        opt_name (str): Optimizer name - either 'adam' or 'sgd_momentum'
        lr (float): Learning rate
        weight_decay (float): L2 regularization coefficient. Default is 0.
    
    Returns:
        torch.optim.Optimizer: Configured optimizer
        
    Raises:
        ValueError: If opt_name is not recognized
    """
    if opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd_momentum':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def run_experiment(config, experiment_name):
    """
    Run a single training experiment with the given configuration.
    
    This function handles the complete training loop including:
    - Data loading with appropriate augmentation
    - Model initialization based on config
    - Training and validation for specified number of epochs
    - Tracking of loss and accuracy metrics
    
    Args:
        config (dict): Configuration dictionary with keys:
            - model_type (str): 'simple' or 'advanced'
            - optimizer (str): 'adam' or 'sgd_momentum'
            - lr (float): Learning rate
            - batch_size (int): Batch size for data loaders
            - epochs (int): Number of training epochs
            - augmentation (bool, optional): Use data augmentation. Default True.
            - use_bn (bool, optional): Use batch norm (for advanced model). Default True.
            - dropout (float, optional): Dropout rate (for advanced model). Default 0.5.
            - weight_decay (float, optional): L2 regularization. Default 0.
        experiment_name (str): Name of the experiment for logging
    
    Returns:
        dict: Training history with keys:
            - 'train_loss': List of average training loss per epoch
            - 'train_acc': List of training accuracy per epoch
            - 'val_loss': List of validation loss per epoch
            - 'val_acc': List of validation accuracy per epoch
    """
    print(f"\n=== Running: {experiment_name} ===")
    print(f"Config: {config}")
    set_seed(42) 
    
    use_aug = config.get('augmentation', True)
    train_loader, val_loader = get_data_loaders(DATA_DIR, config['batch_size'], use_aug)
    
    if config['model_type'] == 'simple':
        model = SimpleCNN().to(device)
    else:
        use_bn = config.get('use_bn', True)
        dropout = config.get('dropout', 0.5) 
        model = AdvancedCNN(use_bn=use_bn, dropout_rate=dropout).to(device)
        
    wd = config.get('weight_decay', 0)
    optimizer = get_optimizer(model, config['optimizer'], config['lr'], weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    # Updated History to store everything
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(config['epochs']):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_total += labels.size(0)
            
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct.double() / train_total
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels) # Calculate val loss
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct.double() / val_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc.item())
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc.item())
        
        print(f"Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}")

    return history

def main():
    """
    Main execution function that runs all experiments and generates visualizations.
    
    Experiments conducted:
        1. Simple_Adam: Baseline SimpleCNN with Adam optimizer
        2. Simple_SGD: SimpleCNN with SGD+momentum
        3. Deep_NoBN: AdvancedCNN without batch normalization
        4. Deep_BN_HighDrop: AdvancedCNN with BN and 50% dropout
        5. Deep_BN_LowDrop: AdvancedCNN with BN and 10% dropout
        6. Deep_BN_LowDrop_SGD: Same as #5 but with SGD optimizer
        7. Deep_WeightDecay: #6 with added L2 regularization
    
    Generates two visualization files:
        - comparison_val_accuracy.png: Line plot comparing validation accuracy across experiments
        - detailed_learning_curves.png: Grid of subplots showing train/val loss and accuracy
    """
    experiments = {
        '1_Simple_Adam': {
            'model_type': 'simple', 'optimizer': 'adam', 'lr': 0.001, 
            'batch_size': 32, 'epochs': 8
        },
        '2_Simple_SGD': {
            'model_type': 'simple', 'optimizer': 'sgd_momentum', 'lr': 0.01, 
            'batch_size': 32, 'epochs': 8
        },
        '3_Deep_NoBN': { 
            'model_type': 'advanced', 'optimizer': 'adam', 'lr': 0.001, 
            'batch_size': 32, 'epochs': 8, 'use_bn': False, 'dropout': 0.5
        },
        '4_Deep_BN_HighDrop': { 
            'model_type': 'advanced', 'optimizer': 'adam', 'lr': 0.001, 
            'batch_size': 32, 'epochs': 8, 'use_bn': True, 'dropout': 0.5
        },
        '5_Deep_BN_LowDrop': { 
            'model_type': 'advanced', 'optimizer': 'adam', 'lr': 0.001, 
            'batch_size': 32, 'epochs': 8, 'use_bn': True, 'dropout': 0.1 
        },
        '6_Deep_BN_LowDrop_SGD': { 
            'model_type': 'advanced', 'optimizer': 'sgd_momentum', 'lr': 0.01, 
            'batch_size': 32, 'epochs': 8, 'use_bn': True, 'dropout': 0.1 
        },
        '7_Deep_WeightDecay': { 
            'model_type': 'advanced', 'optimizer': 'sgd_momentum', 'lr': 0.01, 
            'batch_size': 32, 'epochs': 8, 'use_bn': True, 'dropout': 0.1,
            'weight_decay': 1e-4
        }
    }
    
    results = {}
    print(f"Starting execution on: {device}")
    
    for name, config in experiments.items():
        results[name] = run_experiment(config, name)
        
    # --- Plot 1: Validation Accuracy Comparison (The "Main" Graph) ---
    plt.figure(figsize=(10, 6))
    for name, hist in results.items():
        best_acc = max(hist['val_acc'])
        label = f"{name} (Best: {best_acc:.3f})"
        plt.plot(hist['val_acc'], label=label, linewidth=2)
    
    plt.title('Validation Accuracy Comparison: All Experiments')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparison_val_accuracy.png', dpi=300)
    print("Graph saved to 'comparison_val_accuracy.png'")
    
    # --- Plot 2: Detailed Learning Curves (Train vs Val) ---
    # Creating a subplot grid to show detailed behavior for each experiment
    num_exps = len(experiments)
    cols = 2
    rows = (num_exps + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i, (name, hist) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot Loss (Left Y-axis)
        l1 = ax.plot(hist['train_loss'], label='Train Loss', color='tab:blue', linestyle='--')[0]
        l2 = ax.plot(hist['val_loss'], label='Val Loss', color='tab:blue')[0]
        ax.set_ylabel('Loss', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        
        # Plot Accuracy (Right Y-axis)
        ax2 = ax.twinx()
        l3 = ax2.plot(hist['train_acc'], label='Train Acc', color='tab:orange', linestyle='--')[0]
        l4 = ax2.plot(hist['val_acc'], label='Val Acc', color='tab:orange')[0]
        ax2.set_ylabel('Accuracy', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        ax.set_title(f"Experiment: {name}")
        ax.set_xlabel("Epochs")
        ax.grid(True, alpha=0.3)
        
        # Legend
        lines = [l1, l2, l3, l4]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')

    plt.tight_layout()
    plt.savefig('detailed_learning_curves.png', dpi=300)
    print("Detailed curves saved to 'detailed_learning_curves.png'")
    
    plt.show()

if __name__ == "__main__":
    main()