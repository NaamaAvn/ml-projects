"""
Part 2: Transfer Learning - Winter vs Summer Classification

This module demonstrates the power of transfer learning by:
1. Pretraining an AdvancedCNN model on CIFAR-10 (external dataset with 10 classes)
2. Fine-tuning the pretrained model on the target Winter/Summer dataset (2 classes)
3. Comparing performance against training from scratch (Part 1)

Key Concepts Demonstrated:
- Pretraining on a large external dataset to learn general features
- Transfer learning by replacing the final classification layer
- Fine-tuning with a lower learning rate to preserve learned features
- Performance comparison showing the benefits of transfer learning

The approach shows how knowledge from one task (CIFAR-10 classification) can be
transferred to improve performance on a different but related task (season classification).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration & Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path to Summer/Winter dataset
DATA_DIR = '../Exercise1/data/organized/' 


class AdvancedCNN(nn.Module):
    """
    A deeper CNN architecture with 4 convolutional blocks and regularization.
    
    This architecture is designed to be flexible and support transfer learning:
    - Can be trained on different numbers of classes (e.g., 10 for CIFAR-10, 2 for binary)
    - Includes batch normalization for stable training
    - Uses dropout for regularization
    
    Architecture:
        - 4 convolutional blocks, each with:
            - Conv layer (channel progression: 32 -> 64 -> 128 -> 256)
            - Optional Batch Normalization
            - ReLU activation
            - MaxPool (2x2)
        - FC1: flattened -> 512 units
        - Dropout (configurable rate)
        - FC2: 512 -> num_classes
    
    Args:
        num_classes (int): Number of output classes. Default is 2.
        use_bn (bool): Whether to use batch normalization. Default is True.
        dropout_rate (float): Dropout probability after first FC layer. Default is 0.1.
    
    Input size: 256x256x3
    Output size: num_classes (logits)
    """
    def __init__(self, num_classes=2, use_bn=True, dropout_rate=0.1):
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
        
        # Flatten size: Input 256x256 -> pooled 4 times -> 16x16
        self.flatten_size = 256 * 16 * 16
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        
        # Dynamic output layer to support CIFAR-10 (10 classes) and our task (2 classes)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 256, 256)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
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

# --- Helper Functions for Training ---

def train_epoch(model, loader, criterion, optimizer):
    """
    Execute one training epoch.
    
    Args:
        model (nn.Module): The neural network model
        loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer for parameter updates
    
    Returns:
        tuple: (average_loss, accuracy) for the epoch
            - average_loss (float): Mean loss across all samples
            - accuracy (float): Classification accuracy (0-1)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
    return running_loss / total, correct.double() / total

def validate_epoch(model, loader, criterion):
    """
    Execute one validation epoch.
    
    Args:
        model (nn.Module): The neural network model
        loader (DataLoader): Validation data loader
        criterion: Loss function
    
    Returns:
        tuple: (average_loss, accuracy) for the epoch
            - average_loss (float): Mean loss across all samples
            - accuracy (float): Classification accuracy (0-1)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
    return running_loss / total, correct.double() / total

# --- STEP 4: Pretraining on External Dataset (CIFAR-10) ---

def step4_pretraining_cifar():
    """
    Step 4: Pretrain the model on CIFAR-10 dataset.
    
    This function demonstrates the first phase of transfer learning:
    1. Downloads and prepares CIFAR-10 (10-class image classification dataset)
    2. Resizes CIFAR-10 images (32x32) to match our network input (256x256)
    3. Trains the AdvancedCNN model on CIFAR-10 for 5 epochs
    4. Saves the pretrained weights for later fine-tuning
    
    Purpose: The model learns general low-level and mid-level features (edges, textures,
    patterns) from CIFAR-10 that can be useful for the target season classification task.
    
    Note: Uses only 20% of CIFAR-10 (10,000 samples) to reduce training time.
    
    Saves:
        'cifar10_pretrained.pth': Model weights after pretraining
    """
    print("\n=== Step 4: Pretraining on CIFAR-10 ===")
    
    # 1. Data Preparation
    # We must resize CIFAR (32x32) to match our network input (256x256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Downloading CIFAR-10...")
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Optimization: Use a subset (20%) to save time during this exercise
    indices = torch.arange(10000) 
    cifar_subset = Subset(cifar_train, indices)
    
    train_loader = DataLoader(cifar_subset, batch_size=32, shuffle=True, num_workers=2)
    
    # 2. Model Initialization
    # Initialize with 10 classes because CIFAR-10 has 10 categories
    model = AdvancedCNN(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 5 
    print(f"Training on CIFAR-10 for {epochs} epochs...")
    for epoch in range(epochs):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Pretraining Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        
    # 4. Save Weights
    torch.save(model.state_dict(), 'cifar10_pretrained.pth')
    print("Saved pretrained weights to 'cifar10_pretrained.pth'")

# --- STEP 5: Fine-Tuning on Target Dataset (Summer vs Winter) ---

def step5_finetuning_target():
    """
    Step 5: Fine-tune the pretrained model on Winter/Summer dataset.
    
    This function demonstrates the second phase of transfer learning:
    1. Loads the CIFAR-10 pretrained weights
    2. Performs "model surgery" by replacing the final classification layer
       (from 10 classes to 2 classes)
    3. Fine-tunes the entire model on the target dataset with a lower learning rate
    4. Tracks and returns validation accuracy for each epoch
    
    Key Transfer Learning Concepts:
    - Lower learning rate (0.0005 vs 0.001) preserves pretrained features
    - Only the final layer starts with random weights; all other layers retain
      knowledge from CIFAR-10
    - The pretrained features (edges, textures, patterns) help the model learn
      the new task faster and better
    
    Returns:
        dict: Training history containing:
            - 'train_loss': Training loss for each epoch
            - 'train_acc': Training accuracy for each epoch
            - 'val_loss': Validation loss for each epoch
            - 'val_acc': Validation accuracy for each epoch
    """
    print("\n=== Step 5: Fine-Tuning on Summer/Winter Dataset ===")
    
    # 1. Prepare Target Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(), # Augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'validation'), transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    
    # 2. Load Pretrained Model
    print("Loading pretrained weights...")
    # Initialize the model structure with 10 classes to match the saved weights file
    model = AdvancedCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load('cifar10_pretrained.pth'))
    
    # 3. Modify the Classification Layer (The "Surgery")
    # We replace the final layer (10 outputs) with a new layer (2 outputs)
    # The weights for this new layer are initialized randomly.
    # The rest of the network retains the knowledge from CIFAR.
    model.fc2 = nn.Linear(512, 2).to(device)
    print("Replaced final layer for binary classification (2 classes).")
    
    # 4. Optimizer Setup
    criterion = nn.CrossEntropyLoss()
    # Note: We use a lower Learning Rate (0.0005) for fine-tuning 
    # to avoid destroying the pretrained features too quickly.
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # 5. Fine-Tuning Loop
    history_ft = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    epochs = 8
    
    print("Starting Fine-Tuning...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        # Track all metrics
        history_ft['train_loss'].append(train_loss)
        history_ft['train_acc'].append(train_acc.item())
        history_ft['val_loss'].append(val_loss)
        history_ft['val_acc'].append(val_acc.item())
        
        print(f"Fine-Tuning Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    return history_ft

def plot_training_curves(history):
    """
    Plot detailed training and validation curves for Step 5.
    
    Creates a dual-axis plot showing:
    - Training and validation loss (left y-axis, blue)
    - Training and validation accuracy (right y-axis, orange)
    
    Args:
        history (dict): Training history from step5_finetuning_target()
    
    Saves:
        'step5_training_curves.png': Detailed learning curves visualization
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Loss (Left Y-axis)
    l1 = ax1.plot(history['train_loss'], label='Train Loss', color='tab:blue', linestyle='--', linewidth=2)[0]
    l2 = ax1.plot(history['val_loss'], label='Val Loss', color='tab:blue', linewidth=2)[0]
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy (Right Y-axis)
    ax2 = ax1.twinx()
    l3 = ax2.plot(history['train_acc'], label='Train Acc', color='tab:orange', linestyle='--', linewidth=2)[0]
    l4 = ax2.plot(history['val_acc'], label='Val Acc', color='tab:orange', linewidth=2)[0]
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    # Title and Legend
    plt.title('Step 5: Fine-Tuning Training & Validation Curves')
    
    # Combine legends from both axes
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    plt.savefig('step5_training_curves.png', dpi=300)
    print("Step 5 training curves saved to 'step5_training_curves.png'")
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    """
    Main execution pipeline for transfer learning demonstration.
    
    Pipeline:
        1. Pretrain model on CIFAR-10 (external dataset) - Step 4
        2. Fine-tune on Winter/Summer dataset (target task) - Step 5
        3. Compare results with training from scratch (Part 1)
        4. Generate and save comparison visualization
    
    Output:
        - 'cifar10_pretrained.pth': Pretrained model weights
        - 'step5_training_curves.png': Training/validation curves for fine-tuning
        - 'transfer_learning_comparison.png': Performance comparison plot
    
    The visualization demonstrates the advantage of transfer learning by showing
    that fine-tuning a pretrained model typically achieves better accuracy faster
    than training from scratch.
    """
    # Part A: Pretrain on external data
    step4_pretraining_cifar()
    
    # Part B: Fine-tune on our data
    ft_history = step5_finetuning_target()
    
    # Part C: Plot Training & Validation Curves for Step 5
    print("\n=== Generating Training & Validation Curves for Step 5 ===")
    plot_training_curves(ft_history)
    
    # Part D: Comparison Visualization (Transfer Learning vs From Scratch)
    # These values representing our best result from Part 1 
    scratch_results = [0.65, 0.71, 0.74, 0.74, 0.75, 0.74, 0.76, 0.75] 
    
    plt.figure(figsize=(10, 6))
    plt.plot(scratch_results, label='Training from Scratch (Part 1)', linestyle='--', linewidth=2)
    plt.plot(ft_history['val_acc'], label='Transfer Learning (Part 2)', linewidth=2)
    
    plt.title('Impact of Transfer Learning')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('transfer_learning_comparison.png', dpi=300)
    print("Comparison graph saved to 'transfer_learning_comparison.png'")
    plt.show()