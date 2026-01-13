"""
Part 3: Using Pretrained ResNet50 - Winter vs Summer Classification

This module demonstrates the power of using state-of-the-art pretrained models:
1. Adapts ResNet50 (pretrained on ImageNet) for binary season classification
2. Compares two transfer learning strategies:
   - Feature Extraction: Freeze backbone, train only the final layer (faster)
   - Fine-Tuning: Train the entire network with a lower learning rate (more adaptive)
3. Compares results against custom CNN from Part 2

Key Concepts:
- Using professionally trained models (ResNet50 on ImageNet with 1M+ images)
- Feature extraction vs full fine-tuning trade-offs
- Proper preprocessing (ImageNet normalization statistics)
- Performance comparison showing benefits of deep pretrained models

This demonstrates that models trained on large-scale datasets can significantly
outperform custom architectures on smaller specialized tasks.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration & Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_DIR = '../Exercise1/data/organized/' 


# --- Step 6: Adapting ResNet50 ---

def get_resnet_model(freeze_backbone=True, num_classes=2):
    """
    Loads a pretrained ResNet50 and adapts it for binary classification.
    
    This function demonstrates two transfer learning strategies:
    1. Feature Extraction (freeze_backbone=True): Only the new final layer is trained.
       The pretrained backbone acts as a fixed feature extractor. Fast but less adaptive.
    2. Fine-Tuning (freeze_backbone=False): The entire network is trained with a low
       learning rate. Slower but can adapt features to the specific task.
    
    Args:
        freeze_backbone (bool): 
            If True, backbone weights are frozen (Feature Extraction mode).
            If False, all weights are trainable (Fine-Tuning mode).
            Default is True.
        num_classes (int): Number of output classes. Default is 2.
    
    Returns:
        nn.Module: Modified ResNet50 model ready for training on the target task.
    """
    print(f"Loading ResNet50 | Freeze Backbone: {freeze_backbone}...")
    
    # 1. Load Pretrained Weights (ImageNet)
    # Using 'DEFAULT' loads the best available weights (IMAGENET1K_V2)
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # 2. Freeze/Unfreeze Layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 3. Replace the Final Classification Layer (The "Head")
    # ResNet's final layer is called 'fc' and has 2048 input features.
    num_ftrs = model.fc.in_features
    
    # We replace it with a new Linear layer for 2 classes.
    # Note: This new layer implies requires_grad=True by default.
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model.to(device)

# --- Data Preparation (Specific for ResNet) ---

def get_resnet_loaders(data_dir, batch_size=32):
    """
    Creates data loaders with ResNet-specific preprocessing.
    
    ResNet models trained on ImageNet require specific preprocessing:
    - Input size: 224x224 (ResNet standard)
    - Normalization: Using ImageNet mean and std statistics
    
    This ensures the input distribution matches what the model was trained on,
    which is critical for good transfer learning performance.
    
    Args:
        data_dir (str): Path to the directory containing 'train' and 'validation' folders
        batch_size (int): Number of samples per batch. Default is 32.
    
    Returns:
        tuple: (train_loader, val_loader) - DataLoader objects for training and validation
    """
    # ImageNet stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet standard input
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

# --- Generic Training Loop ---

def run_training(model, train_loader, val_loader, epochs, lr, name):
    """
    Generic training loop for ResNet experiments.
    
    Trains the model for specified epochs and tracks all training metrics.
    
    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Number of training epochs
        lr (float): Learning rate for Adam optimizer
        name (str): Experiment name for logging
    
    Returns:
        dict: Training history containing:
            - 'train_loss': Training loss for each epoch
            - 'train_acc': Training accuracy for each epoch
            - 'val_loss': Validation loss for each epoch
            - 'val_acc': Validation accuracy for each epoch
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    print(f"--- Starting Training: {name} ---")
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_total += labels.size(0)
        
        train_loss = running_loss / train_total
        train_acc = train_correct.double() / train_total
            
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
        
        val_loss = val_running_loss / val_total
        val_acc = val_correct.double() / val_total
        
        # Track all metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    return history

def plot_training_curves(history, title, filename):
    """
    Plot detailed training and validation curves for a single experiment.
    
    Creates a dual-axis plot showing:
    - Training and validation loss (left y-axis, blue)
    - Training and validation accuracy (right y-axis, orange)
    
    Args:
        history (dict): Training history from run_training()
        title (str): Plot title
        filename (str): Output filename for saving the plot
    
    Saves:
        The specified filename with the learning curves visualization
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
    plt.title(title)
    
    # Combine legends from both axes
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Training curves saved to '{filename}'")
    plt.show()

# --- Main Execution: Step 7 Experiments ---

if __name__ == "__main__":
    """
    Main execution comparing ResNet50 transfer learning strategies.
    
    Experiments:
        A. Feature Extraction (Frozen Backbone):
           - Backbone weights frozen, only final layer trained
           - Fast training (fewer parameters to update)
           - Learning rate: 0.001
           - Good when target task is similar to ImageNet
        
        B. Fine-Tuning (Unfrozen Backbone):
           - All weights trainable
           - Slower training (entire network updates)
           - Learning rate: 0.0001 (lower to preserve pretrained features)
           - Better when target task differs from ImageNet
    
    Output:
        - 'resnet_frozen_training_curves.png': Training/validation curves for frozen backbone
        - 'resnet_finetuning_training_curves.png': Training/validation curves for fine-tuning
        - 'resnet_comparison.png': Comparison plot showing Custom CNN vs ResNet50
          strategies, demonstrating the power of large-scale pretraining
    
    Expected Result:
        ResNet50 typically outperforms custom CNNs due to:
        - Trained on 1M+ ImageNet images
        - Deeper architecture (50 layers vs 4 in custom CNN)
        - Better learned feature representations
    """
    
    # 1. Prepare Data
    train_loader, val_loader = get_resnet_loaders(DATA_DIR)
    
    # --- Experiment A: Feature Extraction (Frozen Backbone) ---
    # Fast training, only the final layer learns.
    print("\n=== Experiment A: Feature Extraction (Frozen Backbone) ===")
    model_frozen = get_resnet_model(freeze_backbone=True)
    hist_frozen = run_training(
        model_frozen, train_loader, val_loader, 
        epochs=8, lr=0.001, name="ResNet50 Frozen"
    )
    
    # Plot training curves for frozen backbone
    print("\nGenerating training curves for Frozen Backbone...")
    plot_training_curves(
        hist_frozen, 
        'ResNet50 Feature Extraction: Training & Validation Curves',
        'resnet_frozen_training_curves.png'
    )
    
    # --- Experiment B: Fine-Tuning (Unfrozen Backbone) ---
    # Slower, allows the whole network to adapt. 
    print("\n=== Experiment B: Fine-Tuning (Unfrozen Backbone) ===")
    model_unfrozen = get_resnet_model(freeze_backbone=False)
    hist_unfrozen = run_training(
        model_unfrozen, train_loader, val_loader, 
        epochs=8, lr=0.0001, name="ResNet50 Fine-Tuning"
    ) # Note lower LR
    
    # Plot training curves for fine-tuning
    print("\nGenerating training curves for Fine-Tuning...")
    plot_training_curves(
        hist_unfrozen,
        'ResNet50 Fine-Tuning: Training & Validation Curves',
        'resnet_finetuning_training_curves.png'
    )
    
    # --- Comparison Plot: Custom CNN vs ResNet50 Strategies ---
    
    # These values representing the best result from Part 2
    custom_cnn_best = [0.72, 0.74, 0.72, 0.73, 0.73, 0.76, 0.77, 0.78] 
    
    print("\n=== Generating Comparison Plot ===")
    plt.figure(figsize=(10, 6))
    plt.plot(custom_cnn_best, label='Custom CNN (Part 2)', linestyle='--', linewidth=2)
    plt.plot(hist_frozen['val_acc'], label='ResNet50 (Frozen)', linewidth=2)
    plt.plot(hist_unfrozen['val_acc'], label='ResNet50 (Fine-Tuning)', linewidth=2)
    
    plt.title('Performance Comparison: Custom vs. ResNet50')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('resnet_comparison.png', dpi=300)
    print("Comparison graph saved to 'resnet_comparison.png'")
    plt.show()