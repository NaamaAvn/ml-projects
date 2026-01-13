import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import argparse
from typing import List, Tuple, Any

# Import the LanguageModel class from the training script
class LanguageModel(nn.Module):
    """
    Language Model using LSTM architecture for next word prediction.
    """
    
    def __init__(self, vocab: Any, embedding_dim: int = 100, hidden_dim: int = 100, 
                 num_layers: int = 2, dropout: float = 0.2, device: str = 'cpu'):
        super(LanguageModel, self).__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer to predict next word
        self.output_layer = nn.Linear(hidden_dim, self.vocab_size)
        
        # Move model to device
        self.to(device)
    
    def forward(self, input_sequences: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_length = input_sequences.shape
        
        # Convert input indices to embeddings
        embeddings = self.embedding(input_sequences)
        
        # Apply dropout to embeddings
        embeddings = self.dropout(embeddings)
        
        # Pass through LSTM
        lstm_output, (hidden_state, cell_state) = self.lstm(embeddings, hidden)
        
        # Apply dropout to LSTM output
        lstm_output = self.dropout(lstm_output)
        
        # Pass through output layer to get logits
        output_logits = self.output_layer(lstm_output)
        
        return output_logits, (hidden_state, cell_state)
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden_state, cell_state
    
    def calculate_loss(self, output_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Reshape for loss calculation
        batch_size, seq_length, vocab_size = output_logits.shape
        output_logits = output_logits.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # Calculate cross-entropy loss
        loss = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])(output_logits, targets)
        return loss
    
    def generate_text(self, start_tokens: List[str], max_length: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text using the trained model.
        
        Args:
            start_tokens: List of starting tokens
            max_length: Maximum length of generated text
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Generated text as string
        """
        self.eval()
        generated_tokens = start_tokens.copy()
        
        with torch.no_grad():
            # Convert start tokens to indices
            input_indices = [self.vocab[token] for token in start_tokens]
            input_tensor = torch.tensor([input_indices]).to(self.device)
            
            # Initialize hidden state
            hidden = self.init_hidden(1)
            
            for _ in range(max_length):
                # Forward pass
                output_logits, hidden = self.forward(input_tensor, hidden)
                
                # Get the last output (for next word prediction)
                last_output = output_logits[:, -1, :]  # (1, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    last_output = last_output / temperature
                
                # Sample from the distribution
                probs = torch.softmax(last_output, dim=-1)
                next_token_idx = torch.multinomial(probs, 1).item()
                
                # Convert index back to token
                next_token = self.vocab.get_itos()[next_token_idx]
                
                # Stop if we generate an end token
                if next_token in ['<eos>', '<pad>']:
                    break
                
                generated_tokens.append(next_token)
                
                # Update input for next iteration
                input_tensor = torch.tensor([[next_token_idx]]).to(self.device)
        
        return ' '.join(generated_tokens)

# Simple vocabulary wrapper
class SimpleVocab:
    def __init__(self, stoi_dict):
        self.stoi = stoi_dict
        self.itos = {idx: token for token, idx in stoi_dict.items()}
        self._default_index = stoi_dict.get('<unk>', 0)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self._default_index)
    
    def __len__(self):
        return len(self.stoi)
    
    def get_stoi(self):
        return self.stoi
    
    def get_itos(self):
        return self.itos
    
    def set_default_index(self, idx):
        self._default_index = idx

# Custom Dataset class
class IMDBDataset(torch.utils.data.Dataset):
    """Custom Dataset for IMDB language modeling."""
    def __init__(self, data_file: str, vocab: Any, seq_length: int):
        with open(data_file, 'r') as f:
            self.sequences = json.load(f)
        self.vocab = vocab
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.sequences[idx]
        
        # Convert input sequence to indices
        input_tokens = input_seq.split()
        input_indices = [self.vocab[token] for token in input_tokens]
        
        # Convert target sequence to indices
        target_tokens = target_seq.split()
        target_indices = [self.vocab[token] for token in target_tokens]
        
        return torch.tensor(input_indices), torch.tensor(target_indices)

def load_trained_model(model_path: str, data_dir: str) -> LanguageModel:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the saved model
        data_dir: Directory containing vocabulary
        
    Returns:
        Loaded language model
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    # Load vocabulary
    with open(f'{data_dir}/vocab.json', 'r') as f:
        vocab_stoi = json.load(f)
    
    vocab = SimpleVocab(vocab_stoi)
    
    # Check if we have the new format (state_dict + config)
    config_path = model_path.replace('.pth', '_config.json')
    info_path = model_path.replace('.pth', '_info.json')
    
    if os.path.exists(config_path):
        # New format: load state_dict and config separately
        print("Loading model using new format (state_dict + config)...")
        
        # Load model configuration
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Load training info if available
        training_info = {}
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                training_info = json.load(f)
        
        # Initialize model with loaded configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LanguageModel(
            vocab=vocab,
            embedding_dim=model_config['embedding_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config.get('dropout', 0.2),
            device=device
        )
        
        # Load model weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"  - Embedding dimension: {model_config['embedding_dim']}")
        print(f"  - Hidden dimension: {model_config['hidden_dim']}")
        print(f"  - Number of LSTM layers: {model_config['num_layers']}")
        
        if training_info:
            print(f"  - Training completed at epoch {training_info.get('epoch', 'unknown') + 1}")
            print(f"  - Best validation loss: {training_info.get('best_val_loss', 'unknown'):.4f}")
        
        return model
    
    else:
        # Old format: try to load the full checkpoint
        print("Attempting to load model using old format...")
        
        try:
            # Try to load the model checkpoint normally
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Get model configuration
            model_config = checkpoint['model_config']
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LanguageModel(
                vocab=vocab,
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                device=device
            )
            
            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"Model loaded from {model_path}")
            print(f"Training completed at epoch {checkpoint['epoch'] + 1}")
            print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
            
            return model
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Please retrain the model to use the new format.")
            return None

def create_test_dataloader(data_dir: str, vocab: SimpleVocab, batch_size: int = 32) -> DataLoader:
    """Create DataLoader for test set."""
    test_dataset = IMDBDataset(f'{data_dir}/test.json', vocab, 50)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader

def test_model(model: LanguageModel, test_loader: DataLoader, device: str) -> Tuple[float, float]:
    """
    Test the trained language model on test data.
    
    Args:
        model: Trained language model
        test_loader: DataLoader for test data
        device: Device to run the model on
        
    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    print("Evaluating on test set...")
    
    with torch.no_grad():
        for batch_idx, (input_sequences, target_sequences) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to device
            input_sequences = input_sequences.to(device)
            target_sequences = target_sequences.to(device)
            
            # Forward pass
            output_logits, _ = model.forward(input_sequences)
            
            # Calculate loss
            loss = model.calculate_loss(output_logits, target_sequences)
            
            # Calculate accuracy
            batch_size, seq_length, vocab_size = output_logits.shape
            output_logits = output_logits.view(-1, vocab_size)
            targets = target_sequences.view(-1)
            
            # Get predicted tokens
            predicted = torch.argmax(output_logits, dim=1)
            
            # Calculate accuracy (ignore padding tokens)
            mask = targets != model.vocab['<pad>']
            correct = (predicted == targets) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return avg_loss, accuracy

def generate_sample_texts(model: LanguageModel, num_samples: int = 5) -> None:
    """
    Generate sample texts using the trained model.
    
    Args:
        model: Trained language model
        num_samples: Number of sample texts to generate
    """
    print(f"\nGenerating {num_samples} sample texts:")
    print("=" * 60)
    
    sample_prompts = [
        ['the', 'movie', 'was'],
        ['i', 'really', 'liked'],
        ['this', 'film', 'is'],
        ['the', 'acting', 'was'],
        ['the', 'story', 'is'],
        ['this', 'is', 'one'],
        ['i', 'would', 'recommend'],
        ['the', 'director', 'did'],
        ['this', 'movie', 'has'],
        ['the', 'plot', 'was']
    ]
    
    for i in range(min(num_samples, len(sample_prompts))):
        prompt = sample_prompts[i]
        generated_text = model.generate_text(prompt, max_length=30, temperature=0.8)
        print(f"Sample {i+1}:")
        print(f"Prompt: {' '.join(prompt)}")
        print(f"Generated: {generated_text}")
        print("-" * 50)

def save_test_results(test_loss: float, test_accuracy: float, output_dir: str = './model/') -> None:
    """
    Save test results to file.
    
    Args:
        test_loss: Test loss
        test_accuracy: Test accuracy
        output_dir: Directory to save results
    """
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    
    with open(f'{output_dir}/test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Test results saved to {output_dir}/test_results.json")

def main(data_dir: str = './data/', model_dir: str = './model/',
         model_path: str = './model/language_model.pth', batch_size: int = 32,
         num_samples: int = 5):
    """Main function to test the trained model."""
    print("Step 6: Testing the trained Language Model")
    print("=" * 50)
    
    # Check if required files exist
    if not os.path.exists(f'{data_dir}/vocab.json'):
        print(f"Error: vocab.json not found in {data_dir}")
        print("Please run steps 1-4 first.")
        return
    
    if not os.path.exists(f'{data_dir}/processed_lm_data/test.json'):
        print(f"Error: test.json not found in {data_dir}/processed_lm_data/")
        print("Please run steps 1-4 first.")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run step 5 first to train the model.")
        return
    
    # Load vocabulary
    with open(f'{data_dir}/vocab.json', 'r') as f:
        vocab_stoi = json.load(f)
    vocab = SimpleVocab(vocab_stoi)
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Load trained model
    model = load_trained_model(model_path, data_dir)
    if model is None:
        return
    
    # Create test DataLoader
    test_loader = create_test_dataloader(f'{data_dir}/processed_lm_data', vocab, batch_size)
    print(f"Created test DataLoader with {len(test_loader)} batches")
    
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loss, test_accuracy = test_model(model, test_loader, device)
    
    # Print test results
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save test results
    save_test_results(test_loss, test_accuracy, model_dir)
    
    # Generate sample texts
    generate_sample_texts(model, num_samples)
    
    print("\nStep 6 completed successfully!")
    print(f"Test results saved to: {model_dir}/test_results.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the trained Language Model')
    parser.add_argument('--data-dir', type=str, default='./data/',
                       help='Directory containing data (vocab.json and processed_lm_data/)')
    parser.add_argument('--model-dir', type=str, default='./model/',
                       help='Directory containing trained model')
    parser.add_argument('--model-path', type=str, default='./model/language_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of sample texts to generate')
    
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.model_path, args.batch_size, args.num_samples) 