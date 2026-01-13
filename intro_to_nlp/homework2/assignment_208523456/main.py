import json
import time
import csv
import re
from collections import Counter, defaultdict
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import gensim.downloader as api
from gensim.models import KeyedVectors


VOCABULARY_SIZE = 10000
WINDOW_SIZE = 5  # best window size
MAX_LINES = 8000000  # Use first 7-8M lines from Wikipedia dump

# The best pre-trained model to use
PRETRAINED_MODEL = 'word2vec-google-news-300'


def load_train_test_words(train_file: str, test_file: str):
    """Load words from train and test files."""
    words = set()
    train_data = {}
    test_data = {}
    
    # Load train words
    with open(train_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            word = row[0].lower().strip()
            valence = float(row[1])
            words.add(word)
            train_data[word] = valence
    
    # Load test words
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            word = row[0].lower().strip()
            valence = float(row[1])
            words.add(word)
            test_data[word] = valence
    
    return words, train_data, test_data


def tokenize_text(text):
    """Tokenize text into words (lowercase, alphanumeric only)."""
    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words


def build_cooccurrence_matrix(data_path: str, target_words: set, window_size: int, vocab_size: int, max_lines: int):
    """
    Build co-occurrence matrix from Wikipedia dump.
    
    Args:
        data_path: Path to Wikipedia dump file
        target_words: Set of words to create representations for (rows)
        window_size: Size of context window
        vocab_size: Number of most frequent words to use as columns
        max_lines: Maximum number of lines to process
    
    Returns:
        cooccurrence_matrix: Sparse matrix (len(target_words) x vocab_size)
        word_to_idx: Mapping from word to column index
        idx_to_word: Mapping from column index to word
    """
    print("Step 1: Counting word frequencies...")
    word_freq = Counter()
    line_count = 0
    
    # First pass: count word frequencies
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line_count >= max_lines:
                break
            words = tokenize_text(line)
            word_freq.update(words)
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  Processed {line_count} lines...")
    
    print(f"Total lines processed: {line_count}")
    print(f"Total unique words: {len(word_freq)}")
    
    # Get top vocab_size most frequent words
    top_words = [word for word, _ in word_freq.most_common(vocab_size)]
    word_to_idx = {word: idx for idx, word in enumerate(top_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    print(f"Selected top {vocab_size} most frequent words for vocabulary")
    
    # Second pass: build co-occurrence matrix
    print("Step 2: Building co-occurrence matrix...")
    cooccurrence_dict = defaultdict(lambda: defaultdict(int))
    target_words_set = set(target_words)
    
    line_count = 0
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line_count >= max_lines:
                break
            words = tokenize_text(line)
            
            # For each word in the line
            for i, word in enumerate(words):
                # If this is a target word, count co-occurrences
                if word in target_words_set:
                    # Look at context window
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    
                    for j in range(start, end):
                        if j != i:  # Don't count the word itself
                            context_word = words[j]
                            if context_word in word_to_idx:
                                cooccurrence_dict[word][context_word] += 1
            
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  Processed {line_count} lines...")
    
    print("Step 3: Converting to sparse matrix...")
    # Convert to sparse matrix
    target_word_list = sorted(target_words_set)
    target_word_to_row = {word: idx for idx, word in enumerate(target_word_list)}
    
    rows = []
    cols = []
    data = []
    
    for target_word in target_word_list:
        row_idx = target_word_to_row[target_word]
        if target_word in cooccurrence_dict:
            for context_word, count in cooccurrence_dict[target_word].items():
                col_idx = word_to_idx[context_word]
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(count)
    
    # Create sparse matrix
    cooccurrence_matrix = csr_matrix((data, (rows, cols)), 
                                      shape=(len(target_word_list), vocab_size))
    
    print(f"Co-occurrence matrix shape: {cooccurrence_matrix.shape}")
    print(f"Non-zero entries: {cooccurrence_matrix.nnz}")
    
    return cooccurrence_matrix, word_to_idx, idx_to_word, target_word_list


def normalize_cooccurrence_matrix(matrix):
    """
    Normalize co-occurrence matrix to handle word frequency effects.
    Using L2 normalization per row (word representation).
    """
    # L2 normalize each row (each word's representation)
    normalized_matrix = normalize(matrix, norm='l2', axis=1)
    return normalized_matrix


def load_pretrained_embeddings(model_name: str):
    """
    Load pre-trained word embeddings from gensim.
    
    Args:
        model_name: Name of the pre-trained model to load
    
    Returns:
        model: KeyedVectors model with pre-trained embeddings
    """
    print(f"Loading pre-trained model: {model_name}")
    try:
        model = api.load(model_name)
        print(f"  Loaded successfully! Vocabulary size: {len(model)}")
        print(f"  Embedding dimension: {model.vector_size}")
        return model
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
        return None


def train_and_evaluate_regression(X_train, y_train, X_test, y_test):
    """
    Train a linear regression model and evaluate it on test set.
    
    Args:
        X_train: Training embeddings (numpy array)
        y_train: Training valence scores (numpy array)
        X_test: Test embeddings (numpy array)
        y_test: Test valence scores (numpy array)
    
    Returns:
        mse: Mean squared error on test set
        corr: Pearson correlation on test set
    """
    if len(X_train) == 0:
        print("Error: No training words found in embeddings!")
        return float('inf'), 0.0
    
    if len(X_test) == 0:
        print("Error: No test words found in embeddings!")
        return float('inf'), 0.0
    
    print(f"Training on {len(X_train)} words, testing on {len(X_test)} words")
    
    # Convert to numpy arrays if not already
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Train linear regression model
    print("Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    print("Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate Pearson correlation
    corr, _ = pearsonr(y_test, y_pred)
    
    return mse, corr


def evaluate_model_with_embeddings(embeddings, train_data: dict, test_data: dict):
    """
    Evaluate a regression model using given embeddings.
    
    Args:
        embeddings: KeyedVectors model with word embeddings
        train_data: Dictionary mapping words to valence scores (training set)
        test_data: Dictionary mapping words to valence scores (test set)
    
    Returns:
        mse: Mean squared error on test set
        corr: Pearson correlation on test set
    """
    # Extract embeddings and labels for training set
    X_train = []
    y_train = []
    
    for word, valence in train_data.items():
        if word in embeddings:
            embedding = embeddings[word]
            X_train.append(embedding)
            y_train.append(valence)
        else:
            print(f"Warning: Word '{word}' from training set not found in embeddings")
    
    # Extract embeddings and labels for test set
    X_test = []
    y_test = []
    
    for word, valence in test_data.items():
        if word in embeddings:
            embedding = embeddings[word]
            X_test.append(embedding)
            y_test.append(valence)
        else:
            print(f"Warning: Word '{word}' from test set not found in embeddings")
    
    # Use evaluation function
    return train_and_evaluate_regression(X_train, y_train, X_test, y_test)


def predict_words_valence(train_file: str, test_file: str, data_path: str, is_dense_embedding: bool) -> (float, float):
    print(f'starting regression with {train_file}, evaluating on {test_file}, '
          f'and dense word embedding {is_dense_embedding}')

    if is_dense_embedding == False:
        print("\n=== Building sparse word representations ===")
        
        # Load train and test words
        print("Loading train and test words...")
        target_words, train_data, test_data = load_train_test_words(train_file, test_file)
        print(f"Total unique words in train+test: {len(target_words)}")
        
        # Build co-occurrence matrix
        cooccurrence_matrix, word_to_idx, idx_to_word, target_word_list = build_cooccurrence_matrix(
            data_path, target_words, WINDOW_SIZE, VOCABULARY_SIZE, MAX_LINES
        )
        
        # Normalize the matrix to handle word frequency effects
        print("Step 4: Normalizing co-occurrence matrix...")
        normalized_matrix = normalize_cooccurrence_matrix(cooccurrence_matrix)
        
        print("Sparse word representations created successfully!")
        print(f"Matrix shape: {normalized_matrix.shape}")
        print(f"Sparsity: {(1 - normalized_matrix.nnz / (normalized_matrix.shape[0] * normalized_matrix.shape[1])) * 100:.2f}%")
        
        # Step 5: Prepare data for regression
        print("\n=== Training regression model ===")
        
        # Create mapping from word to row index in the matrix
        word_to_row_idx = {word: idx for idx, word in enumerate(target_word_list)}
        
        # Extract embeddings and labels for training set
        X_train = []
        y_train = []
        
        for word, valence in train_data.items():
            if word in word_to_row_idx:
                row_idx = word_to_row_idx[word]
                embedding = normalized_matrix[row_idx, :].toarray().flatten()
                X_train.append(embedding)
                y_train.append(valence)
            else:
                print(f"Warning: Word '{word}' from training set not found in embeddings (not in corpus)")
        
        # Extract embeddings and labels for test set
        X_test = []
        y_test = []
        
        for word, valence in test_data.items():
            if word in word_to_row_idx:
                row_idx = word_to_row_idx[word]
                embedding = normalized_matrix[row_idx, :].toarray().flatten()
                X_test.append(embedding)
                y_test.append(valence)
            else:
                print(f"Warning: Word '{word}' from test set not found in embeddings (not in corpus)")
        
        if len(X_train) == 0:
            print("Error: No training words found in embeddings!")
            return float('inf'), 0.0
        
        if len(X_test) == 0:
            print("Error: No test words found in embeddings!")
            return float('inf'), 0.0
        
        # Use common evaluation function
        mse, corr = train_and_evaluate_regression(X_train, y_train, X_test, y_test)
        
        print(f"\n=== Regression Results ===")
        print(f"MSE: {mse:.3f}")
        print(f"Pearson correlation: {corr:.3f}")
        
        return mse, corr

    elif is_dense_embedding == True:
        print("\n=== Using pre-trained dense word representations ===")
        
        # Load train and test words
        print("Loading train and test words...")
        target_words, train_data, test_data = load_train_test_words(train_file, test_file)
        print(f"Total unique words in train+test: {len(target_words)}")
        
        # Load pre-trained embeddings
        print(f"\n=== Loading pre-trained model: {PRETRAINED_MODEL} ===")
        embeddings = load_pretrained_embeddings(PRETRAINED_MODEL)
        
        if embeddings is None:
            print("Error: Failed to load pre-trained embeddings!")
            return float('inf'), 0.0
        
        # Train regression model and evaluate
        print("\n=== Training regression model ===")
        mse, corr = evaluate_model_with_embeddings(embeddings, train_data, test_data)
        
        print(f"\n=== Regression Results ===")
        print(f"MSE: {mse:.3f}")
        print(f"Pearson correlation: {corr:.3f}")
        
        return mse, corr


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    mse, corr = predict_words_valence(
        config['train'],
        config['test'],
        config['wiki_data'],
        config["word_embedding_dense"])

    elapsed_time = time.time() - start_time
    print(f"elapsed time: {elapsed_time: .2f} seconds")

    print(f'test set evaluation results: MSE: {mse: .3f}, '
          f'Pearsons correlation: {corr: .3f}')
