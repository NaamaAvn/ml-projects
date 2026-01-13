import json
import time
from collections import defaultdict
import re
import random


SOLUTION = ['notation', 'system', 'remote', 'open', 'technologies', 'faster', 'commonly', 'browsers', 'displayed', 'people', 'half', 'methods']


def tokenize(text):
    """Tokenize text by splitting on whitespace and converting to lowercase."""
    return text.lower().split()


def build_relevant_trigram_model(corpus_file, relevant_words, vocab_size=50000):
    """
    Build a 3-gram language model only for relevant words.
    
    Args:
        corpus_file: Path to corpus file
        relevant_words: Set of words we care about (candidates + context words)
        vocab_size: Fixed vocabulary size for Laplace smoothing
    
    Returns:
        trigram_counts: dict of (w1, w2, w3) -> count (only for relevant trigrams)
        bigram_counts: dict of (w1, w2) -> count (only for relevant bigrams)
        vocab_size: fixed vocabulary size
    """
    print(f'Building trigram model for {len(relevant_words)} relevant words...')
    trigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    
    # Also include special tokens as relevant (make a copy to avoid modifying original)
    relevant_words = set(relevant_words)
    relevant_words.add('<s>')
    relevant_words.add('</s>')
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            tokens = tokenize(line)
            if not tokens:
                continue
            
            # Add start/end markers
            padded = ['<s>', '<s>'] + tokens + ['</s>']
            
            # Only count n-grams that contain at least one relevant word
            for i in range(len(padded) - 2):
                w1, w2, w3 = padded[i], padded[i+1], padded[i+2]
                
                # Check if any word in the trigram is relevant
                if w1 in relevant_words or w2 in relevant_words or w3 in relevant_words:
                    bigram = (w1, w2)
                    bigram_counts[bigram] += 1
                    
                    trigram = (w1, w2, w3)
                    trigram_counts[trigram] += 1
            
            if (line_num + 1) % 100000 == 0:
                print(f'Processed {line_num + 1:,} lines...')
    
    print(f'Built {len(trigram_counts):,} relevant trigrams, {len(bigram_counts):,} relevant bigrams')
    return trigram_counts, bigram_counts, vocab_size


def trigram_prob(w1, w2, w3, trigram_counts, bigram_counts, vocab_size):
    """
    Calculate Laplace-smoothed probability: P(w3 | w1, w2)
    Formula: (C(w1,w2,w3) + 1) / (C(w1,w2) + V)
    """
    trigram = (w1, w2, w3)
    bigram = (w1, w2)
    
    trigram_count = trigram_counts[trigram]
    bigram_count = bigram_counts[bigram]
    
    # Laplace smoothing
    numerator = trigram_count + 1
    denominator = bigram_count + vocab_size
    
    return numerator / denominator if denominator > 0 else 1.0 / vocab_size


def score_candidate(left_words, right_words, candidate, trigram_counts, bigram_counts, vocab_size, left_only):
    """
    Score a candidate word for a blank.
    If left_only: use P(candidate | w1, w2) where w1, w2 are last 2 words before blank
    Otherwise: use P(candidate | w1, w2) * P(w3 | w2, candidate) where w3 is first word after blank
    """
    # Get last 2 words from left context (pad with <s> if needed)
    if len(left_words) >= 2:
        w1, w2 = left_words[-2], left_words[-1]
    elif len(left_words) == 1:
        w1, w2 = '<s>', left_words[-1]
    else:
        w1, w2 = '<s>', '<s>'
    
    if left_only:
        # Only use left context
        return trigram_prob(w1, w2, candidate, trigram_counts, bigram_counts, vocab_size)
    else:
        # Use both left and right context
        prob_left = trigram_prob(w1, w2, candidate, trigram_counts, bigram_counts, vocab_size)
        
        if right_words:
            # Also consider: P(right_word | w2, candidate)
            right_word = right_words[0]
            prob_right = trigram_prob(w2, candidate, right_word, trigram_counts, bigram_counts, vocab_size)
            return prob_left * prob_right
        else:
            return prob_left


def accuracy(solution, correct_solution):
    """
    Calculate accuracy of the solution
    """
    if len(solution) != len(correct_solution):
        return 0.0
    correct = sum(1 for i in range(len(solution)) if solution[i] == correct_solution[i])
    return correct / len(solution)


def chance_accuracy(correct_solution):
    "generate  100 random solutions and calculate their mean accuracy"
    random_solutions = [random.sample(correct_solution, len(correct_solution)) for _ in range(100)]
    accuracies = [accuracy(solution, correct_solution) for solution in random_solutions]
    return sum(accuracies) / len(accuracies)


def solve_cloze(input_file, candidates_file, corpus_file, left_only):
    """
    Solve cloze task using 3-gram language model with Laplace smoothing.
    Only builds n-grams for relevant words (candidates + context words).
    """
    print(f'Solving cloze with {"left context only" if left_only else "left and right context"}')
    
    # Step 1: Read candidates
    with open(candidates_file, 'r', encoding='utf-8') as f:
        candidates = [line.strip().lower() for line in f if line.strip()]
    print(f'Loaded {len(candidates)} candidates')
    
    # Step 2: Read cloze document and extract context words
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find all blanks and extract context
    blank_pattern = r'_{10,}'
    segments = re.split(blank_pattern, text)
    
    # Collect all relevant words: candidates + context words around blanks
    relevant_words = set(candidates)  # Start with candidates
    
    # Extract context words around each blank (last 2 words before, first 2 words after)
    for i in range(len(segments) - 1):
        left_text = segments[i]
        right_text = segments[i + 1] if i + 1 < len(segments) else ''
        
        left_words = tokenize(left_text)
        right_words = tokenize(right_text)
        
        # Add last 2 words from left context
        if len(left_words) >= 2:
            relevant_words.update(left_words[-2:])
        elif len(left_words) == 1:
            relevant_words.add(left_words[-1])
        
        # Add first 2 words from right context
        if len(right_words) >= 2:
            relevant_words.update(right_words[:2])
        elif len(right_words) == 1:
            relevant_words.add(right_words[0])
    
    print(f'Found {len(relevant_words)} relevant words (candidates + context)')
    
    # Step 3: Build model only for relevant words 
    trigram_counts, bigram_counts, vocab_size = build_relevant_trigram_model(
        corpus_file, relevant_words, vocab_size=50000
    )
    
    # Step 4: Solve each blank (ensure no duplicate words)
    solution = []
    used_candidates = set()  # Track which candidates have been used
    
    for i in range(len(segments) - 1):
        left_text = segments[i]
        right_text = segments[i + 1] if i + 1 < len(segments) else ''
        
        left_words = tokenize(left_text)
        right_words = tokenize(right_text)
        
        # Score each candidate (excluding already used ones)
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            # Skip candidates that have already been used
            if candidate in used_candidates:
                continue
                
            score = score_candidate(left_words, right_words, candidate, 
                                   trigram_counts, bigram_counts, vocab_size, left_only)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        # If no unused candidate found (shouldn't happen if we have enough candidates)
        if best_candidate is None:
            print(f'Warning: No unused candidate found for blank {i+1}')
            best_candidate = candidates[0]  # Fallback to first candidate
        
        solution.append(best_candidate)
        used_candidates.add(best_candidate)  # Mark as used
        print(f'Blank {i+1}: {best_candidate} (score: {best_score:.6e})')
        
    # print accuracy and chance accuracy
    print(f'Accuracy: {accuracy(solution, SOLUTION):.2f}')
    print(f'Chance accuracy: {chance_accuracy(SOLUTION):.2f}')
    
    return solution


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    solution = solve_cloze(config['input_filename'],
                           config['candidates_filename'],
                           config['corpus'],
                           config['left_only'])

    elapsed_time = time.time() - start_time
    print(f"elapsed time: {elapsed_time:.2f} seconds")

    print('cloze solution:', solution)
