#!/usr/bin/env python3
"""
Lexical Diversity Analysis Script

Analyzes lexical diversity metrics for conversational datasets in JSONL format.
Outputs dataset-level and role-level statistics to CSV.

Metrics:
- Vocabulary richness (MTLD)
- Average sentence length (tokens)
- Median utterance length (assistant only)
- Stopword ratio
- Bigram diversity
"""

import argparse
import json
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple
from transformers import GPT2TokenizerFast
from lexicalrichness import LexicalRichness
import nltk
from nltk.corpus import stopwords
import warnings

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL dataset and extract messages."""
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'messages' not in data:
                    warnings.warn(f"Line {line_num}: Missing 'messages' field, skipping")
                    continue
                conversations.append(data['messages'])
            except json.JSONDecodeError:
                warnings.warn(f"Line {line_num}: Invalid JSON, skipping")
                continue
    return conversations


def extract_texts_by_role(conversations: List[List[Dict]]) -> Tuple[List[str], List[str], List[str]]:
    """Extract message texts grouped by role."""
    all_texts = []
    user_texts = []
    assistant_texts = []

    for messages in conversations:
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')

            # Skip system messages
            if role == 'system':
                continue

            all_texts.append(content)

            if role == 'user':
                user_texts.append(content)
            elif role == 'assistant':
                assistant_texts.append(content)

    return all_texts, user_texts, assistant_texts


def calculate_mtld(texts: List[str]) -> float:
    """Calculate MTLD (Measure of Textual Lexical Diversity)."""
    if not texts:
        return 0.0

    combined_text = ' '.join(texts)
    try:
        lex = LexicalRichness(combined_text)
        return lex.mtld(threshold=0.72)
    except Exception as e:
        warnings.warn(f"MTLD calculation failed: {e}")
        return 0.0


def calculate_avg_sentence_length(texts: List[str], tokenizer: GPT2TokenizerFast) -> float:
    """Calculate average sentence length in tokens."""
    if not texts:
        return 0.0

    token_counts = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(tokens))

    return sum(token_counts) / len(token_counts)


def calculate_median_utterance_length(texts: List[str], tokenizer: GPT2TokenizerFast) -> float:
    """Calculate median utterance length in tokens."""
    if not texts:
        return 0.0

    token_counts = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(tokens))

    token_counts.sort()
    n = len(token_counts)

    if n % 2 == 0:
        return (token_counts[n//2 - 1] + token_counts[n//2]) / 2
    else:
        return token_counts[n//2]


def calculate_stopword_ratio(texts: List[str], tokenizer: GPT2TokenizerFast) -> float:
    """Calculate ratio of stopwords to total tokens."""
    if not texts:
        return 0.0

    total_tokens = 0
    stopword_count = 0

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([t]).strip().lower() for t in tokens]

        total_tokens += len(decoded_tokens)
        stopword_count += sum(1 for token in decoded_tokens if token in STOPWORDS)

    return stopword_count / total_tokens if total_tokens > 0 else 0.0


def calculate_bigram_diversity(texts: List[str], tokenizer: GPT2TokenizerFast) -> float:
    """Calculate bigram diversity (unique bigrams / total bigrams)."""
    if not texts:
        return 0.0

    all_bigrams = []

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Create bigrams
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        all_bigrams.extend(bigrams)

    if not all_bigrams:
        return 0.0

    unique_bigrams = len(set(all_bigrams))
    total_bigrams = len(all_bigrams)

    return unique_bigrams / total_bigrams


def calculate_metrics(texts: List[str], tokenizer: GPT2TokenizerFast,
                     role: str = 'all') -> Dict[str, float]:
    """Calculate all metrics for a given set of texts."""
    metrics = {
        'level': 'dataset' if role == 'all' else 'role',
        'role': role,
        'mtld': calculate_mtld(texts),
        'avg_sentence_length': calculate_avg_sentence_length(texts, tokenizer),
        'median_utterance_length': calculate_median_utterance_length(texts, tokenizer),
        'stopword_ratio': calculate_stopword_ratio(texts, tokenizer),
        'bigram_diversity': calculate_bigram_diversity(texts, tokenizer)
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Analyze lexical diversity metrics for conversational datasets'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )

    args = parser.parse_args()

    print(f"Loading dataset from: {args.input}")
    conversations = load_jsonl(args.input)
    print(f"Loaded {len(conversations)} conversations")

    print("Extracting texts by role...")
    all_texts, user_texts, assistant_texts = extract_texts_by_role(conversations)
    print(f"  All messages: {len(all_texts)}")
    print(f"  User messages: {len(user_texts)}")
    print(f"  Assistant messages: {len(assistant_texts)}")

    print("Initializing tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    print("Calculating metrics...")
    results = []

    # Dataset-level metrics
    print("  Dataset-level...")
    results.append(calculate_metrics(all_texts, tokenizer, role='all'))

    # Role-level metrics
    print("  User role...")
    results.append(calculate_metrics(user_texts, tokenizer, role='user'))

    print("  Assistant role...")
    results.append(calculate_metrics(assistant_texts, tokenizer, role='assistant'))

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, float_format='%.4f')

    print(f"\nResults saved to: {args.output}")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
