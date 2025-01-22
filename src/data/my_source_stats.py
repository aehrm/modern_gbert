import json
import random
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset
import sys
import os

def sample_lines(file_path, prop=0.01, seed=42):
    """
    Randomly sample lines from a file without loading the entire file into memory.
    """
    random.seed(seed)
    sampled_lines = []
    filesize = os.path.getsize(file_path)

    with tqdm(total=filesize, unit="B", unit_scale=True, desc="Sampling lines") as pbar:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                pbar.update(len(line.encode()))

                if random.random() < prop:
                    sampled_lines.append(json.loads(line)['text'])

    return sampled_lines

def tokenize_function(examples, tokenizer):
    """
    Tokenize the input text examples using the provided tokenizer.
    """
    return tokenizer(examples['text'], truncation=False)

def compute_distribution(tokenized_dataset):
    """
    Compute the length distribution from the tokenized dataset.
    """
    # Calculate tokenized lengths
    tokenized_lengths = [len(tokens) for tokens in tokenized_dataset['input_ids']]
    
    # Calculate quantiles
    quantiles = np.percentile(tokenized_lengths, [75,80,85,90,95,99])
    
    # Calculate proportion of sequences longer than 1024
    longer_than_1024 = sum(1 for length in tokenized_lengths if length > 1024) / len(tokenized_lengths)
    
    # Calculate proportion of tokens dropped if sequences are cut off at length 1024
    total_tokens = sum(tokenized_lengths)
    tokens_dropped = sum(max(0, length - 1024) for length in tokenized_lengths) / total_tokens
    
    return quantiles, longer_than_1024, tokens_dropped

def main():
    # Path to the JSONL file
    file_path = sys.argv[1]
    
    # Sample lines from the dataset
    sampled_texts = sample_lines(file_path)
    
    # Create a Hugging Face Dataset from sampled texts
    dataset = Dataset.from_dict({"text": sampled_texts})
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
    
    # Tokenize the dataset using the map function
    tokenized_dataset = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True, )

    # Compute the token length distribution
    quantiles, longer_than_1024, tokens_dropped = compute_distribution(tokenized_dataset)
    
    # Print the results
    print("Quantiles (75,80,85,90,95,99) of tokenized lengths:", quantiles)
    print("Proportion of sequences longer than 1024:", longer_than_1024)
    print("Proportion of tokens dropped (if cut at 1024):", tokens_dropped)

if __name__ == '__main__':
    main()
