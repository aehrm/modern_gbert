import json
import random
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset
from streaming import StreamingDataset
import sys
import os
from torch.utils.data import DataLoader

def sample_lines(file_path, num_samples=100000, seed=42):
    """
    Randomly sample lines from a file without loading the entire file into memory.
    """
    streaming_ds = StreamingDataset(local=file_path, shuffle_seed=seed, batch_size=64, shuffle=True)
    dl = DataLoader(streaming_ds, num_workers=8, batch_size=1)
    print('total number of samples', len(streaming_ds))
    sampled_lines = []

    print('preparing streaming dataset')
    it = iter(dl)

    for _ in tqdm(range(num_samples)):
        sampled_lines.append(next(it)['text'])

    return sampled_lines

    #random.seed(seed)
    #sampled_lines = []
    #filesize = os.path.getsize(file_path)

    #with tqdm(total=filesize, unit="B", unit_scale=True, desc="Sampling lines") as pbar:
    #    with open(file_path, 'r', encoding='utf-8') as file:
    #        for line in file:
    #            pbar.update(len(line.encode()))

    #            if random.random() < prop:
    #                sampled_lines.append(json.loads(line)['text'])

    #return sampled_lines

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
    tokenized_lengths = [len(tokens) for tokens in tokenized_dataset['input_ids']]
    quantiles = np.percentile(tokenized_lengths, [75,80,85,90,95,99])
    longer_than_1024 = sum(1 for length in tokenized_lengths if length > 1024) / len(tokenized_lengths)
    total_tokens = sum(tokenized_lengths)
    tokenized_lengths_trunc = [min(1024, x) for x in tokenized_lengths]
    tokens_dropped = sum(max(0, length - 1024) for length in tokenized_lengths) / total_tokens

    std = np.std(tokenized_lengths_trunc)
    t_value = 2.576
    margin_of_error = t_value * (std / (len(tokenized_lengths_trunc) ** 0.5))
    
    # Print the results
    print("Quantiles (75,80,85,90,95,99) of tokenized lengths:", quantiles)
    print("Proportion of sequences longer than 1024:", longer_than_1024)
    print("Proportion of tokens dropped (if cut at 1024):", tokens_dropped)
    print("Average sequence length:", total_tokens / len(tokenized_lengths))
    print("Average sequence length after truncation:", np.mean(tokenized_lengths_trunc))
    print("confidence range:", margin_of_error)

if __name__ == '__main__':
    main()
