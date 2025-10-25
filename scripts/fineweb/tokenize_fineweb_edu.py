from datasets import load_dataset, Dataset
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import gc

# Use streaming to avoid loading entire dataset into RAM
dataset = load_dataset('karpathy/fineweb-edu-100b-shuffle', split='train', streaming=True, trust_remote_code=True)
model_path = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

output_dir = '/mnt/data/users/ivan.rodkin/lab/datasets/fineweb_edu_100b_tokenized'
os.makedirs(output_dir, exist_ok=True)

# Process and save in chunks to avoid RAM issues
chunk_size = 1_000_000  # Process 1M samples at a time
chunk_num = 0
buffer = []

print(f"Starting tokenization in streaming mode (chunk size: {chunk_size})")
print(f"Output directory: {output_dir}")

for i, sample in enumerate(tqdm(dataset, desc="Tokenizing", total=97_200_000)):
    try:
        tokens = tokenizer.encode(sample['text'], return_tensors='pt')[0].tolist()
        buffer.append({'tokens': tokens})
        
        # Save chunk when buffer is full
        if len(buffer) >= chunk_size:
            chunk_dataset = Dataset.from_list(buffer)
            chunk_path = os.path.join(output_dir, f'chunk_{chunk_num:06d}')
            chunk_dataset.save_to_disk(chunk_path)
            print(f"Saved chunk {chunk_num} ({len(buffer)} samples) to {chunk_path}")
            
            buffer = []
            chunk_num += 1
            gc.collect()
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        continue

# Save remaining samples
if buffer:
    chunk_dataset = Dataset.from_list(buffer)
    chunk_path = os.path.join(output_dir, f'chunk_{chunk_num:06d}')
    chunk_dataset.save_to_disk(chunk_path)
    print(f"Saved final chunk {chunk_num} ({len(buffer)} samples) to {chunk_path}")

print(f"Tokenization complete! Saved {chunk_num + 1} chunks to {output_dir}")

