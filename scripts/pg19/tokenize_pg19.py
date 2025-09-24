from datasets import load_dataset
import numpy as np
from transformers import AutoModel, AutoTokenizer

dataset = load_dataset('pg19', trust_remote_code=True)
model_path = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def tokenize(sample):
    sample['tokens'] = tokenizer.encode(sample['text'], return_tensors='pt')[0]
    return sample

new_ds = dataset.map(tokenize, batch_size=256)
new_ds.save_to_disk('/mnt/data/users/XXXX/lab/datasets/pg19_tokenized')