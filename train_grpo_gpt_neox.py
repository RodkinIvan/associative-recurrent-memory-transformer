import os
import torch
from datasets import load_dataset
from transformers import AutoConfig, GPTNeoXForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# Set environment
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

# Load tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained("./accel_configs/neox_tiny", use_fast=True)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<sep>", "<gen>"],
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
})

SEP_TOKEN = "<sep>"
GEN_TOKEN = "<gen>"

# Load datasets
train_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="train")
val_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="validation")
test_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="test")

# Process: Insert <sep> between input states, <gen> before target
def process_sample(sample):
    input_chunks = [sample[f"input_ids_{i}"] for i in range(10)]
    prompt_ids = []
    for chunk in input_chunks:
        prompt_ids.extend(chunk)
        prompt_ids.append(SEP_TOKEN)

    prompt_str = " ".join(map(str, prompt_ids))
    target_ids = sample["input_ids_10"]
    target_str = " ".join(map(str, target_ids))
    is_eos = [False] * (len(target_ids) - 1) + [True] if target_ids else []

    return {
        "prompt": f"{prompt_str} {GEN_TOKEN}",
        "target": target_str,
        "is_eos": is_eos,
    }

# Apply processing
train_dataset = train_dataset.map(process_sample, batched=False)

# Reward: token-level accuracy
def reward_token_accuracy(completions, **kwargs):
    targets = kwargs.get("target", [""] * len(completions))
    rewards = []
    for comp_str, tgt_str in zip(completions, targets):
        pred_tokens = comp_str.strip().split()
        tgt_tokens = tgt_str.strip().split()
        correct = sum(p == t for p, t in zip(pred_tokens, tgt_tokens))
        acc = correct / len(tgt_tokens) if tgt_tokens else 0.0
        rewards.append(acc)
    return rewards

# Load model config and model
config = AutoConfig.from_pretrained("./accel_configs/neox_tiny")
model = GPTNeoXForCausalLM(config)
model.resize_token_embeddings(len(tokenizer))

# GRPO Training config
training_args = GRPOConfig(
    output_dir="optimized-GRPO",
    logging_steps=2,
    per_device_train_batch_size=2,
    num_generations=2,
    max_steps=40000,
    fp16=True,
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=reward_token_accuracy,
    train_dataset=train_dataset,
)

# Train
trainer.train()
