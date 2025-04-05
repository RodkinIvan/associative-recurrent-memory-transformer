import os
import torch
import difflib
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Clear GPU memory and pick one device
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load dataset splits
train_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="train")
val_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="validation")
test_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="test")

# Pick a tokenizer that matches your model
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Convert token IDs into text prompts
def process_sample(sample):
    # Combine input_ids_0..9
    prompt_tokens = []
    for i in range(10):
        prompt_tokens += sample[f"input_ids_{i}"]
    prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)

    # Decode the target (input_ids_10)
    target_tokens = sample["input_ids_10"]
    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)

    # Store both for training and reward calculation
    return {
        "prompt": prompt_text,
        "target": target_text
    }

train_dataset = train_dataset.map(process_sample, batched=False)
val_dataset = val_dataset.map(process_sample, batched=False)
test_dataset = test_dataset.map(process_sample, batched=False)

# Reward function comparing generation to the target using difflib
def reward_similarity(completions, **kwargs):
    prompts = kwargs.get("prompt", [""] * len(completions))
    targets = kwargs.get("target", [""] * len(completions))
    rewards = []
    for comp, tgt in zip(completions, targets):
        ratio = difflib.SequenceMatcher(None, comp.strip(), tgt.strip()).ratio()
        rewards.append(ratio)
    return rewards

# Prepare config
training_args = GRPOConfig(
    output_dir="optimized-GRPO",
    logging_steps=2,
    per_device_train_batch_size=2,
    num_generations=2,
    max_steps=25,
    use_vllm=False,
    fp16=True
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=reward_similarity,
    args=training_args,
    train_dataset=train_dataset
)

# Start fine tuning
trainer.train()
