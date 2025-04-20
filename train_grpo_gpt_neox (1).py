import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from transformers.integrations import WandbCallback
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd
import wandb
import accelerate
from trl import GRPOTrainer, GRPOConfig
from lm_experiments_tools.utils import get_cls_by_name

# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--model_cfg', type=str, required=True)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--beta_kl', type=float, default=0.0)
parser.add_argument('--iters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_generations', type=int, default=8)
parser.add_argument('--prediction_shift', type=int, default=1)
parser.add_argument('--model_cls', type=str, required=True)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--max_length', type=int, default=256)
parser.add_argument('--from_pretrained', type=str, required=True, help='Initial checkpoint to load from')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save or resume checkpoints')
parser.add_argument('--reasoning', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# --- Reproducibility & setup ---
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
logger = accelerate.logging.get_logger('')

# --- Tokenizer ---
if os.path.exists(os.path.join(args.output_dir, "tokenizer_config.json")):
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_cfg, use_fast=True)
SEP_TOKEN = "<sep>"
GEN_TOKEN = "<gen>"
EOS_TOKEN = "<eos>"

# --- Datasets ---
dataset_name = "irodkin/1dCA_r2s20T20"
train_dataset = load_dataset(dataset_name, split="train")
val_dataset = load_dataset(dataset_name, split="validation")
test_dataset = load_dataset(dataset_name, split="test")

def process_sample(sample):
    input_chunks = [sample[f"input_ids_{i}"] for i in range(10)]
    prompt_ids = []
    for chunk in input_chunks:
        prompt_ids.append(SEP_TOKEN)
        prompt_ids.extend(chunk)
    prompt_str = " ".join(map(str, prompt_ids))
    t = 9 + args.prediction_shift
    target_ids = sample[f"input_ids_{t}"]
    target_str = " ".join(map(str, target_ids))
    is_eos = [False] * len(target_ids) + [True] if target_ids else []
    return {
        "prompt": f"{prompt_str} {GEN_TOKEN}",
        "target": f"{target_str} {EOS_TOKEN}",
        "is_eos": is_eos,
    }

print("*** Start processing dataset ***")
train_dataset = train_dataset.map(process_sample, batched=False, desc='Processing train')
val_dataset = val_dataset.map(process_sample, batched=False, desc='Processing eval')
test_dataset = test_dataset.map(process_sample, batched=False, desc='Processing test')
print("*** Done processing dataset ***")

# --- Reward function ---
def reward_token_accuracy(completions, **kwargs):
    targets = kwargs.get("target")
    if targets is None:
        raise ValueError("Targets are required for reward calculation.")
    rewards = []
    for comp_str, tgt_str in zip(completions, targets):
        answer = comp_str.split(GEN_TOKEN)[-1] if args.reasoning else comp_str
        pred_tokens = answer.strip().split()
        tgt_tokens = tgt_str.strip().split()
        correct = sum(p == t for p, t in zip(pred_tokens, tgt_tokens))
        acc = correct / len(tgt_tokens) if tgt_tokens else 0.0
        rewards.append(acc)
    return rewards

# --- Model loading ---
save_path = "./check_points/neox_grpo_run1"
model_cls = get_cls_by_name(args.model_cls)
from transformers import AutoModelForCausalLM

if os.path.exists(os.path.join(args.output_dir, "config.json")):
    print(f"🔁 Resuming from checkpoint at {args.output_dir}")
    model = model_cls.from_pretrained(args.output_dir)
else:
    print(f"🆕 Loading from initial checkpoint at {args.from_pretrained}")
    model = model_cls.from_pretrained(args.from_pretrained)


model.resize_token_embeddings(len(tokenizer))

# --- Training config ---
training_args = GRPOConfig(
    output_dir=save_path,                    # Checkpoints go here
    learning_rate=args.lr,
    logging_steps=10,
    save_strategy="steps",                   # ← Save checkpoints periodically
    save_steps=500,                          # ← Save every 500 steps
    save_total_limit=3,                      # ← Optional: keep only last 3 checkpoints
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_generations=args.num_generations,
    max_steps=args.iters,
    fp16=True,
    beta=args.beta_kl,
    eval_strategy='steps',
    eval_steps=100,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    report_to='wandb',
    max_completion_length=args.max_length,
    skip_memory_metrics=True
)


trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=reward_token_accuracy,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# --- Train ---
trainer.train()
trainer.evaluate(test_dataset)
