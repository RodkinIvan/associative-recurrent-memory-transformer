import os
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import argparse
from lm_experiments_tools.utils import get_cls_by_name
# Set environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('--model_cfg', type=str, help='path to model configuration file')
parser.add_argument('--lr', type=float, help='learning_rate', default=1e-4)
parser.add_argument('--beta_kl', type=float, help='KL coefficient', default=0.0)
parser.add_argument('--iters', type=int, help='number of iterations', default=40000)
parser.add_argument('--batch_size', type=int, help='batch size', default=128)
parser.add_argument('--num_generations', type=int, help='num grpo generations', default=32)
parser.add_argument('--prediction_shift', type=int, help='t, so that we predict 10+t\'s state, in 1-indexing', default=1)
parser.add_argument('--model_cls', type=str, help='path to model class implementation')
# parser.add_argument('--reasoner_cls', type=str, help='path to reasoner class implementation')
parser.add_argument('--gradient_accumulation_steps', type=int, help='', default=1)
parser.add_argument('--max_length', type=int, help='maximum completion length', default=256)




args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

# Load tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained(args.model_cfg, use_fast=True)
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
    t = 9 + args.prediction_shift
    target_ids = sample[f"input_ids_{t}"]
    target_str = " ".join(map(str, target_ids))
    is_eos = [False] * (len(target_ids) - 1) + [True] if target_ids else []

    return {
        "prompt": f"{prompt_str} {GEN_TOKEN}",
        "target": target_str,
        "is_eos": is_eos,
    }

# Apply processing
print("*** Start processing dataset ***")
train_dataset = train_dataset.map(process_sample, batched=False, desc='Processing train')
val_dataset = val_dataset.map(process_sample, batched=False, desc='Processing eval')
test_dataset = test_dataset.map(process_sample, batched=False, desc='Processing test')

print("*** Done processing dataset ***")

# Reward: token-level accuracy
def reward_token_accuracy(completions, **kwargs):
    targets = kwargs.get("target", [""] * len(completions))
    rewards = []
    for comp_str, tgt_str in zip(completions, targets):
        answer = comp_str.split("<sep>")[-1]
        pred_tokens = answer.strip().split()
        tgt_tokens = tgt_str.strip().split()
        correct = sum(p == t for p, t in zip(pred_tokens, tgt_tokens))
        acc = correct / len(tgt_tokens) if tgt_tokens else 0.0
        rewards.append(acc)
    return rewards

# Load model config and model
print(f"*** Loading model of class {args.model_cls} ***")
config = AutoConfig.from_pretrained(args.model_cfg)

model_cls = get_cls_by_name(args.model_cls)
model = model_cls.from_config(config)
model.resize_token_embeddings(len(tokenizer))

# reasoner_cls = get_cls_by_name(args.reasoner_cls)
# model = reasoner_cls(model, think_token_id=124)

# model.generate = None

print("*** Done loaing Model ***")
# GRPO Training config
training_args = GRPOConfig(
    output_dir="optimized-GRPO",
    learning_rate=args.lr,
    logging_steps=10,
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
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=reward_token_accuracy,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

trainer.evaluate(test_dataset)
