#!/usr/bin/env python
import json
import logging
import os
import math
import random
import numpy as np
from pathlib import Path

import torch
import datasets
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader

# Import TRL’s GRPO components
from trl import GRPOTrainer  # Removed GRPOConfig and AutoModelForCausalLMWithValueHead remains below
from trl import AutoModelForCausalLMWithValueHead

from transformers import AutoConfig, HfArgumentParser

# Our utilities
from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run
import lm_experiments_tools.optimizers as optimizers

import accelerate
from accelerate.logging import get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = get_logger("")

# Ensure CUDA_VISIBLE_DEVICES is set (if not already in the environment)
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])

# ----------------------------------------------------------------------------
# Define argument parser and register arguments
# ----------------------------------------------------------------------------
parser = HfArgumentParser([])

# Basic task and model arguments
parser.add_argument("--task_name", type=str, default="CA_GRPO", help="Task name")
parser.add_argument("--model_path", type=str, required=True, help="Path where model & metrics are saved")
parser.add_argument("--model_cfg", type=str, required=True, help="Path to model configuration file")
parser.add_argument("--dataset_path", type=str, default="irodkin/1dCA_r2s20T20", help="Dataset path")
parser.add_argument("--model_type", type=str, default="decoder", help="Model type (e.g., decoder)")
parser.add_argument("--memory_cell_cls", type=str, required=True, help="Memory cell class for RMT")
parser.add_argument("--recurrent_wrapper_cls", type=str, required=True, help="Wrapper class for RMT")
parser.add_argument("--model_cls", type=str, required=True, help="Main model class (e.g., transformers:GPTNeoXForCausalLM)")
parser.add_argument("--model_cpt", type=str, default="None", help="Checkpoint path for pretrained model")
parser.add_argument("--from_pretrained", type=str, default=None, help="Pretrained model name or path")
parser.add_argument("--backbone_cpt", type=str, default=None, help="Path to backbone checkpoint (optional)")
parser.add_argument("--segment_alignment", type=str, default=None, help="How segments are aligned in the input")
parser.add_argument("--time_penalty", type=float, default=0.0, help="Coefficient for time penalty in the recurrent wrapper")

# Data processing arguments
parser.add_argument("--segment_size", type=int, default=1000, help="Segment size")
parser.add_argument("--input_size", type=int, default=1000, help="Input size")
parser.add_argument("--max_n_segments", type=int, default=10, help="Max number of segments")
parser.add_argument("--num_mem_tokens", type=int, default=1, help="Number of memory tokens")
parser.add_argument("--num_timesteps", type=int, default=10, help="Number of timesteps in train sample")
parser.add_argument("--num_test_timesteps", type=int, default=10, help="Number of timesteps in test sample")
parser.add_argument("--prediction_shift", type=int, default=3, help="Prediction shift")

# Optimization and training hyperparameters
parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer name")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Optimizer weight decay")
parser.add_argument("--data_n_workers", type=int, default=2, help="Number of dataloader workers")
parser.add_argument("--show_valid_examples", type=int, default=5, help="How many valid examples to log")
parser.add_argument("--seed", type=int, default=8, help="Random seed")
parser.add_argument("--d_mem", type=int, default=1, help="Memory dimension")
parser.add_argument("--layers_attr", type=str, default="gpt_neox.layers", help="Layers attribute for memory")
parser.add_argument("--freeze_mem", action="store_true", help="Freeze memory parameters")
parser.add_argument("--validate_only", action="store_true", help="Run validation only")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--iters", type=int, default=40000, help="Total iterations")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--lr_scheduler", type=str, default="linear", help="Learning rate scheduler")
parser.add_argument("--num_warmup_steps", type=int, default=1000, help="Number of warmup steps")
parser.add_argument("--num_training_steps", type=int, default=80000, help="Total training steps")
parser.add_argument("--log_interval", type=int, default=50, help="Logging interval")
parser.add_argument("--valid_interval", type=int, default=250, help="Validation interval")
parser.add_argument("--early_stopping_patience", type=int, default=30, help="Early stopping patience")
parser.add_argument("--clip_grad_value", type=float, default=0.1, help="Gradient clipping value")
parser.add_argument("--save_best", action="store_true", help="Save best model")
parser.add_argument("--tokenizer", type=str, default=None, help="Pretrained tokenizer path or name")

# Additional parameters
parser.add_argument("--optimize_metric", type=str, default="exact_match", help="Metric to optimize")
parser.add_argument("--optimize_mode", type=str, default="max", help="Optimize mode (max or min)")

# ----------------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()
    
    args.working_dir = str(Path(args.working_dir).expanduser().absolute()) if hasattr(args, "working_dir") else os.getcwd()
    os.chdir(args.working_dir)
    
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    logger = get_logger("")
    logger.info(f"Using model class: {args.model_cls}")
    logger.info(f"Num processes: {accelerator.num_processes}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    
    if not args.model_path:
        logger.warning("model_path is not set: logs and checkpoints will not be saved.")
    prepare_run(args, logger, logger_fmt)
    
    # -------------------------------------------------------------------------
    # Set up tokens and collate function for Cellular Automata (CA)
    # -------------------------------------------------------------------------
    left = None
    right = None
    rule_left = None
    rule_right = None
    
    if args.model_type == "decoder":
        block_size = (args.segment_size + 1) * (1 + getattr(args, "repeat_state", False))
        sep_token, gen_token, eos_token = 100, 101, 102
        rule_token = 103
        
        def ca_collate_fn(batch, sample_length=False, array_size=getattr(args, "valid_array_size", 40), valid=False):
            for i, b in enumerate(batch):
                steps = args.num_test_timesteps if valid else args.num_timesteps
                shift = args.prediction_shift
                if getattr(args, "repeat_state", False):
                    batch[i] = {
                        "input_ids": [x for t in range(steps - 1)
                                       if f"input_ids_{t}" in b
                                       for x in [sep_token] + b[f"input_ids_{t}"] + [sep_token] + b[f"input_ids_{t+1}"]]
                    }
                    if getattr(args, "learn_rule", False):
                        batch[i]["input_ids"] += [gen_token] + b["rule_ids"]
                    batch[i]["input_ids"] += [sep_token if getattr(args, "learn_rule", False) else gen_token] + \
                                             b[f"input_ids_{steps-1}"] + [sep_token] + b[f"input_ids_{steps+shift-1}"]
                else:
                    batch[i] = {
                        "input_ids": [x for t in range(steps)
                                       if f"input_ids_{t}" in b
                                       for x in [sep_token] + b[f"input_ids_{t}"]]
                    }
                    if getattr(args, "learn_rule", False):
                        batch[i]["input_ids"] += [gen_token] + b["rule_ids"] + [sep_token] + b[f"input_ids_{steps+shift-1}"]
                    else:
                        batch[i]["input_ids"] += [gen_token] + b[f"input_ids_{steps+shift-1}"]
                
                batch[i]["labels"] = batch[i]["input_ids"][:]
                batch[i]["attention_mask"] = [1] * len(batch[i]["input_ids"])
            
            input_ids = torch.stack([torch.tensor(b["input_ids"]) for b in batch], dim=0)
            labels = torch.stack([torch.tensor(b["labels"]) for b in batch], dim=0)
            if getattr(args, "learn_rule", False):
                input_ids[:, rule_left:rule_right] = rule_token
            attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch], dim=0)
            labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            labels_mask[:, -(getattr(args, "array_size", 40) + 1 +
                             (getattr(args, "learn_rule", False)) * (getattr(args, "rule_len", 0) + 1)):] = True
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "labels_mask": labels_mask,
            }
    else:
        raise NotImplementedError(f"Unknown model type {args.model_type}")
    
    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    logger.info(f"Preparing dataset for: {args.task_name}")
    with accelerator.main_process_first():
        train_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="train")
        args.rule_len = len(train_dataset[0]["rule_ids"])
        logger.info(f"Rule len: {args.rule_len}")
        valid_dataset = load_dataset(args.dataset_path, split="validation")
        test_dataset = load_dataset(args.dataset_path, split="test")
        args.array_size = len(train_dataset[0]["input_ids_0"])
    
    right = 0
    left = -args.array_size
    if getattr(args, "learn_rule", False):
        rule_left = -(2 * args.array_size + 2 + args.rule_len) + (1 - getattr(args, "repeat_state", False)) * (args.array_size + 1)
        rule_right = rule_left + args.rule_len

    train_rnd_generator = torch.Generator().manual_seed(args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {"pin_memory": True, "num_workers": args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size,
                                  generator=train_rnd_generator,
                                  collate_fn=lambda x: ca_collate_fn(x),
                                  drop_last=True, **kwargs)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size,
                                  collate_fn=lambda x: ca_collate_fn(x),
                                  drop_last=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size,
                                 collate_fn=lambda x: ca_collate_fn(x),
                                 drop_last=True, **kwargs)
    
    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # -------------------------------------------------------------------------
    # Build the model
    # -------------------------------------------------------------------------
    model_cls_ = get_cls_by_name(args.model_cls)
    logger.info(f"Using model class: {model_cls_}")

    if not args.from_pretrained:
        model_cfg_ = AutoConfig.from_pretrained(args.model_cfg)
        base_model = transformers.AutoModelForCausalLM.from_config(model_cfg_)
        if hasattr(base_model, "transformer") and not hasattr(base_model, "gpt_neox"):
            base_model.gpt_neox = base_model.transformer
        
        print("\n--- BASE MODEL ARCHITECTURE ---")
        print(base_model)
        
        model = AutoModelForCausalLMWithValueHead(base_model)
        
        print("\n--- VALUE-HEAD MODEL ARCHITECTURE ---")
        print(model)
        
        if not hasattr(model, "get_input_embeddings"):
            model.get_input_embeddings = lambda: base_model.get_input_embeddings()
    else:
        logger.info(f"Loading pretrained model: {args.from_pretrained}")
        model = model_cls_.from_pretrained(args.from_pretrained)
    
    if args.backbone_cpt:
        ckpt_path = os.path.join(args.backbone_cpt, "model_best/pytorch_model.bin")
        cpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(cpt["model_state_dict"])
        logger.info(f"Loaded baseline state dict from: {args.backbone_cpt}")
    
    # -------------------------------------------------------------------------
    # Wrap model with memory settings
    # -------------------------------------------------------------------------
    memory_cell_cls_ = get_cls_by_name(args.memory_cell_cls)
    recurrent_wrapper_cls_ = get_cls_by_name(args.recurrent_wrapper_cls)
    logger.info(f"Wrapping in: {memory_cell_cls_} and {recurrent_wrapper_cls_}")
    
    mem_cell_args = {"base_model": model}
    if args.d_mem is not None:
        mem_cell_args["d_mem"] = args.d_mem
    if getattr(args, "act_on", False):
        mem_cell_args["act_on"] = args.act_on
        mem_cell_args["max_hop"] = args.max_hop
        if getattr(args, "act_type", None):
            mem_cell_args["act_type"] = args.act_type
        if getattr(args, "act_format", None):
            mem_cell_args["act_format"] = args.act_format
        if getattr(args, "noisy_halting", False):
            mem_cell_args["noisy_halting"] = args.noisy_halting
    if args.num_mem_tokens is not None:
        mem_cell_args["num_mem_tokens"] = args.num_mem_tokens
        mem_cell_args["wrap_pos"] = getattr(args, "wrap_pos", False)
    if args.layers_attr is not None:
        mem_cell_args["layers_attr"] = args.layers_attr
    if getattr(args, "no_denom", False):
        mem_cell_args["use_denom"] = False
    if getattr(args, "freeze_mem", False):
        mem_cell_args["freeze_mem"] = True
    if getattr(args, "no_correction", False):
        mem_cell_args["correction"] = False

    cell = memory_cell_cls_(**mem_cell_args)
    model = recurrent_wrapper_cls_(cell,
                                   segment_size=block_size,
                                   max_n_segments=args.max_n_segments,
                                   segment_alignment=args.segment_alignment,
                                   act_on=getattr(args, "act_on", False),
                                   time_penalty=args.time_penalty)
    
    if args.model_cpt and args.model_cpt != "None":
        cpt_path = os.path.join(args.model_cpt, "model_best/pytorch_model.bin")
        cpt = torch.load(cpt_path, map_location="cpu")
        model.load_state_dict(cpt)
        logger.info(f"Loaded RMT state dict from: {args.model_cpt}")
    
    if getattr(args, "freeze_model_weights", False):
        for n, p in model.named_parameters():
            if "memory" not in n and "lora" not in n:
                p.requires_grad = False
        logger.info("Frozen model weights")
        logger.info(f"Remaining trainable: {[n for n, p in model.named_parameters() if p.requires_grad]}")
    
    # -------------------------------------------------------------------------
    # Define optimizer
    # -------------------------------------------------------------------------
    optimizer_cls_ = get_optimizer(args.optimizer)
    if optimizer_cls_ is None:
        raise RuntimeError(f"{args.optimizer} not found in known optimizers or torch")
    logger.info(f"Using optimizer class: {optimizer_cls_}")
    
    if optimizer_cls_ in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        optimizer = optimizer_cls_(model.parameters(), lr=args.lr,
                                   scale_parameter=getattr(args, "scale_parameter", False),
                                   relative_step=getattr(args, "relative_step", False),
                                   warmup_init=getattr(args, "warmup_init", False),
                                   weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls_(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # -------------------------------------------------------------------------
    # Define reward function for GRPOTrainer
    # -------------------------------------------------------------------------
    def compute_ca_rewards(generation_outputs, labels):
        generation_outputs = generation_outputs.cpu().numpy()
        labels = labels.cpu().numpy()
        return torch.tensor([1.0 if np.array_equal(pred, ref) else -1.0
                             for pred, ref in zip(generation_outputs, labels)],
                            dtype=torch.float32)
    
    # -------------------------------------------------------------------------
    # Set up GRPO trainer (remove GRPOConfig usage)
    # -------------------------------------------------------------------------
    from trl import GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=None,
        reward_fn=compute_ca_rewards,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=lambda x: ca_collate_fn(x),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        num_train_epochs=math.ceil(args.iters / (len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps))),
    )
    
    # -------------------------------------------------------------------------
    # Prepare with Accelerator
    # -------------------------------------------------------------------------
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, None
    )
    
    # -------------------------------------------------------------------------
    # Train or validate
    # -------------------------------------------------------------------------
    logger.info("Starting GRPO training...")
    trainer.train()
    accelerator.wait_for_everyone()
    if args.save_best:
        best_model_path = str(Path(args.model_path) / "model_best")
        logger.info(f"Loading best saved model from {best_model_path}")
        trainer.load(best_model_path)
    if valid_dataloader is not None:
        logger.info("Running evaluation on validation data:")
        trainer.evaluate()
    trainer.save_metrics(save_path=args.model_path)
    print("Done!")
