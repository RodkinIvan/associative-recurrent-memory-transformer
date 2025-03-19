import os
import math
import logging
from pathlib import Path

import torch
import numpy as np
import datasets
import transformers

from datasets import load_dataset
from torch.utils.data import DataLoader
from lm_experiments_tools import Trainer, TrainerArgs
import accelerate

# Import WandB
import wandb

# Set up logging.
logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')

# Set CUDA_VISIBLE_DEVICES if not already set.
if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(torch.cuda.device_count()))
logger.info("CUDA_VISIBLE_DEVICES: " + os.environ['CUDA_VISIBLE_DEVICES'])
logger.info("CUDA DEVICE COUNT: " + str(torch.cuda.device_count()))

from transformers import AutoConfig, HfArgumentParser
from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run

# Define only the necessary arguments.
parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--working_dir', type=str, default='.', help="Working directory")
parser.add_argument('--dataset_path', type=str, default="irodkin/1dCA_r2s20T20", help="Dataset path")
parser.add_argument('--model_cfg', type=str, required=True, help="Path to model configuration JSON")
parser.add_argument('--model_cls', type=str, default="transformers:GPTNeoXForCausalLM", help="Model class")
parser.add_argument('--model_type', type=str, default="decoder", help="Model type")
parser.add_argument('--from_pretrained', type=str, default=None, help="Pretrained model name (default: None)")
# Memory-related arguments.
parser.add_argument('--segment_size', type=int, default=128, help="Tokens per segment")
parser.add_argument('--repeat_state', action='store_true', default=False, help="Repeat state in input")
parser.add_argument('--learn_rule', action='store_true', default=False, help="Enable rule learning")
parser.add_argument('--d_mem', type=int, default=None, help="Rows in associative matrix")
parser.add_argument('--memory_cell_cls', type=str, default="modeling_amt.language_modeling:AssociativeMemoryCell", help="Memory cell class")
parser.add_argument('--recurrent_wrapper_cls', type=str, default="modeling_amt.language_modeling:AssociativeRecurrentWrapper", help="Recurrent wrapper class")
parser.add_argument('--array_size', type=int, default=None, help="Length of one input segment (set from dataset)")
parser.add_argument('--num_mem_tokens', type=int, default=None, help="Number of memory tokens (default: None)")
parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay (default: 0.0)")
parser.add_argument('--report_to', type=str, default="none", help="Logging backend (e.g., wandb, none)")
# Added missing argument
parser.add_argument('--max_n_segments', type=int, default=10, help="Maximum number of segments")

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()

    # Fix: If valid_interval is None, set a default value.
    if args.valid_interval is None:
        args.valid_interval = 250

    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    from accelerate.logging import get_logger as acc_get_logger
    logger = acc_get_logger('')
    logger.info("Using model class: " + args.model_cls)
    logger.info("Num processes: " + str(accelerator.num_processes))
    logger.info("Mixed precision: " + accelerator.mixed_precision)
    
    prepare_run(args, logger, logger_fmt)
    
    # Initialize WandB if selected and on the main process.
    if args.report_to.lower() == "wandb" and accelerator.is_main_process:
        wandb.init(project="your_project_name", config=vars(args))
        logger.info("WandB initialized.")
    
    # Define special tokens.
    sep_token, gen_token, eos_token = 100, 101, 102
    rule_token = 103

    # Minimal collate function.
    def collate_fn(batch, valid=False):
        input_ids = torch.stack([torch.tensor(b['input_ids_0']) for b in batch], dim=0)
        return {'input_ids': input_ids, 'labels': input_ids, 'attention_mask': torch.ones_like(input_ids)}

    logger.info("Preparing dataset from: " + args.dataset_path)
    with accelerator.main_process_first():
        train_dataset = load_dataset(args.dataset_path, split='train')
        valid_dataset = load_dataset(args.dataset_path, split='validation')
        test_dataset  = load_dataset(args.dataset_path, split='test')
    args.array_size = len(train_dataset[0]['input_ids_0'])

    # Hard-code number of dataloader workers to 2.
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size,
                                  collate_fn=collate_fn, pin_memory=True, num_workers=2, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size,
                                  collate_fn=collate_fn, pin_memory=True, num_workers=2, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size,
                                 collate_fn=collate_fn, pin_memory=True, num_workers=2, drop_last=True)
    
    model_cls = get_cls_by_name(args.model_cls)
    logger.info("Loading model with class: " + str(model_cls))
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model = model_cls(config=model_cfg)
    else:
        logger.info("Loading pretrained model: " + args.from_pretrained)
        model = model_cls.from_pretrained(args.from_pretrained)
    
    optimizer_cls = get_optimizer(getattr(args, "optimizer", "AdamW"))
    if optimizer_cls is None:
        raise RuntimeError("Optimizer not found in available optimizers.")
    logger.info("Using optimizer: " + str(optimizer_cls))
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    def keep_for_metrics_fn(batch, output):
        predictions = torch.argmax(output['logits'].detach(), dim=-1)
        # Optional: log batch accuracy to WandB.
        if args.report_to.lower() == "wandb" and accelerator.is_main_process:
            batch_acc = np.mean(batch['labels'].cpu().numpy() == predictions.cpu().numpy())
            wandb.log({"batch_accuracy": batch_acc})
        return {'predictions': predictions, 'labels': batch['labels']}
    
    def metrics_fn(data):
        y = data['labels']
        p = data['predictions']
        accuracy = np.mean(y.cpu().numpy() == p.cpu().numpy())
        return {'accuracy': accuracy}
    
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader)
    
    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, valid_dataloader,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn)
    
    # Train the model.
    trainer.train()
    
    # Run validation on validation and test splits.
    logger.info("Running validation on validation data:")
    trainer.validate(valid_dataloader, split='validation')
    logger.info("Running evaluation on test data:")
    trainer.validate(test_dataloader, split='test')
    
    if args.report_to.lower() == "wandb" and accelerator.is_main_process:
        wandb.finish()
    print("Done!")
