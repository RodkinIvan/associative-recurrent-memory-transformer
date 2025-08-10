import logging
import os
import math
from pathlib import Path
import torch
import random
import numpy as np
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from lm_experiments_tools import  Trainer,  TrainerArgs
from transformers import AutoConfig, HfArgumentParser
from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run
import lm_experiments_tools.optimizers as optimizers
import accelerate
from accelerate.logging import get_logger
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])


parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help='Scrolls task name: "gov_report", "summ_screen_fd", "qmsum", '
                                                  '"narrative_qa", "qasper", "quality", "contract_nli"')
parser.add_argument('--report_to', type=str, default='wandb', help='')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--wrap_pos', action='store_true', default=False,
                    help='Wrap positional encoding for memory tokens (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=420, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')
parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attention mask, '
                    'eval on last segment only', default=False)

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')

# Dataset args
parser.add_argument('--num_timesteps', type=int, default=None, help='number of timesteps in train sample')
parser.add_argument('--num_test_timesteps', type=int, default=None, help='number of timesteps in test sample')
parser.add_argument('--prediction_shift', type=int, default=1, help='num_timesteps between the last training steps and the predicted timestep')
parser.add_argument('--repeat_state', action='store_true', default=False,
                    help='repeat state in the input so the input look like: [s0, s1, s1, s2, s2, s3...]')
parser.add_argument('--dataset_name', type=str, default="ca", help="path to saved datasets")
parser.add_argument('--segment_size', type=int, default=128, help='number of useful tokens in a segment')
parser.add_argument('--d_mem', type=int, default=None, help='number of rows in associative matrix')
parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')
parser.add_argument('--rewrite_setting', action='store_true', default=False,
                    help='keys can occur several times')
parser.add_argument('--act_on', action='store_true', default=False,
                    help='use Adaptive Computation Time')
parser.add_argument('--max_hop', type=int, default=4, help='number of cycles in ACT')
parser.add_argument('--time_penalty', type=float, default=0.0, help='time penalty coefficient in ACT loss')
parser.add_argument('--act_type', type=str, default=None, help='what is in ACT (options: layer, associative)')


parser.add_argument('--no_denom', action='store_true', default=False,
                    help='use no denominator in ARMT')
parser.add_argument('--freeze_mem', action='store_true', default=False,
                    help='Freeze memory parameters in ARMT')
parser.add_argument('--no_correction', action='store_true', default=False,
                    help='ARMT shmidhuber correction for rewriting')
parser.add_argument('--desired_metric', type=float, default=1.0, help='metric to stop training')

parser.add_argument('--output_last_segment_only', action='store_true', default=False,
                    help='')
# Aydar # RMT args 
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--segment_alignment', type=str, default=None, help="How to align segments when splitting input")
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()
    
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    args.block_size = (args.segment_size + 1) * (1 + args.repeat_state)
    sep_token, gen_token, eos_token = 100, 101, 102
    
    logger = get_logger('')
    logger.info(args.model_cls)
    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    dataset_name_to_path = {
        "ca": "irodkin/1dCA_r2s20T20",
        "reverse_binary": "steeldream/binary",
        "reverse_decimal": "steeldream/decimal",
        "copy_binary": "steeldream/binary",
        "copy_decimal": "steeldream/decimal",
    }
    dataset_path = dataset_name_to_path[args.dataset_name]

    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    logger.info(f'preparing dataset for: {args.task_name}')

    with accelerator.main_process_first():
        train_dataset = load_dataset(dataset_path, split='train')
        valid_dataset = load_dataset(dataset_path, split='validation')
        test_dataset = load_dataset(dataset_path, split='test')
        if args.dataset_name == "ca":
            args.train_array_size = len(train_dataset[0]['input_ids_0'])
            args.valid_array_size = len(valid_dataset[0]['input_ids_0'])
        elif args.dataset_name in ["reverse_binary", "reverse_decimal", "copy_binary", "copy_decimal"]:
            args.train_array_size = len(train_dataset[0]['input_ids'])
            args.valid_array_size = len(valid_dataset[0]['input_ids'])
        else:
            raise ValueError('Unknown dataset name')

    prepare_run(args, logger, logger_fmt)

    def copy_collate_fn(batch, min_length=5, valid=False, reverse=False):
        batch_array_size = args.valid_array_size if valid else random.randint(min_length, args.train_array_size) 

        for i, b in enumerate(batch):
            X1 = b['input_ids'][:batch_array_size]
            X2 = X1[::-1] if reverse else X1
            batch[i] = {
                'input_ids': [eos_token] + X1 + [gen_token] + X2 ,
                'labels': [eos_token] +  X1 + [gen_token] + X2 ,
                'attention_mask': [1] * (2 * batch_array_size + 2)
            }
        
        input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch], dim=0)
        labels = torch.stack([torch.tensor(b['labels']) for b in batch], dim=0)
        attention_mask = torch.stack([torch.tensor(b['attention_mask']) for b in batch], dim=0)
    
        labels_mask = torch.zeros_like(input_ids).bool()
        labels_mask[:, -batch_array_size:] = True
        B, L = input_ids.shape
        logger.info(L)
        if 'armt' in args.model_path:
            input_ids = input_ids.reshape(B, 2, L // 2)
            labels = labels.reshape(B, 2, L // 2)
            labels_mask = labels_mask.reshape(B, 2, L // 2)
            attention_mask = attention_mask.reshape(B, 2, L // 2)
            
        collated = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'labels_mask': labels_mask,
        }
        return collated

    def reverse_collate_fn(batch, min_length=5, valid=False):
        return copy_collate_fn(batch, min_length=min_length, valid=valid, reverse=True)

    def ca_collate_fn(batch, valid=False):
        batch_array_size = args.valid_array_size if valid else args.train_array_size
        for i, b in enumerate(batch):
            steps = args.num_test_timesteps if valid else args.num_timesteps
            shift = args.prediction_shift
            if args.repeat_state:
                batch[i] = {
                    'input_ids': [i for t in range(steps-1) if f'input_ids_{t}' in b \
                                  for i in [sep_token,] + b[f'input_ids_{t}'] + [sep_token,]  + b[f'input_ids_{t+1}']]
                }
                batch[i]['input_ids'] = batch[i]['input_ids'] + [gen_token,] + b[f'input_ids_{steps-1}'] + \
                    [sep_token,] + b[f'input_ids_{steps+shift-1}']
            else:
                batch[i] = {
                    'input_ids': [i for t in range(steps) if f'input_ids_{t}' in b \
                                  for i in [sep_token,] + b[f'input_ids_{t}']]
                }
                batch[i]['input_ids'] = batch[i]['input_ids'] + [gen_token,] + b[f'input_ids_{steps+shift-1}']
            batch[i]['labels'] = batch[i]['input_ids'].copy()
            batch[i]['attention_mask'] = [1 for _ in batch[i]['input_ids']] 
            
        input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch], dim=0)
        labels = torch.stack([torch.tensor(b['labels']) for b in batch], dim=0)
        attention_mask = torch.stack([torch.tensor(b['attention_mask']) for b in batch], dim=0)
        
        labels_mask = torch.zeros_like(input_ids).bool()
        labels_mask[:, -batch_array_size-1:] = True
        collated = {
            'input_ids': input_ids,
            'labels': labels, 
            'attention_mask': attention_mask,
            'labels_mask': labels_mask,
        }
        return collated

    collate_fn_dict = {
        "ca": ca_collate_fn,
        "reverse_binary": reverse_collate_fn,
        "reverse_decimal": reverse_collate_fn,
        "copy_binary": copy_collate_fn,
        "copy_decimal": copy_collate_fn,
    }

    collate_fn = collate_fn_dict[args.dataset_name]

    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    train_dataloader = DataLoader(
        train_dataset, batch_size=per_worker_batch_size, generator=train_rnd_generator,
        collate_fn=collate_fn, **kwargs, drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=per_worker_batch_size,
        collate_fn=lambda x: collate_fn(x, valid=True), **kwargs, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=per_worker_batch_size,
        collate_fn=collate_fn, **kwargs, drop_last=True
    )

    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.model_cls)

    logger.info(f'Using model class: {model_cls}')
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model = model_cls(config=model_cfg)
    else:
        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained)
    
    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'])
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model 
    memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
    recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
    logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
    
    mem_cell_args = dict(base_model=model)

    if args.d_mem is not None:
        mem_cell_args['d_mem'] = args.d_mem

    if args.act_on:
        mem_cell_args['act_on'] = args.act_on
        mem_cell_args['max_hop'] = args.max_hop
        if args.act_type is not None:
            mem_cell_args['act_type'] = args.act_type

    if args.num_mem_tokens is not None:
        mem_cell_args['num_mem_tokens'] = args.num_mem_tokens
        mem_cell_args['wrap_pos'] = args.wrap_pos
    if args.layers_attr is not None:
        mem_cell_args['layers_attr'] = args.layers_attr
    if args.no_denom:
        mem_cell_args['use_denom'] = not args.no_denom
    if args.freeze_mem:
        mem_cell_args['freeze_mem'] = args.freeze_mem

    if args.no_correction:
        mem_cell_args['correction'] = False

    cell = memory_cell_cls(**mem_cell_args)
    model = recurrent_wrapper_cls(cell, 
                                    segment_size=args.block_size,
                                    max_n_segments=args.max_n_segments, 
                                    k2=args.k2,
                                    segment_alignment=args.segment_alignment,
                                    act_on=args.act_on,
                                    time_penalty=args.time_penalty
    )
                                
    ## load cpt of rmt
    if args.model_cpt and args.model_cpt != 'None':
        model_cpt = os.path.join(args.model_cpt, "model_best/pytorch_model.bin")
        cpt = torch.load(model_cpt, map_location='cpu')
        model.load_state_dict(cpt)
        logger.info(f'Loaded RMT state dict from: {args.model_cpt}')

    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            # if 'memory' not in n and 'wte' not in n:
            if 'memory' not in n and 'lora' not in n:
                p.requires_grad = False
        logger.info(f'Frozen moodel weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  scale_parameter=args.scale_parameter,
                                  relative_step=args.relative_step,
                                  warmup_init=args.warmup_init,
                                  weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        data = {}
        data['labels'] = batch['labels']
        data['labels_mask'] = batch['labels_mask']
        
        if 'generation_outputs' in output:
            data['generation_outputs'] = output['generation_outputs']
        data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        for key in batch.keys():
            if 'loss' in key: 
                data[key] = batch[key]
        if args.act_on:
            data['n_updates'] = output['n_updates']
            data['remainders'] = output['remainders']
        return data
    

    def batch_metrics_fn(batch, output):
        data = keep_for_metrics_fn(batch, output)

        if args.dataset_name == "ca":
            array_size = args.valid_array_size
        if args.dataset_name in ["reverse_binary", "reverse_decimal", "copy_binary", "copy_decimal"]:
            array_size = data['labels'][0].shape[0] // 2 - 1 if 'armt' not in args.model_path else data['labels'][0].shape[-1] - 1
        metrics = {}
        y = data['labels'][:, -array_size:] if 'armt' not in args.model_path else data['labels'][:, -1, -array_size:]
        p = data['predictions'][:, -array_size-1:-1]

        metrics['bit_accuracy'] = np.mean((y.cpu().numpy()) == (p.cpu().numpy()))
        metrics['exact_match'] = np.mean([np.array_equal(p_, y_) for p_, y_ in zip(p.cpu().numpy(), y.cpu().numpy())])

        if 'loss' in output:
            metrics['loss'] = output['loss'].mean().item()  # Store the loss
        
        if 'ce_loss' in data:
            metrics['ce_loss'] = data['ce_loss'].mean().item()
            try:
                metrics['perplexity'] = math.exp(metrics['ce_loss'])
            except OverflowError:
                metrics['perplexity'] = float("inf")
        
        if 'dist' in data:
            metrics['dist'] = data['dist'].mean().item()
        
        for i in range(args.max_n_segments):
            if f'ce_loss_{i}' in data:
                metrics[f'ce_loss_{i}'] = data[f'ce_loss_{i}'].mean().item()

        return metrics

    # acceleratems-appid:undefined
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, None)

    fwd_kwargs = dict()
    if args.output_last_segment_only:
        fwd_kwargs['output_only_last_segment'] = True
    if 'armt' in args.model_path and args.dataset_name != 'ca':
        fwd_kwargs['input_segmented'] = True
    trainer = Trainer(
        args, accelerator, model, optimizer, train_dataloader, valid_dataloader,
        keep_for_metrics_fn=keep_for_metrics_fn,
        batch_metrics_fn=batch_metrics_fn,
        stop_metric_condition=lambda m: m >= args.desired_metric,
        forward_kwargs=fwd_kwargs,
    )

    if not args.validate_only:
        trainer.train()
        accelerator.wait_for_everyone()
        
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best')
            logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False, split='valid')

        trainer.save_metrics(save_path=args.model_path)
    else:

        if valid_dataloader is not None:
            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=True, split='valid')
        else:
            raise "No valid dataset"

    print('Done!')