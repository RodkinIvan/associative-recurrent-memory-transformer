import logging
import os
import math
from pathlib import Path
import torch
import random
import numpy as np
import transformers
from datasets import load_dataset
from datasets.features.features import _FEATURE_TYPES, Sequence
if "List" not in _FEATURE_TYPES:
    _FEATURE_TYPES["List"] = Sequence
from torch.utils.data import DataLoader
from lm_experiments_tools import Trainer, TrainerArgs
from transformers import AutoConfig, HfArgumentParser
from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run
import lm_experiments_tools.optimizers as optimizers
import accelerate
from accelerate.logging import get_logger
from fvcore.nn import FlopCountAnalysis
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])


parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help='Scrolls task name: "gov_report", "summ_screen_fd", "qmsum", '
                                                  '"narrative_qa", "qasper", "quality", "contract_nli", "ca_adaptive", ...')
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
parser.add_argument('--num_timesteps', type=int, default=10, help='number of timesteps in train sample')
parser.add_argument('--num_test_timesteps', type=int, default=None, help='number of timesteps in test sample')
parser.add_argument('--prediction_shift', type=int, default=1, help='num_timesteps between the last training steps and the predicted timestep')
parser.add_argument('--repeat_state', action='store_true', default=False,
                    help='repeat state in the input so the input look like: [s0, s1, s1, s2, s2, s3...]')
parser.add_argument('--dataset_name', type=str, default="ca", help="path to saved datasets")
parser.add_argument('--sample_length', action='store_true', default=False, help='sample input length')
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

parser.add_argument('--act_format', type=str, default=None, help='')

parser.add_argument('--no_denom', action='store_true', default=False,
                    help='use no denominator in ARMT')
parser.add_argument('--freeze_mem', action='store_true', default=False,
                    help='Freeze memory parameters in ARMT')
parser.add_argument('--no_correction', action='store_true', default=False,
                    help='ARMT shmidhuber correction for rewriting')
parser.add_argument('--desired_metric', type=float, default=1.0, help='metric to stop training')

parser.add_argument('--output_last_segment_only', action='store_true', default=False,
                    help='')
# XXXX # RMT args 
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
parser.add_argument('--num_predict', type=int, default=4, help='number of predicted states')
parser.add_argument('--cot_setting', action='store_true', default=False, help='use CoT')
parser.add_argument('--learn_rule', action='store_true', default=False,
                    help='learn rule')
parser.add_argument('--constant_depth', action='store_true', default=False, help='ACT depth type')
parser.add_argument('--input_rule', action='store_true', default=False, help='RO-S learning')



if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()
    
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    args.block_size = (args.segment_size + 1) * (1 + args.repeat_state)
    sep_token, gen_token, eos_token, mask_token = 100, 101, 102, 104
    rule_token = 103
    
    logger = get_logger('')
    logger.info(args.model_cls)
    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    dataset_name_to_path = {
        "ca": "irodkin/1dCA_r2s20T20",
        # put your dataset_name -> dataset_path mappings here
    }
    dataset_path = dataset_name_to_path[args.dataset_name]

    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    logger.info(f'preparing dataset for: {args.task_name}')
    logger.info(f'sampling the length of the input: {args.sample_length}')

    with accelerator.main_process_first():
        train_dataset = load_dataset(dataset_path, split='train')
        args.rule_len = len(train_dataset[0]['rule_ids'])

        valid_dataset = load_dataset(dataset_path, split='validation')
        test_dataset = load_dataset(dataset_path, split='test')
        if args.dataset_name in ["ca", "addition_binary", "addition_decimal"]: 
            args.train_array_size = len(train_dataset[0]['input_ids_0'])
            args.valid_array_size = len(valid_dataset[0]['input_ids_0'])
        elif args.dataset_name in ["reverse_binary", "reverse_decimal", "copy_binary", "copy_decimal"]:
            args.train_array_size = len(train_dataset[0]['input_ids'])
            args.valid_array_size = len(valid_dataset[0]['input_ids'])
        else:
            raise ValueError('Unknown dataset name')

    prepare_run(args, logger, logger_fmt)
 
    # --- Collate function for the "ca_oo" task ---
    def ca_oo_collate_fn(
        batch,
        array_size,
        num_timesteps=args.num_timesteps,
        num_predict=args.num_predict,
        valid=False,
    ):
        for i, b in enumerate(batch):
            input_ids_seq = []

            for t in range(num_timesteps - args.repeat_state):
                input_ids_seq += [sep_token] + b[f'input_ids_{t}']
                if args.repeat_state:
                    input_ids_seq += [sep_token] + b[f'input_ids_{t+1}']
            
            if args.input_rule:
                input_ids_seq = b['rule_ids'] + input_ids_seq
            if args.learn_rule:
                input_ids_seq += [gen_token] + b['rule_ids']
            if args.repeat_state:
                input_ids_seq += [gen_token] + b[f'input_ids_{num_timesteps - 1}']
            else:
                input_ids_seq += [gen_token] 
            labels_seq = input_ids_seq.copy()
            
            if args.learn_rule:
                input_ids_seq[-args.rule_len:] = [rule_token] * args.rule_len

            labels_mask_seq = [0] * len(input_ids_seq)
            if args.learn_rule:
                labels_mask_seq[-len(b['rule_ids']) - 1:] = [1] * (len(b['rule_ids']) + 1)
            input_ids_generate_seq = input_ids_seq.copy() + [sep_token,]

            for t in range(num_timesteps, num_timesteps + num_predict):
                if not args.cot_setting:
                    input_ids_seq += [sep_token] + [mask_token] * array_size 
                    labels_mask_seq += [1,] * array_size + [0,] 
                else:
                    input_ids_seq += [sep_token] + b[f'input_ids_{t}']
                    labels_mask_seq += [1,] * array_size + [1,] 
                labels_seq += [sep_token] + b[f'input_ids_{t}'] 

            batch[i]['input_ids'] = input_ids_seq
            batch[i]['input_ids_generate'] = input_ids_generate_seq
            batch[i]['labels'] = labels_seq
            batch[i]['attention_mask'] = [1] * len(input_ids_seq)
            batch[i]['labels_mask'] = labels_mask_seq
            
        input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch], dim=0)
        labels = torch.stack([torch.tensor(b['labels']) for b in batch], dim=0)
        attention_mask = torch.stack([torch.tensor(b['attention_mask']) for b in batch], dim=0)
        input_ids_generate = torch.stack([torch.tensor(b['input_ids_generate']) for b in batch], dim=0)

        # store a mask region for the labels

        # labels_mask = torch.zeros_like(input_ids).bool()
        # labels_mask[:, -(num_predict * (array_size + 1)):] = True
        labels_mask = torch.stack([torch.tensor(b['labels_mask']) for b in batch], dim=0).bool()
        collated = {
            'input_ids': input_ids,
            'labels': labels, 
            'attention_mask': attention_mask,
            'labels_mask': labels_mask,
            'input_ids_generate': input_ids_generate,
            'attention_mask_generate': torch.ones_like(input_ids_generate)
        }
        return collated

    # --- Collate function for the "ca_adaptive" task ---
    def ca_adaptive_collate_fn(
        batch,
        array_size,
        num_timesteps=args.num_timesteps,  # how many states are "given" at the start
        valid=False,
    ):
        assert not args.learn_rule, 'learn_rule is not supported for ca_adaptive'
        # mapping from shift to special token
        shift2token = {
            1: 106,
            2: 107,
            3: 108,
            4: 109,
        }

        # We'll store the shift for each item so that we can compute metrics later
        for i, b in enumerate(batch):
            input_ids_seq = []
            for t in range(num_timesteps - args.repeat_state):
                input_ids_seq += [sep_token] + b[f'input_ids_{t}']
                if args.repeat_state:
                    input_ids_seq += [sep_token,] + b[f'input_ids_{t+1}']
            if args.repeat_state:
                input_ids_seq += [gen_token,] + b[f'input_ids_{num_timesteps - 1}']

            # If we're in validation, do something deterministic:
            if valid:
                # For instance, pick shift = (i % 4) + 1
                shift = (i % 4) + 1
            else:
                # Training can remain random
                shift = random.randint(1, 4)
            
            # Save the shift in this sample so that we can retrieve it later
            b['shift'] = shift

            shift_token_id = shift2token[shift]
            # (num_timesteps + shift) is presumably the next state to reveal
            input_ids_seq += [shift_token_id, gen_token if not args.repeat_state else sep_token] + b[f'input_ids_{num_timesteps + shift - 1}']

            labels_seq = input_ids_seq.copy()

            b['input_ids'] = input_ids_seq
            b['labels'] = labels_seq
            b['attention_mask'] = [1] * len(input_ids_seq)

        # Collate the batch
        input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch], dim=0)
        labels = torch.stack([torch.tensor(b['labels']) for b in batch], dim=0)
        attention_mask = torch.stack([torch.tensor(b['attention_mask']) for b in batch], dim=0)

        # Mark the region where labels apply
        labels_mask = torch.zeros_like(input_ids).bool()
        labels_mask[:, -array_size - 1:] = True

        # We also want to keep track of the shift in the final collated batch. 
        # We can store it as a tensor that parallels the batch dimension.
        shifts = torch.tensor([b['shift'] for b in batch], dtype=torch.long)

        collated = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'labels_mask': labels_mask,
            'shift': shifts,  # <--- so we can see each sample's shift
        }
        return collated


    # dictionary of collate functions:
    collate_fn_dict = {
        "ca_oo": ca_oo_collate_fn,
        "ca_adaptive": ca_adaptive_collate_fn,
        # add others if needed
    }

    collate_fn = collate_fn_dict[args.task_name]

    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    train_dataloader = DataLoader(
        train_dataset, batch_size=per_worker_batch_size, generator=train_rnd_generator,
        collate_fn=lambda x: collate_fn(x, array_size=args.train_array_size, valid=False),
        **kwargs, drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=per_worker_batch_size,
        collate_fn=lambda x: collate_fn(x, array_size=args.valid_array_size, valid=True),
        **kwargs, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=per_worker_batch_size,
        collate_fn=lambda x: collate_fn(x, array_size=args.valid_array_size, valid=True),
        **kwargs, drop_last=True
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
    
    # load backbone checkpoint if needed
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'])
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Wrap with memory cell if needed
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
        if args.constant_depth:
            mem_cell_args['constant_depth'] = args.constant_depth
        if args.act_format is not None:
            mem_cell_args['act_format'] = args.act_format

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
    model = recurrent_wrapper_cls(
        cell,
        segment_size=args.block_size,
        max_n_segments=args.max_n_segments,
        k2=args.k2,
        segment_alignment=args.segment_alignment,
        act_on=args.act_on,
        time_penalty=args.time_penalty
    )
    block_size = (args.segment_size + 1) * (1 + args.repeat_state)
    state_size = args.segment_size
    if 'armt' in args.model_path:
            assert args.num_timesteps == args.num_test_timesteps
            assert not args.learn_rule
            def spliter(x):
                if args.task_name == 'ca_oo':
                    assert (x.size(1) == (args.num_timesteps - args.repeat_state) * block_size + (state_size + 1) * (args.num_predict+args.repeat_state)) or x.size(1) == (args.num_timesteps - args.repeat_state) * block_size + (state_size + 1) * (args.repeat_state) + 1, f'{x.size(1)} != {(args.num_timesteps - args.repeat_state) * block_size + (state_size + 1) * (args.num_predict+args.repeat_state)} and {x.size(1)} != {(args.num_timesteps - args.repeat_state) * block_size + (state_size + 1) * (args.repeat_state) + 1}'
                elif args.task_name == 'ca_adaptive':
                    assert x.size(1) == (args.num_timesteps - args.repeat_state) * block_size + state_size * 2 + 3, f'{x.size(1)} != {(args.num_timesteps - args.repeat_state) * block_size + state_size * 2 + 3}'
                segmented = [x[:, i*block_size:(i+1)*block_size] for i in range(args.num_timesteps - args.repeat_state)] + [x[:, (args.num_timesteps-args.repeat_state)*block_size:],]
                assert sum([s.size(1) for s in segmented]) == x.size(1)
                return segmented
            if args.task_name in ['ca_oo', 'ca_adaptive']:
                model.split_tensor = spliter
    # load RMT checkpoint if needed
    if args.model_cpt and args.model_cpt != 'None':
        try:
            model_cpt = os.path.join(args.model_cpt, "model_best.pth")
            cpt = torch.load(model_cpt, map_location='cpu')
            model.load_state_dict(cpt)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')
        except:
            import safetensors
            model_cpt = os.path.join(args.model_cpt, "model_best/model.safetensors")
            cpt = safetensors.torch.load_file(model_cpt)
            w = model.load_state_dict(cpt, strict=True)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')
            logger.info(f'loaded model with mis w {w}')

    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            if 'memory' not in n and 'lora' not in n:
                p.requires_grad = False
        logger.info(f'Frozen model weights')
        logger.info(f'Remaining trainable parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        optimizer = optimizer_cls(
            model.parameters(), lr=args.lr,
            scale_parameter=args.scale_parameter,
            relative_step=args.relative_step,
            warmup_init=args.warmup_init,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        data = {}
        data['labels'] = batch['labels']
        data['labels_mask'] = batch['labels_mask']
        if 'generation_outputs' in output:
            data['generation_outputs'] = output['generation_outputs']
        # Shift is stored ONLY for ca_adaptive in our code, but won't break for other tasks
        # so we can just keep it if it exists:
        if 'shift' in batch:
            data['shift'] = batch['shift']

        if 'generation_outputs' in output:
            data['generation_outputs'] = output['generation_outputs']
        if 'logits' in output:
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

        # We only have "ca_oo" and "ca_adaptive" tasks
        # For "ca", we still rely on array_size from valid_array_size
        if args.dataset_name == "ca":
            array_size = args.valid_array_size

        # figure out how many states are predicted, and how large the predicted region is
        if args.task_name == "ca_adaptive":
            # for ca_adaptive, we predict exactly 1 state
            num_predict = 1
            total_pred_size = array_size
        elif args.task_name == "ca_oo":
            # for ca_oo, we have multiple predicted states
            num_predict = args.num_predict
            total_pred_size = num_predict * (array_size + 1) - 1
        else:
            raise ValueError(f'Unknown task_name: {args.task_name}')

        metrics = {}

        # region of reference ( ground truth ), same for ARMT or other variants
        y = data['labels'][:, -total_pred_size:]
        if args.learn_rule:
            rule = data['labels'][:, -total_pred_size - 1 - args.rule_len:-total_pred_size - 1]
        # region of predictions
        # shift by -1 because we typically ignore the last token for next-token prediction
        if 'generation_outputs' not in output:
            p = data['predictions'][:, -total_pred_size - 1 : -1]
            if args.learn_rule:
                predicted_rule = data['predictions'][:, -total_pred_size - 2 - args.rule_len:-total_pred_size - 2]
        else:
            if args.learn_rule:
                p = output['generation_outputs'][:, args.rule_len:args.rule_len + total_pred_size]
                if args.learn_rule:
                    predicted_rule = output['generation_outputs'][:, :args.rule_len]
            else:
                p = output['generation_outputs'][:, :total_pred_size]


        # ==============
        # Overall metrics
        # ==============
        # 1) bit_accuracy
        if args.learn_rule:
            metrics['rule_bit_accuracy'] = np.mean((rule.cpu().numpy()) == (predicted_rule.cpu().numpy()))
            metrics['rule_exact_match'] = np.mean([
                np.array_equal(p_, y_) for p_, y_ in zip(predicted_rule.cpu().numpy(), rule.cpu().numpy())
            ])
        metrics['bit_accuracy'] = np.mean((y.cpu().numpy()) == (p.cpu().numpy()))
        # 2) exact_match
        metrics['exact_match'] = np.mean([
            np.array_equal(p_, y_) for p_, y_ in zip(p.cpu().numpy(), y.cpu().numpy())
        ])

        # ================================================
        # SHIFT-based metrics (for ca_adaptive only)
        # ================================================
        # We only do this if we're in ca_adaptive
        if args.task_name == "ca_adaptive" and 'shift' in data:
            shift_vals = data['shift'].cpu().numpy()  # shape: (batch_size,)
            for shift_id in range(1, args.num_predict+1):
                mask = (shift_vals == shift_id)
                if not np.any(mask):
                    continue  # no samples had this shift in this batch
                y_shift = y[mask]
                p_shift = p[mask]

                metrics[f'bit_accuracy_s{shift_id}'] = np.mean(
                    (y_shift.cpu().numpy()) == (p_shift.cpu().numpy())
                )
                metrics[f'exact_match_s{shift_id}'] = np.mean([
                    np.array_equal(ps, ys)
                    for ps, ys in zip(p_shift.cpu().numpy(), y_shift.cpu().numpy())
                ])

        # ================================================
        # Per-state metrics (for ca_oo only)
        # ================================================
        if args.task_name == 'ca_oo':
            # For each predicted state, compute separate bit_accuracy and exact_match
            p_slices = []
            y_slices = []
            for i in range(num_predict):
                block_start = i * (array_size + 1)
                block_end   = block_start + array_size  # includes [sep_token] + array_size

                # predictions for state i (excluding the final sep, so -1)
                p_slice = p[:, block_start : block_end]  
                # ground truth for state i
                y_slice = y[:, block_start : block_end]

                y_slices.append(y_slice)
                p_slices.append(p_slice)

                # e.g. if we had 10 "given" states, then the predicted states might be 11..14
                state_number = args.num_timesteps + 1 + i

                bit_acc_i = np.mean((y_slice.cpu().numpy()) == (p_slice.cpu().numpy()))
                exact_match_i = np.mean([
                    np.array_equal(ps, ys)
                    for ps, ys in zip(p_slice.cpu().numpy(), y_slice.cpu().numpy())
                ])
                metrics[f'bit_accuracy_s{state_number}'] = bit_acc_i
                metrics[f'exact_match_s{state_number}'] = exact_match_i
            y_slices = torch.stack(y_slices, dim=0)
            p_slices = torch.stack(p_slices, dim=0)

            metrics['bit_accuracy'] = np.mean((y_slices == p_slices).cpu().numpy())
            metrics['exact_match'] = np.mean(torch.all(y_slices == p_slices, dim=(0, 2)).cpu().numpy())

        # ==============
        # Other metrics
        # ==============
        if 'loss' in output:
            metrics['loss'] = output['loss'].mean().item()

        if 'ce_loss' in data:
            metrics['ce_loss'] = data['ce_loss'].mean().item()
            try:
                metrics['perplexity'] = math.exp(metrics['ce_loss'])
            except OverflowError:
                metrics['perplexity'] = float("inf")

        if 'dist' in data:
            metrics['dist'] = data['dist'].mean().item()

        # if you have multiple segments, we keep track of per-segment CE losses
        for i in range(args.max_n_segments):
            if f'ce_loss_{i}' in data:
                metrics[f'ce_loss_{i}'] = data[f'ce_loss_{i}'].mean().item()

        # ACT (Adaptive Computation Time) metrics
        if args.act_on:
            metrics['n_updates'] = data['n_updates'].mean().item()
            metrics['remainders'] = data['remainders'].mean().item()

        return metrics


    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, None
    )

    fwd_kwargs = {}
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
        generate_kwargs={'max_new_tokens': args.num_predict * (args.valid_array_size + 1) + 1}
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
        from functools import partial
        import inspect
        class UnpackWrapper(torch.nn.Module):
            def __init__(self, model):
                super(UnpackWrapper, self).__init__()
                self.model = model

            def forward(self, batch):
                args = self.get_function_arguments(self.model.forward)
                # print(args)
                args = [a for a in args if a in batch]
                # print(batch, args)
                batch = dict(zip(args, [batch[a] for a in args]))
                return self.model(**batch)
            
            def get_function_arguments(self, func):
                sig = inspect.signature(func)
                return [param.name for param in sig.parameters.values()]
        batch = next(iter(valid_dataloader))
        # partial_model = partial(trainer.model.forward, **next(iter(valid_dataloader)))
        flop_analysis = FlopCountAnalysis(UnpackWrapper(trainer.model.module), batch)
        logger.info(f"FLOPs: {flop_analysis.total()}")
        trainer.run.log({'FLOPs': flop_analysis.total()})
        if valid_dataloader is not None:
            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=True, split='valid')
        else:
            raise "No valid dataset"

    print('Done!')
