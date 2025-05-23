import json
import logging
import os
import math
import shutil
from pathlib import Path
from itertools import chain

# from dotenv import load_dotenv
import torch
import numpy as np
import datasets
import transformers

from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from lm_experiments_tools import  Trainer,  TrainerArgs

from torch.nn.utils.rnn import pad_sequence

import accelerate

from abstract_algebra.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
)
# load_dotenv()

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...

parser = HfArgumentParser(TrainerArgs)

parser.add_argument('--report_to', type=str, default='wandb', help='')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')

parser.add_argument('--grad_cp',action='store_true', default=False, help='enable gradient_checkpointing')
parser.add_argument('--noisy_halting', action='store_true', default=False,
                    help='add noise to halting')
parser.add_argument('--output_last_segment_only', action='store_true', default=False,
                    help='')
parser.add_argument('--wrap_pos', action='store_true', default=False,
                    help='Wrap positional encoding for memory tokens (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
# parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
# parser.add_argument('--target_seq_len', type=int, default=16, help='target sequnce length, should be set to '
                                                                #    'max(len(target))+1 for EOS (default: 16).')
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
parser.add_argument('--model_type', type=str, default='decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: decoder)')

# Dataset args
parser.add_argument('--length', type=int, default=10, help='number of group elements in a sequence')

parser.add_argument('--dataset_path', type=str, default="XXXX/groupmul_A5_split")
parser.add_argument('--num_samples', type=int, default=100000, help='number of samples in a dataset')
parser.add_argument('--segment_size', type=int, default=128, help='number of useful tokens in a segment')
parser.add_argument('--d_mem', type=int, default=None, help='number of rows in associative matrix')
parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')

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
# XXXX # RMT args 
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--segment_alignment', type=str, default=None, help="How to align segments when splitting input")
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')


# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')
parser.add_argument('--predict_from_mask', action='store_true', default=False,
                    help='Diables autoregressive generation')
parser.add_argument('--generate_gen_token', action='store_true', default=False,
                    help='Generate gen token')



from tqdm.auto import tqdm


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    from accelerate.logging import get_logger
    logger = get_logger('')
    logger.info(args.model_cls)

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    prepare_run(args, logger, logger_fmt)

    # if not args.from_pretrained:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)


    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def collate_fn(batch, valid=False):
        # assert False, "I should handle the autoregressive shift"
        for i, b in enumerate(batch):
            batch[i]['input_ids'] = np.array(b['input_ids'] + [0,])
            batch[i]['labels'] = np.array([0,] + b['labels'])
            batch[i]['attention_mask'] = np.ones_like(batch[i]['input_ids']).astype(np.int64)
            
        input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch], dim=0)
        labels = torch.stack([torch.tensor(b['labels']) for b in batch], dim=0)
        attention_mask = torch.stack([torch.tensor(b['attention_mask']) for b in batch], dim=0)
        labels_mask = torch.ones_like(input_ids).bool()
        collated = {'input_ids': input_ids,
                    'labels': labels, 
                    'attention_mask': attention_mask,
                    'labels_mask': labels_mask
        }
        return collated

    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    # get train dataset
    logger.info(f'preparing dataset for: Groupmul with dataset {args.dataset_path} and length {args.length}')
    with accelerator.main_process_first():
        train_dataset = load_dataset(args.dataset_path, f'length_{args.length}', split='train')
        valid_dataset = load_dataset(args.dataset_path, f'length_{args.length}', split='validation')
        test_dataset = load_dataset(args.dataset_path, f'length_{args.length}', split='test')
    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size,  generator=train_rnd_generator,
                                  collate_fn=collate_fn, **kwargs, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size,
                                  collate_fn=collate_fn, **kwargs, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size,
                                  collate_fn=collate_fn, **kwargs, drop_last=True)
    

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
        model_args = dict()
        if args.grad_cp:
            model_args['grad_cp'] = args.grad_cp
        model = model_cls.from_pretrained(args.from_pretrained, **model_args)

    # ## add [GEN] token
    # model.resize_token_embeddings(len(tokenizer))
    
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
    
    
    mem_cell_args = dict(
        base_model=model,
    )
    if args.d_mem is not None:
        mem_cell_args['d_mem'] = args.d_mem

    if args.act_on:
        mem_cell_args['act_on'] = args.act_on
        mem_cell_args['max_hop'] = args.max_hop
        
        if args.act_type is not None:
            mem_cell_args['act_type'] = args.act_type

        if args.act_format is not None:
            mem_cell_args['act_format'] = args.act_format
        if args.noisy_halting:
            mem_cell_args['noisy_halting'] = args.noisy_halting


    if args.num_mem_tokens is not None:
        mem_cell_args['num_mem_tokens'] = args.num_mem_tokens
        mem_cell_args['wrap_pos'] = args.wrap_pos
    if args.layers_attr is not None:
        mem_cell_args['layers_attr'] = args.layers_attr
    if args.no_denom:
        mem_cell_args['use_denom'] = not args.no_denom
    if args.freeze_mem is not None:
        mem_cell_args['freeze_mem'] = args.freeze_mem

    if args.no_correction:
        mem_cell_args['correction'] = False

    

    cell = memory_cell_cls(**mem_cell_args)

    model = recurrent_wrapper_cls(cell, 
                                    segment_size=args.segment_size,
                                    max_n_segments=args.max_n_segments, 
                                #   vary_n_segments=args.vary_n_segments,
                                    k2=args.k2,
                                    segment_alignment=args.segment_alignment,
                                    act_on=args.act_on,
                                    time_penalty=args.time_penalty
    )
                                
    if 'armt' in args.model_path:

        assert False, "ARMT is not supported yet"
    
    ## load cpt of rmt
    if args.model_cpt and args.model_cpt != 'None':
        
        model_cpt = os.path.join(args.model_cpt, "model_best/pytorch_model.bin")
        if os.path.exists(model_cpt):
            cpt = torch.load(model_cpt, map_location='cpu')
            model.load_state_dict(cpt)
        else:
            import safetensors
            model_cpt = os.path.join(args.model_cpt, "model_best/model.safetensors")
            cpt = safetensors.torch.load_file(model_cpt)
            w = model.load_state_dict(cpt, strict=False)
            logger.info(f'loaded model with mis w {w}')
        logger.info(f'Loaded model state dict from: {args.model_cpt}')
    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            # if 'memory' not in n and 'wte' not in n:
            if 'memory' not in n and 'lora' not in n:
                p.requires_grad = False
        logger.info(f'Frozen moodel weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    # # fix the not-contiguous error with loralib and horovod
    # def make_contiguous(module):
    #     with torch.no_grad():
    #         for param in module.parameters():
    #             param.set_(param.contiguous())
    # make_contiguous(model)
    
    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
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
        # select data from batch and model output that would be used to compute metrics
        data = {}

        data['labels'] = batch['labels']
        data['labels_mask'] = batch['labels_mask']
        if 'generation_outputs' in output:

            data['generation_outputs'] = output['generation_outputs']
            # if 'labels_mask' in batch:
            #     data['generation_outputs'] = [data['generation_outputs'][i, mask] for i, mask in enumerate(batch['labels_mask'])]
        # if args.model_type == 'encoder':
            
        ##### XXXX
        data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        # data['labels'] = batch['labels']
        for key in batch.keys():
            if 'loss' in key: 
                data[key] = batch[key]
        # else:
        if args.act_on:
            data['n_updates'] = output['n_updates']
            data['remainders'] = output['remainders']
        return data

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        
        metrics = {}
        l = data['labels'].size(1)
        y, p = data['labels'][:, 1:], data['predictions'][:, :-1]

        if accelerator.is_main_process and args.show_valid_examples > 0:
            for i in range(min(args.show_valid_examples, len(y))):
                y_ = np.array(y[i])
                p_ = np.array(p[i])
                logger.info(f'y: {y_}')
                logger.info(f'p: {p_}')
                logger.info(f'y: {y[i]}')
                logger.info(f'p: {p[i]}')
                logger.info('-' * 50)
        if 'ce_loss' in data:
            metrics['ce_loss'] = data['ce_loss'].mean()
            try:
                perplexity = math.exp(metrics['ce_loss'])
            except OverflowError:
                perplexity = float("inf")

            metrics["perplexity"] = perplexity
        
        if 'dist' in data:
            metrics['dist'] = data['dist'].mean()
            
        for i in range(args.max_n_segments):
            if f'ce_loss_{i}' in data:
                metrics[f'ce_loss_{i}'] = data[f'ce_loss_{i}'].mean()
        metrics['bit_accuracy'] = np.mean(np.array(y) == np.array(p))
        metrics['exact_match'] = np.mean([np.array_equal(p_, y_) for p_, y_ in zip(p, y)])

        
        if args.act_on:
            metrics['n_updates'] = torch.mean(data['n_updates']).item()
            metrics['remainders'] = torch.mean(data['remainders']).item()
        return metrics

    # accelerate
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, None)

    ### XXXX

    fwd_kwargs = dict()
    if args.output_last_segment_only:
        fwd_kwargs['output_only_last_segment'] = True
    
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, valid_dataloader,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ###XXXX
                      batch_metrics_fn=batch_metrics_fn,
                      stop_metric_condition=lambda m: m >= args.desired_metric,
                      forward_kwargs=fwd_kwargs,
                    )

    # try:
    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        accelerator.wait_for_everyone()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best')
            logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False, split='valid')
        # if test_dataloader is not None:
        #     logger.info('Runnning validation on test data:')
            # trainer.validate(test_dataloader, write_tb=True, split='test')
        trainer.save_metrics(save_path=args.model_path)
    else:
        from fvcore.nn import FlopCountAnalysis
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
        # run validation, do not write to tensorboard
        # logger.info('Running validation on train set:')
        # trainer.validate(train_dataloader, split='train', write_tb=True)
        if valid_dataloader is not None:
            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=True, split='valid')
        else:
            raise "No valid dataset"
        # if test_dataloader is not None:
        #     logger.info('Runnning validation on test data:')
        #     trainer.validate(test_dataloader, write_tb=True, split='test')
    # except Exception as e:
    #     print(f"Got exception: {e}")
    print('Done!')