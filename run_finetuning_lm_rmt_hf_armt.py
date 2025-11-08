import logging
from pathlib import Path
from itertools import chain
import os
import torch
import numpy as np
import random
import datasets
from torch.utils.data import DataLoader
import datetime
from itertools import chain
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.nn.utils.rnn import pad_sequence
from datasets.distributed import split_dataset_by_node

import accelerate
from accelerate.utils import InitProcessGroupKwargs
from peft import get_peft_model, LoraConfig, TaskType
from transformers import modeling_utils
from torch.utils.data import IterableDataset
import heapq
from tqdm import tqdm
import gc

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')

# Suppress verbose torch.distributed warnings
logging.getLogger('torch.distributed.distributed_c10d').setLevel(logging.ERROR)


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])
# if "LOCAL_RANK" in os.environ:
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model

from lm_experiments_tools.utils import get_cls_by_name


parser = HfArgumentParser(TrainingArguments)
parser.add_argument('--task_name', type=str, help="Task name, wikitext, ...")
parser.add_argument('--valid_task_name', type=str, help="Task name, wikitext, ...")
parser.add_argument('--tokenized_dataset', type=str, help="path to folder with tokenized hf dataset")
parser.add_argument('--valid_tokenized_dataset', type=str, help="path to folder with tokenized valid hf dataset", default=None)
parser.add_argument('--train_tokens', type=str, default="input_ids")
parser.add_argument('--valid_tokens', type=str, default="input_ids")

parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')

parser.add_argument('--attn_implementation', type=str, default='flash_attention_2',
                    help='')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--sample_size', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--val_sample_size', type=int, default=128, help='input sequnce length for validation (default: 128).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attentinon mask, '
                    'eval on last segment only', default=False)
parser.add_argument('--attend_to_previous_input', action='store_true', help='attend to the previous segment', default=False)

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:AutoModel',
                    help='model class name to use (default: transformers:AutoModel)')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--checkpoint', type=str, default=None, help='full experiment checkpoint')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')


# ARMT args
parser.add_argument('--segment_size', type=int, default=None, help='number of real tokens in block')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--loss_from_last_seg_only', action='store_true', default=False, help='take loss from last segment only')
parser.add_argument('--no_loss_from_first_segment', action='store_true', default=False, help='turn off loss from first segment')

parser.add_argument('--min_sample_len', type=int, default=16000, help='min sample len in tokens')


parser.add_argument('--sum_loss', action='store_true', default=False,
                    help='with this flag task loss from all segments is summed')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--retain_graph', action='store_true', help='Retain computation graph during backward pass', default=False)
parser.add_argument('--use_truncated_backward', action='store_true', default=False,
                    help='whether to use RMT truncated bptt method in backward')
parser.add_argument('--k1', type=int, default=-1, help='(not implemented) If not -1, gradient update is done each k1 segments')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')


# tokenizer
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')

# LoRA args
parser.add_argument('--use_lora', action='store_true', default=False, help='')
parser.add_argument('--lora_attn_dim', type=int, default=8, help='')
parser.add_argument('--lora_attn_alpha', type=int, default=32, help='')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='')

parser.add_argument('--d_mem', type=int, default=None, help='number of rows in associative matrix')
parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')

parser.add_argument('--prev_seg_kv', action='store_true', default=False, help='propagate kv from previous segment')
parser.add_argument('--use_sink', action='store_true', default=False, help='use_attention_sink_token')
parser.add_argument('--armt_impl', type=str, choices=['outer', 'inner', 'mem_params'], default='outer',
                    help='ARMT implementation: outer (AssociativeRecurrentWrapper) or inner (per-layer inner-loop)')
parser.add_argument('--streaming', action='store_true', default=False, help='use streaming dataset')
parser.add_argument('--stream_chunk_docs', type=int, default=5000, help='number of raw samples per streaming tokenization chunk')
parser.add_argument('--alternate_layers', action='store_true', default=False,
                    help='If set, wrap alternating transformer layers (1,0,1,0,...) in ARMT')
os.environ['HF_Trainer'] = '1'
if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir

    training_args_dict = {key: value for key, value in vars(args).items() if hasattr(TrainingArguments('.'), key)}

    training_args_dict['remove_unused_columns'] = False
    training_args_dict['save_safetensors'] = False
    training_args_dict['bf16'] = True
    training_args_dict['label_names'] = ['labels']
    
    # Debug: Add average_tokens_across_devices=False to see if this affects loss
    # training_args_dict['average_tokens_across_devices'] = False

    training_args_dict['eval_strategy'] = 'steps'
    training_args_dict['per_device_eval_batch_size'] = training_args_dict.get('per_device_train_batch_size') # // 2
    training_args_dict['eval_accumulation_steps'] = training_args_dict['gradient_accumulation_steps']
    # print("="*20, training_args_dict['gradient_accumulation_steps'], "="*20)
    if args.d_mem is None:
        # for now, gradient checkpointing is not supported for ARMT
        training_args_dict['gradient_checkpointing'] = True
    else:
        training_args_dict['gradient_checkpointing'] = False
    
    # training_args_dict['gradient_checkpointing_kwargs'] = {'use_reentrant':False}
    # training_args_dict['log_level'] = 'debug'
    training_args_dict['report_to'] = 'wandb'
    # Push checkpoints to Hugging Face Hub every 1000 steps
    training_args_dict['save_strategy'] = 'steps'
    training_args_dict['save_steps'] = 1000
    training_args_dict['push_to_hub'] = True
    training_args_dict['hub_strategy'] = 'every_save'
    # Avoid duplicating iterable streams across dataloader workers when streaming
    if args.streaming:
        training_args_dict['dataloader_num_workers'] = 0
    else:
        training_args_dict['dataloader_num_workers'] = args.data_n_workers
    training_args = TrainingArguments(**training_args_dict)

    if args.valid_tokenized_dataset is None:
        args.valid_tokenized_dataset = args.tokenized_dataset
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(1))
    from accelerate.logging import get_logger
    logger = get_logger('')


    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained, trust_remote_code=True)

    # Prepare datasets
    logger.info(f'preparing dataset for {args.task_name}')

    # Helper function to check if dataset directory contains chunks
    def is_chunked_dataset(dataset_path):
        """Check if dataset directory contains chunk subdirectories"""
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            return False
        chunk_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('chunk_')]
        return len(chunk_dirs) > 0

    # Resolve rank/world size robustly
    def _get_rank_world_size():
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass
        rank = getattr(training_args, 'process_index', None)
        world_size = getattr(training_args, 'world_size', None)
        if rank is None:
            rank = int(os.environ.get('RANK', '0'))
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
        return int(rank), int(world_size)

    with training_args.main_process_first(desc="dataset prep"):
        if args.tokenized_dataset is not None:
            # Check if this is a chunked dataset
            is_chunked = is_chunked_dataset(args.tokenized_dataset)
            
            if is_chunked:
                logger.info(f"Detected chunked dataset format in {args.tokenized_dataset}")
                
                # If no separate validation dataset provided, extract from first chunk
                if args.valid_tokenized_dataset == args.tokenized_dataset:
                    logger.info("No separate validation dataset provided - will extract from first chunk")
                    
                    # Load first chunk to extract validation/test
                    first_chunk_dir = sorted([
                        d for d in Path(args.tokenized_dataset).iterdir() 
                        if d.is_dir() and d.name.startswith('chunk_')
                    ])[0]
                    
                    logger.info(f"Loading first chunk {first_chunk_dir.name} to extract validation/test sets...")
                    first_chunk = datasets.load_from_disk(str(first_chunk_dir))
                    logger.info(f"First chunk loaded: {len(first_chunk)} samples")
                    
                    # Take first 2000 samples for val/test
                    val_test_samples = min(2000, len(first_chunk) // 10)  # At most 2000 or 10% of chunk
                    logger.info(f"Extracting first {val_test_samples} samples for validation/test")
                    
                    val_test_data = first_chunk.select(range(val_test_samples))
                    
                    # Split into validation and test
                    val_size = val_test_samples // 2
                    validation_data = val_test_data.select(range(val_size))
                    test_data = val_test_data.select(range(val_size, val_test_samples))
                    
                    logger.info(f"Created validation set: {len(validation_data)} samples, test set: {len(test_data)} samples")
                    
                    # Create validation dataset dict
                    validation_dataset = datasets.DatasetDict({
                        'validation': validation_data,
                        'test': test_data
                    })
                    
                    # Store the number of samples to skip from first chunk during training
                    chunked_skip_samples = val_test_samples
                    
                    # Clean up
                    del first_chunk
                    del val_test_data
                    gc.collect()
                else:
                    # Use provided validation dataset
                    logger.info(f"Using provided validation dataset from {args.valid_tokenized_dataset}")
                    validation_dataset = datasets.load_from_disk(args.valid_tokenized_dataset)
                    chunked_skip_samples = 0
                
                # For chunked datasets, we'll create a special placeholder
                dataset = {'train': 'chunked'}  # Placeholder to indicate chunked format
            else:
                logger.info(f"Loading regular tokenized dataset from {args.tokenized_dataset}")
                dataset = datasets.load_from_disk(args.tokenized_dataset)
                validation_dataset = datasets.load_from_disk(args.valid_tokenized_dataset)
                chunked_skip_samples = 0  # Not used for non-chunked datasets
            
            logger.info("Tokenized Dataset loaded")
            if not is_chunked and args.valid_tokens != args.train_tokens:
                validation_dataset = validation_dataset.rename_column(args.valid_tokens, args.train_tokens)
        else:
            # Not using tokenized dataset - streaming or on-the-fly tokenization
            chunked_skip_samples = 0  # Not applicable for non-tokenized datasets
            
            # Load dataset with streaming=True to load samples on the fly
            train_dataset = datasets.load_dataset(args.task_name, split='train', streaming=args.streaming, trust_remote_code=True)
            if args.valid_task_name is not None:
                validation_dataset = datasets.load_dataset(args.valid_task_name, split='validation', trust_remote_code=True)
                test_dataset = datasets.load_dataset(args.valid_task_name, split='test', trust_remote_code=True)
                if args.streaming:
                    rank, world_size = _get_rank_world_size()
                    train_dataset = train_dataset.shard(num_shards=world_size, index=rank)
            else:
                # Take the first 1000 samples from train dataset for validation and test
                if args.streaming:
                    # For streaming datasets, use take() and skip()
                    # Shard by rank to avoid all ranks pulling the same samples
                    rank, world_size = _get_rank_world_size()

                    validation_dataset = train_dataset.take(1000)
                    test_dataset = train_dataset.skip(1000).take(1000)
                    train_dataset = train_dataset.skip(2000).shard(num_shards=world_size, index=rank)
                else:
                    # For regular datasets, create random train/val/test split
                    logger.info("Creating random train/validation/test split from train dataset")
                    
                    # First split: separate out 2000 samples for val+test
                    split_data = train_dataset.train_test_split(test_size=2000, seed=args.seed)
                    train_dataset = split_data['train']
                    val_test_dataset = split_data['test']
                    
                    # Second split: divide val+test into validation and test
                    val_test_split = val_test_dataset.train_test_split(test_size=0.5, seed=args.seed)
                    validation_dataset = val_test_split['train']
                    test_dataset = val_test_split['test']
                    
                    logger.info(f"Split sizes - Train: {len(train_dataset)}, Val: {len(validation_dataset)}, Test: {len(test_dataset)}")
            logger.info("Dataset loaded")
            # Create a function to tokenize on the fly
            
            def tokenize_function(examples):
                result = tokenizer.encode(examples['text'], return_tensors='pt')
                examples[args.train_tokens] = result[0]
                return examples
            if not args.streaming:
                
                # Apply tokenization on the fly
                train_dataset = train_dataset.map(
                    tokenize_function,
                    batched=False,
                    remove_columns=['text'],
                )
            validation_dataset = validation_dataset.map(
                tokenize_function,
                batched=False,
                remove_columns=['text'],
            )
            test_dataset = test_dataset.map(
                tokenize_function,
                batched=False,
                remove_columns=['text'],
            )
            
            # Create a DatasetDict with the processed splits
            dataset = datasets.DatasetDict({
                'train': train_dataset.with_format("torch"),
                'validation': validation_dataset.with_format("torch"),
                'test': test_dataset.with_format("torch")
            })
            validation_dataset = dataset


    segment_size = args.segment_size
    history_size = args.sample_size - segment_size

    if args.val_sample_size is not None:
        val_history_size = args.val_sample_size - segment_size
    else:
        val_history_size = history_size

    class ChunkedDatasetIterator(IterableDataset):
        """
        Loads chunked datasets (chunk_000000, chunk_000001, ...) sequentially.
        Each chunk is processed with group_texts just like a normal dataset.
        Dynamically detects new chunks that appear during training.
        """
        def __init__(self, dataset_dir, segment_size, history_size, token_column='tokens', seed=42, 
                     skip_first_n_samples=0):
            self.dataset_dir = Path(dataset_dir)
            self.seg = segment_size
            self.hist = history_size
            self.token_column = token_column
            self.seed = seed
            self.skip_first_n_samples = skip_first_n_samples  # For excluding val/test from first chunk
            
            # Find initial chunk directories
            initial_chunks = self._get_chunk_dirs()
            
            if not initial_chunks:
                raise ValueError(f"No chunks found in {dataset_dir}. Expected directories like 'chunk_000000', 'chunk_000001', etc.")
            
            logger.info(f"Found {len(initial_chunks)} initial chunks in {dataset_dir}")
            logger.info(f"Chunks will be loaded and processed sequentially: {initial_chunks[0].name} ... {initial_chunks[-1].name}")
            logger.info(f"NOTE: Will dynamically check for new chunks during training")
            if self.skip_first_n_samples > 0:
                logger.info(f"NOTE: First {self.skip_first_n_samples} samples from chunk_000000 will be skipped (reserved for validation/test)")
        
        def _get_chunk_dirs(self):
            """Get sorted list of chunk directories"""
            return sorted([
                d for d in self.dataset_dir.iterdir() 
                if d.is_dir() and d.name.startswith('chunk_')
            ])
        
        def _get_chunk_index(self, chunk_dir):
            """Extract chunk index from directory name (e.g., 'chunk_000000' -> 0)"""
            # Split by underscore and take the numeric part
            return int(chunk_dir.name.split('_')[1])
        
        def _get_max_chunk_index(self):
            """Get the maximum chunk index currently available"""
            chunk_dirs = self._get_chunk_dirs()
            if not chunk_dirs:
                return -1
            return self._get_chunk_index(chunk_dirs[-1])
        
        def _get_chunk_dir_by_index(self, target_idx):
            """Get chunk directory path for a given index, returns None if not found"""
            for chunk_dir in self._get_chunk_dirs():
                if self._get_chunk_index(chunk_dir) == target_idx:
                    return chunk_dir
            return None
        
        def __iter__(self):
            """Iterate through all chunks, loading and processing one at a time.
            Dynamically checks for new chunks during training."""
            chunk_idx = 0
            
            # Use while loop to dynamically check for new chunks
            while chunk_idx <= self._get_max_chunk_index():
                # Get actual chunk directory by index
                chunk_dir = self._get_chunk_dir_by_index(chunk_idx)
                
                # Check if this chunk exists
                if chunk_dir is None:
                    logger.warning(f"[Chunk {chunk_idx + 1}] chunk with index {chunk_idx} does not exist yet, skipping...")
                    chunk_idx += 1
                    continue
                
                chunk_name = chunk_dir.name
                max_chunk_idx = self._get_max_chunk_index()
                logger.info(f"[Chunk {chunk_idx + 1}/{max_chunk_idx + 1}] Loading {chunk_name}...")
                
                # Load this chunk
                chunk_dataset = datasets.load_from_disk(str(chunk_dir))
                logger.info(f"[Chunk {chunk_idx + 1}/{max_chunk_idx + 1}] Loaded {len(chunk_dataset)} samples from {chunk_name}")
                
                # Skip validation/test samples from the first chunk
                if chunk_idx == 0 and self.skip_first_n_samples > 0:
                    original_size = len(chunk_dataset)
                    chunk_dataset = chunk_dataset.select(range(self.skip_first_n_samples, len(chunk_dataset)))
                    logger.info(f"[Chunk {chunk_idx + 1}/{max_chunk_idx + 1}] Skipped first {self.skip_first_n_samples} samples (val/test), using {len(chunk_dataset)}/{original_size} samples")
                
                # Process chunk with group_texts (same as normal pipeline)
                logger.info(f"[Chunk {chunk_idx + 1}/{max_chunk_idx + 1}] Processing with group_texts (segment_size={self.seg}, history_size={self.hist})...")
                
                processed = chunk_dataset.select_columns([self.token_column]).map(
                    lambda x: group_texts(x, self.seg, self.hist),
                    batched=True,
                    batch_size=4096
                )
                
                # Shuffle the processed chunk
                processed = processed.shuffle(seed=self.seed + chunk_idx)
                
                logger.info(f"[Chunk {chunk_idx + 1}/{max_chunk_idx + 1}] Processed into {len(processed)} windows, yielding...")
                
                # Yield all samples from this processed chunk
                for sample in processed:
                    yield sample
                
                # Explicitly delete the chunk datasets and run garbage collection
                del chunk_dataset
                del processed
                gc.collect()
                logger.info(f"[Chunk {chunk_idx + 1}/{max_chunk_idx + 1}] Completed and unloaded {chunk_name}")
                
                # Move to next chunk
                chunk_idx += 1
                
                # Check if new chunks appeared
                new_max_chunk_idx = self._get_max_chunk_index()
                if new_max_chunk_idx > max_chunk_idx:
                    logger.info(f"[Dynamic Update] Detected {new_max_chunk_idx - max_chunk_idx} new chunks! Will continue training on them.")

    class ChunkedWindowStream(IterableDataset):
        def __init__(self, raw_ds, segment_size, history_size, chunk_tokens, dataset_length, seed=0):
            self.raw_ds       = raw_ds
            self.seg          = segment_size
            self.hist         = history_size
            self.block        = segment_size + history_size
            self.chunk_tokens = chunk_tokens
            self.seed         = seed
            self.dataset_length = dataset_length

            # how many windows per "epoch" (used to satisfy __len__)
            # self.windows_per_epoch = self.chunk_tokens // self.block

        # def __len__(self):
        #     return self.dataset_length

        def __iter__(self):
            buf = []
            rng = random.Random(self.seed)
            
            for sample in self.raw_ds:
                # accumulate tokens
                buf.extend(sample[args.train_tokens])
                
                # once we have enough, build ALL windows at once
                if len(buf) >= self.chunk_tokens:
                    flat     = np.array(buf, dtype=np.int32)
                    starts   = np.arange(self.hist, len(flat) - self.seg + 1, self.seg, dtype=int)
                    idx      = starts[:, None] + np.arange(-self.hist, self.seg, dtype=int)
                    windows  = flat[idx]   # shape (n_windows, hist+seg)
                    
                    # shuffle the windows
                    windows = windows.tolist()
                    rng.shuffle(windows)
                    
                    # yield them in random order
                    for w in windows:
                        yield {args.train_tokens: w}
                    
                    # clear buffer for next chunk
                    buf = []

    class OnlineWindowStream(IterableDataset):
        def __init__(self, raw_ds, segment_size, history_size,
                    chunk_tokens, seed=0):
            self.raw_ds  = raw_ds
            self.seg     = segment_size
            self.hist    = history_size
            self.block   = segment_size + history_size
            self.B       = chunk_tokens // segment_size
            self.seed    = seed
            self.stats   = {
                'raw_samples_consumed': 0,
                'total_tokens_consumed': 0,
                'windows_yielded': 0
            }

        def __iter__(self):
            rng  = random.Random(self.seed)
            buf  = []          # holds ≤ B windows
            tail = []          # rolling token tail for windowing
            bar = tqdm(total=self.B, desc="Filling buffer")
            
            for sample in self.raw_ds:
                sample_tokens = sample[args.train_tokens]
                tail.extend(sample_tokens)
                
                # Track statistics
                self.stats['raw_samples_consumed'] += 1
                self.stats['total_tokens_consumed'] += len(sample_tokens)
                
                # Log every 1000 samples
                if self.stats['raw_samples_consumed'] % 1000 == 0:
                    logger.info(f"[Dataset Stats] Consumed {self.stats['raw_samples_consumed']:,} raw samples, "
                              f"{self.stats['total_tokens_consumed']:,} tokens, "
                              f"yielded {self.stats['windows_yielded']:,} windows")
                
                # emit as many full windows as we can
                while len(tail) >= self.block:
                    win  = tail[: self.block]
                    tail = tail[self.seg :]          # slide by segment_size

                    # ───── Fisher–Yates with fixed buffer ─────
                    if len(buf) < self.B:
                        buf.append(win)              # just fill
                        bar.update(1)                # update progress bar
                        if len(buf) == self.B:
                            bar.set_description("Buffer is full")
                            bar.close()
                    else:
                        j = rng.randrange(self.B)    # 0 … B-1
                        yield {args.train_tokens: buf[j]}  # emit old window
                        self.stats['windows_yielded'] += 1
                        buf[j] = win
                                      # insert new one
                    # -------------------------------------------

            rng.shuffle(buf)
            for w in buf:
                yield {args.train_tokens: w}
                self.stats['windows_yielded'] += 1
            
            # Final stats
            logger.info(f"[Dataset Final Stats] Total consumed: {self.stats['raw_samples_consumed']:,} samples, "
                      f"{self.stats['total_tokens_consumed']:,} tokens, "
                      f"yielded {self.stats['windows_yielded']:,} windows")

    class HashedWindowStream(IterableDataset):
        """
        Online sliding-window builder + hash-based bounded shuffle.
        Produces a near-perfect random permutation using only `B` windows of RAM.
        """
        def __init__(
            self,
            raw_ds,
            segment_size,
            history_size,
            chunk_tokens,
            seed=0,
        ):
            self.raw_ds   = raw_ds
            self.seg      = segment_size
            self.hist     = history_size
            self.block    = segment_size + history_size
            self.B        = chunk_tokens // segment_size
            self.seed     = seed
            random.seed(seed)

        def __iter__(self):
            tail = []
            heap = []                     # min-heap of (key, window)

            for sample in self.raw_ds:
                tail.extend(sample["input_ids"])

                # build windows on-the-fly
                while len(tail) >= self.block:
                    win = tail[: self.block]
                    tail = tail[self.seg :]

                    key  = random.random()
                    # keep key positive so heapq is happy
                    heapq.heappush(heap, (key, win.copy()))

                    if len(heap) > self.B:
                        _, w = heapq.heappop(heap)   # smallest key
                        yield {"input_ids": w.tolist()}

            # end-of-stream → flush the heap
            heap.sort()        # turn heap into sorted list by key
            for _, w in heap:
                yield {"input_ids": w.tolist()}

    def group_texts(examples, segment_size, history_size=None):
        # concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # total_length = len(concatenated_examples[list(examples.keys())[0]])

        # if history_size is None:
        #     result = {
        #         k: [t[i : i + segment_size] for i in range(0, total_length, segment_size)]
        #         for k, t in concatenated_examples.items()
        #     }
        # else:
        #     result = {
        #         k: [t[max({0, i - history_size}) : i + segment_size] for i in range(history_size, total_length, segment_size)]
        #         for k, t in concatenated_examples.items()
        #     }
        # return result
        # 1. flatten once, in C
        col = 'input_ids'
        result = dict()
        for col in examples.keys():
            flat = np.fromiter(chain.from_iterable(examples[col]), dtype=np.int32)

            if history_size is None:
                usable = (len(flat) // segment_size) * segment_size          # trim ragged tail
                flat   = flat[:usable].reshape(-1, segment_size)
                return {col: flat.tolist()}

            # 2. sliding-window with stride = segment)suze
            starts  = np.arange(history_size, len(flat) - segment_size - history_size + 1, segment_size + history_size, dtype=np.int32)
            idx     = starts[:, None] + np.arange(-history_size, segment_size, dtype=np.int32)
            windows = flat[idx]                                # (n_windows, history_size+block)
            result[col] = windows.tolist()
        return result



    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def collate_fn(batch):
        tokens = args.train_tokens
        input_ids = labels = [torch.tensor(b[tokens]) for b in batch]
        attention_mask = [torch.ones_like(b, dtype=int) for b in input_ids]


        labels_mask = [torch.ones_like(b, dtype=int) for b in input_ids]
        
        if getattr(args, 'loss_from_last_seg_only', False):
            for m in labels_mask:
                m[:-args.segment_size] = False

        if getattr(args, 'no_loss_from_first_segment', False):
            for m in labels_mask:
                m[:args.segment_size] = False

        input_ids = pad_sequence(input_ids, padding_value=id_pad_value, batch_first=True)
        labels = pad_sequence(labels, padding_value=-100, batch_first=True)
        attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
        labels_mask = pad_sequence(labels_mask, padding_value=0, batch_first=True)
        # logger.info(f"\n\n\n\n{input_ids.shape}, \n\n {tokenizer.decode(input_ids[0])}\n\n\n\n")
        collated = {'input_ids': input_ids,
                    'labels': labels, 
                    'attention_mask': attention_mask,
                    'labels_mask': labels_mask.bool()
                    }

        # print(len(collated['input_ids']), len(collated['input_ids'][0]), (collated['input_ids'][0] != -100).sum())
        return collated

    def filter_by_len(sample, min_len=16000):
        return len(sample[args.train_tokens]) > min_len
    
    def filter_by_16k(sample):
        return len(sample[args.train_tokens]) > 16000
    
    # Check if we're using a chunked dataset
    is_chunked_train = (args.tokenized_dataset is not None and 
                        is_chunked_dataset(args.tokenized_dataset))
    
    if is_chunked_train:
        # For chunked datasets, skip filtering and use ChunkedDatasetIterator
        logger.info("Using chunked dataset - will load and process chunks sequentially")
        # Note: We'll create the ChunkedDatasetIterator below in the dataset prep section
        train_dataset = None  # Placeholder
    # else:
    #     # Normal filtering for non-chunked datasets
    #     if args.min_sample_len not in {16000, None}:
    #         train_dataset = dataset['train'].filter(lambda sample: filter_by_len(sample, args.min_sample_len))
    #     else:
    #         train_dataset = dataset['train'].filter(filter_by_16k)
    

    

    with training_args.main_process_first(desc="dataset prep"):
        n_cpus = max(os.cpu_count() - 1, 1)
        BATCH = 4096
        
        if is_chunked_train:
            # Create ChunkedDatasetIterator which handles loading, processing, and yielding
            logger.info(f"Creating ChunkedDatasetIterator for {args.tokenized_dataset}")
            train_dataset = ChunkedDatasetIterator(
                dataset_dir=args.tokenized_dataset,
                segment_size=segment_size,
                history_size=history_size,
                token_column=args.train_tokens,
                seed=args.seed,
                skip_first_n_samples=chunked_skip_samples
            )
            logger.info("ChunkedDatasetIterator created - chunks will be processed on-the-fly during training")
        elif not args.streaming:
            # Normal tokenized dataset processing
            train_dataset = train_dataset.select_columns([args.train_tokens]).map(lambda x: group_texts(x, segment_size, history_size,),
                                                            batched=True, batch_size=BATCH)
            # BUFFER = 1024
            train_dataset = train_dataset.shuffle(seed=args.seed)
        else:
            # Streaming: per-chunk pipeline → non-iterable HF Dataset → shuffle → tokenize (batched) → group_texts → shuffle
            raw_stream = train_dataset  # this is the remaining stream after skipping 2000 docs for eval

            def tokenize_batch(examples):
                ids = tokenizer.batch_encode_plus(examples['text'], return_tensors=None, add_special_tokens=True)['input_ids']
                return {args.train_tokens: ids}

            class StreamingChunkToWindows(IterableDataset):
                def __init__(self, raw_iterable, seg, hist, docs_per_chunk, seed=0):
                    self.raw = raw_iterable
                    self.seg = seg
                    self.hist = hist
                    self.docs_per_chunk = int(docs_per_chunk)
                    self.seed = seed

                def __iter__(self):
                    rng = random.Random(self.seed)
                    buf = []
                    pbar = tqdm(total=self.docs_per_chunk, desc="Filling chunk", leave=False)
                    for s in self.raw:
                        buf.append(s)
                        pbar.update(1)
                        if len(buf) >= self.docs_per_chunk:
                            pbar.set_description("Processing chunk")
                            pbar.refresh()
                            hf_ds = datasets.Dataset.from_list(buf)
                            hf_ds = hf_ds.shuffle(seed=rng.randrange(1 << 30))
                            tok_ds = hf_ds.map(tokenize_batch, batched=True, remove_columns=['text'], desc="Tokenizing chunk")
                            win_ds = tok_ds.select_columns([args.train_tokens]).map(
                                lambda x: group_texts(x, segment_size, history_size), batched=True, batch_size=BATCH, desc="Grouping chunk"
                            )
                            win_ds = win_ds.shuffle(seed=rng.randrange(1 << 30))
                            for ex in win_ds:
                                yield ex
                            del hf_ds, tok_ds, win_ds
                            gc.collect()
                            buf = []
                            pbar.close()
                            pbar = tqdm(total=self.docs_per_chunk, desc="Filling chunk", leave=False)
                    if buf:
                        pbar.set_description("Processing tail chunk")
                        pbar.refresh()
                        hf_ds = datasets.Dataset.from_list(buf)
                        hf_ds = hf_ds.shuffle(seed=rng.randrange(1 << 30))
                        tok_ds = hf_ds.map(tokenize_batch, batched=True, remove_columns=['text'])
                        win_ds = tok_ds.select_columns([args.train_tokens]).map(
                            lambda x: group_texts(x, segment_size, history_size), batched=True, batch_size=BATCH
                        )
                        win_ds = win_ds.shuffle(seed=rng.randrange(1 << 30))
                        for ex in win_ds:
                            yield ex
                        del hf_ds, tok_ds, win_ds
                        gc.collect()
                    pbar.close()

            _rank, _world = _get_rank_world_size()
            _total_docs_cfg = getattr(args, 'stream_chunk_docs', 1_000_000)
            _docs_per_rank = max(1000, (_total_docs_cfg + _world - 1) // _world)  # ceil divide with floor cap
            train_dataset = StreamingChunkToWindows(
                raw_iterable=raw_stream,
                seg=segment_size,
                hist=history_size,
                docs_per_chunk=_docs_per_rank,
                seed=(args.seed + int(_rank)),
            )
        # Convert validation/test to standard Datasets before grouping (important for group_texts)
        val_base = validation_dataset["validation"].select_columns([args.train_tokens])
        test_base = validation_dataset["test"].select_columns([args.train_tokens])
        if isinstance(val_base, IterableDataset):
            val_list = [ex for ex in val_base]
            val_base = datasets.Dataset.from_list(val_list)
        if isinstance(test_base, IterableDataset):
            test_list = [ex for ex in test_base]
            test_base = datasets.Dataset.from_list(test_list)

        valid_dataset = val_base.map(
            lambda x: group_texts(x, segment_size, val_history_size),
            batched=True,
            batch_size=BATCH,
        )
        test_dataset = test_base.map(
            lambda x: group_texts(x, segment_size, val_history_size),
            batched=True,
            batch_size=BATCH,
        )

    
    num_valid_examples = 1000
    if args.streaming or isinstance(valid_dataset, IterableDataset):
        # For streaming/iterable datasets, just take the first N examples
        valid_dataset = valid_dataset.take(min(num_valid_examples, len(valid_dataset)))
    else:
        # For regular datasets, sample evenly across the dataset
        valid_inds = np.linspace(1, len(valid_dataset)-1, num_valid_examples).astype(int).tolist()
        valid_dataset = valid_dataset.select(valid_inds)

    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    # Log expected training statistics
    logger.info("="*80)
    logger.info("TRAINING DATASET STATISTICS")
    logger.info("="*80)
    logger.info(f"Dataset: {args.task_name if args.task_name else args.tokenized_dataset}")
    if is_chunked_train:
        logger.info(f"Dataset format: CHUNKED (sequential chunk loading with group_texts processing)")
    elif args.streaming:
        logger.info(f"Dataset format: STREAMING")
    else:
        logger.info(f"Dataset format: REGULAR")
    logger.info(f"Segment size: {segment_size}")
    logger.info(f"History size: {history_size}")
    logger.info(f"Window size: {segment_size + history_size}")
    logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"Number of devices: {training_args.world_size if hasattr(training_args, 'world_size') else 'unknown'}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * (training_args.world_size if hasattr(training_args, 'world_size') else 1)}")
    logger.info(f"Max training steps: {training_args.max_steps}")
    
    # Calculate expected tokens
    expected_windows = training_args.max_steps * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * (training_args.world_size if hasattr(training_args, 'world_size') else 1)
    expected_tokens = expected_windows * segment_size  # Each window processes segment_size new tokens
    logger.info(f"Expected windows to process: {expected_windows:,}")
    logger.info(f"Expected tokens to process: {expected_tokens:,} ({expected_tokens/1e9:.2f}B)")
    
    # Dataset-specific info
    if is_chunked_train:
        logger.info(f"NOTE: Chunks will be loaded sequentially, each processed with group_texts, then trained on until exhausted")
    if args.task_name and 'fineweb' in args.task_name.lower():
        logger.info(f"NOTE: FineWeb-Edu contains ~1.3 trillion tokens across ~billions of documents")
        logger.info(f"      You will process approximately {100 * expected_tokens / 1.3e12:.4f}% of the full dataset")
    logger.info("="*80)

    # define model
    model_cls = get_cls_by_name(args.model_cls)

    logger.info(f'Using model class: {model_cls}')

    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model = model_cls(config=model_cfg)
    else:
        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained, attn_implementation=args.attn_implementation,)
    # try:
    #     model.parallelize()
    # except Exception as e:
    #     logger.error(f'Error in parallelize: {e}')

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_attn_dim, 
            lora_alpha=args.lora_attn_alpha, 
            lora_dropout=args.lora_dropout
            )
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()
    

    ## load cpt of backbone model
    if args.backbone_cpt:
        cpt = torch.load(args.backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=False)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Use HF-compatible ARMT instead of original RMT classes
    if args.num_mem_tokens is not None:
        from modeling_amt.model import ARMTConfig, ARMTForCausalLM
        if args.armt_impl == 'inner':
            from modeling_amt.inner_loop import InnerLoopARMTForCausalLM

        logger.info(f'Creating HF-compatible ARMT model (impl={args.armt_impl})')

        # Optionally compute alternating wrap pattern for layers
        wrap_layers_arg = None
        if args.alternate_layers:
            try:
                # Prefer counting layers via the actual model's layers container
                layers_attr_path = args.layers_attr if args.layers_attr is not None else "model.layers"
                container = model
                for attr in layers_attr_path.split('.'):
                    container = getattr(container, attr)
                n_layers = len(container)
            except Exception:
                # Fallback to common config attributes
                cfg = getattr(model, 'config', None)
                n_layers = None
                for field in ("num_hidden_layers", "n_layer", "n_layers"):
                    val = getattr(cfg, field, None) if cfg is not None else None
                    if val is not None:
                        n_layers = int(val)
                        break
                if n_layers is None:
                    n_layers = 0
            # Start with 1 to wrap the first layer, then alternate 1,0,1,0,...
            wrap_layers_arg = [1 if (i % 2 == 0) else 0 for i in range(n_layers)]

        # Create ARMT config
        armt_config = ARMTConfig(
            base_model_name=args.from_pretrained,
            num_mem_tokens=args.num_mem_tokens,
            d_mem=args.d_mem if args.d_mem is not None else 512,
            segment_size=segment_size,
            segment_alignment="left",
            sliding_window=args.prev_seg_kv,
            attend_to_previous_input=args.attend_to_previous_input,
            use_sink=args.use_sink,
            layers_attr=args.layers_attr if args.layers_attr is not None else "model.layers",
            wrap_layers=wrap_layers_arg,
            wrap_pos=False,
            correction=True,
            n_heads=1,
            use_denom=True,
            gating=False,
            freeze_mem=args.freeze_model_weights,
            act_on=False,
            max_hop=4,
            act_type="associative",
            time_penalty=0.0
        )

        # Create ARMT model (outer vs inner loop)
        if args.armt_impl == 'inner':
            armt_model_cls = InnerLoopARMTForCausalLM
        elif args.armt_impl == 'mem_params':
            from modeling_amt.armt_memory_params import MemoryParamsARMTForCausalLM
            armt_model_cls = MemoryParamsARMTForCausalLM
        else:
            armt_model_cls = ARMTForCausalLM

        ## load cpt of ARMT
        if args.model_cpt and args.model_cpt != 'None':
            logger.info(f'Loading ARMT checkpoint from: {args.model_cpt}')
            model = armt_model_cls.from_pretrained(args.model_cpt, config=armt_config)
            logger.info(f'Loaded HF-compatible ARMT model from checkpoint (impl={args.armt_impl})')
        else:
            model = armt_model_cls(config=armt_config)
            logger.info(f'Created HF-compatible ARMT model (impl={args.armt_impl})')


    
    # args.gradient_checkpointing = True
    print("="*20, training_args.deepspeed, "="*20)

    training_args.bf16 = True
    training_args.fp16 = False
    training_args.ddp_find_unused_parameters = False
    
    # Custom callback to log dataset consumption statistics
    class DatasetStatsCallback(TrainerCallback):
        def __init__(self, train_dataset, expected_tokens):
            self.train_dataset = train_dataset
            self.expected_tokens = expected_tokens
            
        def on_train_end(self, args, state, control, **kwargs):
            if hasattr(self.train_dataset, 'stats'):
                stats = self.train_dataset.stats
                logger.info("="*80)
                logger.info("ACTUAL TRAINING DATASET CONSUMPTION")
                logger.info("="*80)
                logger.info(f"Raw samples consumed: {stats['raw_samples_consumed']:,}")
                logger.info(f"Total tokens consumed: {stats['total_tokens_consumed']:,} ({stats['total_tokens_consumed']/1e9:.2f}B)")
                logger.info(f"Windows yielded: {stats['windows_yielded']:,}")
                logger.info(f"Expected tokens: {self.expected_tokens:,} ({self.expected_tokens/1e9:.2f}B)")
                logger.info(f"Actual vs Expected: {100 * stats['total_tokens_consumed'] / self.expected_tokens:.2f}%")
                
                if 'fineweb' in args.task_name.lower() if hasattr(args, 'task_name') else False:
                    logger.info(f"Fraction of FineWeb-Edu (1.3T tokens): {100 * stats['total_tokens_consumed'] / 1.3e12:.4f}%")
                logger.info("="*80)
    
    # Import DeepSpeed callback for handling ZeRO-3 checkpoint consolidation
    from deepspeed_push_callback import DeepSpeedCheckpointCallback
    
    # Determine the model class name for Hub pushing
    model_class_name = "InnerLoopARMTForCausalLM" if args.armt_impl == 'inner' else "ARMTForCausalLM"
    
    # Create callbacks
    dataset_stats_callback = DatasetStatsCallback(train_dataset, expected_tokens)
    deepspeed_callback = DeepSpeedCheckpointCallback(
        consolidate_on_save=training_args.push_to_hub,
        modeling_code_dir=os.path.join(args.working_dir, "modeling_amt"),
        model_class_name=model_class_name if args.num_mem_tokens is not None else None
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # test_dataset=test_dataset,
        # compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[dataset_stats_callback, deepspeed_callback],
    )


    # if training_args.deepspeed:
    #     trainer._setup_deepspeed()  # private but safe in HF; triggers DS engine build
    #     print("is_deepspeed_enabled:", trainer.is_deepspeed_enabled)
    #     print("wrapped type:", type(trainer.model_wrapped))
    
    # model, train_dataset, valid_dataset = trainer.accelerator.prepare(model, train_dataset, valid_dataset)
    print("Trainer Gradient Checkpointing Enabled:", trainer.args.gradient_checkpointing)
    
    # trainer.evaluate()
    if not args.validate_only:
        trainer.train(resume_from_checkpoint=args.checkpoint) 