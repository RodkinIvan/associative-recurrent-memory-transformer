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
parser.add_argument('--armt_impl', type=str, choices=['outer', 'inner'], default='outer',
                    help='ARMT implementation: outer (AssociativeRecurrentWrapper) or inner (per-layer inner-loop)')
parser.add_argument('--streaming', action='store_true', default=False, help='use streaming dataset')
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

    with training_args.main_process_first(desc="dataset prep"):
        if args.tokenized_dataset is not None:
            dataset = datasets.load_from_disk(args.tokenized_dataset)
            validation_dataset = datasets.load_from_disk(args.valid_tokenized_dataset)
            logger.info("Tokenized Dataset loaded")
            if args.valid_tokens != args.train_tokens:
                validation_dataset = validation_dataset.rename_column(args.valid_tokens, args.train_tokens)
        else:
            # Load dataset with streaming=True to load samples on the fly
            train_dataset = datasets.load_dataset(args.task_name, split='train', streaming=args.streaming, trust_remote_code=True)
            if args.valid_task_name is not None:
                validation_dataset = datasets.load_dataset(args.valid_task_name, split='validation', trust_remote_code=True)
                test_dataset = datasets.load_dataset(args.valid_task_name, split='test', trust_remote_code=True)
            else:
                # Take the first 1000 samples from train dataset for validation and test
                if args.streaming:
                    # For streaming datasets, use take() and skip()
                    validation_dataset = train_dataset.take(1000)
                    test_dataset = train_dataset.skip(1000).take(1000)
                    train_dataset = train_dataset.skip(2000)
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
            
            # Apply tokenization on the fly
            train_dataset = train_dataset.map(
                tokenize_function,
                batched=False,
                # batch_size=256,
                remove_columns=['text'],
            )
            validation_dataset = validation_dataset.map(
                tokenize_function,
                batched=False,
                remove_columns=['text'],
                # desc="Tokenizing eval split",
                # num_proc=1,
            )
            test_dataset = test_dataset.map(
                tokenize_function,
                batched=False,
                remove_columns=['text'],
                # desc="Tokenizing test split",
                # num_proc=1,
            )
            
            # Create a DatasetDict with the processed splits
            # nodes = torch.cuda.device_count()
            # train_dataset = split_dataset_by_node(train_dataset, world_size=nodes, rank=0)
            # validation_dataset = split_dataset_by_node(validation_dataset, world_size=nodes, rank=0)
            # test_dataset = split_dataset_by_node(test_dataset, world_size=nodes, rank=0)

            dataset = datasets.DatasetDict({
                'train': train_dataset.with_format("torch"),
                'validation': validation_dataset.with_format("torch"),
                'test': test_dataset.with_format("torch")
            })
            validation_dataset = dataset
            # validation_dataset = datasets.load_from_disk('/mnt/data/users/ivan.rodkin/lab/datasets/pg19_tokenized')


    segment_size = args.segment_size
    history_size = args.sample_size - segment_size

    if args.val_sample_size is not None:
        val_history_size = args.val_sample_size - segment_size
    else:
        val_history_size = history_size


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
            starts  = np.arange(history_size, len(flat) - segment_size + 1, segment_size, dtype=np.int32)
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
    
    if args.min_sample_len not in {16000, None}:
        train_dataset = dataset['train'].filter(lambda sample: filter_by_len(sample, args.min_sample_len))
    else:
        train_dataset = dataset['train'].filter(filter_by_16k)
    

    

    with training_args.main_process_first(desc="dataset prep"):
        n_cpus = max(os.cpu_count() - 1, 1)
        BATCH = 1024
        if not args.streaming:
            
            train_dataset = train_dataset.select_columns([args.train_tokens]).map(lambda x: group_texts(x, segment_size, history_size,),
                                                            batched=True, batch_size=BATCH)
            # BUFFER = 1024
            train_dataset = train_dataset.shuffle(seed=args.seed)
        else:
            # Estimate number of tokens to consume per epoch and derive number of windows
            tokens_per_chunk = 50_000_000  # adjust this estimate as needed
            # tokens_per_chunk = 1_000_000
            # Use a buffer at least as large as the number of windows for effective shuffling
            # BUFFER = 2048
            # if args.streaming:
            #     train_dataset = train_dataset.shuffle(buffer_size=BUFFER, seed=args.seed)
            # else:
            #     train_dataset = train_dataset.shuffle(seed=args.seed)
            # Wrap the raw stream in windowed iterable and shuffle windows
            # length = 5_451_448
            # train_dataset = ChunkedWindowStream(train_dataset, segment_size, history_size, tokens_per_chunk, length, args.seed)
            train_dataset = OnlineWindowStream(
                raw_ds=train_dataset, 
                segment_size=segment_size, 
                history_size=history_size, 
                chunk_tokens=tokens_per_chunk, 
                seed=args.seed
            )
        valid_dataset = validation_dataset["validation"].select_columns([args.train_tokens]).map(
            lambda x: group_texts(x, segment_size, val_history_size ), 
            batched=True, 
            # batch_size=BATCH, 
            # desc=f"Grouping valid in chunks of {segment_size} and history {val_history_size}", 
            # num_proc=n_cpus
        )
        test_dataset = validation_dataset["test"].select_columns([args.train_tokens]).map(
            lambda x: group_texts(x, segment_size, val_history_size), 
            batched=True, 
            # batch_size=BATCH, 
            # desc=f"Grouping test in chunks of {segment_size} and history {val_history_size}", 
            # num_proc=n_cpus
        )

    
    num_valid_examples = 1000
    if args.streaming or isinstance(valid_dataset, IterableDataset):
        # For streaming/iterable datasets, just take the first N examples
        valid_dataset = valid_dataset.take(num_valid_examples)
    else:
        # For regular datasets, sample evenly across the dataset
        valid_inds = np.linspace(1, len(valid_dataset)-1, num_valid_examples).astype(int).tolist()
        valid_dataset = valid_dataset.select(valid_inds)

    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    # Log expected training statistics
    logger.info("="*80)
    logger.info("TRAINING DATASET STATISTICS")
    logger.info("="*80)
    logger.info(f"Dataset: {args.task_name}")
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
    
    # FineWeb-Edu info
    if 'fineweb' in args.task_name.lower():
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
        armt_model_cls = InnerLoopARMTForCausalLM if args.armt_impl == 'inner' else ARMTForCausalLM

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
    
    # Create callback with expected tokens
    dataset_stats_callback = DatasetStatsCallback(train_dataset, expected_tokens)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # test_dataset=test_dataset,
        # compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[dataset_stats_callback],
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