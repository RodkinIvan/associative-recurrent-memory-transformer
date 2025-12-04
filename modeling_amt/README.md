# ARMT implementations in `modeling_amt/`

This folder contains several Hugging Face compatible variants of the Associative Recurrent Memory Transformer (ARMT). They share the same high level API (a `PretrainedConfig` plus a `PreTrainedModel` subclass exposing `forward` and `generate`) but differ in how the associative memory is threaded through the network.

## Common usage
- Every variant expects either `base_model_name` (loads a pretrained HF causal LM) or `base_model_config` (a config dict/instance) plus ARMT-specific hyperparameters such as `num_mem_tokens`, `d_mem`, `segment_size`, `sliding_window`, and `use_sink`.
- Standard construction pattern:
  ```python
  from modeling_amt.model import ARMTConfig, ARMTForCausalLM  # swap module for other variants

  cfg = ARMTConfig(
      base_model_name="meta-llama/Llama-3-1b",
      num_mem_tokens=16,
      d_mem=512,
      segment_size=512,
      sliding_window=True,
  )
  model = ARMTForCausalLM(cfg)
  out = model(input_ids, attention_mask=attn_mask)
  ```
- All models return Hugging Face-compatible outputs; `generate` is implemented where possible. Helpers such as `zero_mem()` and `detach_mem()` reset or detach the associative state across batches.
- Sliding-window caching is implemented for `model.py` and `inner_loop.py`. Other variants ignore `sliding_window` even if set.

## Which ARMT should I use?
- **Paper baseline (`model.py`)** – Outer-loop driver that segments the input before the base model. Use when you want the exact implementation from the paper with minimal additional controls.
- **Inner-loop (`inner_loop.py`)** – Same math moved inside each transformer block. Choose between:
  - Horizontal mode (default): segments inside each layer; good for training and model parallelism.
  - Vertical mode (`model.vertical_mode = True`): feeds pre-augmented segments through the base model; preferred for long-form inference because memory persists across segments with less Python overhead.
- **Disjoint memory (`disjoint_memory.py`)** – Inner-loop style, but memory tokens do not propagate between layers; each layer re-initializes its memory tokens. Useful for experiments where you want layer-local memories without memory tokens accumulating through depth.
- **Separate memory parameters (`armt_memory_params.py`)** – Memory tokens and sequence tokens are processed by different parameter sets. Each wrapper keeps a `mem_layer` (a deep copy of the base layer) that only touches memory tokens; optionally freeze the base model (`freeze_base_model=True`) and train just the memory pathway for language modeling–style fine-tuning.
- **Thinking controls (`thinking.py`)** – Adds knobs to increase read/write effort:
  - `reading_depth_multiplier`, `writing_depth_multiplier` scale the number of times layers process a segment when reading from or writing to memory.
  - `repeat_read_segments`, `repeat_write_segments` repeat segmentation passes.
  - Use to trade latency for better recall/precision on hard prompts by deepening associative updates without changing the base model.

## Implementation-specific snippets
- **Paper baseline (`model.py`, sliding window supported)**
  ```python
  import torch
  from modeling_amt.model import ARMTConfig, ARMTForCausalLM

  cfg = ARMTConfig(
      base_model_name="meta-llama/Llama-3-1b",
      segment_size=256,
      num_mem_tokens=16,
      sliding_window=True,  # effective here
  )
  model = ARMTForCausalLM(cfg)
  input_ids = torch.randint(0, model.config.vocab_size, (1, 256))
  attn_mask = torch.ones_like(input_ids)
  out = model(input_ids=input_ids, attention_mask=attn_mask)
  logits = out.logits
  ```
- **Inner-loop (`inner_loop.py`, sliding window supported)**
  ```python
  import torch
  from modeling_amt.inner_loop import ARMTConfig, InnerLoopARMTForCausalLM

  cfg = ARMTConfig(
      base_model_name="meta-llama/Llama-3-1b",
      segment_size=256,
      sliding_window=True,  # efficient caching in horizontal/vertical modes
  )
  model = InnerLoopARMTForCausalLM(cfg)
  model.vertical_mode = False  # set True for long-context inference
  input_ids = torch.randint(0, model.config.vocab_size, (1, 256))
  out = model(input_ids=input_ids)
  logits = out.logits
  ```
- **Disjoint memory (`disjoint_memory.py`, no sliding window)**
  ```python
  import torch
  from modeling_amt.inner_loop import ARMTConfig  # config schema reused
  from modeling_amt.disjoint_memory import InnerLoopARMTForCausalLM

  cfg = ARMTConfig(
      base_model_name="meta-llama/Llama-3-1b",
      segment_size=256,
      num_mem_tokens=32,
      sliding_window=False,
  )
  model = InnerLoopARMTForCausalLM(cfg)
  logits = model(input_ids=torch.randint(0, model.config.vocab_size, (1, 128))).logits
  ```
- **Separate memory parameters (`armt_memory_params.py`, no sliding window)**
  ```python
  import torch
  from modeling_amt.armt_memory_params import MemParamsARMTConfig, MemoryParamsARMTForCausalLM

  cfg = MemParamsARMTConfig(
      base_model_name="meta-llama/Llama-3-1b",
      segment_size=256,
      freeze_base_model=True,  # train only memory pathway
  )
  model = MemoryParamsARMTForCausalLM(cfg)
  logits = model(input_ids=torch.randint(0, model.config.vocab_size, (1, 128))).logits
  ```
- **Thinking controls (`thinking.py`, no sliding window)**
  ```python
  import torch
  from modeling_amt.thinking import ThinkingARMTConfig, ThinkingARMTForCausalLM

  cfg = ThinkingARMTConfig(
      base_model_name="meta-llama/Llama-3-1b",
      segment_size=256,
      reading_depth_multiplier=2,
      writing_depth_multiplier=3,
      repeat_read_segments=2,
      repeat_write_segments=1,
  )
  model = ThinkingARMTForCausalLM(cfg)
  logits = model(input_ids=torch.randint(0, model.config.vocab_size, (1, 128))).logits
  ```

## Push to the Hugging Face Hub and load back
- Use the trainer callback from `deepspeed_push_callback.py` to inline any ARMT implementation (outer/inner/mem-params/thinking) and push with `trust_remote_code=True` (same pattern as `run_finetuning_lm_rmt_hf_armt.py`):
  ```python
  from transformers import Trainer, TrainingArguments
  from deepspeed_push_callback import PushToHubCallback
  from modeling_amt.inner_loop import ARMTConfig, InnerLoopARMTForCausalLM  # swap module/class for other variants

  armt_impl = "inner"  # one of: "outer" (default), "inner", "mem_params", "thinking"
  model_class_by_impl = {
      "outer": "ARMTForCausalLM",
      "inner": "InnerLoopARMTForCausalLM",
      "mem_params": "MemoryParamsARMTForCausalLM",
      "thinking": "ThinkingARMTForCausalLM",
  }
  model_class = model_class_by_impl[armt_impl]
  cfg = ARMTConfig(base_model_name="meta-llama/Llama-3-1b")
  model = InnerLoopARMTForCausalLM(cfg)

  training_args = TrainingArguments(
      output_dir="checkpoints/armt",
      push_to_hub=True,
      hub_strategy="every_save",
      hub_model_id="your-username/armt-llama-3-1b-inner",
      save_steps=1000,
      logging_steps=100,
  )

  callbacks = [PushToHubCallback(modeling_code_dir="modeling_amt", model_class_name=model_class)]
  trainer = Trainer(model=model, args=training_args, train_dataset=..., eval_dataset=..., callbacks=callbacks)
  trainer.train()
  # The callback inlines all ARMT code (model.py, inner_loop.py, armt_memory_params.py, thinking.py, utils/act_utils/language_modeling)
  # and updates config.auto_map so Hub users can load with trust_remote_code=True.
  ```
- Load directly from the Hub with remote code enabled:
  ```python
  from transformers import AutoModelForCausalLM

  repo_id = "your-username/armt-llama-3-1b"
  model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
  ```

## Notes and tips
- For sliding-window decoding, only `model.py` and `inner_loop.py` honor `sliding_window=True`; other variants fall back to full-sequence attention.
- `use_sink=True` prepends a learned sink token per segment; memory tokens always trail each segment.
- `wrap_layers` (where available) lets you skip wrapping selected base layers (e.g., to leave embeddings/head untouched).
- When loading checkpoints, variants try the base HF loader first and fall back to their own ARMT-specific layout, so you can reuse standard HF saving/loading flows.
