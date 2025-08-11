import math
import torch
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.cache_utils import Cache, DynamicCache
from torch.nn.functional import relu as r
import torch.nn.functional as F
import os

from modeling_amt.language_modeling import (
    AssociativeMemoryCell, 
    AssociativeRecurrentWrapper,
    attn_mask_to_4d,
    invert_attn_mask
)


class ARMTConfig(PretrainedConfig):
    model_type = "armt"

    def __init__(self,
                 base_model_name=None,
                 base_model_config=None,
                 num_mem_tokens=16,
                 d_mem=512,

                 segment_size=512,
                 segment_alignment="left",
                 sliding_window=False,
                 attend_to_previous_input=False,
                 use_sink=False,
                 layers_attr="model.layers",
                 wrap_pos=False,
                 correction=True,
                 n_heads=1,
                 use_denom=True,
                 gating=False,
                 freeze_mem=False,
                 act_on=False,
                 max_hop=4,
                 act_type="associative",
                 act_format="linear",
                 noisy_halting=False,
                 constant_depth=False,
                 time_penalty=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        # Validate mutual exclusivity
        if (base_model_name is not None) and (base_model_config is not None):
            raise ValueError("Exactly one of `base_model_name` or `base_model_config` must be provided. Set the other to None.")
        self.base_model_name = base_model_name
        # Optional alternative to base_model_name: a config (dict/PretrainedConfig/name-or-path)
        self.base_model_config = base_model_config
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem

        self.segment_size = segment_size
        self.segment_alignment = segment_alignment
        self.sliding_window = sliding_window
        self.attend_to_previous_input = attend_to_previous_input
        self.use_sink = use_sink
        self.layers_attr = layers_attr
        self.wrap_pos = wrap_pos
        self.correction = correction
        self.n_heads = n_heads
        self.use_denom = use_denom
        self.gating = gating
        self.freeze_mem = freeze_mem
        self.act_on = act_on
        self.max_hop = max_hop
        self.act_type = act_type
        self.act_format = act_format
        self.noisy_halting = noisy_halting
        self.constant_depth = constant_depth
        self.time_penalty = time_penalty

    def get(self, attr: str, default=None):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return default


class ARMTForCausalLM(PreTrainedModel):
    config_class = ARMTConfig

    def __init__(self, config: ARMTConfig, **kwargs):
        super().__init__(config, **kwargs)
        from transformers import AutoConfig, AutoModelForCausalLM
        
        # Build base model either from name (pretrained weights) or from provided config
        base_model = None
        if getattr(config, 'base_model_name', None) is not None and getattr(config, 'base_model_config', None) is not None:
            raise ValueError("Exactly one of `base_model_name` or `base_model_config` must be provided in ARMTConfig.")
        bm_cfg = getattr(config, 'base_model_config', None)
        if bm_cfg is not None:
            # Prefer explicit config when provided
            if isinstance(bm_cfg, PretrainedConfig) and getattr(bm_cfg, 'model_type', None) != ARMTConfig.model_type:
                resolved_cfg = bm_cfg
            elif isinstance(bm_cfg, dict):
                if 'model_type' not in bm_cfg:
                    raise ValueError("`base_model_config` dict must include a 'model_type' key (e.g., 'gpt_neox', 'llama').")
                config_cls_or_instance = AutoConfig.for_model(bm_cfg['model_type'])
                # If an instance was returned, update it; if a class was returned, construct from dict
                if isinstance(config_cls_or_instance, PretrainedConfig):
                    resolved_cfg = config_cls_or_instance
                    for k, v in bm_cfg.items():
                        setattr(resolved_cfg, k, v)
                else:
                    resolved_cfg = config_cls_or_instance.from_dict(bm_cfg)
            elif isinstance(bm_cfg, str):
                # Treat as a name or path to load a config
                resolved_cfg = AutoConfig.from_pretrained(bm_cfg)
            else:
                raise TypeError("`base_model_config` must be a transformers.PretrainedConfig, dict, or str (name/path)")
            base_model = AutoModelForCausalLM.from_config(resolved_cfg)
        elif getattr(config, 'base_model_name', None):
            base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        else:
            raise ValueError("ARMTForCausalLM requires either `base_model_config` or `base_model_name` in ARMTConfig.")

        self.armt_config = config
        
        # Create the associative memory cell
        memory_cell = AssociativeMemoryCell(
            base_model=base_model,
            num_mem_tokens=config.num_mem_tokens,
            d_mem=config.d_mem,
            layers_attr=config.layers_attr,
            wrap_pos=config.wrap_pos,
            correction=config.correction,
            n_heads=config.n_heads,
            use_denom=config.use_denom,
            gating=config.gating,
            freeze_mem=config.freeze_mem,
            act_on=config.act_on,
            max_hop=config.max_hop,
            act_type=config.act_type,
            # Optional extras
            constant_depth=config.get('constant_depth', False),
            act_format=config.get('act_format', 'linear'),
            noisy_halting=config.get('noisy_halting', False),
            attend_to_previous_input=config.attend_to_previous_input,
            use_sink=config.use_sink
        )
        
        # Create the associative recurrent wrapper
        self.armt = AssociativeRecurrentWrapper(
            memory_cell,
            segment_size=config.segment_size,
            segment_alignment=config.segment_alignment,
            sliding_window=config.sliding_window,
            attend_to_previous_input=config.attend_to_previous_input,
            act_on=config.act_on,
            time_penalty=config.time_penalty
        )

    def forward(
        self,
        input_ids=None,
        labels=None,
        labels_mask=None,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        input_segmented=False,
        output_only_last_segment=False,
        num_items_in_batch=None,
    ):
        return self.armt(
            input_ids=input_ids,
            labels=labels,
            labels_mask=labels_mask,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            input_segmented=input_segmented,
            output_only_last_segment=output_only_last_segment,
            num_items_in_batch=num_items_in_batch,
        )

    def generate(self, *args, **kwargs):
        return self.armt.generate(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            return super().load_state_dict(state_dict, strict, assign)
        except RuntimeError:
            print("Failed to load state, retrying with ARMT loader.")
            self.armt.load_state_dict(state_dict, strict=True, assign=assign)
            print("Success!")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, *args, **kwargs):
        # Delegate to the base class to benefit from full shard/format support
        return super().from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.armt.gradient_checkpointing_enable(*args, **kwargs) 