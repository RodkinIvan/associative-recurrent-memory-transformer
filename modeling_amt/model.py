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
                 base_model_name="HuggingFaceTB/SmolLM2-135M",
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
                 time_penalty=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
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
        
        # base_config = AutoConfig.from_pretrained(config.base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)

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
        from transformers.utils.hub import cached_file, HfHubHTTPError
        import torch

        if config is None:
            config = ARMTConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model = cls(config)

        state_dict = None
        try:
            weights_path = cached_file(pretrained_model_name_or_path, "model.safetensors", **kwargs)
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device="cpu")
        except (OSError, HfHubHTTPError):
            try:
                weights_path = cached_file(pretrained_model_name_or_path, "pytorch_model.bin", **kwargs)
                state_dict = torch.load(weights_path, map_location="cpu")
            except (OSError, HfHubHTTPError):
                print(f"Warning: Could not find weights for {pretrained_model_name_or_path}. "
                      f"The model is initialized randomly.")

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)

        return model

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.armt.gradient_checkpointing_enable(*args, **kwargs) 