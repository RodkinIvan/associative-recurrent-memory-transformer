import math
import os
from typing import Optional, Tuple, Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

# Reuse utilities from the existing implementation to ensure identical math
from modeling_amt.language_modeling import DPFP, invert_attn_mask, attn_mask_to_4d


class InnerLoopAssociativeLayerWrapper(nn.Module):
    """
    A per-layer wrapper that performs associative read/write within the layer by
    splitting the incoming full sequence into fixed-size segments on the fly.

    Unlike the outer-loop design (which segments inputs before the model), this
    module receives the full, unsplit hidden sequence and internally iterates
    over segments:
      1) Optional associative READ is applied to the segment's hidden states
         based on the current associative memory (W_mem, z).
      2) Memory tokens are appended to the segment and the underlying transformer
         layer is executed only on this augmented segment.
      3) The resulting memory token outputs are used to WRITE/update the
         associative memory.
      4) The transformed real-token outputs replace the corresponding slice in
         the layer output for the full sequence.

    This preserves identical behavior w.r.t. memory math while avoiding any
    outer recurrent wrapper.
    """

    def __init__(
        self,
        layer: nn.Module,
        d_model: int,
        num_mem_tokens: int,
        d_mem: int,
        segment_size: int,
        n_heads: int = 1,
        correction: bool = True,
        use_denom: bool = True,
        gating: bool = False,
        use_sink: bool = False,
        sliding_window: bool = False,
        get_memory_fn: Optional[Callable[[], torch.Tensor]] = None,
        get_sink_fn: Optional[Callable[[], Optional[torch.Tensor]]] = None,
        rotary_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        read_prev_states_fn: Optional[Callable[[int, int, torch.device, torch.dtype], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
        write_states_fn: Optional[Callable[[int, torch.Tensor, Optional[torch.Tensor]], None]] = None,
        info: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.info = info
        self.layer = layer
        self.d_model = d_model
        self.num_mem_tokens = int(num_mem_tokens or 0)
        self.d_mem = d_mem
        self.segment_size = int(segment_size)
        self.n_heads = n_heads
        self.gating = gating
        self.use_denom = use_denom
        self.correction = correction
        self.use_sink = bool(use_sink)
        self.sliding_window = bool(sliding_window)

        # DPFP feature map dimensions
        nu = 3
        self.d_key = 2 * nu * d_mem

        assert self.d_mem % n_heads == 0 and self.d_model % n_heads == 0

        # Match the dtype to the wrapped layer
        layer_dtype = next(self.layer.parameters()).dtype

        # Readout/query/key/value projections for associative memory
        self.W_mq = nn.Linear(d_model, d_mem, bias=False, dtype=layer_dtype)
        self.W_mk = nn.Linear(d_model, d_mem, bias=False, dtype=layer_dtype)
        self.W_mv = nn.Linear(d_model, d_model, bias=False, dtype=layer_dtype)
        if gating:
            self.W_mb = nn.Linear(d_model, d_model, dtype=layer_dtype)
        else:
            self.W_mb = nn.Linear(d_model, n_heads, dtype=layer_dtype)
        torch.nn.init.zeros_(self.W_mv.weight)

        self.phi = DPFP(nu)

        # Associative memory state (not parameters, not trained by optimizer)
        # self.register_buffer(
        #     "W_mem",
        #     torch.zeros(1, n_heads, self.d_key // n_heads, d_model // n_heads, dtype=layer_dtype),
        #     persistent=False,
        # )
        # if self.use_denom:
        #     self.register_buffer(
        #         "z",
        #         torch.zeros(1, n_heads, self.d_key // n_heads, dtype=layer_dtype),
        #         persistent=False,
        #     )

        # Runtime flags/counters
        self.generate_mode = False
        self.seg_num = 0

        # Lightweight accessors to shared trainable memory tensors owned by the top-level model.
        # These are callables, not Modules/Parameters stored as attributes, to avoid submodule cycles.
        self._get_memory = get_memory_fn
        self._get_sink = get_sink_fn
        self._rotary_fn = rotary_fn
        self._read_prev_states = read_prev_states_fn
        self._write_states = write_states_fn

    # ----- helpers for heads reshaping -----
    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        x = x.reshape(bsz, seq_len, self.n_heads, d_model // self.n_heads)
        x = x.permute(0, 2, 1, 3)
        return x

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, seq_len, d_head = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * d_head)
        return x

    # ----- associative read -----
    def associate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("associate() is unused in inner-loop; uses local memory helpers instead")

    # ----- associative write -----
    def update_mem(self, mem_tokens: torch.Tensor) -> None:
        raise NotImplementedError("update_mem() is unused in inner-loop; uses local memory helpers instead")

    # ----- memory state management -----
    def zero_mem(self) -> None:
        return None

    def detach_mem(self) -> None:
        return None

    def freeze_mem(self) -> None:
        self.W_mb.weight.requires_grad = False
        self.W_mb.bias.requires_grad = False
        self.W_mq.weight.requires_grad = False
        self.W_mk.weight.requires_grad = False
        self.W_mv.weight.requires_grad = False

    # ----- utilities -----
    def _get_segment_positions(
        self, position_ids: Optional[torch.LongTensor], start: int, end: int, device: torch.device
    ) -> torch.LongTensor:
        # If original absolute positions are provided, slice and extend for sink/memory
        if position_ids is not None:
            seg_pos = position_ids[:, start:end]
        else:
            # Fallback: local positions starting at offset (1 if sink) for this segment
            seg_len = end - start
            offset = int(self.use_sink)
            seg_pos = (
                torch.arange(offset, offset + seg_len, device=device)
                .long()
                .unsqueeze(0)
                .expand(-1, seg_len)
            )

        if self.num_mem_tokens == 0 and not self.use_sink:
            return seg_pos

        last_pos = seg_pos[:, -1:] if seg_pos.size(1) > 0 else torch.zeros_like(seg_pos[:, :1])
        # Memory tokens continue after the last real token positions
        mem_pos = last_pos + torch.arange(1, self.num_mem_tokens + 1, device=device).long().unsqueeze(0)

        if self.use_sink:
            sink_pos = torch.zeros_like(seg_pos[:, :1])
            pos = torch.cat([sink_pos, seg_pos, mem_pos], dim=1)
        else:
            pos = torch.cat([seg_pos, mem_pos], dim=1)
        return pos

    def _build_segment_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # attention_mask: (bsz, seg_len) for real tokens only
        bsz, seg_len = attention_mask.shape
        extra = self.num_mem_tokens + int(self.use_sink)
        total = seg_len + extra
        # Base causal mask over real tokens, expanded to total query length
        base4d = attn_mask_to_4d(attention_mask.to(dtype), upper=False, query_len=total)  # (b,1,total,seg_len)
        mask = torch.ones(bsz, 1, total, total, dtype=dtype, device=device)
        # Place real-token columns starting after sink (if any)
        start_col = int(self.use_sink)
        mask[:, :, :, start_col:start_col + seg_len] = base4d
        # Sink cannot attend to others
        if self.use_sink:
            mask[:, :, 0, 1:] = 0
        # Real tokens cannot attend to memory tokens
        if self.num_mem_tokens > 0:
            mask[:, :, : total - self.num_mem_tokens, total - self.num_mem_tokens :] = 0
        # Invert to additive mask as expected by attention: allowed->0, masked->-inf
        mask = invert_attn_mask(mask, dtype)
        return mask

    def pad_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        shape = list(attention_mask.shape)
        if len(shape) == 4:
            # For 4D masks (from attn_mask_to_4d), only extend the KV dimension (dim=-1)
            # The query dimension (dim=-2) should remain unchanged
            shape[-1] += self.num_mem_tokens + int(self.use_sink)
            mask = torch.ones(*shape, dtype=dtype).to(attention_mask.device)
            mask[..., int(self.use_sink):-self.num_mem_tokens] = attention_mask
            if self.use_sink:
                mask[..., 0, 1:] = 0
            mask[..., :-self.num_mem_tokens, -self.num_mem_tokens:] = 0
            if not os.environ.get("NOT_INVERT_ATTN_MASK"):
                mask = invert_attn_mask(mask, dtype)
        else:
            shape[-1] += self.num_mem_tokens + int(self.use_sink)
            mask = torch.ones(*shape, dtype=dtype).to(attention_mask.device)
            mask[..., int(self.use_sink):-self.num_mem_tokens] = attention_mask
        return mask.to(dtype)

    def pad_prev_seg_attn_mask(self, prev_seg_attn_mask: torch.Tensor, dtype: torch.dtype):
        if self.num_mem_tokens in {0, None}:
            return prev_seg_attn_mask
        # For previous segment masks, we don't need to extend the KV dimension
        # since prev_aug_len already includes the sink and memory tokens
        # Just return the mask as-is, but ensure it has the right dtype
        return prev_seg_attn_mask.to(dtype)

    def _build_sliding_window_mask(self, prev4d: torch.Tensor, cur4d: torch.Tensor, prev_aug_len: int, cur_aug_len: int) -> torch.Tensor:
        """
        Build a sliding window attention mask that prevents target leak.
        
        Args:
            prev4d: Previous segments mask (bsz, 1, query_len, prev_kv_len)
            cur4d: Current segment mask (bsz, 1, query_len, cur_kv_len)
            prev_aug_len: Length of previous augmented segments
            cur_aug_len: Length of current augmented segment
            
        Returns:
            Combined mask with proper isolation between memory and context tokens
        """
        bsz, _, query_len, _ = cur4d.shape
        device = cur4d.device
        dtype = cur4d.dtype
        
        # Calculate positions of memory tokens in previous and current segments
        prev_mem_start = prev_aug_len - self.num_mem_tokens if self.num_mem_tokens > 0 else prev_aug_len
        cur_mem_start = cur_aug_len - self.num_mem_tokens if self.num_mem_tokens > 0 else cur_aug_len
        
        # Create the combined mask: [prev_segments, current_segment]
        total_kv_len = prev_aug_len + cur_aug_len
        combined_mask = torch.ones(bsz, 1, query_len, total_kv_len, dtype=dtype, device=device)
        
        # Fill in previous segments mask
        combined_mask[:, :, :, :prev_aug_len] = prev4d
        
        # Fill in current segment mask
        combined_mask[:, :, :, prev_aug_len:] = cur4d
        
        # Now apply the key isolation rules to prevent target leak:
        
        # 1. Memory tokens in current segment cannot attend to context tokens from previous segments
        if self.num_mem_tokens > 0:
            # For memory tokens in current segment (last num_mem_tokens positions)
            # Block attention to all context tokens from previous segments
            combined_mask[:, :, cur_mem_start:, :prev_mem_start] = 0
            
        # 2. Context tokens in current segment can attend to previous context tokens (causal)
        # This is already handled by the upper=True in prev4d
        
        # 3. Memory tokens in current segment can attend to previous memory tokens
        # This is allowed and already handled by the mask construction
        
        # 4. Sink token (if used) cannot attend to others
        if self.use_sink:
            combined_mask[:, :, 0, 1:] = 0
            
        # Invert to additive mask as expected by attention: allowed->0, masked->-inf
        combined_mask = invert_attn_mask(combined_mask, dtype)
        
        return combined_mask

    def _get_memory_tokens(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._get_memory is None or self.num_mem_tokens == 0:
            return None, None
        memory = self._get_memory()
        sink = self._get_sink() if self.use_sink and self._get_sink is not None else None
        mem = memory.unsqueeze(0).expand(batch_size, -1, -1)
        if sink is not None:
            sink = sink.unsqueeze(0).expand(batch_size, -1, -1)
        return mem, sink

    # ----- helpers operating on provided memory tensors (no buffers) -----
    def _alloc_initial_mem(self, device: torch.device, dtype: torch.dtype):
        W_mem = torch.zeros(
            1,
            self.n_heads,
            self.d_key // self.n_heads,
            self.d_model // self.n_heads,
            device=device,
            dtype=dtype,
        )
        z = torch.zeros(1, self.n_heads, self.d_key // self.n_heads, device=device, dtype=dtype) if self.use_denom else None
        return W_mem, z

    def _associate_with_mem(self, hidden_states: torch.Tensor, W_mem: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        q = self._to_heads(self.W_mq(hidden_states))
        mq = self.phi(q)
        mq = F.normalize(mq, dim=-1, p=2.0)
        num = torch.einsum("ihjk,ihkt->ihjt", mq, W_mem)
        if self.use_denom and z is not None:
            denom = torch.einsum("ihk,ihjk->ihj", z, mq)[..., None] + 1e-5
            hs = num / denom
        else:
            hs = num
        return self._from_heads(hs)

    def _update_mem_with_mem(
        self,
        mem_tokens: torch.Tensor,
        W_mem: torch.Tensor,
        z: Optional[torch.Tensor],
        first_seg: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        k = self._to_heads(self.W_mk(mem_tokens))
        mk = self.phi(k)
        mk = F.normalize(mk, dim=-1, p=2.0)

        new_mv = self._to_heads(self.W_mv(mem_tokens))
        if not first_seg:
            num = torch.einsum("ihjk,ihkt->ihjt", mk, W_mem)
            if self.use_denom and z is not None:
                denom = torch.einsum("ihj,ihkj->ihk", z, mk)[..., None] + 1e-5
                prev_mv = num / denom
                if self.correction:
                    new_info_coef = (
                        1 - denom / (torch.linalg.norm(mk, dim=-1) ** 2)[..., None]
                    )
                    new_info_coef = torch.clip(new_info_coef, 0, 1).detach()
                else:
                    new_info_coef = 1
            else:
                prev_mv = num
                new_info_coef = 1
        else:
            prev_mv = torch.zeros_like(new_mv, device=new_mv.device)
            new_info_coef = 1

        mv = new_mv - prev_mv
        mb = self._to_heads(torch.sigmoid(self.W_mb(mem_tokens)))
        einop = f"ihjk,ihjt,ihj{'t' if self.gating else 'x'}->ihkt"
        associations = torch.einsum(einop, mk, mv, mb)
        W_mem = W_mem + associations.detach()
        if self.use_denom and z is not None:
            z = z + (new_info_coef * mk).sum(dim=-2).detach()
        return W_mem, z, False

    # ----- main forward (inner-loop segmentation) -----
    def forward(self, hidden_states: torch.Tensor, attention_mask=None, *args, **kwargs):
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        bsz, seq_len, _ = hidden_states.shape

        out_full = []

        W_mem, z = self._alloc_initial_mem(hidden_states.device, hidden_states.dtype)
        first_seg = True

        # cache plumbing
        # Match language_modeling: enable caching only when sliding_window is on
        use_cache = self.sliding_window
        kwargs.pop('use_cache', None)

        past_key_values = kwargs.get("past_key_values", None)
        last_attn = None
        present_kv = None

        # helper to segment arbitrary tensor-like by time dim
        def segment_tensor(t: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
            if not isinstance(t, torch.Tensor):
                return t
            # common cases: (bsz, seq_len, ...), (bsz, seq_len), (seq_len, ...)
            if t.dim() >= 2 and t.size(1) == seq_len:
                return t[:, start_idx:end_idx, ...]
            if t.dim() >= 1 and t.size(0) == seq_len and (t.dim() == 1 or t.size(0) == t.shape[0]):
                return t[start_idx:end_idx, ...]
            return t

        prev_aug_len = 0
        for start in range(0, seq_len, self.segment_size):
            end = min(start + self.segment_size, seq_len)
            seg = hidden_states[:, start:end, :]
            attn_mask = attention_mask[:, start:end] if attention_mask is not None else torch.ones_like(seg[:, :, 0])

            # Check if this is the last segment and we're in generate mode
            is_last_segment = (end >= seq_len)
            should_add_memory_tokens = not (self.generate_mode and is_last_segment)

            mem, sink = self._get_memory_tokens(seg.size(0))
            if self.use_sink and sink is not None:
                seg_aug = torch.cat([sink.to(seg.dtype).to(seg.device), seg, mem.to(seg.dtype).to(seg.device)], dim=1)
            elif self.num_mem_tokens > 0 and mem is not None:
                seg_aug = torch.cat([seg, mem.to(seg.dtype).to(seg.device)], dim=1)
            else:
                seg_aug = seg

            if not first_seg:
                assoc = self._associate_with_mem(seg_aug, W_mem, z)
                seg_aug = assoc + seg_aug

            # Build attention mask for this augmented segment
            seg_aug_len = seg_aug.size(1)
            
            # Check if we're using cached forward pass (past_key_values provided)
            using_cache = kwargs.get('past_key_values') is not None
            
            if self.sliding_window and not using_cache:
                # Current segment: build 4D mask at query_len = seg_aug_len
                base_cur4d = attn_mask_to_4d(attn_mask.to(seg_aug.dtype), upper=False, query_len=seg_aug_len)
                cur4d = self.pad_attention_mask(base_cur4d, dtype=seg_aug.dtype)
                if prev_aug_len > 0:
                    # Previous segments: build 4D mask at query_len = seg_aug_len
                    # The prev2d should have length prev_aug_len, representing previous segments
                    prev2d = torch.ones(attn_mask.size(0), prev_aug_len, dtype=seg_aug.dtype, device=attn_mask.device)
                    base_prev4d = attn_mask_to_4d(prev2d, upper=True, query_len=seg_aug_len)
                    prev4d = self.pad_prev_seg_attn_mask(base_prev4d, dtype=seg_aug.dtype)
                    
                    # Create a proper sliding window mask that prevents target leak
                    # We need to ensure memory tokens don't attend to context tokens across segments
                    seg_mask = self._build_sliding_window_mask(prev4d, cur4d, prev_aug_len, seg_aug_len)
                else:
                    seg_mask = cur4d
            elif using_cache:
                # When using cache, attention mask should already be properly formatted
                # Just ensure it has the right shape for the current query length
                if attention_mask.dim() == 4:
                    seg_mask = attention_mask
                else:
                    # Convert 2D mask to 4D if needed
                    base_cur4d = attn_mask_to_4d(attention_mask.to(seg_aug.dtype), upper=False, query_len=seg_aug_len)
                    seg_mask = self.pad_attention_mask(base_cur4d, dtype=seg_aug.dtype)
            else:
                base_cur4d = attn_mask_to_4d(attn_mask.to(seg_aug.dtype), upper=False, query_len=seg_aug_len)
                seg_mask = self.pad_attention_mask(base_cur4d, dtype=seg_aug.dtype)

            seg_pos_ids = self._get_segment_positions(kwargs.get("position_ids", None), start, end, seg_aug.device)

            # Segment incoming args/kwargs by time where applicable
            seg_args = tuple(segment_tensor(a, start, end) if isinstance(a, torch.Tensor) else a for a in args)
            seg_kwargs = {k: segment_tensor(v, start, end) for k, v in kwargs.items()}


            
            # Override with our computed fields
            seg_kwargs["attention_mask"] = seg_mask
            if seg_pos_ids is not None:
                seg_kwargs["position_ids"] = seg_pos_ids
            seg_kwargs["use_cache"] = use_cache
            if past_key_values is not None:
                seg_kwargs["past_key_values"] = self.update_past_key_values_sw(past_key_values, self.segment_size)
            if self._rotary_fn is not None and seg_pos_ids is not None:
                cos, sin = self._rotary_fn(seg_aug, seg_pos_ids)
                seg_kwargs["position_embeddings"] = (cos, sin)

            layer_out = self.layer(seg_aug, *seg_args, **seg_kwargs)
            if isinstance(layer_out, tuple):
                seg_out = layer_out[0]
                if len(layer_out) > 1:
                    last_attn = layer_out[1]
                if len(layer_out) > 2:
                    present_kv = layer_out[2]
                    past_key_values = present_kv
            else:
                seg_out = layer_out

            real_start = int(self.use_sink)
            real_end = real_start + seg.size(1)
            seg_real_out = seg_out[:, real_start:real_end, :]
            
            # Update memory only if we have memory tokens and we're not in generate mode for the last segment
            should_update_memory = (self.num_mem_tokens > 0 and 
                                  not (self.generate_mode and is_last_segment))
            
            if should_update_memory:
                seg_mem_out = seg_out[:, -self.num_mem_tokens :, :]
                W_mem, z, first_seg = self._update_mem_with_mem(
                    seg_mem_out, W_mem, z, first_seg
                )
            else:
                first_seg = False

            out_full.append(seg_real_out)
            # Update prev_aug_len for next iteration - this should be the total augmented length so far
            prev_aug_len += seg_aug_len

        merged = torch.cat(out_full, dim=1) if len(out_full) > 1 else out_full[0]

        # Return tensor or HF-like tuple (hidden_states, attn, present_kv)
        # if use_cache or ("output_attentions" in kwargs and kwargs["output_attentions"]):
        #     return (merged, last_attn, present_kv)
        return merged

    def update_past_key_values_sw(self, past_key_values, window_size):
        """
        Update past key values for sliding window attention.
        This keeps only the most recent tokens within the window size.
        """
        if past_key_values is None:
            return None
            
        # Convert to legacy cache format for easier manipulation
        if hasattr(past_key_values, 'to_legacy_cache'):
            past_key_values = past_key_values.to_legacy_cache()
        
        # Keep only the most recent tokens within the window size
        updated_past_key_values = [
            [
                k_or_v[..., -window_size:, :]
                for k_or_v in seg_kv
            ]
            for seg_kv in past_key_values
        ]
        
        # Convert back to DynamicCache if possible
        try:
            from transformers.cache_utils import DynamicCache
            return DynamicCache.from_legacy_cache(updated_past_key_values)
        except ImportError:
            return updated_past_key_values


class InnerLoopARMTForCausalLM(PreTrainedModel):
    """
    Drop-in ARMT model that installs InnerLoopAssociativeLayerWrapper into a base
    HF Causal LM. All segmentation happens inside each wrapped layer; no outer
    recurrent driver is needed.
    """

    # Reuse the config used by the outer-loop variant for parity
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, **kwargs)
        from transformers import AutoConfig, AutoModelForCausalLM

        # Resolve base model from either provided name or config
        base_model = None
        bm_cfg = getattr(config, "base_model_config", None)
        bm_name = getattr(config, "base_model_name", None)
        if bm_cfg is not None and bm_name is not None:
            raise ValueError("Exactly one of `base_model_name` or `base_model_config` must be provided in config.")
        if bm_cfg is not None:
            if isinstance(bm_cfg, PretrainedConfig) and getattr(bm_cfg, "model_type", None) != getattr(config, "model_type", None):
                resolved_cfg = bm_cfg
            elif isinstance(bm_cfg, dict):
                from transformers import AutoConfig as HF_AutoConfig

                if "model_type" not in bm_cfg:
                    raise ValueError("`base_model_config` dict must include a 'model_type' key.")
                cfg_or_inst = HF_AutoConfig.for_model(bm_cfg["model_type"])  # type: ignore[arg-type]
                if isinstance(cfg_or_inst, PretrainedConfig):
                    resolved_cfg = cfg_or_inst
                    for k, v in bm_cfg.items():
                        setattr(resolved_cfg, k, v)
                else:
                    resolved_cfg = cfg_or_inst.from_dict(bm_cfg)
            elif isinstance(bm_cfg, str):
                from transformers import AutoConfig as HF_AutoConfig

                resolved_cfg = HF_AutoConfig.from_pretrained(bm_cfg)
            else:
                raise TypeError("`base_model_config` must be a transformers.PretrainedConfig, dict, or str.")
            base_model = AutoModelForCausalLM.from_config(resolved_cfg)
        elif bm_name is not None:
            from transformers import AutoModelForCausalLM as HF_AutoModelForCausalLM

            base_model = HF_AutoModelForCausalLM.from_pretrained(bm_name)
        else:
            raise ValueError("InnerLoopARMTForCausalLM requires either `base_model_config` or `base_model_name` in the config.")

        # Install wrappers
        self.model = base_model

        # Extract hyperparameters (fall back to sane defaults if missing)
        self.num_mem_tokens = int(getattr(config, "num_mem_tokens", 0) or 0)
        self.d_mem = int(getattr(config, "d_mem", 512))
        self.segment_size = int(getattr(config, "segment_size", 512))
        self.segment_alignment = getattr(config, "segment_alignment", "left")
        self.layers_attr = getattr(config, "layers_attr", "model.layers")
        self.correction = bool(getattr(config, "correction", True))
        self.n_heads = int(getattr(config, "n_heads", 1))
        self.use_denom = bool(getattr(config, "use_denom", True))
        self.gating = bool(getattr(config, "gating", False))
        self.freeze_mem_flag = bool(getattr(config, "freeze_mem", False))
        self.use_sink = bool(getattr(config, "use_sink", False))
        self.sliding_window = bool(getattr(config, "sliding_window", False))

        # Shared trainable memory embeddings (used by all layers)
        emb = self.model.get_input_embeddings()
        d_model = emb.embedding_dim
        memory_dim = getattr(self.model.config, "n_embd", getattr(self.model.config, "hidden_size", d_model))
        memory_weights = torch.randn(
            (self.num_mem_tokens, memory_dim), device=emb.weight.device, dtype=emb.weight.dtype
        ) * emb.weight.data.std()
        self.memory = nn.Parameter(memory_weights, requires_grad=True)
        if self.use_sink:
            self.sink = nn.Parameter(
                torch.randn((1, memory_dim), device=emb.weight.device, dtype=emb.weight.dtype), requires_grad=True
            )
        # function to access layers container
        def _get_layers_from_model(model_root: nn.Module):
            obj = model_root
            for attr in self.layers_attr.split("."):
                obj = getattr(obj, attr)
            return obj

        layers = _get_layers_from_model(self.model)
        # Try to obtain the base model's rotary embedding function
        rotary_fn = None
        # Llama-style: model.model.rotary_emb
        if hasattr(self.model, "model") and hasattr(self.model.model, "rotary_emb"):
            rotary_fn = self.model.model.rotary_emb
        # GPT-NeoX-style: model.gpt_neox.rotary_emb
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "rotary_emb"):
            rotary_fn = self.model.gpt_neox.rotary_emb
        # Per-forward context for propagating sink/mem states across layers
        self._il_ctx = None

        for i in range(len(layers)):
            layers[i] = InnerLoopAssociativeLayerWrapper(
                layer=layers[i],
                d_model=d_model,
                num_mem_tokens=self.num_mem_tokens,
                d_mem=self.d_mem,
                segment_size=self.segment_size,
                n_heads=self.n_heads,
                correction=self.correction,
                use_denom=self.use_denom,
                gating=self.gating,
                use_sink=self.use_sink,
                sliding_window=self.sliding_window,
                get_memory_fn=lambda self_ref=self: self_ref.memory,
                get_sink_fn=lambda self_ref=self: getattr(self_ref, "sink", None),
                rotary_fn=rotary_fn,
                info={"layer": i},
            )

        if self.freeze_mem_flag:
            for layer in _get_layers_from_model(self.model):
                layer.freeze_mem()

        # Expose convenience accessor
        self.get_layers = lambda: _get_layers_from_model(self.model)

    # ----- control helpers -----
    def generate_mode(self, is_on: bool):
        for layer in self.get_layers():
            layer.generate_mode = is_on

    def zero_mem(self):
        """Reset memory state for all layers."""
        for layer in self.get_layers():
            layer.zero_mem()

    def detach_mem(self):
        """Detach memory state for all layers."""
        for layer in self.get_layers():
            layer.detach_mem()

    # ----- hf api -----
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
        use_cache=None,
        past_key_values=None,
    ):
        out = self.model(
            input_ids=input_ids,
            labels=labels,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        return out

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        """
        Generate tokens using the inner-loop model with proper sliding window attention.
        This method should produce the same logits as the forward method for alignment.
        """
        if self.sliding_window:
            # Temporarily use inefficient implementation, as efficient implementation is not working
            return self._generate_sliding_window_inefficient(input_ids, attention_mask, **generate_kwargs)
        else:
            return self._generate_standard(input_ids, attention_mask, **generate_kwargs)
    
    def _generate_standard(self, input_ids, attention_mask=None, **generate_kwargs):
        """Standard generation without sliding window."""
        self.generate_mode(True)
        try:
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        finally:
            self.generate_mode(False)
    
    def _generate_sliding_window_inefficient(self, input_ids, attention_mask=None, **generate_kwargs):
        """
        Generate tokens using sliding window attention that matches the forward method.
        This ensures alignment between generate and forward methods.
        INEFFICIENT: recomputes the entire sequence on every token generation.
        Kept for reference and testing purposes.
        """
        self.generate_mode(True)
        try:
            max_new_tokens = generate_kwargs.get('max_new_tokens', 1)
            eos_token_id = generate_kwargs.get('eos_token_id', None)
            return_logits = generate_kwargs.get('return_logits', False)
            
            generated_ids = None
            all_logits = []

            # Process tokens one by one to ensure perfect alignment
            for i in range(max_new_tokens):
                # Prepare the full sequence for this step
                if generated_ids is not None:
                    current_input_ids = torch.cat([input_ids, generated_ids], dim=-1)
                    current_attention_mask = torch.cat([attention_mask, torch.ones_like(generated_ids)], dim=-1)
                else:
                    current_input_ids = input_ids
                    current_attention_mask = attention_mask
                
                # Process the full sequence through the inner loop
                # Reset memory state before each forward pass to ensure complete independence
                self.zero_mem()
                
                with torch.no_grad():
                    outputs = self.forward(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                
                # Get next token
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                if generated_ids is not None:
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                else:
                    generated_ids = next_token_id
                
                # Store the logits that were actually used to generate the next token
                if return_logits:
                    all_logits.append(next_token_logits)
                
                # Check for EOS
                if eos_token_id is not None and (next_token_id == eos_token_id).all():
                    break
            
            if return_logits:
                # Return the logits that were actually used for generation during the loop
                return generated_ids, torch.stack(all_logits, dim=1)
            else:
                return generated_ids
        finally:
            self.generate_mode(False)

    def _generate_sliding_window(self, input_ids, attention_mask=None, **generate_kwargs):
        """
        Generate tokens using sliding window attention with efficient caching.
        Uses the base model directly with past_key_values to avoid recomputing the entire sequence.
        This method should produce the same logits as the forward method for alignment.
        """
        self.generate_mode(True)
        try:
            max_new_tokens = generate_kwargs.get('max_new_tokens', 1)
            eos_token_id = generate_kwargs.get('eos_token_id', None)
            return_logits = generate_kwargs.get('return_logits', False)
            
            # Initialize memory state
            self.zero_mem()
            
            # Process the input sequence through inner loop to get memory state
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            # Get initial outputs using forward method (without caching for now)
            initial_outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract last logits
            next_token_logits = initial_outputs.logits[:, -1, :]
            
            generated_ids = None
            all_logits = []
            
            # Now implement truly efficient generation using past_key_values
            # First, we need to get the base model's past_key_values from the initial forward pass
            # But since our inner loop doesn't return past_key_values, we need a different approach
            
            base_model = self.model
            window_size = self.segment_size + self.num_mem_tokens + int(self.use_sink)
            
            # Let me try to use the base model directly with the initial sequence to get past_key_values
            try:
                # Get past_key_values from base model for the initial sequence
                base_outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
                past_key_values = base_outputs.past_key_values
                
                # Now we can use efficient generation
                for i in range(max_new_tokens):
                    # Get next token
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    
                    if generated_ids is not None:
                        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                    else:
                        generated_ids = next_token_id
                    
                    # Store logits if requested
                    if return_logits:
                        all_logits.append(next_token_logits)
                    
                    # Check for EOS
                    if eos_token_id is not None and (next_token_id == eos_token_id).all():
                        break
                    
                    # Use efficient generation with past_key_values
                    with torch.no_grad():
                        next_outputs = base_model(
                            input_ids=next_token_id,
                            attention_mask=torch.ones_like(next_token_id),
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                        next_token_logits = next_outputs.logits[:, -1, :]
                        past_key_values = next_outputs.past_key_values
                        
                        # Update past_key_values for sliding window
                        if past_key_values is not None:
                            past_key_values = self.update_past_key_values_sw(past_key_values, window_size)
                            
            except Exception as e:
                # If this fails, we need to understand why
                print(f"Error implementing efficient generation: {e}")
                print("This suggests the base model doesn't support the expected interface")
                print("Why could this happen?")
                print("1. The base model might not support past_key_values")
                print("2. The attention mask handling might be incompatible")
                print("3. The memory tokens might interfere with caching")
                print("4. The inner loop wrapper might not be compatible with base model caching")
                raise RuntimeError(f"Efficient generation failed: {e}")
            
            if return_logits:
                return generated_ids, torch.stack(all_logits, dim=1)
            else:
                return generated_ids
        finally:
            self.generate_mode(False)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        try:
            return super().load_state_dict(state_dict, strict, assign)
        except RuntimeError:
            # Fallback: some checkpoints may target only the wrapped model
            self.model.load_state_dict(state_dict, strict=True)
            return


