import math
import os
import inspect
from typing import Optional, Tuple, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.cache_utils import DynamicCache
import warnings

# Reuse utilities from the existing implementation to ensure identical math
from modeling_amt.utils import DPFP, invert_attn_mask, attn_mask_to_4d

class ThinkingARMTConfig(PretrainedConfig):
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
                 wrap_layers=None,
                 reading_depth_multiplier=1,
                 writing_depth_multiplier=1,
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
        self.wrap_layers = wrap_layers
        self.reading_depth_multiplier = reading_depth_multiplier
        self.writing_depth_multiplier = writing_depth_multiplier
    def get(self, attr: str, default=None):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return default

try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    print("*** Can't import liger_kernel ***")
    LIGER_KERNEL_AVAILABLE = False
except Exception as e:
    print("*** Can't import liger_kernel ***")
    raise e


def reverse_invert_attn_mask(mask: torch.Tensor) -> torch.Tensor:
    if os.environ.get("NOT_INVERT_ATTN_MASK"):
        return mask
    mask = mask.clone().long()
    mask[mask > -1] = 1
    mask[mask < -1] = 0
    return mask

def attn_mask_to_2d(mask: torch.Tensor) -> torch.Tensor:
    mask = reverse_invert_attn_mask(mask)
    mask = torch.any(mask, dim=-2)
    mask = torch.any(mask, dim=1)
    return mask.long()

def is_empty_past_key_values(past_key_values: Optional[DynamicCache], layer_idx: int) -> bool:
    if past_key_values is None:
        return True
    if len(past_key_values.layers) == 0:
        return True
    if len(past_key_values.layers) <= layer_idx:
        return True
    if past_key_values.layers[layer_idx].keys is None:
        return True
    return False

def segment_tensor(t, start_idx: int, end_idx: int, seq_len: int):
    if isinstance(t, tuple):
        return tuple(segment_tensor(item, start_idx, end_idx, seq_len) for item in t)
    if not isinstance(t, torch.Tensor):
        return t
    if t.dim() == 1 and t.size(0) == seq_len:
        return t[start_idx:end_idx]
    # common cases: (bsz, seq_len, ...), (bsz, seq_len), (seq_len, ...)
    if t.dim() >= 2 and t.size(1) == seq_len:
        return t[:, start_idx:end_idx, ...]
    return t

class ThinkingAssociativeLayerWrapper(nn.Module):
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

        self.memory_state = None

        self.depth_multiplier = 1

    def set_depth_multiplier(self, depth_multiplier: int):
        self.depth_multiplier = depth_multiplier
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
        self.memory_state = None

    def detach_mem(self) -> None:
        self.memory_state = (self.memory_state[0].detach(), self.memory_state[1].detach()) if self.memory_state is not None else None

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
            return position_ids[:, start:end]
        else:
            position_ids = torch.arange(start, end, device=device).long().unsqueeze(0)
            return position_ids


    def pad_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype):
        if self.num_mem_tokens in {0, None} and not self.use_sink:
            return attention_mask
        shape = list(attention_mask.shape)
        if len(shape) == 4:
            shape[-1] += self.num_mem_tokens + int(self.use_sink)
            shape[-2] += self.num_mem_tokens + int(self.use_sink)
            mask = torch.ones(*shape, dtype=dtype).to(attention_mask.device)
            mask[..., int(self.use_sink):-self.num_mem_tokens, int(self.use_sink):-self.num_mem_tokens] = attention_mask
            if self.use_sink:
                mask[..., 0, 1:] = 0
            mask[..., :-self.num_mem_tokens, -self.num_mem_tokens:] = 0
        elif len(shape) == 2:
            shape[-1] += self.num_mem_tokens + int(self.use_sink)
            mask = torch.ones(*shape, dtype=dtype).to(attention_mask.device)
            mask[..., int(self.use_sink):-self.num_mem_tokens] = attention_mask
        else:
            raise ValueError("Attention mask must be 2D or 4D")
        return mask.to(dtype)


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
        W_mem = W_mem + associations
        if self.use_denom and z is not None:
            z = z + (new_info_coef * mk).sum(dim=-2)
        return W_mem, z, False

    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Convert positional args of the wrapped HF block into keyword args by
        introspecting the block's forward signature. This prevents accidental
        misplacement (e.g., a cache object being treated as attention_mask).
        """
        # Map positional args to their parameter names (excluding self & hidden_states)
        try:
            sig = inspect.signature(self.layer.forward)
            params = list(sig.parameters.values())
            # Drop the first param which should be 'self' for bound method
            param_names = [p.name for p in params[1:]]
            # If the next parameter is hidden_states, drop it as well
            if len(param_names) > 0 and param_names[0] in {"hidden_states", "x"}:
                param_names = param_names[1:]
        except Exception:
            param_names = []

        for idx, arg in enumerate(args):
            if idx >= len(param_names):
                break
            name = param_names[idx]
            if name not in kwargs:
                kwargs[name] = arg

        # Normalize cache kwarg name to 'past_key_values'
        if "layer_past" in kwargs and "past_key_values" not in kwargs:
            layer_past = kwargs.pop("layer_past")
            try:
                if isinstance(layer_past, DynamicCache):
                    kwargs["past_key_values"] = layer_past
                else:
                    kwargs["past_key_values"] = DynamicCache.from_legacy_cache(layer_past)
            except Exception:
                kwargs["past_key_values"] = layer_past

        # Extract attention mask (avoid passing both positional & kwarg duplicates)
        attention_mask = kwargs.pop("attention_mask", None)
        for _ in range(self.depth_multiplier):
            layer_out = self.forward_horizontal(hidden_states, attention_mask, **kwargs)
            if isinstance(layer_out, tuple):
                hidden_states = layer_out[0]
                rest = layer_out[1:]
            else:
                hidden_states = layer_out
                rest = tuple()
        return hidden_states, *rest

    
    # ----- main forward (inner-loop segmentation) -----
    def forward_horizontal(self, hidden_states: torch.Tensor, attention_mask=None, *args, **kwargs):
        assert not self.generate_mode, "Generate mode is not supported for horizontal forward"
        assert attention_mask is None or attention_mask.dim() == 4, "Attention mask must be 4D"
        using_cache = not is_empty_past_key_values(kwargs.get("past_key_values"), self.info['layer'])
        assert not using_cache or (kwargs.get('past_attn_mask') is not None and kwargs.get('past_attn_mask').shape[-1] == self.segment_size), "When using cache, past_attn_mask must be provided and have the same length as the segment size"

        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        bsz, seq_len, _ = hidden_states.shape

        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
            attention_mask = attn_mask_to_4d(attention_mask, upper=False, query_len=seq_len)
            attention_mask = invert_attn_mask(attention_mask, hidden_states.dtype)
        out_full = []

        # Initialize associative memory from persisted state if available
        if self.memory_state is not None:
            W_mem, z = self.memory_state
            first_seg = False
        else:
            W_mem, z = self._alloc_initial_mem(hidden_states.device, hidden_states.dtype)
            first_seg = True


        # Always use provided cache object if present, even if currently empty,
        # so upstream callers can observe in-place mutations across segments.
        provided_cache = kwargs.get("past_key_values")
        past_key_values = provided_cache if provided_cache is not None else DynamicCache()
        past_attn_mask = kwargs.get('past_attn_mask') if using_cache else None
        present_kv = None

        # helper to segment arbitrary tensor-like by time dim
        
        seg_num = 0
        for start in range(0, seq_len, self.segment_size+self.num_mem_tokens+int(self.use_sink)):
            real_start = start+int(self.use_sink)
            real_end = min(real_start + self.segment_size, seq_len-self.num_mem_tokens)
            end = real_end+self.num_mem_tokens
            seg_aug = hidden_states[:, start:end, :]
            seg_len = real_end - real_start

            attn_mask = attention_mask[:, :, real_start:real_end, real_start:real_end]

            # print("attn_mask", attn_mask[0][0])

            # Check if this is the last segment and we're in generate mode
            is_last_segment = (end >= seq_len)


            if not first_seg:
                assoc = self._associate_with_mem(seg_aug, W_mem, z)
                seg_aug = assoc + seg_aug

            # Build attention mask for this augmented segment
            seg_aug_len = seg_aug.size(1)
            
            if self.sliding_window:
                # print(attn_mask.shape, "attn_mask", "*"*100)
                # print(base_cur4d.shape, "base_cur4d", "*"*100)
                base_cur4d = reverse_invert_attn_mask(attn_mask)
                seg_mask = self.pad_attention_mask(base_cur4d, dtype=seg_aug.dtype)
                seg_mask = invert_attn_mask(seg_mask, seg_aug.dtype)

                if past_attn_mask is not None:

                    base_past4d = attn_mask_to_4d(attn_mask_to_2d(past_attn_mask), upper=True, query_len=seg_aug_len)
                    if self.use_sink:
                        base_past4d[:, :, 0, :] = 0 # sink cannot attend to others
                    # base_past4d = torch.ones_like(base_past4d)
                    base_past4d = invert_attn_mask(base_past4d, seg_aug.dtype)

                    # print(base_past4d.shape, "base_past4d", "*"*100)
                    # print(seg_mask.shape, "seg_mask", "*"*100)
                    seg_mask = torch.cat([base_past4d, seg_mask], dim=-1)
                if os.environ.get("ARMT_DEBUG_SW"):
                    print(f"[H-SEG] L{self.info['layer']} seg_len={seg_len} seg_aug_len={seg_aug_len} mask={tuple(seg_mask.shape)}")
            else:
                base_cur4d = reverse_invert_attn_mask(attn_mask)
                seg_mask = self.pad_attention_mask(base_cur4d, dtype=seg_aug.dtype)
                seg_mask = invert_attn_mask(seg_mask, seg_aug.dtype)
            # print("seg_mask", reverse_invert_attn_mask(seg_mask)[0][0])
            # print("seg_mask", seg_mask.shape)
            seg_pos_ids = self._get_segment_positions(kwargs.get("position_ids", None), start, end, seg_aug.device)

            # Segment incoming args/kwargs by time where applicable
            seg_args = tuple(segment_tensor(a, start, end, seq_len) if isinstance(a, torch.Tensor) else a for a in args)
            seg_kwargs = {k: segment_tensor(v, start, end, seq_len) for k, v in kwargs.items()}


            
            # Override with our computed fields
            seg_kwargs["attention_mask"] = seg_mask.to(seg_aug.dtype)
            if seg_pos_ids is not None:
                seg_kwargs["position_ids"] = seg_pos_ids
            seg_kwargs["use_cache"] = self.sliding_window
            
            if self.sliding_window:
                seg_kwargs["past_key_values"] = past_key_values
            else:
                # In non-sliding mode, ensure no cache is used by the underlying layer
                seg_kwargs.pop("layer_past", None)
                seg_kwargs.pop("cache_position", None)
                seg_kwargs.pop("past_key_values", None)
                seg_kwargs["use_cache"] = False

            if self._rotary_fn is not None and seg_pos_ids is not None:
                cos, sin = self._rotary_fn(seg_aug, seg_pos_ids)
                seg_kwargs["position_embeddings"] = (cos, sin)

            
            layer_out = self.layer(seg_aug, *seg_args, **seg_kwargs)
            if self.sliding_window:
                assert past_key_values is not None, "Past key values object must be provided"
                # In-place update & trim so outer references observe changes
                if os.environ.get("ARMT_DEBUG_SW"):
                    k = past_key_values.layers[self.info['layer']].keys
                    v = past_key_values.layers[self.info['layer']].values
                    print(f"[H-CACHE:pre] L{self.info['layer']} K={tuple(k.shape) if k is not None else None} V={tuple(v.shape) if v is not None else None}")
                past_key_values = self.update_past_key_values_sw(past_key_values, self.segment_size)
                if os.environ.get("ARMT_DEBUG_SW"):
                    k = past_key_values.layers[self.info['layer']].keys
                    v = past_key_values.layers[self.info['layer']].values
                    print(f"[H-CACHE:post] L{self.info['layer']} K={tuple(k.shape) if k is not None else None} V={tuple(v.shape) if v is not None else None}")
            if isinstance(layer_out, tuple):
                seg_out = layer_out[0]
            else:
                seg_out = layer_out

            seg_mem_out = seg_out[:, -self.num_mem_tokens:, :]
            W_mem, z, first_seg = self._update_mem_with_mem(
                seg_mem_out, W_mem, z, first_seg
            )
            first_seg = False

            out_full.append(seg_out)

            past_attn_mask = attn_mask
            seg_num += 1

        merged = torch.cat(out_full, dim=1)

        # Persist updated memory state for vertical mode to reuse across segments
        self.memory_state = (W_mem, z)

        if isinstance(layer_out, tuple):
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            if len(layer_out) == 1:
                return (merged,)
            elif len(layer_out) == 2:
                warnings.warn(f"{YELLOW}Last attention was not tested for horizontal forward{RESET}")
                return (merged, None)
            elif len(layer_out) == 3:
                warnings.warn(f"{YELLOW}Last attention and kv states were not tested for horizontal forward{RESET}")
                return (merged, None, present_kv)
            else:
                raise ValueError(f"Expected 1, 2 or 3 elements in layer output, got {len(layer_out)}")
        else:
            return merged

    def update_past_key_values_sw(self, past_key_values, window_size):
        """
        Update past key values for sliding window attention.
        This keeps only the most recent tokens within the window size.
        """
        if is_empty_past_key_values(past_key_values, self.info['layer']):
            return None
            
        # Convert to legacy cache format for easier manipulation
        if hasattr(past_key_values, 'to_legacy_cache'):
            legacy = past_key_values.to_legacy_cache()
        
        # Keep only the most recent real tokens within the window size
        k, v = legacy[self.info['layer']]
        k = k[..., -window_size-self.num_mem_tokens:-self.num_mem_tokens, :]
        v = v[..., -window_size-self.num_mem_tokens:-self.num_mem_tokens, :]
        
        past_key_values.layers[self.info['layer']].keys = k
        past_key_values.layers[self.info['layer']].values = v
        return past_key_values


class ThinkingARMTForCausalLM(PreTrainedModel):
    """
    Drop-in ARMT model that installs InnerLoopAssociativeLayerWrapper into a base
    HF Causal LM. All segmentation happens inside each wrapped layer; no outer
    recurrent driver is needed.
    """

    # Reuse the config used by the outer-loop variant for parity
    config_class = ThinkingARMTConfig

    def __init__(self, config: ThinkingARMTConfig, **kwargs):
        global LIGER_KERNEL_AVAILABLE
        super().__init__(config, **kwargs)
        from transformers import AutoConfig, AutoModelForCausalLM

        # Resolve base model from either provided name or config
        base_model = None
        bm_cfg = getattr(config, "base_model_config", None)
        bm_name = getattr(config, "base_model_name", None)

        if bm_name is None or 'llama' not in bm_name:
            LIGER_KERNEL_AVAILABLE = False
            os.environ["ARMT_DISABLE_LIGER_KERNEL"] = "1"
        if LIGER_KERNEL_AVAILABLE and not os.environ.get("ARMT_DISABLE_LIGER_KERNEL"):
            apply_liger_kernel_to_llama()

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
        if self.segment_alignment != 'left':
            raise 
        self.layers_attr = getattr(config, "layers_attr", "model.layers")
        self.correction = bool(getattr(config, "correction", True))
        self.n_heads = int(getattr(config, "n_heads", 1))
        self.use_denom = bool(getattr(config, "use_denom", True))
        self.gating = bool(getattr(config, "gating", False))
        self.freeze_mem_flag = bool(getattr(config, "freeze_mem", False))
        self.use_sink = bool(getattr(config, "use_sink", False))
        self.sliding_window = bool(getattr(config, "sliding_window", False))
        self.reading_depth_multiplier = int(getattr(config, "reading_depth_multiplier", 1))
        self.writing_depth_multiplier = int(getattr(config, "writing_depth_multiplier", 1))
        # Shared trainable memory embeddings (used by all layers)
        emb = self.model.get_input_embeddings()
        d_model = emb.embedding_dim
        memory_dim = getattr(self.model.config, "n_embd", getattr(self.model.config, "hidden_size", d_model))
        # Robust std in float32 with sane fallback
        # with torch.no_grad():
        #     emb_std32 = emb.weight.detach().float().std()
        #     if not torch.isfinite(emb_std32):
        #         emb_std32 = torch.tensor(0.02, device=emb.weight.device)
        #     emb_std32 = torch.clamp(emb_std32, min=1e-3, max=0.1)
        memory_weights = torch.empty(
            (self.num_mem_tokens, memory_dim), device=emb.weight.device, dtype=emb.weight.dtype
        )
        # torch.nn.init.normal_(memory_weights, mean=0.0, std=emb_std32.to(memory_weights.dtype))
        torch.nn.init.normal_(memory_weights, mean=0.0, std=0.02)
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
        wrap_layers = config.get("wrap_layers")
        self.wrap_layers = wrap_layers if wrap_layers is not None else [1,] * len(layers)
        assert len(self.wrap_layers) == len(layers)
        rotary_fn = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "rotary_emb"):
            rotary_fn = self.model.model.rotary_emb
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "rotary_emb"):
            rotary_fn = self.model.gpt_neox.rotary_emb

        for i in range(len(layers)):
            if self.wrap_layers[i]:
                layers[i] = ThinkingAssociativeLayerWrapper(
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
            for i, layer in enumerate(_get_layers_from_model(self.model)):
                if self.wrap_layers[i]:
                    layer.freeze_mem()


        # Expose convenience accessor
        self.get_layers = lambda: _get_layers_from_model(self.model)

        self.vertical_mode = True

    # ----- control helpers -----
    def generate_mode(self, is_on: bool):
        for i, layer in enumerate(self.get_layers()):
            if self.wrap_layers[i]:
                layer.generate_mode = is_on

    def zero_mem(self):
        """Reset memory state for all layers."""
        for i, layer in enumerate(self.get_layers()):
            if self.wrap_layers[i]:
                layer.zero_mem()

    def detach_mem(self):
        """Detach memory state for all layers."""
        for i, layer in enumerate(self.get_layers()):
            if self.wrap_layers[i]:
                layer.detach_mem()

    def augment_sequence(self, hidden_states: torch.Tensor, mem: torch.Tensor, sink: torch.Tensor = None, starts = None, ends = None):
        segments = self.split_tensor(hidden_states, starts, ends)
        if sink is not None:
            augmented_segments = [torch.cat([sink.to(segment.dtype).to(segment.device), segment, mem.to(segment.dtype).to(segment.device)], dim=1) for segment in segments]
        else:
            augmented_segments = [torch.cat([segment, mem.to(segment.dtype).to(segment.device)], dim=1) for segment in segments]
        augmented_sequence = torch.cat(augmented_segments, dim=1)

        return augmented_sequence

    def clean_sequence(self, hidden_states: torch.Tensor, aug_starts, aug_ends):
        segments = []
        for s, e in zip(aug_starts, aug_ends):
            segment = hidden_states[:, s+self.use_sink:e-self.num_mem_tokens]
            segments.append(segment)
        return torch.cat(segments, dim=1)

    def augment_attention_mask(self, attention_mask: torch.Tensor, starts, ends):
        segments = self.split_tensor(attention_mask, starts, ends)

        if self.use_sink:
            augmented_segments = [torch.cat([
                torch.ones(segment.shape[0], 1, device=segment.device, dtype=segment.dtype), 
                segment, 
                torch.ones(segment.shape[0], self.num_mem_tokens, device=segment.device, dtype=segment.dtype)
            ], dim=1) for segment in segments]
        else:
            augmented_segments = [torch.cat([
                segment, 
                torch.ones(segment.shape[0], self.num_mem_tokens, device=segment.device, dtype=segment.dtype)
            ], dim=1) for segment in segments]
        augmented_attention_mask = torch.cat(augmented_segments, dim=1)
        return augmented_attention_mask

    def augment_labels(self, labels, starts, ends):
        if labels is None:
            return None
        first = labels[:, :1]

        # add -100 token to ensure the correct splitting
        labels = torch.cat([labels, -100 * torch.ones_like(first)], dim=1)

        segments = self.split_tensor(labels[:, 1:], starts, ends)
        if self.use_sink:
            augmented_segments = [torch.cat([
                -100 * torch.ones(segment.shape[0], 1, device=segment.device, dtype=segment.dtype),
                segment,
                -100 * torch.ones(segment.shape[0], self.num_mem_tokens, device=segment.device, dtype=segment.dtype)
            ], dim=1) for segment in segments]
        else:
            augmented_segments = [torch.cat([
                segment,
                -100 * torch.ones(segment.shape[0], self.num_mem_tokens, device=segment.device, dtype=segment.dtype)
            ], dim=1) for segment in segments]
        augmented_segments = torch.cat(augmented_segments, dim=1)

        # remove -100 token and concatenate the original first label (it is not supposed to be used in loss computation, though)
        augmented_labels = torch.cat([first, augmented_segments[:, :-1]], dim=1)
        return augmented_labels

    def augment(self, input_ids, inputs_embeds, attention_mask, labels):
        if input_ids is not None:
            assert inputs_embeds is None, "input_ids and inputs_embeds cannot be provided together"
            hidden_states = self.model.get_input_embeddings()(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        labels_start = torch.min(torch.where(labels != -100)[1]).item() if labels is not None else 0
        # print("labels_start", labels_start, len(labels[0]) if labels is not None else "None")
        starts = []
        first_labels_segment = -1
        for start in range(0, hidden_states.size(1), self.segment_size):
            starts.append(start)
            if start < labels_start and start + self.segment_size >= labels_start:
                first_labels_segment = len(starts)
                for start in range(labels_start, hidden_states.size(1), self.segment_size):
                    starts.append(start)
                break
        ends = [s for s in starts[1:]] + [hidden_states.size(1),]
        offsets = [(i+1) * self.use_sink + i * self.num_mem_tokens for i in range(len(starts))]
        aug_starts = [s + o - self.use_sink for s,o in zip(starts, offsets)]
        aug_ends = [e + o + self.num_mem_tokens  for e,o in zip(ends, offsets)]
        
        mem = self.memory.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        sink = self.sink.unsqueeze(0).expand(hidden_states.size(0), -1, -1) if self.use_sink else None

        augmented_hidden_states = self.augment_sequence(hidden_states, mem, sink, starts, ends)
        augmented_attention_mask = self.augment_attention_mask(attention_mask, starts, ends)
        augmented_labels = self.augment_labels(labels, starts, ends)
        return augmented_hidden_states, augmented_attention_mask, augmented_labels, aug_starts, aug_ends, first_labels_segment

    def forward(
        self,
        input_ids=None,
        labels=None,
        labels_mask=None,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        output_only_last_segment=False,
        num_items_in_batch=None,
        use_cache=None,
        past_key_values=None,
    ):  
        
        self.zero_mem()
        if labels_mask is not None:
            assert labels_mask.any(), "labels_mask must not be all zeros"
        # Apply labels_mask by mapping masked positions to -100 (ignored by loss)
        effective_labels = labels
        if labels is not None and labels_mask is not None:
            if isinstance(labels_mask, torch.Tensor):
                mask_bool = labels_mask.bool() if labels_mask.dtype != torch.bool else labels_mask
                effective_labels = labels.masked_fill(~mask_bool, -100)
            else:
                raise ValueError("labels_mask must be a torch.Tensor")

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], device=input_ids.device, dtype=input_ids.dtype)
            else:
                attention_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        
        if self.vertical_mode:
            return self.forward_vertical(
                input_ids=input_ids,
                labels=effective_labels,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, 
                output_only_last_segment=output_only_last_segment,
                num_items_in_batch=num_items_in_batch,
                use_cache=use_cache,
                past_key_values=past_key_values,
                past_attn_mask=None
        )
        else:
            return self.forward_horizontal(
                input_ids=input_ids,
                labels=effective_labels,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                output_only_last_segment=output_only_last_segment,
                num_items_in_batch=num_items_in_batch,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
        
    def set_depth_multiplier(self, depth_multiplier: int):
        for layer in self.get_layers():
            layer.set_depth_multiplier(depth_multiplier)
    
    def split_tensor(self, tensor, starts, ends):
        return [tensor[:, s:e] for s,e in zip(starts,ends)]
    
    def forward_vertical(
        self,
        input_ids=None,
        labels=None,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        output_only_last_segment=False,
        num_items_in_batch=None,
        use_cache=None,
        past_key_values=None,
        past_attn_mask=None,
    ):
        assert not self.training or os.environ.get("ARMT_DISABLE_LIGER_KERNEL"), "Liger kernel is not supported for training in vertical mode, to disable liger kernel, set ARMT_DISABLE_LIGER_KERNEL=1"
        # Establish batch/seq info
        if input_ids is not None:
            assert inputs_embeds is None
            B, L = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            B, L, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        dtype = next(self.model.parameters()).dtype

        augmented_hidden_states, augmented_attention_mask, augmented_labels, aug_starts, aug_ends, first_labels_segment = self.augment(input_ids, inputs_embeds, attention_mask, labels)

        # print(aug_starts, aug_ends, first_labels_segment)
        # Build segmented inputs
        # Split all provided tensors consistently
        seg_inputs_embeds = self.split_tensor(augmented_hidden_states, aug_starts, aug_ends)
        seg_attention_mask = self.split_tensor(augmented_attention_mask, aug_starts, aug_ends) if attention_mask is not None else None
        # seg_labels = self.split_tensor(augmented_labels, aug_starts, aug_ends) if labels is not None else None
        # Assemble list of per-segment dicts
        num_segments = len(seg_inputs_embeds) 
        segments = []
        for i in range(num_segments):
            segments.append({
                "inputs_embeds": seg_inputs_embeds[i],
                "attention_mask": None if seg_attention_mask is None else seg_attention_mask[i],
                "labels": None # if seg_labels is None else seg_labels[i],
            })

        # Sliding window state across segments
        use_sliding = bool(self.sliding_window)
        shared_cache = past_key_values if (use_sliding and past_key_values is not None) else (DynamicCache() if use_sliding else None)
        past_attn_mask = past_attn_mask if use_sliding else None
        # Absolute positions across segments
        pos_offset = 0

        # Run each segment through the base model; per-layer memory persists inside wrappers
        seg_outputs = []
        layers = self.get_layers()
        for seg_idx, seg in enumerate(segments):
            seg_len = seg["inputs_embeds"].size(1)
            if seg.get("attention_mask") is None:
                base_2d = torch.ones(B, seg_len, device=device, dtype=dtype)
            else:
                base_2d = seg["attention_mask"]
            cur4d = attn_mask_to_4d(base_2d, upper=False, query_len=seg_len)
            cur4d = invert_attn_mask(cur4d, dtype=dtype)

            # Absolute position ids (match horizontal behavior when given position_ids=None)
            position_ids = torch.arange(pos_offset, pos_offset + seg_len, device=device).long().unsqueeze(0)

            # Temporarily wrap each layer to inject past_attn_mask into kwargs
            orig_forwards = [ly.forward for ly in layers]
            seg_past_attn_mask = past_attn_mask
            def _inject_mask(orig_fn, mask):
                def _wrapped(hs, *a, **k):
                    # Inject past attention mask and shared cache at layer level to mirror horizontal
                    if mask is not None:
                        if 'past_attn_mask' not in k:
                            k['past_attn_mask'] = mask
                        # Ensure using shared DynamicCache for this segment
                        if 'past_key_values' not in k or k['past_key_values'] is None:
                            k['past_key_values'] = shared_cache
                        # Guard against blocks that expect a tuple per layer
                        if hasattr(k['past_key_values'], 'layers') and len(k['past_key_values'].layers) < len(layers):
                            # Extend layers with empty entries up to current depth
                            needed = len(layers) - len(k['past_key_values'].layers)
                            k['past_key_values'].layers.extend([type(k['past_key_values'].layers[0])() for _ in range(needed)])
                        k['use_cache'] = True
                    return orig_fn(hs, *a, **k)
                return _wrapped
            for i, ly in enumerate(layers):
                ly.forward = _inject_mask(orig_forwards[i], seg_past_attn_mask)

            if seg_idx < first_labels_segment - 1:
                self.set_depth_multiplier(self.writing_depth_multiplier)
            elif seg_idx == first_labels_segment - 1:
                self.set_depth_multiplier(self.reading_depth_multiplier)
            else:
                self.set_depth_multiplier(1)
            out = self.model(
                input_ids=seg.get("input_ids"),
                inputs_embeds=seg.get("inputs_embeds"),
                attention_mask=cur4d,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_sliding,
                past_key_values=shared_cache if use_sliding else None,
            )
            if os.environ.get("ARMT_DEBUG_SW"):
                print(f"[V-SEG] seg_len={seg_len} cur4d={tuple(cur4d.shape)} pos=({int(position_ids[0,0])},{int(position_ids[0,-1])})")
                if hasattr(out, 'past_key_values') and out.past_key_values is not None:
                    try:
                        k = out.past_key_values.layers[0].keys
                        v = out.past_key_values.layers[0].values
                        print(f"[V-CACHE:out] L0 K={tuple(k.shape) if k is not None else None} V={tuple(v.shape) if v is not None else None}")
                    except Exception:
                        pass
            # Restore original forwards
            for i, ly in enumerate(layers):
                ly.forward = orig_forwards[i]
            seg_outputs.append(out)

            if use_sliding:
                # Update cache and past attention for next segment
                shared_cache = out.past_key_values if hasattr(out, 'past_key_values') else shared_cache
                if os.environ.get("ARMT_DEBUG_SW") and shared_cache is not None:
                    try:
                        k = shared_cache.layers[0].keys
                        v = shared_cache.layers[0].values
                        print(f"[V-CACHE:posttrim] L0 K={tuple(k.shape) if k is not None else None} V={tuple(v.shape) if v is not None else None}")
                    except Exception:
                        pass
                past_attn_mask = cur4d[:, :, int(self.use_sink):-self.num_mem_tokens, int(self.use_sink):-self.num_mem_tokens]
            pos_offset += seg_len

        # Aggregate outputs across segments
        # Concatenate logits along time dimension
        full_logits = torch.cat([o.logits for o in seg_outputs], dim=1) if len(seg_outputs) > 1 else seg_outputs[0].logits

        result = {}
        result["logits"] = self.clean_sequence(full_logits, aug_starts, aug_ends)
        # Compute loss similar to outer wrapper
        if labels is not None:
            labels = labels[:, -result['logits'].size(1):]
            shift_labels = labels[..., 1:].contiguous()
            flat_labels = shift_labels.view(-1)

            shift_logits = result['logits'][..., :-1, :].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            loss_fct = CrossEntropyLoss(reduction='sum')
            loss = loss_fct(flat_logits, flat_labels)

            denom = (flat_labels != -100).sum()
            denom = torch.clamp(denom, min=1)
            result["loss"] = loss / denom
        
        if output_hidden_states:
            if all(getattr(o, 'hidden_states', None) is not None for o in seg_outputs):
                # Concatenate last layer hidden states across segments per layer index
                full_hidden_states = tuple([
                    torch.cat(layer_hs, dim=1)
                    for layer_hs in zip(*[o.hidden_states for o in seg_outputs])
                ])
                result["hidden_states"] = full_hidden_states
        return result
    
    # ----- hf api -----
    def forward_horizontal(
        self,
        input_ids=None,
        labels=None,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        output_only_last_segment=False,
        num_items_in_batch=None,
        use_cache=None,
        past_key_values=None,
    ):
        raise NotImplementedError("Horizontal forward is not implemented for ThinkingARMTForCausalLM")

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        """
        Generate tokens using the inner-loop model with proper sliding window attention.
        This method should produce the same logits as the forward method for alignment.
        """

        warnings.warn("Efficient generation is not implemented")
        if self.sliding_window:
            return self._generate_inefficient(input_ids, attention_mask, **generate_kwargs)
        else:
            # return self._generate_standard(input_ids, attention_mask, **generate_kwargs) 
            return self._generate_inefficient(input_ids, attention_mask, **generate_kwargs)
            # raise NotImplementedError("Non-sliding window generation is not implemented")
    
    def _generate_standard(self, input_ids, attention_mask=None, **generate_kwargs):
        """Standard generation without sliding window."""
        generate_kwargs['output_scores'] = generate_kwargs.get('return_logits', False)
        generate_kwargs['return_dict_in_generate'] = generate_kwargs.get('return_logits', False)
        generate_kwargs.pop('return_logits')
        out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        if generate_kwargs.get('output_scores', False):
            print(out.scores)
            return out.sequences, out.scores
        else:
            return out.sequences
    
    def _generate_inefficient(self, input_ids, attention_mask=None, **generate_kwargs):
        """
        Generate tokens using sliding window attention that matches the forward method.
        This ensures alignment between generate and forward methods.
        INEFFICIENT: recomputes the entire sequence on every token generation.
        Kept for reference and testing purposes.
        """
        max_new_tokens = generate_kwargs.get('max_new_tokens', 1)
        eos_token_id = generate_kwargs.get('eos_token_id', None)
        return_logits = generate_kwargs.get('return_logits', False)
        
        generated_ids = None
        all_logits = []
        fake_labels = -100 * torch.ones_like(input_ids)
        fake_labels[:, -1] = 0
        # Process tokens one by one to ensure perfect alignment
        for i in range(max_new_tokens):
            # Prepare the full sequence for this step
            if generated_ids is not None:
                current_input_ids = torch.cat([input_ids, generated_ids], dim=-1)
                current_attention_mask = torch.cat([attention_mask, torch.ones_like(generated_ids)], dim=-1)
                current_fake_labels = torch.cat([fake_labels, torch.zeros_like(generated_ids)], dim=-1)
            else:
                current_input_ids = input_ids
                current_attention_mask = attention_mask
                current_fake_labels = fake_labels
            
            # Process the full sequence through the inner loop
            # Reset memory state before each forward pass to ensure complete independence
            self.zero_mem()
            
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    labels=current_fake_labels
                )
                next_token_logits = outputs["logits"][:, -1, :]
            
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

    def zero_mem(self):
        for layer in self.get_layers():
            layer.zero_mem()

    def detach_mem(self):
        for layer in self.get_layers():
            layer.detach_mem()

    def freeze_mem(self):
        for layer in self.get_layers():
            layer.freeze_mem()
