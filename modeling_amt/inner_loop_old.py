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
import logging
import warnings
# Reuse utilities from the existing implementation to ensure identical math
from modeling_amt.utils import DPFP, invert_attn_mask, attn_mask_to_4d
from transformers.generation.utils import GenerationMixin

logger = logging.getLogger(__name__)

def _resolve_torch_dtype(value, default=None):
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.lower().replace("torch.", "")
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
            "double": torch.float64,
        }
        if key in mapping:
            return mapping[key]
    raise ValueError(f"Unsupported dtype value: {value}")

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
                 model_dtype="float32",
                 memory_dtype=None,
                 wrap_layers=None,
                 attn_implementation=None,
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
        self.sliding_window_enabled = sliding_window
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
        self.model_dtype = model_dtype
        self.memory_dtype = memory_dtype
        self.wrap_layers = wrap_layers
        self.attn_implementation = attn_implementation
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
    # Handle tuples of tensors (like position_embeddings which are (cos, sin))
    if isinstance(t, tuple):
        return tuple(segment_tensor(item, start_idx, end_idx, seq_len) for item in t)
    if not isinstance(t, torch.Tensor):
        return t
    # Handle 1D tensors (like cache_position)
    if t.dim() == 1 and t.size(0) == seq_len:
        return t[start_idx:end_idx]
    # common cases: (bsz, seq_len, ...), (bsz, seq_len), (seq_len, ...)
    if t.dim() >= 2 and t.size(1) == seq_len:
        return t[:, start_idx:end_idx, ...]
    return t

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
        attn_implementation: str = "flash_attention_2",
        memory_dtype: Optional[torch.dtype] = None,
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
        self.sliding_window_enabled = bool(sliding_window)
        self.attn_implementation = attn_implementation

        # DPFP feature map dimensions
        nu = 3
        self.d_key = 2 * nu * d_mem

        assert self.d_mem % n_heads == 0 and self.d_model % n_heads == 0

        # Match the dtype to the wrapped layer
        layer_dtype = next(self.layer.parameters()).dtype
        self.memory_dtype = memory_dtype if memory_dtype is not None else layer_dtype

        # Readout/query/key/value projections for associative memory
        self.W_mq = nn.Linear(d_model, d_mem, bias=False, dtype=self.memory_dtype)
        self.W_mk = nn.Linear(d_model, d_mem, bias=False, dtype=self.memory_dtype)
        self.W_mv = nn.Linear(d_model, d_model, bias=False, dtype=self.memory_dtype)
        if gating:
            self.W_mb = nn.Linear(d_model, d_model, dtype=self.memory_dtype)
        else:
            self.W_mb = nn.Linear(d_model, n_heads, dtype=self.memory_dtype)
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

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped layer if not found in wrapper."""
        # First let nn.Module try to find the attribute (in _modules, _parameters, _buffers)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Then delegate to the wrapped layer
        try:
            layer = super().__getattr__('layer')
            return getattr(layer, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
        # Match old SW behavior: positions are local to each segment.
        seg_len = end - start
        return torch.arange(0, seg_len, device=device).long().unsqueeze(0)


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
            # Make memory→memory attention causal (matching SW behavior)
            mem_causal = torch.tril(torch.ones(self.num_mem_tokens, self.num_mem_tokens, dtype=dtype, device=attention_mask.device))
            mask[..., -self.num_mem_tokens:, -self.num_mem_tokens:] = mem_causal
        elif len(shape) == 2:
            shape[-1] += self.num_mem_tokens + int(self.use_sink)
            mask = torch.ones(*shape, dtype=dtype).to(attention_mask.device)
            mask[..., int(self.use_sink):-self.num_mem_tokens] = attention_mask
        else:
            raise ValueError("Attention mask must be 2D or 4D")
        return mask.to(dtype)

    def pad_prev_seg_attn_mask(self, prev_seg_attn_mask: torch.Tensor, dtype: torch.dtype):
        if self.num_mem_tokens in {0, None} and not self.use_sink:
            return prev_seg_attn_mask
        shape = list(prev_seg_attn_mask.shape)
        if len(shape) == 4:
            shape[-2] += self.num_mem_tokens + int(self.use_sink)
            mask = torch.ones(*shape, dtype=dtype, device=prev_seg_attn_mask.device)
            mask[..., int(self.use_sink):-self.num_mem_tokens, :] = prev_seg_attn_mask
            if self.use_sink:
                mask[..., 0, :] = 0
            return mask.to(dtype)
        return prev_seg_attn_mask.to(dtype)


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
    def _alloc_initial_mem(self, device: torch.device, dtype: Optional[torch.dtype] = None):
        dtype = self.memory_dtype if dtype is None else dtype
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
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.memory_dtype)
        q = self._to_heads(self.W_mq(hidden_states))
        mq = self.phi(q)
        mq = F.normalize(mq, dim=-1, p=2.0)
        num = torch.einsum("ihjk,ihkt->ihjt", mq, W_mem)
        if self.use_denom and z is not None:
            denom = torch.einsum("ihk,ihjk->ihj", z, mq)[..., None] + 1e-5
            hs = num / denom
        else:
            hs = num
        return self._from_heads(hs).to(original_dtype)

    def _update_mem_with_mem(
        self,
        mem_tokens: torch.Tensor,
        W_mem: torch.Tensor,
        z: Optional[torch.Tensor],
        first_seg: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        mem_tokens = mem_tokens.to(self.memory_dtype)
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

        return self.forward_horizontal(hidden_states, attention_mask, **kwargs)
    
    # ----- main forward (inner-loop segmentation) -----
    def forward_horizontal(self, hidden_states: torch.Tensor, attention_mask=None, *args, **kwargs):
        assert not self.generate_mode, "Generate mode is not supported for horizontal forward"
        if self.sliding_window_enabled:
            assert attention_mask is None or attention_mask.dim() == 4, "Attention mask must be 4D when sliding window is enabled"
        using_cache = not is_empty_past_key_values(kwargs.get("past_key_values"), self.info['layer'])
        assert not using_cache or (kwargs.get('past_attn_mask') is not None and kwargs.get('past_attn_mask').shape[-1] == self.segment_size), "When using cache, past_attn_mask must be provided and have the same length as the segment size"

        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        bsz, seq_len, _ = hidden_states.shape

        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
            if self.sliding_window_enabled or self.attn_implementation != "flash_attention_2":
                attention_mask = attn_mask_to_4d(attention_mask, upper=False, query_len=seq_len)
                attention_mask = invert_attn_mask(attention_mask, hidden_states.dtype)
        out_full = []

        # Initialize associative memory from persisted state if available
        if self.memory_state is not None:
            W_mem, z = self.memory_state
            W_mem = W_mem.to(device=hidden_states.device, dtype=self.memory_dtype)
            if z is not None:
                z = z.to(device=hidden_states.device, dtype=self.memory_dtype)
            first_seg = False
        else:
            W_mem, z = self._alloc_initial_mem(hidden_states.device, self.memory_dtype)
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

            if attention_mask.dim() == 4:
                attn_mask = attention_mask[:, :, real_start:real_end, real_start:real_end]
            else:
                attn_mask = attention_mask[:, real_start:real_end]

            # print("attn_mask", attn_mask[0][0])

            # Check if this is the last segment and we're in generate mode
            is_last_segment = (end >= seq_len)


            if not first_seg:
                assoc = self._associate_with_mem(seg_aug, W_mem, z)
                seg_aug = assoc + seg_aug

            # Build attention mask for this augmented segment
            seg_aug_len = seg_aug.size(1)
            
            if self.sliding_window_enabled:
                # print(attn_mask.shape, "attn_mask", "*"*100)
                # print(base_cur4d.shape, "base_cur4d", "*"*100)
                base_cur4d = reverse_invert_attn_mask(attn_mask)
                seg_mask = self.pad_attention_mask(base_cur4d, dtype=seg_aug.dtype)
                seg_mask = invert_attn_mask(seg_mask, seg_aug.dtype)

                if past_attn_mask is not None:

                    base_past4d = attn_mask_to_4d(attn_mask_to_2d(past_attn_mask), upper=True, query_len=seg_len)
                    base_past4d = self.pad_prev_seg_attn_mask(base_past4d, dtype=seg_aug.dtype)
                    base_past4d = invert_attn_mask(base_past4d, seg_aug.dtype)

                    # print(base_past4d.shape, "base_past4d", "*"*100)
                    # print(seg_mask.shape, "seg_mask", "*"*100)
                    seg_mask = torch.cat([base_past4d, seg_mask], dim=-1)
                if os.environ.get("ARMT_DEBUG_SW"):
                    print(f"[H-SEG] L{self.info['layer']} seg_len={seg_len} seg_aug_len={seg_aug_len} mask={tuple(seg_mask.shape)}")
            else:
                if attn_mask.dim() == 4:
                    if self.attn_implementation == "flash_attention_2":
                        attn_mask = attn_mask_to_2d(attn_mask)
                        seg_mask = self.pad_attention_mask(attn_mask, dtype=seg_aug.dtype)
                    else:
                        base_cur4d = reverse_invert_attn_mask(attn_mask)
                        seg_mask = self.pad_attention_mask(base_cur4d, dtype=seg_aug.dtype)
                        seg_mask = invert_attn_mask(seg_mask, seg_aug.dtype)
                else:
                    seg_mask = self.pad_attention_mask(attn_mask, dtype=seg_aug.dtype)
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
            seg_kwargs["use_cache"] = self.sliding_window_enabled
            
            if self.sliding_window_enabled:
                seg_kwargs["past_key_values"] = past_key_values
            else:
                # When not using sliding window, clear cache-related params to avoid shape mismatches
                seg_kwargs["past_key_values"] = None
                seg_kwargs["use_cache"] = False

            if self._rotary_fn is not None and seg_pos_ids is not None:
                cos, sin = self._rotary_fn(seg_aug, seg_pos_ids)
                seg_kwargs["position_embeddings"] = (cos, sin)

            layer_out = self.layer(seg_aug, *seg_args, **seg_kwargs)
            if self.sliding_window_enabled:
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
        else:
            legacy = past_key_values
        
        # Keep only the most recent real tokens within the window size
        k, v = legacy[self.info['layer']]
        k = k[..., -window_size-self.num_mem_tokens:-self.num_mem_tokens, :]
        v = v[..., -window_size-self.num_mem_tokens:-self.num_mem_tokens, :]
        
        past_key_values.layers[self.info['layer']].keys = k
        past_key_values.layers[self.info['layer']].values = v
        return past_key_values


class InnerLoopARMTForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Drop-in ARMT model that installs InnerLoopAssociativeLayerWrapper into a base
    HF Causal LM. All segmentation happens inside each wrapped layer; no outer
    recurrent driver is needed.
    """

    # Reuse the config used by the outer-loop variant for parity
    config_class = ARMTConfig

    def __init__(self, config: ARMTConfig, **kwargs):
        global LIGER_KERNEL_AVAILABLE
        super().__init__(config, **kwargs)
        from transformers import AutoConfig, AutoModelForCausalLM

        # Resolve base model from either provided name or config
        base_model = None
        bm_cfg = getattr(config, "base_model_config", None)
        bm_name = getattr(config, "base_model_name", None)

        if bm_name is None or 'llama' not in bm_name.lower():
            LIGER_KERNEL_AVAILABLE = False
            os.environ["ARMT_DISABLE_LIGER_KERNEL"] = "1"
        if os.environ.get("ARMT_DISABLE_LIGER_KERNEL"):
            LIGER_KERNEL_AVAILABLE = False
        if LIGER_KERNEL_AVAILABLE:
            apply_liger_kernel_to_llama()

        if bm_cfg is not None and bm_name is not None:
            raise ValueError("Exactly one of `base_model_name` or `base_model_config` must be provided in config.")
        model_dtype = _resolve_torch_dtype(getattr(config, "model_dtype", "float32"), default=torch.float32)
        memory_dtype = _resolve_torch_dtype(getattr(config, "memory_dtype", None), default=model_dtype)
        attn_implementation = getattr(config, "attn_implementation", None)
        if attn_implementation is None:
            if getattr(config, "sliding_window_enabled", False):
                attn_implementation = "eager"
            else:
                attn_implementation = "flash_attention_2"
        if attn_implementation == "flash_attention_2" and getattr(config, "sliding_window_enabled", False):
            attn_implementation = "eager"
            warnings.warn("Flash attention 2 is not supported for sliding window attention. Using eager instead.")

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
            base_model = base_model.to(dtype=model_dtype)
        elif bm_name is not None:
            from transformers import AutoModelForCausalLM as HF_AutoModelForCausalLM
            if attn_implementation == "flash_attention_2":
                assert model_dtype == "bfloat16" or model_dtype == "bf16" or model_dtype == "torch.bfloat16" or model_dtype == torch.bfloat16, f"Model dtype {model_dtype} is not supported for flash attention 2"
            base_model = HF_AutoModelForCausalLM.from_pretrained(
                bm_name,
                torch_dtype=model_dtype,
                attn_implementation=attn_implementation
            )
        else:
            raise ValueError("InnerLoopARMTForCausalLM requires either `base_model_config` or `base_model_name` in the config.")


        # Install wrappers
        self.model = base_model

        # Extract hyperparameters (fall back to sane defaults if missing)
        self.num_mem_tokens = int(getattr(config, "num_mem_tokens", 0) or 0)
        self.d_mem = int(getattr(config, "d_mem", 64))
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
        self.sliding_window_enabled = bool(getattr(config, "sliding_window_enabled", False))
        self.model_dtype = model_dtype
        self.memory_dtype = memory_dtype
        self._attn_implementation = attn_implementation

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
            (self.num_mem_tokens, memory_dim), device=emb.weight.device, dtype=self.memory_dtype
        )
        # torch.nn.init.normal_(memory_weights, mean=0.0, std=emb_std32.to(memory_weights.dtype))
        torch.nn.init.normal_(memory_weights, mean=0.0, std=0.02)
        self.memory = nn.Parameter(memory_weights, requires_grad=True)
        if self.use_sink:
            self.sink = nn.Parameter(
                torch.randn((1, memory_dim), device=emb.weight.device, dtype=self.memory_dtype), requires_grad=True
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
        # Only use external rotary_fn for models that need it (Llama, GPT-NeoX)
        # Gemma3 handles rotary embeddings internally via position_ids
        model_type = getattr(self.model.config, "model_type", "").lower()
        if "gemma" not in model_type:
            if hasattr(self.model, "model") and hasattr(self.model.model, "rotary_emb"):
                rotary_fn = self.model.model.rotary_emb
            elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "rotary_emb"):
                rotary_fn = self.model.gpt_neox.rotary_emb

        for i in range(len(layers)):
            if self.wrap_layers[i]:
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
                    sliding_window=self.sliding_window_enabled,
                    attn_implementation=self._attn_implementation,
                    memory_dtype=self.memory_dtype,
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

        self.vertical_mode = False

    # ----- control helpers -----
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        logger.info("[ARMT] Gradient checkpointing enabled on base model")

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

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

    def augment_sequence(self, hidden_states: torch.Tensor, mem: torch.Tensor, sink: torch.Tensor = None):
        segments = torch.split(hidden_states, self.segment_size, dim=1)
        if sink is not None:
            augmented_segments = [torch.cat([sink.to(segment.dtype).to(segment.device), segment, mem.to(segment.dtype).to(segment.device)], dim=1) for segment in segments]
        else:
            augmented_segments = [torch.cat([segment, mem.to(segment.dtype).to(segment.device)], dim=1) for segment in segments]
        augmented_sequence = torch.cat(augmented_segments, dim=1)

        return augmented_sequence

    def clean_sequence(self, hidden_states: torch.Tensor):
        augmented_segments = torch.split(hidden_states, self.segment_size+self.num_mem_tokens+int(self.use_sink), dim=1)
        segments = [segment[:, int(self.use_sink):-self.num_mem_tokens] for segment in augmented_segments]
        return torch.cat(segments, dim=1)

    def augment_attention_mask(self, attention_mask: torch.Tensor):
        segments = torch.split(attention_mask, self.segment_size, dim=1)
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

    def augment_labels(self, labels):
        if labels is None:
            return None
        first = labels[:, :1]

        # add -100 token to ensure the correct splitting
        labels = torch.cat([labels, -100 * torch.ones_like(first)], dim=1)

        segments = torch.split(labels[:, 1:], self.segment_size, dim=1)
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
        mem = self.memory.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        sink = self.sink.unsqueeze(0).expand(hidden_states.size(0), -1, -1) if self.use_sink else None

        augmented_hidden_states = self.augment_sequence(hidden_states, mem, sink)
        augmented_attention_mask = self.augment_attention_mask(attention_mask)
        augmented_labels = self.augment_labels(labels)
        return augmented_hidden_states, augmented_attention_mask, augmented_labels

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
        return_dict=True,
    ):  
        if labels_mask is not None:
            assert labels_mask.any(), "labels_mask must not be all zeros"

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], device=input_ids.device, dtype=input_ids.dtype)
            else:
                attention_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        
        if self.vertical_mode:
            return self.forward_vertical(
                input_ids=input_ids,
                labels=labels,
                labels_mask=labels_mask,
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
                labels=labels,
                labels_mask=labels_mask,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                output_only_last_segment=output_only_last_segment,
                num_items_in_batch=num_items_in_batch,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
    def forward_vertical(
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
        past_attn_mask=None,
        reset_memory=True,
        return_internal_state=False,
    ):  
        if reset_memory:
            self.zero_mem()
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

        augmented_hidden_states, augmented_attention_mask, augmented_labels = self.augment(input_ids, inputs_embeds, attention_mask, labels)

        # Helper to split tensors into segments
        def split_tensor(tensor: torch.Tensor, segment_size: int):
            return torch.split(tensor, segment_size+self.num_mem_tokens+int(self.use_sink), dim=1)

        # Build segmented inputs
        # Split all provided tensors consistently
        seg_inputs_embeds = split_tensor(augmented_hidden_states, self.segment_size)
        seg_attention_mask = split_tensor(augmented_attention_mask, self.segment_size) if attention_mask is not None else None
        seg_labels = split_tensor(augmented_labels, self.segment_size) if labels is not None else None
        # Assemble list of per-segment dicts
        num_segments = len(seg_inputs_embeds) 
        segments = []
        for i in range(num_segments):
            segments.append({
                "inputs_embeds": seg_inputs_embeds[i],
                "attention_mask": None if seg_attention_mask is None else seg_attention_mask[i],
                "labels": None if seg_labels is None else seg_labels[i],
            })

        # Sliding window state across segments
        use_sliding = bool(self.sliding_window_enabled)
        shared_cache = past_key_values if (use_sliding and past_key_values is not None) else (DynamicCache() if use_sliding else None)
        past_attn_mask = past_attn_mask if use_sliding else None
        # Absolute positions across segments
        pos_offset = 0

        # Run each segment through the base model; per-layer memory persists inside wrappers
        seg_outputs = []
        layers = self.get_layers()
        for seg in segments:
            seg_len = seg["inputs_embeds"].size(1)
            if seg.get("attention_mask") is None:
                base_2d = torch.ones(B, seg_len, device=device, dtype=dtype)
            else:
                base_2d = seg["attention_mask"]
            if use_sliding:
                cur4d = attn_mask_to_4d(base_2d, upper=False, query_len=seg_len)
                cur4d = invert_attn_mask(cur4d, dtype=dtype)
            else:
                cur4d = base_2d

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
        result["logits"] = self.clean_sequence(full_logits)

        # Compute loss similar to outer wrapper
        if labels is not None:
            labels = labels[:, -full_logits.size(1):]
            shift_labels = labels[..., 1:].contiguous()
            if labels_mask is not None:
                if not isinstance(labels_mask, torch.Tensor):
                    raise ValueError("labels_mask must be a torch.Tensor")
                labels_mask = labels_mask[:, -full_logits.size(1):]
                shift_mask = labels_mask[..., :-1].contiguous()
                shift_mask = shift_mask.bool() if shift_mask.dtype != torch.bool else shift_mask
                shift_labels = shift_labels.masked_fill(~shift_mask, -100)
            flat_labels = shift_labels.view(-1)

            # if labels_mask is not None:
            #     labels_mask = labels_mask[:, -full_logits.size(1):]
            #     shift_mask = labels_mask[..., :-1].contiguous()
            # else:
            #     shift_mask = None

            shift_logits = result['logits'][..., :-1, :].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            # if shift_mask is not None:
            #     flat_logits = flat_logits[shift_mask.view(-1)]
            #     flat_labels = flat_labels[shift_mask.view(-1)]
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

        if return_internal_state:
            result["_past_key_values"] = shared_cache
            result["_past_attn_mask"] = past_attn_mask

        return result
    
    # ----- hf api -----
    def forward_horizontal(
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
        augmented_hidden_states, augmented_attention_mask, augmented_labels = self.augment(input_ids, inputs_embeds, attention_mask, labels)

        if LIGER_KERNEL_AVAILABLE and labels is not None:
            if labels_mask is not None:
                if not isinstance(labels_mask, torch.Tensor):
                    raise ValueError("labels_mask must be a torch.Tensor")
                effective_labels = labels.clone()
                mask_bool = labels_mask.bool() if labels_mask.dtype != torch.bool else labels_mask
                effective_labels[..., 1:] = effective_labels[..., 1:].masked_fill(~mask_bool[..., :-1], -100)
                augmented_labels = self.augment_labels(effective_labels)

            out = self.model(
                labels=augmented_labels,
                inputs_embeds=augmented_hidden_states,
                attention_mask=augmented_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        else:
            out = self.model(
                labels=None,
                inputs_embeds=augmented_hidden_states,
                attention_mask=augmented_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
            out.logits = self.clean_sequence(out.logits)
            logits_for_loss = out.logits
            if labels is not None:
                labels = labels[:, -logits_for_loss.size(1):]
                shift_labels = labels[..., 1:].contiguous()
                if labels_mask is not None:
                    if not isinstance(labels_mask, torch.Tensor):
                        raise ValueError("labels_mask must be a torch.Tensor")
                    labels_mask = labels_mask[:, -logits_for_loss.size(1):]
                    shift_mask = labels_mask[..., :-1].contiguous()
                    shift_mask = shift_mask.bool() if shift_mask.dtype != torch.bool else shift_mask
                    shift_labels = shift_labels.masked_fill(~shift_mask, -100)
                flat_labels = shift_labels.view(-1)
                shift_logits = logits_for_loss[..., :-1, :].contiguous()
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                loss_fct = CrossEntropyLoss(reduction='sum')
                loss = loss_fct(flat_logits, flat_labels)
                denom = (flat_labels != -100).sum()
                denom = torch.clamp(denom, min=1)
                out['loss'] = loss / denom
        self.zero_mem()
        return out

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        """
        Generate tokens using the inner-loop model with proper sliding window attention.
        This method should produce the same logits as the forward method for alignment.
        """

        if self.sliding_window_enabled:
            return self._generate_sliding_window(input_ids, attention_mask, **generate_kwargs)
        else:
            return self._generate_non_sliding_window(input_ids, attention_mask, **generate_kwargs)
    
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
                next_token_logits = outputs['logits'][:, -1, :]
            
            # Get next token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            if generated_ids is not None:
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            else:
                generated_ids = next_token_id
            
            # Store the logits that were actually used to generate the next token
            if return_logits:
                all_logits.append(next_token_logits)
            
            # Check for EOS (handle both single int and list of eos token ids)
            if eos_token_id is not None:
                if isinstance(eos_token_id, list):
                    is_eos = any((next_token_id == eid).all() for eid in eos_token_id)
                else:
                    is_eos = (next_token_id == eos_token_id).all()
                if is_eos:
                    break
        
        if return_logits:
            # Return the logits that were actually used for generation during the loop
            return generated_ids, torch.stack(all_logits, dim=1)
        else:
            return generated_ids

    def _clone_memory_states(self):
        states = []
        for i, layer in enumerate(self.get_layers()):
            if not self.wrap_layers[i]:
                states.append(None)
                continue
            state = layer.memory_state
            if state is None:
                states.append(None)
            else:
                W_mem, z = state
                states.append((W_mem.clone(), None if z is None else z.clone()))
        return states

    def _set_memory_states(self, states):
        for i, layer in enumerate(self.get_layers()):
            if not self.wrap_layers[i]:
                continue
            state = None if (states is None or i >= len(states)) else states[i]
            if state is None:
                layer.memory_state = None
            else:
                W_mem, z = state
                layer.memory_state = (W_mem.clone(), None if z is None else z.clone())

    def _clone_cache(self, past_key_values):
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "to_legacy_cache"):
            return DynamicCache.from_legacy_cache(past_key_values.to_legacy_cache())
        if isinstance(past_key_values, (tuple, list)):
            return DynamicCache.from_legacy_cache(past_key_values)
        return past_key_values

    def _is_eos_hit(self, next_token_id, eos_token_id):
        if eos_token_id is None:
            return False
        if isinstance(eos_token_id, (list, tuple, set)):
            return any((next_token_id == eid).all() for eid in eos_token_id)
        return (next_token_id == eos_token_id).all()

    def _generate_non_sliding_window(self, input_ids, attention_mask=None, **generate_kwargs):
        """
        Efficient non-sliding generation:
        - process completed segments once to build committed memory state;
        - decode by recomputing only the active segment from committed state.
        """
        max_new_tokens = int(generate_kwargs.get("max_new_tokens", 1))
        eos_token_id = generate_kwargs.get("eos_token_id", None)
        return_logits = bool(generate_kwargs.get("return_logits", False))

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        batch_size = input_ids.size(0)
        device = input_ids.device
        generated_ids = torch.empty(batch_size, 0, dtype=input_ids.dtype, device=device)
        all_logits = []

        input_segments = list(torch.split(input_ids, self.segment_size, dim=1))
        attn_segments = list(torch.split(attention_mask, self.segment_size, dim=1))

        self.zero_mem()
        committed_memory_states = self._clone_memory_states()

        try:
            with torch.no_grad():
                # Process completed prompt segments once.
                for seg_ids, seg_mask in zip(input_segments[:-1], attn_segments[:-1]):
                    _ = self.forward_vertical(
                        input_ids=seg_ids,
                        attention_mask=seg_mask,
                        use_cache=False,
                        reset_memory=False,
                        return_internal_state=False,
                    )
                    committed_memory_states = self._clone_memory_states()

                # Decode on active segment.
                active_input_ids = input_segments[-1].clone()
                active_attention_mask = attn_segments[-1].clone()

                for _ in range(max_new_tokens):
                    # Recompute active segment from committed prefix state.
                    self._set_memory_states(committed_memory_states)
                    seg_out = self.forward_vertical(
                        input_ids=active_input_ids,
                        attention_mask=active_attention_mask,
                        use_cache=False,
                        reset_memory=False,
                        return_internal_state=False,
                    )

                    next_token_logits = seg_out["logits"][:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                    if return_logits:
                        all_logits.append(next_token_logits)

                    if self._is_eos_hit(next_token_id, eos_token_id):
                        break

                    active_input_ids = torch.cat([active_input_ids, next_token_id], dim=-1)
                    active_attention_mask = torch.cat(
                        [active_attention_mask, torch.ones_like(next_token_id, dtype=active_attention_mask.dtype)],
                        dim=-1,
                    )

                    # Segment rollover: commit finished active segment to prefix memory.
                    if active_input_ids.size(1) > self.segment_size:
                        committed_memory_states = self._clone_memory_states()
                        active_input_ids = active_input_ids[:, self.segment_size:]
                        active_attention_mask = active_attention_mask[:, self.segment_size:]

            if return_logits:
                if len(all_logits) == 0:
                    vocab_size = int(getattr(self.model.config, "vocab_size"))
                    empty_logits = torch.empty(
                        batch_size,
                        0,
                        vocab_size,
                        device=device,
                        dtype=next(self.model.parameters()).dtype,
                    )
                    return generated_ids, empty_logits
                return generated_ids, torch.stack(all_logits, dim=1)
            return generated_ids
        finally:
            self.zero_mem()

    def _generate_sliding_window(self, input_ids, attention_mask=None, **generate_kwargs):
        """
        Efficient generation via vertical mode:
        - completed segments are processed once;
        - active segment is recomputed token-by-token from committed prefix state.
        """
        max_new_tokens = int(generate_kwargs.get("max_new_tokens", 1))
        eos_token_id = generate_kwargs.get("eos_token_id", None)
        return_logits = bool(generate_kwargs.get("return_logits", False))

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        batch_size = input_ids.size(0)
        device = input_ids.device
        generated_ids = torch.empty(batch_size, 0, dtype=input_ids.dtype, device=device)
        all_logits = []

        input_segments = list(torch.split(input_ids, self.segment_size, dim=1))
        attn_segments = list(torch.split(attention_mask, self.segment_size, dim=1))

        self.zero_mem()
        committed_memory_states = self._clone_memory_states()
        prefix_cache = None
        prefix_attn_mask = None

        try:
            with torch.no_grad():
                # Process completed prompt segments once.
                for seg_ids, seg_mask in zip(input_segments[:-1], attn_segments[:-1]):
                    seg_out = self.forward_vertical(
                        input_ids=seg_ids,
                        attention_mask=seg_mask,
                        use_cache=True,
                        past_key_values=prefix_cache,
                        past_attn_mask=prefix_attn_mask,
                        reset_memory=False,
                        return_internal_state=True,
                    )
                    prefix_cache = seg_out["_past_key_values"]
                    prefix_attn_mask = seg_out["_past_attn_mask"]
                    committed_memory_states = self._clone_memory_states()

                # Decode on active segment.
                active_input_ids = input_segments[-1].clone()
                active_attention_mask = attn_segments[-1].clone()

                for _ in range(max_new_tokens):
                    # Recompute active segment from committed prefix state.
                    self._set_memory_states(committed_memory_states)
                    run_cache = self._clone_cache(prefix_cache)
                    seg_out = self.forward_vertical(
                        input_ids=active_input_ids,
                        attention_mask=active_attention_mask,
                        use_cache=True,
                        past_key_values=run_cache,
                        past_attn_mask=prefix_attn_mask,
                        reset_memory=False,
                        return_internal_state=True,
                    )

                    next_token_logits = seg_out["logits"][:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                    if return_logits:
                        all_logits.append(next_token_logits)

                    if self._is_eos_hit(next_token_id, eos_token_id):
                        break

                    active_input_ids = torch.cat([active_input_ids, next_token_id], dim=-1)
                    active_attention_mask = torch.cat(
                        [active_attention_mask, torch.ones_like(next_token_id, dtype=active_attention_mask.dtype)],
                        dim=-1,
                    )

                    # Segment rollover: commit finished active segment to prefix state.
                    if active_input_ids.size(1) > self.segment_size:
                        prefix_cache = seg_out["_past_key_values"]
                        prefix_attn_mask = seg_out["_past_attn_mask"]
                        committed_memory_states = self._clone_memory_states()
                        active_input_ids = active_input_ids[:, self.segment_size:]
                        active_attention_mask = active_attention_mask[:, self.segment_size:]

            if return_logits:
                if len(all_logits) == 0:
                    vocab_size = int(getattr(self.model.config, "vocab_size"))
                    empty_logits = torch.empty(
                        batch_size,
                        0,
                        vocab_size,
                        device=device,
                        dtype=next(self.model.parameters()).dtype,
                    )
                    return generated_ids, empty_logits
                return generated_ids, torch.stack(all_logits, dim=1)
            return generated_ids
        finally:
            self.zero_mem()

    def zero_mem(self):
        for layer in self.get_layers():
            layer.zero_mem()

    def detach_mem(self):
        for layer in self.get_layers():
            layer.detach_mem()

    def freeze_mem(self):
        for layer in self.get_layers():
            layer.freeze_mem()
