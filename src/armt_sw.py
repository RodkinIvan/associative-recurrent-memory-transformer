"""Sliding-window ARMT variant."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from transformers import DynamicCache

from .armt import ARMTConfig, ARMTForCausalLM, AssociativeLayer, _slice_sequence


class ARMTSlidingWindowConfig(ARMTConfig):
    model_type = "armt_sw"
    removed_options = ARMTConfig.removed_options - {"use_sink"}

    def __init__(self, use_sink: bool = False, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        super().__init__(**kwargs)
        self.use_sink = use_sink


class SlidingWindowAssociativeLayer(AssociativeLayer):
    def __init__(self, *args, use_sink: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sink = use_sink
        self.cache_state: tuple[Tensor, Tensor] | None = None
        self.past_attention_mask: Tensor | None = None

    def reset_memory(self) -> None:
        super().reset_memory()
        self.cache_state = None
        self.past_attention_mask = None

    def clone_state(self):
        memory = super().clone_state()
        cache = None if self.cache_state is None else tuple(value.clone() for value in self.cache_state)
        mask = None if self.past_attention_mask is None else self.past_attention_mask.clone()
        return memory, cache, mask

    def restore_state(self, state) -> None:
        memory, cache, mask = state
        super().restore_state(memory)
        self.cache_state = None if cache is None else tuple(value.clone() for value in cache)
        self.past_attention_mask = None if mask is None else mask.clone()

    def detach_memory(self) -> None:
        super().detach_memory()
        if self.cache_state is not None:
            self.cache_state = tuple(value.detach() for value in self.cache_state)
        if self.past_attention_mask is not None:
            self.past_attention_mask = self.past_attention_mask.detach()

    @staticmethod
    def _to_2d(mask: Tensor | None, batch: int, length: int, device: torch.device) -> Tensor:
        if mask is None:
            return torch.ones((batch, length), dtype=torch.bool, device=device)
        if mask.ndim == 2:
            return mask.bool()
        if mask.ndim == 4:
            return (mask == 0).any(dim=1).any(dim=-2)
        raise ValueError("Transformer layers must receive a 2D or 4D attention mask")

    def _window_mask(self, current: Tensor, past: Tensor | None, dtype: torch.dtype) -> Tensor:
        batch, length = current.shape
        allowed = torch.tril(torch.ones((length, length), dtype=torch.bool, device=current.device))
        allowed = allowed[None, None] & current[:, None, None, :].bool()
        if past is not None:
            past_length = past.shape[1]
            previous = torch.zeros((batch, 1, length, past_length), dtype=torch.bool, device=current.device)
            real_start = int(self.use_sink)
            real_length = length - real_start - self.num_mem_tokens
            diagonal = torch.triu(
                torch.ones((real_length, past_length), dtype=torch.bool, device=current.device)
            )
            previous[:, :, real_start : real_start + real_length] = (
                diagonal[None, None] & past[:, None, None, :].bool()
            )
            previous[:, :, -self.num_mem_tokens :] = past[:, None, None, :].bool()
            allowed = torch.cat((previous, allowed), dim=-1)
        value = torch.zeros((), dtype=dtype, device=current.device)
        blocked = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=current.device)
        return torch.where(allowed, value, blocked)

    def _cache(self, state, device) -> DynamicCache:
        cache = DynamicCache()
        if state is not None:
            key, value = (tensor.to(device) for tensor in state)
            cache.update(key, value, self.layer_index)
        return cache

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        self._captured_hidden_input = (
            hidden_states if kwargs.get("output_hidden_states", False) else None
        )
        if args:
            import inspect

            parameters = list(inspect.signature(self.layer.forward).parameters)[1:]
            kwargs.update({name: value for name, value in zip(parameters, args) if name not in kwargs})

        batch, length = hidden_states.shape[:2]
        full_mask = self._to_2d(kwargs.get("attention_mask"), batch, length, hidden_states.device)
        memory, denominator, first = self._initial_state(hidden_states)
        cache_state = self.cache_state if self.persist_memory else None
        past_mask = self.past_attention_mask if self.persist_memory else None
        outputs, last_output = [], None
        width = self.segment_size + self.num_mem_tokens + int(self.use_sink)

        for start in range(0, length, width):
            end = min(start + width, length)
            segment = hidden_states[:, start:end]
            current_mask = full_mask[:, start:end]
            if self.associative and not first:
                segment = segment + self._read(segment, memory, denominator)

            cache = self._cache(cache_state, segment.device)
            past_length = 0 if cache_state is None else cache_state[0].shape[-2]
            layer_kwargs: dict[str, Any] = {
                key: _slice_sequence(value, start, end, length) for key, value in kwargs.items()
            }
            layer_kwargs.pop("layer_past", None)
            layer_kwargs.pop("past_key_values", None)
            layer_kwargs.update(
                attention_mask=self._window_mask(current_mask, past_mask, segment.dtype),
                cache_position=torch.arange(past_length, past_length + end - start, device=segment.device),
                use_cache=True,
            )
            layer_kwargs[self.cache_parameter] = cache
            positions = torch.arange(end - start, device=segment.device).unsqueeze(0)
            layer_kwargs["position_ids"] = positions
            if self.rotary_fn is not None:
                layer_kwargs["position_embeddings"] = self.rotary_fn(segment, positions)
            last_output = self.layer(segment, **layer_kwargs)
            transformed = last_output[0] if isinstance(last_output, tuple) else last_output
            if self.associative:
                memory, denominator = self._write(
                    transformed[:, -self.num_mem_tokens :], memory, denominator, first
                )
            first = False
            outputs.append(transformed)

            cached = cache.layers[self.layer_index]
            real_start = past_length + int(self.use_sink)
            real_end = end - start - self.num_mem_tokens + past_length
            key = torch.cat((cached.keys[..., :past_length, :], cached.keys[..., real_start:real_end, :]), dim=-2)
            value = torch.cat((cached.values[..., :past_length, :], cached.values[..., real_start:real_end, :]), dim=-2)
            cache_state = key[..., -self.segment_size :, :], value[..., -self.segment_size :, :]
            real_mask = current_mask[:, int(self.use_sink) : -self.num_mem_tokens]
            past_mask = torch.cat((past_mask, real_mask), dim=1) if past_mask is not None else real_mask
            past_mask = past_mask[:, -self.segment_size :]

        if self.persist_memory:
            if self.associative:
                self.memory_state = memory, denominator, first
            self.cache_state = cache_state
            self.past_attention_mask = past_mask
        merged = torch.cat(outputs, dim=1)
        return (merged, *last_output[1:]) if isinstance(last_output, tuple) else merged


class ARMTSlidingWindowForCausalLM(ARMTForCausalLM):
    config_class = ARMTSlidingWindowConfig

    def __init__(self, config: ARMTSlidingWindowConfig):
        if config.base_attn_implementation != "eager":
            raise ValueError("Sliding-window ARMT requires eager attention")
        self.use_sink = config.use_sink
        super().__init__(config)
        if self.use_sink:
            self.sink = nn.Parameter(torch.empty(1, self.get_input_embeddings().embedding_dim, dtype=self.memory_dtype))
            nn.init.normal_(self.sink, std=0.02)

    def _make_layer(
        self, layer: nn.Module, d_model: int, index: int, associative: bool = True
    ) -> AssociativeLayer:
        return SlidingWindowAssociativeLayer(
            layer,
            d_model=d_model,
            d_mem=self.config.d_mem,
            num_mem_tokens=self.num_mem_tokens,
            segment_size=self.segment_size,
            memory_dtype=self.memory_dtype,
            layer_index=index,
            correction=self.config.correction,
            use_denom=self.config.use_denom,
            use_sink=self.use_sink,
            rotary_fn=self.rotary_fn,
            associative=associative,
        )

    def _augment(self, embeddings: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        memory = self.memory.to(embeddings.dtype).unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        sink = None
        if self.use_sink:
            sink = self.sink.to(embeddings.dtype).unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        embedded, masks = [], []
        for segment, mask in zip(
            embeddings.split(self.segment_size, dim=1), attention_mask.split(self.segment_size, dim=1)
        ):
            parts = (segment, memory) if sink is None else (sink, segment, memory)
            embedded.append(torch.cat(parts, dim=1))
            extra = mask.new_ones((mask.shape[0], self.num_mem_tokens + int(self.use_sink)))
            masks.append(torch.cat((extra[:, : int(self.use_sink)], mask, extra[:, int(self.use_sink) :]), dim=1))
        return torch.cat(embedded, dim=1), torch.cat(masks, dim=1)

    def _clean(self, value: Tensor) -> Tensor:
        width = self.segment_size + self.num_mem_tokens + int(self.use_sink)
        return torch.cat(
            [segment[:, int(self.use_sink) : -self.num_mem_tokens] for segment in value.split(width, dim=1)], dim=1
        )


InnerLoopARMTForCausalLM = ARMTSlidingWindowForCausalLM

__all__ = [
    "ARMTSlidingWindowConfig",
    "ARMTSlidingWindowForCausalLM",
    "InnerLoopARMTForCausalLM",
]
