"""Associative Recurrent Memory Transformer for causal language models."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import math

def _dtype(value: str | torch.dtype | None, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    aliases = {"fp16": "float16", "half": "float16", "bf16": "bfloat16", "fp32": "float32"}
    name = aliases.get(value.lower().removeprefix("torch."), value.lower().removeprefix("torch."))
    result = getattr(torch, name, None)
    if not isinstance(result, torch.dtype):
        raise ValueError(f"Unsupported dtype: {value}")
    return result


def _dpfp(x: Tensor, nu: int = 3) -> Tensor:
    x = torch.cat((F.relu(x), F.relu(-x)), dim=-1)
    return torch.cat((x,) * nu, dim=-1) * torch.cat(
        tuple(x.roll(i, dims=-1) for i in range(1, nu + 1)), dim=-1
    )


def _slice_sequence(value: Any, start: int, end: int, length: int) -> Any:
    if isinstance(value, tuple):
        return tuple(_slice_sequence(item, start, end, length) for item in value)
    if not isinstance(value, Tensor):
        return value
    if value.ndim == 1 and value.shape[0] == length:
        return value[start:end]
    if value.ndim >= 2 and value.shape[1] == length:
        return value[:, start:end, ...]
    return value


class ARMTConfig(PretrainedConfig):
    model_type = "armt"
    removed_options = {
        "act_format",
        "act_on",
        "act_type",
        "attend_to_previous_input",
        "constant_depth",
        "gating",
        "max_hop",
        "n_heads",
        "noisy_halting",
        "sliding_window",
        "time_penalty",
        "use_sink",
        "wrap_pos",
    }

    def __init__(
        self,
        base_model_name: str | None = None,
        base_model_config: dict | PretrainedConfig | str | None = None,
        num_mem_tokens: int = 16,
        d_mem: int = 512,
        segment_size: int = 512,
        segment_alignment: str = "left",
        layers_attr: str = "model.layers",
        freeze_mem: bool = False,
        correction: bool = True,
        use_denom: bool = True,
        model_dtype: str | torch.dtype = "float32",
        memory_dtype: str | torch.dtype | None = None,
        wrap_layers: list[bool] | None = None,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        attn_implementation = kwargs.pop("base_attn_implementation", attn_implementation)
        obsolete = self.removed_options.intersection(kwargs)
        if obsolete:
            raise TypeError(f"Removed ARMT options: {', '.join(sorted(obsolete))}")
        super().__init__(**kwargs)
        if base_model_name is not None and base_model_config is not None:
            raise ValueError("Only one of base_model_name and base_model_config may be set")
        if isinstance(base_model_config, PretrainedConfig):
            base_model_config = base_model_config.to_dict()
        self.base_model_name = base_model_name
        self.base_model_config = base_model_config
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.segment_size = segment_size
        self.segment_alignment = segment_alignment
        self.layers_attr = layers_attr
        self.freeze_mem = freeze_mem
        self.correction = correction
        self.use_denom = use_denom
        self.model_dtype = str(model_dtype).removeprefix("torch.")
        self.memory_dtype = None if memory_dtype is None else str(memory_dtype).removeprefix("torch.")
        self.wrap_layers = wrap_layers
        self.attn_implementation = attn_implementation
        self.base_attn_implementation = attn_implementation


class AssociativeLayer(nn.Module):
    """Adds segment-local memory tokens and a recurrent associative read/write."""

    def __init__(
        self,
        layer: nn.Module,
        d_model: int,
        d_mem: int,
        num_mem_tokens: int,
        segment_size: int,
        memory_dtype: torch.dtype,
        layer_index: int,
        correction: bool = True,
        use_denom: bool = True,
        rotary_fn=None,
        associative: bool = True,
    ):
        super().__init__()
        self.layer = layer
        self.d_model = d_model
        self.d_key = 6 * d_mem
        self.num_mem_tokens = num_mem_tokens
        self.segment_size = segment_size
        self.memory_dtype = memory_dtype
        self.layer_index = layer_index
        self.correction = correction
        self.use_denom = use_denom
        self.rotary_fn = rotary_fn
        self.associative = associative
        parameters = inspect.signature(layer.forward).parameters
        self.cache_parameter = "layer_past" if "layer_past" in parameters else "past_key_values"

        if associative:
            self.W_mq = nn.Linear(d_model, d_mem, bias=False, dtype=memory_dtype)
            self.W_mk = nn.Linear(d_model, d_mem, bias=False, dtype=memory_dtype)
            self.W_mv = nn.Linear(d_model, d_model, bias=False, dtype=memory_dtype)
            self.W_mb = nn.Linear(d_model, 1, dtype=memory_dtype)

            nn.init.zeros_(self.W_mv.weight)
            s = 1/math.sqrt(d_model)
            torch.nn.init.uniform_(self.W_mq.weight, -s, s)
            torch.nn.init.uniform_(self.W_mk.weight, -s, s)
            torch.nn.init.uniform_(self.W_mb.weight, -s, s)
            nn.init.ones_(self.W_mb.bias)

        self.memory_state: tuple[Tensor, Tensor | None, bool] | None = None
        self.persist_memory = False

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("layer"), name)

    def reset_memory(self) -> None:
        self.memory_state = None

    def clone_state(self):
        if self.memory_state is None:
            return None
        memory, denominator, first = self.memory_state
        return memory.clone(), None if denominator is None else denominator.clone(), first

    def restore_state(self, state) -> None:
        self.memory_state = (
            None
            if state is None
            else (state[0].clone(), None if state[1] is None else state[1].clone(), state[2])
        )

    def detach_memory(self) -> None:
        if self.memory_state is not None:
            memory, denominator, first = self.memory_state
            self.memory_state = (
                memory.detach(),
                None if denominator is None else denominator.detach(),
                first,
            )

    def freeze_memory(self) -> None:
        if not self.associative:
            return
        for projection in (self.W_mq, self.W_mk, self.W_mv, self.W_mb):
            projection.requires_grad_(False)

    def _initial_state(self, hidden_states: Tensor) -> tuple[Tensor | None, Tensor | None, bool]:
        if not self.associative:
            return None, None, True
        batch = hidden_states.shape[0]
        if self.persist_memory and self.memory_state is not None:
            memory, denominator, first = self.memory_state
            if memory.shape[0] != batch:
                raise ValueError("The recurrent batch size changed; reset memory before continuing")
            memory = memory.to(hidden_states.device, self.memory_dtype)
            if denominator is not None:
                denominator = denominator.to(hidden_states.device, self.memory_dtype)
            return memory, denominator, first
        memory = hidden_states.new_zeros(
            (batch, self.d_key, self.d_model), dtype=self.memory_dtype
        )
        denominator = (
            hidden_states.new_zeros((batch, self.d_key), dtype=self.memory_dtype)
            if self.use_denom
            else None
        )
        return memory, denominator, True

    def _read(self, hidden_states: Tensor, memory: Tensor, denominator: Tensor | None) -> Tensor:
        query = F.normalize(_dpfp(self.W_mq(hidden_states.to(self.memory_dtype))), dim=-1)
        value = torch.einsum("blk,bkd->bld", query, memory)
        if denominator is not None:
            value = value / (torch.einsum("bk,blk->bl", denominator, query)[..., None] + 1e-5)
        return value.to(hidden_states.dtype)

    def _write(
        self, tokens: Tensor, memory: Tensor, denominator: Tensor | None, first: bool
    ) -> tuple[Tensor, Tensor | None]:
        tokens = tokens.to(self.memory_dtype)
        key = F.normalize(_dpfp(self.W_mk(tokens)), dim=-1)
        value = self.W_mv(tokens)
        coefficient = 1
        if first:
            previous = torch.zeros_like(value)
        else:
            previous = torch.einsum("bmk,bkd->bmd", key, memory)
            if denominator is not None:
                normalizer = torch.einsum("bk,bmk->bm", denominator, key)[..., None] + 1e-5
                previous = previous / normalizer
                if self.correction:
                    coefficient = torch.clip(
                        1 - normalizer / (torch.linalg.norm(key, dim=-1) ** 2)[..., None],
                        0,
                        1,
                    ).detach()
        gate = torch.sigmoid(self.W_mb(tokens)).squeeze(-1)
        memory = memory + torch.einsum("bmk,bmd,bm->bkd", key, value - previous, gate)
        if denominator is not None:
            denominator = denominator + (coefficient * key).sum(dim=1)
        return memory, denominator

    @staticmethod
    def _segment_mask(mask: Tensor | None, start: int, end: int) -> Tensor | None:
        if mask is None:
            return None
        if mask.ndim == 2:
            return mask[:, start:end]
        if mask.ndim == 4:
            return mask[..., start:end, start:end]
        raise ValueError("Transformer layers must receive a 2D or 4D attention mask")

    def _layer_kwargs(
        self, kwargs: dict[str, Any], start: int, end: int, length: int, segment: Tensor
    ) -> dict[str, Any]:
        result = {key: _slice_sequence(value, start, end, length) for key, value in kwargs.items()}
        result["attention_mask"] = self._segment_mask(kwargs.get("attention_mask"), start, end)
        positions = torch.arange(end - start, device=segment.device).unsqueeze(0)
        result["position_ids"] = positions
        if self.rotary_fn is not None:
            result["position_embeddings"] = self.rotary_fn(segment, positions)
        result.pop("layer_past", None)
        result.pop("past_key_values", None)
        result[self.cache_parameter] = None
        result["use_cache"] = False
        return result

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        self._captured_hidden_input = (
            hidden_states if kwargs.get("output_hidden_states", False) else None
        )
        if args:
            parameters = list(inspect.signature(self.layer.forward).parameters)[1:]
            kwargs.update({name: value for name, value in zip(parameters, args) if name not in kwargs})
        length = hidden_states.shape[1]
        memory, denominator, first = self._initial_state(hidden_states)
        outputs = []
        last_output = None
        width = self.segment_size + self.num_mem_tokens

        for start in range(0, length, width):
            end = min(start + width, length)
            segment = hidden_states[:, start:end]
            if self.associative and not first:
                segment = segment + self._read(segment, memory, denominator)
            last_output = self.layer(segment, **self._layer_kwargs(kwargs, start, end, length, segment))
            transformed = last_output[0] if isinstance(last_output, tuple) else last_output
            if self.associative:
                memory, denominator = self._write(
                    transformed[:, -self.num_mem_tokens :], memory, denominator, first
                )
            first = False
            outputs.append(transformed)

        if self.persist_memory and self.associative:
            self.memory_state = memory, denominator, first
        merged = torch.cat(outputs, dim=1)
        return (merged, *last_output[1:]) if isinstance(last_output, tuple) else merged


class ARMTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ARMTConfig
    base_model_prefix = "model"

    def __init__(self, config: ARMTConfig):
        super().__init__(config)
        if config.segment_alignment != "left":
            raise ValueError("Only left segment alignment is supported")
        if config.num_mem_tokens <= 0 or config.segment_size <= 0 or config.d_mem <= 0:
            raise ValueError("num_mem_tokens, segment_size, and d_mem must be positive")

        model_dtype = _dtype(config.model_dtype, torch.float32)
        self.memory_dtype = _dtype(config.memory_dtype, model_dtype)
        self.model = self._build_base_model(config, model_dtype)
        self.config.base_model_config = self.model.config.to_dict()
        self.config.base_model_name = None
        self.num_mem_tokens = config.num_mem_tokens
        self.segment_size = config.segment_size
        self.layers_attr = config.layers_attr
        model_type = self.model.config.model_type.lower()
        self.rotary_fn = None
        if "gemma" not in model_type:
            backbone = getattr(self.model, "model", None)
            self.rotary_fn = getattr(backbone, "rotary_emb", None)
            if self.rotary_fn is None:
                self.rotary_fn = getattr(getattr(self.model, "gpt_neox", None), "rotary_emb", None)

        embeddings = self.model.get_input_embeddings()
        d_model = embeddings.embedding_dim
        self.memory = nn.Parameter(torch.empty(config.num_mem_tokens, d_model, dtype=self.memory_dtype))
        nn.init.normal_(self.memory, std=0.02)

        layers = self.get_layers()
        self.wrap_layers = config.wrap_layers if config.wrap_layers is not None else [True] * len(layers)
        if len(self.wrap_layers) != len(layers):
            raise ValueError("wrap_layers must have one entry per transformer layer")
        for index, enabled in enumerate(self.wrap_layers):
            layers[index] = self._make_layer(layers[index], d_model, index, associative=enabled)
        if config.freeze_mem:
            self.freeze_memory()
        self.vertical_mode = False
        self.memory_position = 0
        self.pending_embeddings = None
        self.pending_attention_mask = None

    @staticmethod
    def _build_base_model(config: ARMTConfig, model_dtype: torch.dtype) -> nn.Module:
        implementation = config.base_attn_implementation
        if config.base_model_config is not None:
            value = config.base_model_config
            if isinstance(value, str):
                base_config = AutoConfig.from_pretrained(value)
            elif isinstance(value, dict):
                value = dict(value)
                try:
                    model_type = value.pop("model_type")
                except KeyError as error:
                    raise ValueError("base_model_config must include model_type") from error
                base_config = AutoConfig.for_model(model_type, **value)
            else:
                raise TypeError("base_model_config must be a dict or model name/path")
            return AutoModelForCausalLM.from_config(
                base_config, attn_implementation=implementation, dtype=model_dtype
            )
        if config.base_model_name is None:
            raise ValueError("Set base_model_name or base_model_config")
        return AutoModelForCausalLM.from_pretrained(
            config.base_model_name, attn_implementation=implementation, dtype=model_dtype
        )

    def _make_layer(
        self, layer: nn.Module, d_model: int, index: int, associative: bool = True
    ) -> AssociativeLayer:
        return AssociativeLayer(
            layer,
            d_model=d_model,
            d_mem=self.config.d_mem,
            num_mem_tokens=self.num_mem_tokens,
            segment_size=self.segment_size,
            memory_dtype=self.memory_dtype,
            layer_index=index,
            correction=self.config.correction,
            use_denom=self.config.use_denom,
            rotary_fn=self.rotary_fn,
            associative=associative,
        )

    def get_layers(self):
        value = self.model
        for name in self.layers_attr.split("."):
            value = getattr(value, name)
        return value

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, value):
        return self.model.set_output_embeddings(value)

    def reset_memory(self) -> None:
        for layer in self.get_layers():
            layer.reset_memory()
        self.memory_position = 0
        self.pending_embeddings = None
        self.pending_attention_mask = None

    zero_mem = reset_memory

    def detach_memory(self) -> None:
        for layer in self.get_layers():
            layer.detach_memory()

    detach_mem = detach_memory

    def freeze_memory(self) -> None:
        for layer, enabled in zip(self.get_layers(), self.wrap_layers):
            if enabled:
                layer.freeze_memory()

    freeze_mem = freeze_memory

    @contextmanager
    def _memory_mode(self, persistent: bool, reset: bool = False):
        if reset:
            self.reset_memory()
        layers = list(self.get_layers())
        previous = [layer.persist_memory for layer in layers]
        for layer in layers:
            layer.persist_memory = persistent
        try:
            yield
        finally:
            for layer, value in zip(layers, previous):
                layer.persist_memory = value
            if not persistent:
                self.reset_memory()

    def _augment(self, embeddings: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        memory = self.memory.to(embeddings.dtype).unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        embedded, masks = [], []
        for segment, mask in zip(
            embeddings.split(self.segment_size, dim=1), attention_mask.split(self.segment_size, dim=1)
        ):
            embedded.append(torch.cat((segment, memory), dim=1))
            memory_mask = mask.new_ones((mask.shape[0], self.num_mem_tokens))
            masks.append(torch.cat((mask, memory_mask), dim=1))
        return torch.cat(embedded, dim=1), torch.cat(masks, dim=1)

    def _clean(self, value: Tensor) -> Tensor:
        width = self.segment_size + self.num_mem_tokens
        return torch.cat([segment[:, : -self.num_mem_tokens] for segment in value.split(width, dim=1)], dim=1)

    def _captured_hidden_states(self, output):
        inputs = [getattr(layer, "_captured_hidden_input", None) for layer in self.get_layers()]
        if all(state is not None for state in inputs):
            return tuple(self._clean(state) for state in inputs) + (
                self._clean(output.hidden_states[-1]),
            )
        return tuple(self._clean(state) for state in output.hidden_states)

    def _inputs(self, input_ids, inputs_embeds, attention_mask) -> tuple[Tensor, Tensor]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids and inputs_embeds")
        embeddings = self.get_input_embeddings()(input_ids) if input_ids is not None else inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], dtype=torch.long, device=embeddings.device)
        if attention_mask.shape != embeddings.shape[:2]:
            raise ValueError("attention_mask must match the input sequence shape")
        return embeddings, attention_mask

    @staticmethod
    def _loss(logits: Tensor, labels: Tensor | None, labels_mask: Tensor | None, denominator=None):
        if labels is None:
            return None
        targets = labels[:, 1:].contiguous()
        if labels_mask is not None:
            if labels_mask.shape != labels.shape:
                raise ValueError("labels_mask must match labels")
            targets = targets.masked_fill(~labels_mask[:, :-1].bool(), -100)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
        count = denominator if denominator is not None else (targets != -100).sum()
        return loss / torch.as_tensor(count, device=loss.device).clamp_min(1)

    def _output(
        self,
        output,
        logits,
        labels,
        labels_mask,
        num_items_in_batch,
        return_dict,
        last_only=False,
        hidden_states=None,
        loss=None,
        compute_loss=True,
    ):
        if compute_loss:
            loss = self._loss(logits, labels, labels_mask, num_items_in_batch)
        if hidden_states is None and output.hidden_states is not None:
            hidden_states = tuple(self._clean(state) for state in output.hidden_states)
        if last_only:
            logits = logits[:, -self.segment_size :]
            if hidden_states is not None:
                hidden_states = tuple(state[:, -self.segment_size :] for state in hidden_states)
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=None, hidden_states=hidden_states, attentions=output.attentions
        )
        return result if return_dict else result.to_tuple()

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
        reset_memory=True,
    ):
        if output_attentions:
            raise NotImplementedError("Segmented attention outputs are not supported")
        method = self.forward_vertical if self.vertical_mode else self.forward_horizontal
        return method(
            input_ids=input_ids,
            labels=labels,
            labels_mask=labels_mask,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_only_last_segment=output_only_last_segment,
            num_items_in_batch=num_items_in_batch,
            return_dict=return_dict,
            reset_memory=reset_memory,
        )

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
        return_dict=True,
        **_,
    ):
        embeddings, attention_mask = self._inputs(input_ids, inputs_embeds, attention_mask)
        augmented, augmented_mask = self._augment(embeddings, attention_mask)
        with self._memory_mode(False, reset=True):
            output = self.model(
                inputs_embeds=augmented,
                attention_mask=augmented_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=False,
                return_dict=True,
            )
        logits = self._clean(output.logits)
        hidden_states = self._captured_hidden_states(output) if output_hidden_states else None
        return self._output(
            output,
            logits,
            labels,
            labels_mask,
            num_items_in_batch,
            return_dict,
            output_only_last_segment,
            hidden_states,
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
        return_dict=True,
        reset_memory=True,
        **_,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids and inputs_embeds")
        source = input_ids if input_ids is not None else inputs_embeds
        batch, input_length = source.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((batch, input_length), dtype=torch.long, device=source.device)
        if attention_mask.shape != (batch, input_length):
            raise ValueError("attention_mask must match the input sequence shape")
        if reset_memory:
            self.reset_memory()
        pending_embeddings = self.pending_embeddings
        pending_mask = self.pending_attention_mask
        pending_length = 0 if pending_embeddings is None else pending_embeddings.shape[1]
        if pending_embeddings is not None:
            if pending_embeddings.shape[0] != batch:
                raise ValueError("The recurrent batch size changed; reset memory before continuing")
            self.pending_embeddings = None
            self.pending_attention_mask = None

        def embed(start, end):
            if input_ids is not None:
                return self.get_input_embeddings()(input_ids[:, start:end])
            return inputs_embeds[:, start:end]

        def segments():
            offset = 0
            if pending_embeddings is not None:
                take = min(self.segment_size - pending_length, input_length)
                current = embed(0, take)
                yield (
                    torch.cat((pending_embeddings.to(current), current), dim=1),
                    torch.cat((pending_mask.to(attention_mask), attention_mask[:, :take]), dim=1),
                    pending_length,
                )
                offset = take
            while offset < input_length:
                end = min(offset + self.segment_size, input_length)
                yield embed(offset, end), attention_mask[:, offset:end], 0
                offset = end

        outputs, logits = [], []
        last_output = None
        rolling_logits = None
        rolling_hidden_states = None
        collected_hidden_states = []
        loss_sum = None
        loss_count = None
        seen_new_tokens = 0
        position = self.memory_position
        track_gradients = torch.is_grad_enabled()
        accepts_positions = "position_ids" in inspect.signature(self.model.forward).parameters
        with self._memory_mode(True):
            for segment, mask, skip in segments():
                partial = segment.shape[1] < self.segment_size
                if partial:
                    committed_states = [layer.clone_state() for layer in self.get_layers()]
                    committed_position = position
                augmented, augmented_mask = self._augment(segment, mask)
                kwargs = {}
                if accepts_positions:
                    kwargs["position_ids"] = torch.arange(
                        position, position + augmented.shape[1], device=augmented.device
                    ).unsqueeze(0)
                output = self.model(
                    inputs_embeds=augmented,
                    attention_mask=augmented_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    use_cache=False,
                    return_dict=True,
                    **kwargs,
                )
                cleaned_logits = self._clean(output.logits)
                new_logits = cleaned_logits[:, skip:]
                seen_new_tokens += cleaned_logits.shape[1] - skip
                last_output = output
                if output_only_last_segment:
                    rolling_logits = (
                        new_logits
                        if rolling_logits is None
                        else torch.cat((rolling_logits, new_logits), dim=1)[:, -self.segment_size :]
                    )
                    if output_hidden_states:
                        new_hidden_states = tuple(
                            state[:, skip:] for state in self._captured_hidden_states(output)
                        )
                        rolling_hidden_states = (
                            new_hidden_states
                            if rolling_hidden_states is None
                            else tuple(
                                torch.cat((previous, current), dim=1)[:, -self.segment_size :]
                                for previous, current in zip(
                                    rolling_hidden_states,
                                    new_hidden_states,
                                )
                            )
                        )
                    if labels is not None:
                        prediction_start = seen_new_tokens - new_logits.shape[1]
                        prediction_count = min(
                            new_logits.shape[1], labels.shape[1] - 1 - prediction_start
                        )
                        if prediction_count > 0:
                            targets = labels[
                                :, prediction_start + 1 : prediction_start + 1 + prediction_count
                            ].contiguous()
                            if labels_mask is not None:
                                target_mask = labels_mask[
                                    :, prediction_start : prediction_start + prediction_count
                                ].bool()
                                targets = targets.masked_fill(~target_mask, -100)
                            part = F.cross_entropy(
                                new_logits[:, :prediction_count].reshape(-1, new_logits.shape[-1]),
                                targets.reshape(-1),
                                reduction="sum",
                            )
                            count = (targets != -100).sum()
                            loss_sum = part if loss_sum is None else loss_sum + part
                            loss_count = count if loss_count is None else loss_count + count
                else:
                    outputs.append(output)
                    logits.append(cleaned_logits)
                    if output_hidden_states:
                        collected_hidden_states.append(self._captured_hidden_states(output))
                position += augmented.shape[1]
                if partial:
                    for layer, state in zip(self.get_layers(), committed_states):
                        layer.restore_state(state)
                    position = committed_position
                    self.pending_embeddings = segment if track_gradients else segment.detach().clone()
                    self.pending_attention_mask = mask.detach()
                self.memory_position = position

        if output_only_last_segment:
            merged_logits = rolling_logits
            merged = last_output
            hidden_states = rolling_hidden_states
            if labels is None:
                loss = None
            elif loss_sum is None:
                loss = merged_logits.sum().detach() * 0
            else:
                denominator = num_items_in_batch if num_items_in_batch is not None else loss_count
                loss = loss_sum / torch.as_tensor(denominator, device=loss_sum.device).clamp_min(1)
        else:
            merged_logits = torch.cat(logits, dim=1)[:, pending_length:]
            merged = outputs[-1]
            hidden_states = None
            if output_hidden_states:
                hidden_states = tuple(
                    torch.cat([states[index] for states in collected_hidden_states], dim=1)[
                        :, pending_length:
                    ]
                    for index in range(len(collected_hidden_states[-1]))
                )
            loss = None
        return self._output(
            merged,
            merged_logits,
            labels,
            labels_mask,
            num_items_in_batch,
            return_dict,
            output_only_last_segment,
            hidden_states,
            loss,
            not output_only_last_segment,
        )

    def _memory_states(self):
        layers = [layer.clone_state() for layer in self.get_layers()]
        pending_embeddings = None if self.pending_embeddings is None else self.pending_embeddings.clone()
        pending_mask = (
            None if self.pending_attention_mask is None else self.pending_attention_mask.clone()
        )
        return layers, self.memory_position, pending_embeddings, pending_mask

    def _restore_memory_states(self, states) -> None:
        states, self.memory_position, pending_embeddings, pending_mask = states
        for layer, state in zip(self.get_layers(), states):
            layer.restore_state(state)
        self.pending_embeddings = None if pending_embeddings is None else pending_embeddings.clone()
        self.pending_attention_mask = None if pending_mask is None else pending_mask.clone()

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=None,
        max_length=None,
        eos_token_id=None,
        pad_token_id=None,
        return_logits=False,
        **kwargs,
    ):
        if kwargs.pop("do_sample", False) or kwargs.pop("num_beams", 1) != 1:
            raise NotImplementedError("ARMT generation currently supports greedy decoding only")
        if kwargs:
            raise NotImplementedError(f"Unsupported generation options: {', '.join(sorted(kwargs))}")
        if max_new_tokens is not None and max_length is not None:
            raise ValueError("Set max_new_tokens or max_length, not both")
        if max_new_tokens is None:
            max_new_tokens = max(0, (max_length or input_ids.shape[1] + 1) - input_ids.shape[1])
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        eos_ids = None
        if eos_token_id is not None:
            eos_ids = torch.as_tensor(eos_token_id, device=input_ids.device).reshape(-1)
            if pad_token_id is None:
                pad_token_id = int(eos_ids[0])
        pad_token_id = 0 if pad_token_id is None else pad_token_id
        generated = input_ids.new_empty((input_ids.shape[0], 0))
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        scores = []
        segments = list(input_ids.split(self.segment_size, dim=1))
        masks = list(attention_mask.split(self.segment_size, dim=1))
        self.reset_memory()
        committed = self._memory_states()
        try:
            with torch.no_grad():
                for segment, mask in zip(segments[:-1], masks[:-1]):
                    self.forward_vertical(segment, attention_mask=mask, reset_memory=False)
                    committed = self._memory_states()
                active, active_mask = segments[-1], masks[-1]
                for _ in range(max_new_tokens):
                    self._restore_memory_states(committed)
                    output = self.forward_vertical(active, attention_mask=active_mask, reset_memory=False)
                    score = output.logits[:, -1]
                    token = score.argmax(dim=-1, keepdim=True)
                    active_rows = ~finished
                    token = torch.where(active_rows[:, None], token, token.new_full((), pad_token_id))
                    generated = torch.cat((generated, token), dim=1)
                    if return_logits:
                        scores.append(score)
                    if eos_ids is not None:
                        finished |= active_rows & (token.squeeze(-1)[:, None] == eos_ids[None, :]).any(dim=1)
                    if finished.all():
                        break
                    active = torch.cat((active, token), dim=1)
                    active_mask = torch.cat((active_mask, active_rows[:, None].to(active_mask.dtype)), dim=1)
                    if active.shape[1] > self.segment_size:
                        committed = self._memory_states()
                        active, active_mask = active[:, self.segment_size :], active_mask[:, self.segment_size :]
        finally:
            self.reset_memory()
        if return_logits:
            vocab = self.model.config.vocab_size
            stacked = torch.stack(scores, dim=1) if scores else self.memory.new_empty((input_ids.shape[0], 0, vocab))
            return generated, stacked
        return generated

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()


# Backward-compatible name for existing training scripts.
InnerLoopARMTForCausalLM = ARMTForCausalLM
