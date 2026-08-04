import inspect
import tempfile
import unittest

import torch

from src.armt import ARMTConfig, ARMTForCausalLM
from src.armt_sw import ARMTSlidingWindowConfig, ARMTSlidingWindowForCausalLM


BASE = {
    "model_type": "gpt2",
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "n_positions": 128,
    "vocab_size": 97,
}
NEOX = {
    "model_type": "gpt_neox",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 128,
    "vocab_size": 97,
}


def build(windowed=False, use_sink=False, wrap_layers=None):
    common = dict(
        base_model_config=BASE,
        layers_attr="transformer.h",
        num_mem_tokens=2,
        d_mem=8,
        segment_size=5,
        wrap_layers=wrap_layers,
    )
    if windowed:
        model = ARMTSlidingWindowForCausalLM(
            ARMTSlidingWindowConfig(**common, use_sink=use_sink)
        )
    else:
        model = ARMTForCausalLM(ARMTConfig(**common))
    for layer in model.get_layers():
        if layer.associative:
            torch.nn.init.normal_(layer.W_mv.weight, std=0.02)
    return model.eval()


class EquivalenceTest(unittest.TestCase):
    @torch.no_grad()
    def test_horizontal_vertical_and_chunked_vertical_are_equivalent(self):
        cases = ((False, False), (True, False), (True, True))
        for windowed, use_sink in cases:
            for wrap_layers in (None, [False, True], [True, False], [False, False]):
                with self.subTest(windowed=windowed, use_sink=use_sink, wrap_layers=wrap_layers):
                    self._assert_equivalent(windowed, use_sink, wrap_layers)

    def _assert_equivalent(self, windowed, use_sink, wrap_layers):
        torch.manual_seed(23)
        model = build(windowed, use_sink, wrap_layers)
        input_ids = torch.randint(0, 97, (2, 13))
        attention_mask = torch.tensor([[1] * 13, [0, 0, 0] + [1] * 10])

        horizontal = model(
            input_ids, labels=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        self.assertEqual(model.memory_position, 0)
        self.assertTrue(all(layer.memory_state is None for layer in model.get_layers()))

        model.vertical_mode = True
        vertical = model(
            input_ids, labels=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        torch.testing.assert_close(horizontal.logits, vertical.logits, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(horizontal.loss, vertical.loss, atol=2e-6, rtol=2e-6)
        for horizontal_state, vertical_state in zip(horizontal.hidden_states, vertical.hidden_states):
            torch.testing.assert_close(horizontal_state, vertical_state, atol=2e-6, rtol=2e-6)

        model.reset_memory()
        first = model(input_ids[:, :10], attention_mask=attention_mask[:, :10], reset_memory=False)
        second = model(input_ids[:, 10:], attention_mask=attention_mask[:, 10:], reset_memory=False)
        chunked = torch.cat((first.logits, second.logits), dim=1)
        torch.testing.assert_close(vertical.logits, chunked, atol=2e-6, rtol=2e-6)

        for boundaries in ((7, 13), (2, 6, 13)):
            model.reset_memory()
            start = 0
            outputs = []
            for end in boundaries:
                outputs.append(
                    model(
                        input_ids[:, start:end],
                        attention_mask=attention_mask[:, start:end],
                        reset_memory=False,
                    ).logits
                )
                start = end
            torch.testing.assert_close(
                vertical.logits, torch.cat(outputs, dim=1), atol=2e-6, rtol=2e-6
            )

    @torch.no_grad()
    def test_sliding_window_supports_gpt_neox_cache_api(self):
        torch.manual_seed(27)
        model = ARMTSlidingWindowForCausalLM(
            ARMTSlidingWindowConfig(
                base_model_config=NEOX,
                layers_attr="gpt_neox.layers",
                num_mem_tokens=2,
                d_mem=8,
                segment_size=5,
            )
        ).eval()
        input_ids = torch.randint(0, 97, (1, 13))
        horizontal = model(input_ids).logits
        model.vertical_mode = True
        vertical = model(input_ids).logits
        torch.testing.assert_close(horizontal, vertical, atol=2e-6, rtol=2e-6)

    def test_last_segment_vertical_mode_keeps_only_bounded_outputs(self):
        torch.manual_seed(28)
        input_ids = torch.randint(0, 97, (2, 53))
        for model in (build(), build(windowed=True, use_sink=True)):
            with self.subTest(model=type(model).__name__):
                model.vertical_mode = True
                with torch.no_grad():
                    full = model(input_ids, labels=input_ids, output_hidden_states=True)
                embedded_lengths = []
                hook = model.get_input_embeddings().register_forward_pre_hook(
                    lambda _, inputs: embedded_lengths.append(inputs[0].shape[1])
                )
                try:
                    with torch.no_grad():
                        bounded = model(
                            input_ids,
                            labels=input_ids,
                            output_hidden_states=True,
                            output_only_last_segment=True,
                        )
                finally:
                    hook.remove()
                self.assertEqual(bounded.logits.shape[1], model.segment_size)
                self.assertLessEqual(max(embedded_lengths), model.segment_size)
                torch.testing.assert_close(
                    bounded.logits, full.logits[:, -model.segment_size :], atol=2e-6, rtol=2e-6
                )
                torch.testing.assert_close(bounded.loss, full.loss, atol=2e-6, rtol=2e-6)
                self.assertIsNone(bounded.loss.grad_fn)
                for bounded_state, full_state in zip(bounded.hidden_states, full.hidden_states):
                    torch.testing.assert_close(
                        bounded_state,
                        full_state[:, -model.segment_size :],
                        atol=2e-6,
                        rtol=2e-6,
                    )
                for layer in model.get_layers():
                    if layer.memory_state is not None:
                        self.assertIsNone(layer.memory_state[0].grad_fn)
                    if hasattr(layer, "cache_state") and layer.cache_state is not None:
                        self.assertTrue(all(value.grad_fn is None for value in layer.cache_state))

    def test_horizontal_training_keeps_no_recurrent_state(self):
        for windowed in (False, True):
            with self.subTest(windowed=windowed):
                torch.manual_seed(29)
                model = build(windowed=windowed).train()
                input_ids = torch.randint(0, 97, (2, 13))
                output = model(input_ids, labels=input_ids)
                output.loss.backward()
                self.assertTrue(all(layer.memory_state is None for layer in model.get_layers()))
                self.assertTrue(any(layer.W_mv.weight.grad is not None for layer in model.get_layers()))

    def test_last_segment_mode_preserves_vertical_training_gradients(self):
        torch.manual_seed(30)
        input_ids = torch.randint(0, 97, (2, 13))
        for windowed in (False, True):
            with self.subTest(windowed=windowed):
                full = build(windowed=windowed)
                bounded = build(windowed=windowed)
                bounded.load_state_dict(full.state_dict())
                full.vertical_mode = bounded.vertical_mode = True
                full(input_ids, labels=input_ids).loss.backward()
                bounded(
                    input_ids, labels=input_ids, output_only_last_segment=True
                ).loss.backward()
                for (name, parameter), (_, reference) in zip(
                    bounded.named_parameters(), full.named_parameters()
                ):
                    if reference.grad is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        torch.testing.assert_close(
                            parameter.grad, reference.grad, atol=2e-6, rtol=2e-6, msg=name
                        )

    def test_configuration_surface_excludes_removed_options(self):
        removed = {
            "attend_to_previous_input",
            "wrap_pos",
            "correction",
            "use_denom",
            "gating",
            "n_heads",
            "act_on",
            "max_hop",
            "act_type",
            "act_format",
            "noisy_halting",
            "constant_depth",
            "time_penalty",
            "sliding_window",
            "use_sink",
        }
        self.assertTrue(removed.isdisjoint(inspect.signature(ARMTConfig).parameters))
        self.assertNotIn("sliding_window", inspect.signature(ARMTSlidingWindowConfig).parameters)
        self.assertIn("use_sink", inspect.signature(ARMTSlidingWindowConfig).parameters)
        with self.assertRaises(TypeError):
            ARMTConfig(correction=True)
        with self.assertRaises(TypeError):
            ARMTSlidingWindowConfig(sliding_window=True)

    @torch.no_grad()
    def test_hugging_face_save_and_load_round_trip(self):
        input_ids = torch.randint(0, 97, (1, 8))
        for model, model_class in (
            (build(wrap_layers=[False, True]), ARMTForCausalLM),
            (
                build(windowed=True, use_sink=True, wrap_layers=[True, False]),
                ARMTSlidingWindowForCausalLM,
            ),
        ):
            with self.subTest(model=model_class.__name__), tempfile.TemporaryDirectory() as directory:
                expected = model(input_ids).logits
                model.save_pretrained(directory)
                restored = model_class.from_pretrained(directory).eval()
                torch.testing.assert_close(restored(input_ids).logits, expected, atol=2e-6, rtol=2e-6)
