import os
import unittest

import torch
from transformers import LlamaConfig

os.environ["ARMT_DISABLE_LIGER_KERNEL"] = "1"

from modeling_amt.inner_loop_old import ARMTConfig as LegacyConfig
from modeling_amt.inner_loop_old import InnerLoopARMTForCausalLM as LegacyARMT
from src.armt import ARMTConfig, ARMTForCausalLM
from src.armt_sw import ARMTSlidingWindowConfig, ARMTSlidingWindowForCausalLM


GPT2 = {
    "model_type": "gpt2",
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "n_positions": 96,
    "vocab_size": 97,
}
LLAMA = LlamaConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    vocab_size=97,
)


def build_pair(windowed, base, use_sink=False, correction=False, use_denom=False):
    layers_attr = "transformer.h" if isinstance(base, dict) else "model.layers"
    common = dict(
        base_model_config=base,
        layers_attr=layers_attr,
        num_mem_tokens=2,
        d_mem=8,
        segment_size=5,
        model_dtype="float32",
        memory_dtype="float32",
        attn_implementation="eager",
    )
    torch.manual_seed(11)
    legacy = LegacyARMT(
        LegacyConfig(
            **common,
            sliding_window=windowed,
            use_sink=use_sink,
            correction=correction,
            use_denom=use_denom,
            gating=False,
            n_heads=1,
        )
    ).eval()
    for layer in legacy.get_layers():
        torch.nn.init.normal_(layer.W_mv.weight, std=0.02)

    torch.manual_seed(17)
    if windowed:
        current = ARMTSlidingWindowForCausalLM(
            ARMTSlidingWindowConfig(
                **common, use_sink=use_sink, correction=correction, use_denom=use_denom
            )
        ).eval()
    else:
        current = ARMTForCausalLM(
            ARMTConfig(**common, correction=correction, use_denom=use_denom)
        ).eval()
    current.load_state_dict(legacy.state_dict())
    return legacy, current


class LegacyParityTest(unittest.TestCase):
    @torch.no_grad()
    def test_matches_legacy_fixed_behavior(self):
        cases = [
            (False, GPT2, False, False, False),
            (False, LLAMA, False, False, False),
            (True, GPT2, False, False, False),
            (True, GPT2, True, False, False),
            (True, LLAMA, False, False, False),
            (False, GPT2, False, False, True),
            (False, GPT2, False, True, True),
            (False, LLAMA, False, True, True),
            (True, GPT2, False, True, True),
        ]
        for windowed, base, use_sink, correction, use_denom in cases:
            with self.subTest(
                windowed=windowed,
                model=base.model_type if hasattr(base, "model_type") else "gpt2",
                sink=use_sink,
                correction=correction,
                use_denom=use_denom,
            ):
                legacy, current = build_pair(
                    windowed, base, use_sink, correction, use_denom
                )
                for vertical in (False, True):
                    legacy.vertical_mode = vertical
                    current.vertical_mode = vertical
                    for length in (3, 10, 13):
                        legacy.zero_mem()
                        current.reset_memory()
                        torch.manual_seed(length)
                        input_ids = torch.randint(0, 97, (2, length))
                        labels_mask = torch.ones_like(input_ids, dtype=torch.bool)
                        labels_mask[:, 1::4] = False
                        mask = torch.ones_like(input_ids)
                        expected = legacy(input_ids, labels=input_ids, labels_mask=labels_mask, attention_mask=mask)
                        actual = current(input_ids, labels=input_ids, labels_mask=labels_mask, attention_mask=mask)
                        expected_logits = expected["logits"] if isinstance(expected, dict) else expected.logits
                        expected_loss = expected["loss"] if isinstance(expected, dict) else expected.loss
                        torch.testing.assert_close(actual.logits, expected_logits, atol=2e-6, rtol=2e-6)
                        torch.testing.assert_close(actual.loss, expected_loss, atol=2e-6, rtol=2e-6)

    def test_gradients_match_legacy(self):
        torch.manual_seed(19)
        input_ids = torch.randint(0, 97, (2, 13))
        labels_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels_mask[:, 2::4] = False

        for windowed, use_sink in ((False, False), (True, False), (True, True)):
            with self.subTest(windowed=windowed, use_sink=use_sink):
                legacy, current = build_pair(
                    windowed,
                    GPT2,
                    use_sink=use_sink,
                    correction=True,
                    use_denom=True,
                )
                legacy.train()
                current.train()
                torch.manual_seed(23)
                legacy_loss = legacy(
                    input_ids, labels=input_ids, labels_mask=labels_mask
                ).loss
                torch.manual_seed(23)
                current_loss = current(
                    input_ids, labels=input_ids, labels_mask=labels_mask
                ).loss
                torch.testing.assert_close(current_loss, legacy_loss, atol=2e-6, rtol=2e-6)
                legacy_loss.backward()
                current_loss.backward()

                legacy_parameters = dict(legacy.named_parameters())
                current_parameters = dict(current.named_parameters())
                self.assertEqual(legacy_parameters.keys(), current_parameters.keys())
                for name, reference_parameter in legacy_parameters.items():
                    parameter = current_parameters[name]
                    reference = reference_parameter.grad
                    if reference is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        torch.testing.assert_close(
                            parameter.grad, reference, atol=2e-6, rtol=2e-6, msg=name
                        )
