import unittest

import torch
from transformers import LlamaConfig

from src.armt import ARMTConfig, ARMTForCausalLM
from src.armt_sw import ARMTSlidingWindowConfig, ARMTSlidingWindowForCausalLM


GPT2 = {
    "model_type": "gpt2",
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "n_positions": 128,
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
NEOX = {
    "model_type": "gpt_neox",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 128,
    "vocab_size": 97,
}


class GenerationTest(unittest.TestCase):
    @torch.no_grad()
    def test_generate_uses_forward_logits(self):
        cases = [
            (ARMTForCausalLM, ARMTConfig, GPT2, False),
            (ARMTForCausalLM, ARMTConfig, LLAMA, False),
            (ARMTSlidingWindowForCausalLM, ARMTSlidingWindowConfig, GPT2, False),
            (ARMTSlidingWindowForCausalLM, ARMTSlidingWindowConfig, GPT2, True),
            (ARMTSlidingWindowForCausalLM, ARMTSlidingWindowConfig, LLAMA, False),
            (ARMTSlidingWindowForCausalLM, ARMTSlidingWindowConfig, NEOX, False),
        ]
        for model_class, config_class, base, use_sink in cases:
            with self.subTest(model=model_class.__name__, base=base.model_type if hasattr(base, "model_type") else "gpt2", sink=use_sink):
                torch.manual_seed(31)
                kwargs = dict(base_model_config=base, num_mem_tokens=2, d_mem=8, segment_size=5)
                if isinstance(base, dict):
                    kwargs["layers_attr"] = (
                        "gpt_neox.layers" if base["model_type"] == "gpt_neox" else "transformer.h"
                    )
                if config_class is ARMTSlidingWindowConfig:
                    kwargs["use_sink"] = use_sink
                model = model_class(config_class(**kwargs)).eval()
                for layer in model.get_layers():
                    torch.nn.init.normal_(layer.W_mv.weight, std=0.02)

                prompt = torch.randint(0, 97, (1, 7))
                for attention_mask in (torch.ones_like(prompt), torch.tensor([[0, 0, 1, 1, 1, 1, 1]])):
                    generated, generation_logits = model.generate(
                        prompt, attention_mask=attention_mask, max_new_tokens=8, return_logits=True
                    )
                    full_mask = torch.cat((attention_mask, torch.ones_like(generated)), dim=1)
                    full = model(torch.cat((prompt, generated), dim=1), attention_mask=full_mask).logits
                    forward_logits = full[
                        :, prompt.shape[1] - 1 : prompt.shape[1] - 1 + generated.shape[1]
                    ]

                    torch.testing.assert_close(generation_logits, forward_logits, atol=2e-6, rtol=2e-6)
                    self.assertTrue(torch.equal(generated, forward_logits.argmax(dim=-1)))

    @torch.no_grad()
    def test_batched_eos_uses_padding_for_finished_rows(self):
        torch.manual_seed(37)
        model = ARMTForCausalLM(
            ARMTConfig(
                base_model_config=GPT2,
                layers_attr="transformer.h",
                num_mem_tokens=2,
                d_mem=8,
                segment_size=5,
            )
        ).eval()
        for _ in range(100):
            prompt = torch.randint(0, 97, (2, 7))
            predictions = model(prompt).logits[:, -1].argmax(dim=-1)
            if predictions[0] != predictions[1]:
                break
        else:
            self.fail("Could not construct distinct first-token predictions")

        eos_token_id = int(predictions[0])
        pad_token_id = (eos_token_id + 1) % 97
        generated = model.generate(
            prompt,
            max_new_tokens=4,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        self.assertEqual(int(generated[0, 0]), eos_token_id)
        self.assertTrue((generated[0, 1:] == pad_token_id).all())

    @torch.no_grad()
    def test_greedy_generation_options_are_explicit(self):
        model = ARMTForCausalLM(
            ARMTConfig(
                base_model_config=GPT2,
                layers_attr="transformer.h",
                num_mem_tokens=2,
                d_mem=8,
                segment_size=5,
            )
        ).eval()
        prompt = torch.randint(0, 97, (1, 7))
        by_tokens = model.generate(prompt, max_new_tokens=3)
        by_length = model.generate(prompt, max_length=prompt.shape[1] + 3)
        self.assertTrue(torch.equal(by_tokens, by_length))
        with self.assertRaises(NotImplementedError):
            model.generate(prompt, temperature=0.5)
