import unittest

import torch

from src.armt import ARMTConfig, ARMTForCausalLM
from src.armt_sw import ARMTSlidingWindowConfig, ARMTSlidingWindowForCausalLM


BACKBONES = {
    "gpt2": (
        {
            "model_type": "gpt2",
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 32,
            "n_positions": 128,
            "vocab_size": 97,
        },
        "transformer.h",
    ),
    "gpt_neox": (
        {
            "model_type": "gpt_neox",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 128,
            "vocab_size": 97,
        },
        "gpt_neox.layers",
    ),
    "llama": (
        {
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "vocab_size": 97,
        },
        "model.layers",
    ),
    "gemma": (
        {
            "model_type": "gemma",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "vocab_size": 97,
        },
        "model.layers",
    ),
}


def build(backbone, windowed=False, use_sink=False):
    base_config, layers_attr = BACKBONES[backbone]
    common = dict(
        base_model_config=base_config,
        layers_attr=layers_attr,
        num_mem_tokens=2,
        d_mem=8,
        segment_size=5,
        attn_implementation="eager",
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


class ArchitectureCompatibilityTest(unittest.TestCase):
    @torch.no_grad()
    def test_horizontal_vertical_chunking_and_generation(self):
        variants = ((False, False), (True, False), (True, True))
        for backbone in BACKBONES:
            for windowed, use_sink in variants:
                with self.subTest(
                    backbone=backbone, windowed=windowed, use_sink=use_sink
                ):
                    torch.manual_seed(101)
                    model = build(backbone, windowed, use_sink)
                    input_ids = torch.randint(0, 97, (1, 13))
                    attention_mask = torch.ones_like(input_ids)

                    horizontal = model(
                        input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    model.vertical_mode = True
                    vertical = model(
                        input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    torch.testing.assert_close(
                        vertical.logits, horizontal.logits, atol=2e-6, rtol=2e-6
                    )
                    for actual, expected in zip(
                        vertical.hidden_states, horizontal.hidden_states
                    ):
                        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

                    model.reset_memory()
                    chunks = []
                    for start, end in ((0, 2), (2, 6), (6, 13)):
                        chunks.append(
                            model(
                                input_ids[:, start:end],
                                attention_mask=attention_mask[:, start:end],
                                reset_memory=False,
                            ).logits
                        )
                    torch.testing.assert_close(
                        torch.cat(chunks, dim=1), vertical.logits, atol=2e-6, rtol=2e-6
                    )

                    prompt = input_ids[:, :7]
                    prompt_mask = attention_mask[:, :7]
                    generated, scores = model.generate(
                        prompt,
                        attention_mask=prompt_mask,
                        max_new_tokens=4,
                        return_logits=True,
                    )
                    full_mask = torch.cat((prompt_mask, torch.ones_like(generated)), dim=1)
                    forward_logits = model(
                        torch.cat((prompt, generated), dim=1), attention_mask=full_mask
                    ).logits[:, 6:10]
                    torch.testing.assert_close(scores, forward_logits, atol=2e-6, rtol=2e-6)
                    self.assertTrue(torch.equal(generated, forward_logits.argmax(dim=-1)))


if __name__ == "__main__":
    unittest.main()
