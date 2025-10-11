import os
import torch
from typing import Optional

from transformers import PretrainedConfig

from modeling_amt.inner_loop import InnerLoopARMTForCausalLM


def build_model(sliding_window: bool = False) -> InnerLoopARMTForCausalLM:
    cfg = PretrainedConfig()
    # Base tiny GPT-2 config (randomly initialized, no download required)
    cfg.base_model_config = {
        "model_type": "gpt2",
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "vocab_size": 128,
    }
    # ARMT/inner-loop params
    cfg.layers_attr = "transformer.h"  # GPT-2 layers path
    cfg.num_mem_tokens = 2
    cfg.d_mem = 32
    cfg.segment_size = 16
    cfg.correction = True
    cfg.n_heads = 1
    cfg.use_denom = True
    cfg.gating = False
    cfg.freeze_mem = False
    cfg.use_sink = False
    cfg.sliding_window = bool(sliding_window)

    model = InnerLoopARMTForCausalLM(cfg)
    model.eval()
    return model


@torch.no_grad()
def compare_once(model: InnerLoopARMTForCausalLM, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
    # Horizontal (per-layer segmentation)
    model.vertical_mode = False
    out_h = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    logits_h = out_h.logits

    # Vertical (top-level segmentation)
    model.vertical_mode = True
    out_v = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    logits_v = out_v["logits"] if isinstance(out_v, dict) else out_v.logits

    diff = (logits_h - logits_v).abs()
    max_abs_diff = diff.max().item()
    allclose = torch.allclose(logits_h, logits_v, atol=1e-5, rtol=1e-5)
    return max_abs_diff, bool(allclose)


def main():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create random inputs
    bsz, seqlen, vocab_size = 1, 40, 128
    input_ids = torch.randint(0, vocab_size, (bsz, seqlen), device=device)
    attention_mask = None  # Let wrappers build masks; set to ones if desired

    # Without sliding window
    model_nosw = build_model(sliding_window=False).to(device)
    max_diff_nosw, ok_nosw = compare_once(model_nosw, input_ids, attention_mask)
    print(f"No sliding window: allclose={ok_nosw}, max_abs_diff={max_diff_nosw:.6f}")

    # With sliding window
    model_sw = build_model(sliding_window=True).to(device)
    max_diff_sw, ok_sw = compare_once(model_sw, input_ids, attention_mask)
    print(f"With sliding window: allclose={ok_sw}, max_abs_diff={max_diff_sw:.6f}")


if __name__ == "__main__":
    main()


