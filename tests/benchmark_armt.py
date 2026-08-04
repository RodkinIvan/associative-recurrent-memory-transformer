"""Microbenchmark current ARMT against inner_loop_old.py.

Run from the repository root, for example:
    python tests/benchmark_armt.py --iterations 10 --sequence-length 64

Timing is reported rather than asserted because shared runners and CPU frequency
scaling make performance thresholds unsuitable for correctness tests.
"""

import argparse
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ARMT_DISABLE_LIGER_KERNEL", "1")

from modeling_amt.inner_loop_old import ARMTConfig as LegacyConfig
from modeling_amt.inner_loop_old import InnerLoopARMTForCausalLM as LegacyARMT
from src.armt import ARMTConfig, ARMTForCausalLM
from src.armt_sw import ARMTSlidingWindowConfig, ARMTSlidingWindowForCausalLM
from test_armt_architectures import BACKBONES


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median_ms(model, input_ids, mode, warmup, iterations):
    model.vertical_mode = mode == "vertical"
    with torch.inference_mode():
        for _ in range(warmup):
            model.reset_memory() if hasattr(model, "reset_memory") else model.zero_mem()
            model(input_ids)
        samples = []
        for _ in range(iterations):
            model.reset_memory() if hasattr(model, "reset_memory") else model.zero_mem()
            synchronize(input_ids.device)
            start = time.perf_counter()
            model(input_ids)
            synchronize(input_ids.device)
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def build_pair(backbone, windowed, device):
    base_config_dict, layers_attr = BACKBONES[backbone]
    base_config_dict = dict(base_config_dict)
    model_type = base_config_dict.pop("model_type")
    base_config = AutoConfig.for_model(model_type, **base_config_dict)
    common = dict(
        base_model_config=base_config,
        layers_attr=layers_attr,
        num_mem_tokens=2,
        d_mem=8,
        segment_size=8,
        model_dtype="float32",
        memory_dtype="float32",
        attn_implementation="eager",
    )
    torch.manual_seed(211)
    legacy = LegacyARMT(
        LegacyConfig(
            **common,
            sliding_window=windowed,
            use_sink=False,
            correction=False,
            use_denom=False,
            gating=False,
            n_heads=1,
        )
    ).eval()
    for layer in legacy.get_layers():
        torch.nn.init.normal_(layer.W_mv.weight, std=0.02)

    torch.manual_seed(223)
    if windowed:
        current = ARMTSlidingWindowForCausalLM(
            ARMTSlidingWindowConfig(**common, use_sink=False)
        ).eval()
    else:
        current = ARMTForCausalLM(ARMTConfig(**common)).eval()
    current.load_state_dict(legacy.state_dict())
    return legacy.to(device), current.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    rows = []
    skipped = []

    for backbone in BACKBONES:
        for windowed in (False, True):
            legacy, current = build_pair(backbone, windowed, device)
            torch.manual_seed(227)
            input_ids = torch.randint(
                0, 97, (args.batch_size, args.sequence_length), device=device
            )
            timings = {
                ("current", mode): median_ms(
                    current, input_ids, mode, args.warmup, args.iterations
                )
                for mode in ("horizontal", "vertical")
            }
            variant = "sliding" if windowed else "standard"
            for mode in ("horizontal", "vertical"):
                try:
                    with torch.inference_mode():
                        vertical = mode == "vertical"
                        legacy.vertical_mode = current.vertical_mode = vertical
                        legacy.zero_mem()
                        current.reset_memory()
                        expected = legacy(input_ids)
                        actual = current(input_ids)
                        expected_logits = (
                            expected["logits"] if isinstance(expected, dict) else expected.logits
                        )
                        torch.testing.assert_close(
                            actual.logits, expected_logits, atol=2e-6, rtol=2e-6
                        )
                    old = median_ms(
                        legacy, input_ids, mode, args.warmup, args.iterations
                    )
                except (AssertionError, RuntimeError, TypeError, ValueError, IndexError) as error:
                    skipped.append(
                        f"{backbone}/{variant}/{mode}: {type(error).__name__}: {error}"
                    )
                    old = None
                new = timings["current", mode]
                rows.append(
                    (backbone, variant, mode, old, new, None if old is None else old / new)
                )
            horizontal = timings["current", "horizontal"]
            vertical = timings["current", "vertical"]
            rows.append(
                (backbone, variant, "current V/H", horizontal, vertical, horizontal / vertical)
            )

    print(
        f"device={device} batch={args.batch_size} length={args.sequence_length} "
        f"warmup={args.warmup} iterations={args.iterations} threads={args.threads}"
    )
    print("backbone  variant   comparison    baseline_ms  candidate_ms  speedup")
    for backbone, variant, mode, baseline, candidate, speedup in rows:
        baseline_text = "N/A" if baseline is None else f"{baseline:.3f}"
        speedup_text = "N/A" if speedup is None else f"{speedup:.3f}x"
        print(
            f"{backbone:<9} {variant:<9} {mode:<13} "
            f"{baseline_text:>11} {candidate:>13.3f} {speedup_text:>8}"
        )
    if skipped:
        print("\nLegacy combinations skipped because they do not run or match this environment:")
        for reason in skipped:
            print(f"- {reason.splitlines()[0]}")


if __name__ == "__main__":
    main()
