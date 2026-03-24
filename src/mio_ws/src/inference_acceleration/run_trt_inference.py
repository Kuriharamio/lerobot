#!/usr/bin/env python3
"""
Benchmark and validate TRT vs PyTorch inference for LeRobot PI0 / PI05.

Usage:
  # PyTorch baseline only
  python run_trt_inference.py -m pi0  --mode pytorch

  # TRT only
  python run_trt_inference.py -m pi0  --engine ./outputs/pi0/onnx/model_fp16.engine

  # Full comparison (PyTorch then TRT, same batch)
  python run_trt_inference.py -m pi0  --engine ./outputs/pi0/onnx/model_fp16.engine --mode compare
  python run_trt_inference.py -m pi05 --engine ./outputs/pi05/onnx/model_fp8.engine  --mode compare
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# ── path setup ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.dirname(_SCRIPT_DIR)
_MIO_WS_DIR = os.path.dirname(_SRC_DIR)
DEFAULT_OUTPUT_BASE = os.path.join(_MIO_WS_DIR, "outputs")

sys.path.append(_SRC_DIR)
from utils.inference_utils import get_policy
from utils.print_color import print_green, print_yellow, print_red

MODEL_CHOICES    = ["pi0", "pi05"]
DATASET_REPO_ID  = "lerobot/libero"
EPISODE_INDEX    = 0
NUM_WARMUP       = 5
NUM_TEST         = 20


# ── timing ──────────────────────────────────────────────────────────────────────

def bench(policy, batch, amp_ctx, n: int, label: str):
    """Warm inference loop; returns (mean_ms, std_ms, last_actions)."""
    policy.reset()
    times, actions = [], None
    with torch.inference_mode(), amp_ctx:
        for _ in range(n):
            t0 = time.perf_counter()
            actions = policy.select_action(batch)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
            policy.reset()
    arr = np.array(times)
    print_yellow(f"  [{label}]  {arr.mean():.2f} ± {arr.std():.2f} ms  "
                 f"(min {arr.min():.2f} / max {arr.max():.2f})")
    return float(arr.mean()), float(arr.std()), actions


# ── output comparison ────────────────────────────────────────────────────────────

def compare_outputs(pt: torch.Tensor, trt: torch.Tensor):
    a, b = pt.float().cpu().numpy().flatten(), trt.float().cpu().numpy().flatten()
    cos  = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    diff = np.abs(a - b)
    rel  = diff / (np.abs(a) + 1e-6)
    print_green("\n── output comparison ────────────────────────────────────")
    print_yellow(f"  cosine similarity : {cos:.6f}")
    print_yellow(f"  abs diff   mean={diff.mean():.5f}  max={diff.max():.5f}  std={diff.std():.5f}")
    print_yellow(f"  rel error  mean={rel.mean():.5f}  max={rel.max():.5f}")


# ── shared setup ─────────────────────────────────────────────────────────────────

def _load_dataset_batch(policy, model_id, device):
    preprocess, _ = make_pre_post_processors(
        policy.config, model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    dataset  = LeRobotDataset(DATASET_REPO_ID, video_backend="pyav")
    from_idx = dataset.meta.episodes["dataset_from_index"][EPISODE_INDEX]
    frame    = dict(dataset[from_idx])
    return preprocess(frame)


# ── PyTorch benchmark ────────────────────────────────────────────────────────────

def run_pytorch(args, policy, model_id, device):
    print_green("\n── PyTorch inference ────────────────────────────────────")
    amp_ctx = nullcontext()  # no amp for fair comparison with TRT (fp16 weights already loaded)
    batch   = _load_dataset_batch(policy, model_id, device)

    print_green(f"Warmup ({NUM_WARMUP})...")
    with torch.inference_mode(), amp_ctx:
        for _ in range(NUM_WARMUP):
            policy.select_action(batch)
            policy.reset()
    torch.cuda.synchronize()

    print_green(f"Benchmark ({NUM_TEST})...")
    mean_ms, std_ms, actions = bench(policy, batch, amp_ctx, NUM_TEST, "PyTorch")
    return mean_ms, std_ms, actions, batch


# ── TRT benchmark ────────────────────────────────────────────────────────────────

def run_trt(args, policy, model_id, device, batch):
    print_green("\n── TRT inference ────────────────────────────────────────")
    from utils.trt_model_forward import setup_trt_engine
    policy = setup_trt_engine(policy, args.engine, model_type=args.model)

    print_green(f"Warmup ({NUM_WARMUP})...")
    with torch.inference_mode():
        for _ in range(NUM_WARMUP):
            policy.select_action(batch)
            policy.reset()
    torch.cuda.synchronize()

    print_green(f"Benchmark ({NUM_TEST})...")
    mean_ms, std_ms, actions = bench(policy, batch, nullcontext(), NUM_TEST, "TRT")
    return mean_ms, std_ms, actions


# ── main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TRT vs PyTorch inference benchmark")
    parser.add_argument("-m", "--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--engine", type=str, default=None,
                        help="Path to .engine file (required for trt/compare mode)")
    parser.add_argument("--mode", default="compare",
                        choices=["pytorch", "trt", "compare"])
    args = parser.parse_args()

    if args.mode in ("trt", "compare") and not args.engine:
        print_red("Error: --engine required for trt/compare mode.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_green(f"Loading {args.model} (compile=False, amp=False)...")
    policy, model_id = get_policy(model=args.model, compile=False, amp=False)
    policy = policy.to(device).eval()
    print_yellow(f"  Parameters: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f} M")

    if args.mode == "pytorch":
        run_pytorch(args, policy, model_id, device)

    elif args.mode == "trt":
        batch = _load_dataset_batch(policy, model_id, device)
        run_trt(args, policy, model_id, device, batch)

    else:  # compare
        print_green("═" * 58)
        print_green(f"  {args.model.upper()}  PyTorch vs TRT")
        print_green("═" * 58)

        pt_mean, pt_std, pt_actions, batch = run_pytorch(args, policy, model_id, device)
        trt_mean, trt_std, trt_actions     = run_trt(args, policy, model_id, device, batch)

        speedup = pt_mean / trt_mean if trt_mean > 0 else float("nan")
        print_green("\n═" * 58)
        print_yellow(f"  PyTorch : {pt_mean:.2f} ± {pt_std:.2f} ms")
        print_yellow(f"  TRT     : {trt_mean:.2f} ± {trt_std:.2f} ms")
        print_yellow(f"  Speedup : {speedup:.2f}×")
        if pt_actions is not None and trt_actions is not None:
            compare_outputs(pt_actions, trt_actions)
        print_green("\nDone.")


if __name__ == "__main__":
    main()
