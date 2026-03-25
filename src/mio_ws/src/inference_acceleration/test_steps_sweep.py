import argparse
import os
import sys

import torch
import torch.utils.benchmark as bench

WORKSPACE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE_PATH, ".."))  # lerobot src
sys.path.insert(0, WORKSPACE_PATH)

from utils.inference_utils import get_policy


def make_inputs(policy, device, dtype=torch.bfloat16):
    cfg = policy.config
    h = w = cfg.image_resolution[0] if hasattr(cfg, "image_resolution") else 224
    batch = 1
    token_len = 48

    images = torch.rand(batch, 1, 3, h, w, device=device, dtype=dtype)
    img_masks = torch.ones(batch, 1, dtype=torch.bool, device=device)
    tokens = torch.zeros(batch, token_len, dtype=torch.long, device=device)
    masks = torch.ones(batch, token_len, dtype=torch.bool, device=device)
    state = torch.zeros(batch, cfg.max_state_dim, device=device, dtype=dtype)
    noise = torch.randn(batch, cfg.chunk_size, cfg.max_action_dim, device=device, dtype=torch.float32)
    return images, img_masks, tokens, masks, state, noise


def main():
    parser = argparse.ArgumentParser(description="Sweep num_steps and compare speed + action drift.")
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi0")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile path.")
    parser.add_argument("--compile_mode", default="max-autotune")
    parser.add_argument("--steps", default="10,5,4,3,2,1")
    parser.add_argument("--baseline_steps", type=int, default=10)
    parser.add_argument("--min_run_time", type=float, default=2.5)
    args = parser.parse_args()

    steps_list = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    if args.baseline_steps not in steps_list:
        steps_list = [args.baseline_steps] + steps_list

    device = torch.device("cuda")
    policy, _ = get_policy(
        model=args.model,
        compile=args.compile,
        compile_mode=args.compile_mode,
        amp=True,
        replace_method=True,
    )
    images, img_masks, tokens, masks, state, noise = make_inputs(policy, device)

    @torch.no_grad()
    def run_once(steps: int):
        if args.model == "pi0":
            return policy.model.sample_actions(
                images, img_masks, tokens, masks, state, noise=noise.clone(), num_steps=steps
            )
        return policy.model.sample_actions(
            images, img_masks, tokens, masks, noise=noise.clone(), num_steps=steps
        )

    with torch.no_grad():
        baseline_out = run_once(args.baseline_steps).float()

    results = {}
    for steps in steps_list:
        with torch.no_grad():
            run_once(steps)
            run_once(steps)
            timer = bench.Timer("run_once(steps)", globals=locals())
            timed = timer.blocked_autorange(min_run_time=args.min_run_time)
            out = run_once(steps).float()
            diff = (baseline_out - out).abs()

        results[steps] = timed.mean * 1000
        speedup = results[args.baseline_steps] / results[steps]
        print(
            f"steps={steps:2d}: {results[steps]:6.2f} ms  "
            f"speedup={speedup:4.2f}x  "
            f"action_diff max={diff.max().item():.4f} mean={diff.mean().item():.4f}"
        )


if __name__ == "__main__":
    main()
