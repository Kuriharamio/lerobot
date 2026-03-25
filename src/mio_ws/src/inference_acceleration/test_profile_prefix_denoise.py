import argparse
import os
import sys

import torch
import torch.utils.benchmark as bench

WORKSPACE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE_PATH, ".."))  # lerobot src
sys.path.insert(0, WORKSPACE_PATH)

from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
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
    timestep = torch.tensor([0.5], device=device, dtype=torch.float32)
    return images, img_masks, tokens, masks, state, noise, timestep


def main():
    parser = argparse.ArgumentParser(description="Profile prefix encode vs denoise step timing.")
    parser.add_argument("--model", "-m", choices=["pi0"], default="pi0")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile path.")
    parser.add_argument("--compile_mode", default="max-autotune")
    parser.add_argument("--min_run_time", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=10, help="Used for estimated total.")
    args = parser.parse_args()

    device = torch.device("cuda")
    policy, _ = get_policy(
        model=args.model,
        compile=args.compile,
        compile_mode=args.compile_mode,
        amp=True,
        replace_method=True,
    )
    model = policy.model

    images, img_masks, tokens, masks, state, noise, timestep = make_inputs(policy, device)

    @torch.no_grad()
    def prefix_only():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_mask_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_mask_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        torch.cuda.synchronize()
        return past_key_values, prefix_pad_masks

    with torch.no_grad():
        past_key_values, prefix_pad_masks = prefix_only()
        past_key_values, prefix_pad_masks = prefix_only()
        prefix_timer = bench.Timer("prefix_only()", globals=locals())
        prefix_result = prefix_timer.blocked_autorange(min_run_time=args.min_run_time)

        def denoise_only():
            model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
            out = model.denoise_step(state, prefix_pad_masks, past_key_values, noise.clone(), timestep)
            torch.cuda.synchronize()
            return out

        denoise_only()
        denoise_only()
        denoise_timer = bench.Timer("denoise_only()", globals=locals())
        denoise_result = denoise_timer.blocked_autorange(min_run_time=args.min_run_time)

    prefix_ms = prefix_result.mean * 1000
    denoise_ms = denoise_result.mean * 1000
    total_ms = prefix_ms + denoise_ms * args.num_steps
    denoise_share = (denoise_ms * args.num_steps) / total_ms * 100

    print(f"compile={args.compile}, compile_mode={args.compile_mode}")
    print(f"prefix encode  : {prefix_ms:.2f} ms  (x1)")
    print(f"denoise step   : {denoise_ms:.2f} ms  (x{args.num_steps})")
    print(f"total (est.)   : {total_ms:.2f} ms")
    print(f"denoise share  : {denoise_share:.1f}%")


if __name__ == "__main__":
    main()
