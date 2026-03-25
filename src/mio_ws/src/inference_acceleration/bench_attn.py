"""
Benchmark and correctness test: eager vs sdpa vs fa4 attention for pi0/pi05 inference.

Usage:
  conda run -n lerobot python bench_attn.py --model pi0 --mode eager
  conda run -n lerobot python bench_attn.py --model pi0 --mode sdpa
  conda run -n lerobot python bench_attn.py --model pi0 --mode compare   # eager vs sdpa correctness
  conda run -n lerobot python bench_attn.py --model pi0 --mode bench     # timing
"""

import argparse
import sys
import os
import time
import torch
import numpy as np

WORKSPACE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE_PATH, ".."))  # lerobot src
sys.path.insert(0, WORKSPACE_PATH)

from utils.inference_utils import get_policy


def _register_sdpa_fixed(ALL_ATTENTION_FUNCTIONS):
    """
    Register a fixed SDPA that casts the mask to match query dtype.

    HF's built-in sdpa_attention_forward passes the float32 mask directly to
    F.scaled_dot_product_attention even when q/k/v are bfloat16. In PyTorch 2.10
    this causes wrong results (dtype mismatch). We fix it here.
    """
    from torch.nn.functional import scaled_dot_product_attention
    from transformers.models.gemma.modeling_gemma import repeat_kv

    def sdpa_fixed_forward(module, query, key, value, attention_mask,
                           scaling=None, dropout=0.0, **kwargs):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        # Cast mask to query dtype to avoid PyTorch 2.10 dtype mismatch
        if attention_mask is not None and attention_mask.dtype != query.dtype:
            attention_mask = attention_mask.to(query.dtype)
        # is_causal=False: the mask already encodes all causality
        out = scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous()
        return out, None

    ALL_ATTENTION_FUNCTIONS.register("sdpa_fixed", sdpa_fixed_forward)


def set_attn_impl(policy, impl: str):
    """Patch _attn_implementation on paligemma and gemma_expert at inference time."""
    model = policy.model
    llm_cfg = model.paligemma_with_expert.paligemma.model.language_model.config
    expert_cfg = model.paligemma_with_expert.gemma_expert.model.config
    llm_cfg._attn_implementation = impl       # noqa: SLF001
    expert_cfg._attn_implementation = impl    # noqa: SLF001


def make_synthetic_batch(policy, device, dtype=torch.bfloat16):
    """Build a minimal synthetic batch matching pi0/pi05 expected input format."""
    cfg = policy.config

    # Image: single camera, 224x224 (or from config)
    H = W = cfg.image_resolution[0] if hasattr(cfg, "image_resolution") else 224
    num_cameras = 1
    images = torch.rand(1, num_cameras, 3, H, W, device=device, dtype=dtype)
    img_masks = torch.ones(1, num_cameras, dtype=torch.bool, device=device)

    # Language tokens: pad to typical length
    token_len = 20
    tokens = torch.zeros(1, token_len, dtype=torch.long, device=device)
    masks = torch.ones(1, token_len, dtype=torch.bool, device=device)

    return {
        "observation.images.top": images[:, 0],         # [B, C, H, W]
        "observation.state": torch.zeros(1, cfg.max_state_dim if hasattr(cfg, "max_state_dim") else 8,
                                          device=device, dtype=dtype),
        # packed as expected by the policy's select_action wrapper
    }


def make_raw_model_batch(policy, device, dtype=torch.bfloat16):
    """
    Build raw tensors to call policy.model.sample_actions() directly,
    bypassing the preprocessor. Handles both pi0 and pi05 signatures.
    """
    cfg = policy.config
    H = W = cfg.image_resolution[0] if hasattr(cfg, "image_resolution") else 224

    B = 1
    # Single image (no multi-cam for simplicity): [B, N_cams, C, H, W]
    images = torch.rand(B, 1, 3, H, W, device=device, dtype=dtype)
    img_masks = torch.ones(B, 1, dtype=torch.bool, device=device)

    # Language tokens (just zeros, shape must match tokenizer output)
    token_len = 48
    tokens = torch.zeros(B, token_len, dtype=torch.long, device=device)
    masks = torch.ones(B, token_len, dtype=torch.bool, device=device)

    # State vector (pi0 requires it, pi05 may not)
    state_dim = getattr(cfg, "max_state_dim", 8)
    state = torch.zeros(B, state_dim, device=device, dtype=dtype)

    return images, img_masks, tokens, masks, state


@torch.no_grad()
def run_single_inference(policy, images, img_masks, tokens, masks, state, noise=None):
    """
    noise must be passed explicitly so that eager and SDPA runs see the same noise.
    Otherwise torch.normal inside sample_actions uses different random state each call.
    """
    import inspect
    sig = inspect.signature(policy.model.sample_actions)
    if "state" in sig.parameters:
        return policy.model.sample_actions(images, img_masks, tokens, masks, state,
                                           noise=noise)
    else:
        return policy.model.sample_actions(images, img_masks, tokens, masks,
                                           noise=noise)


def benchmark(policy, images, img_masks, tokens, masks, state, n_warmup=3, n_iter=10):
    # Warmup
    for _ in range(n_warmup):
        _ = run_single_inference(policy, images, img_masks, tokens, masks, state)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = run_single_inference(policy, images, img_masks, tokens, masks, state)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    return np.array(times)


def _register_fa4(ALL_ATTENTION_FUNCTIONS):
    """
    Register FA4 wrapper into HF's AttentionInterface.
    FA4 package (flash-attn-4) installs under flash_attn.cute.
    No-op if FA4 not installed.
    """
    try:
        from flash_attn.cute import flash_attn_func as _fa4  # noqa: F401
    except ImportError:
        return  # FA4 not installed

    def fa4_attention_forward(module, query, key, value, attention_mask,
                              scaling=None, dropout=0.0, **kwargs):
        """
        FA4 drop-in for HF GemmaAttention (Gemma 2B/300M with 8:1 GQA).

        HF format:  query [B, H_q, S_q, D],  key/value [B, H_kv, S_k, D]
        FA4 format: query [B, S_q, H_q, D],  key/value [B, S_k, H_kv, D]
        FA4 supports GQA natively — no need to repeat KV heads.

        Mask strategy for pi0/pi05 inference (no training):
          - prefix encode: S_q == S_k, all tokens bidirectional → causal=False
          - denoise step:  S_q (suffix) < S_k (prefix KV cache + suffix)
                           right-aligned causal mask → causal=True
        """
        from flash_attn.cute import flash_attn_func as _fa4
        S_q, S_k = query.shape[2], key.shape[2]
        q = query.transpose(1, 2)   # [B, S_q, H_q, D]
        k = key.transpose(1, 2)     # [B, S_k, H_kv, D]
        v = value.transpose(1, 2)
        # S_q < S_k  →  decoding with KV cache → right-aligned causal
        # S_q == S_k →  full prefix-encode pass → bidirectional (causal=False)
        causal = S_q != S_k
        out = _fa4(q, k, v, softmax_scale=scaling, causal=causal)
        out = out.transpose(1, 2).contiguous()   # back to [B, H, S, D]
        return out, None

    ALL_ATTENTION_FUNCTIONS.register("flash_attention_4", fa4_attention_forward)
    print("  [FA4] Registered flash_attention_4 in HF AttentionInterface")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi0")
    parser.add_argument("--mode", choices=["eager", "sdpa", "compare", "bench", "fa4"], default="compare")
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    policy, _ = get_policy(model=args.model, compile=True, compile_mode="max-autotune", amp=True)

    # Register custom attention impls into HF's global registry
    from transformers.models.gemma.modeling_gemma import ALL_ATTENTION_FUNCTIONS
    _register_sdpa_fixed(ALL_ATTENTION_FUNCTIONS)
    _register_fa4(ALL_ATTENTION_FUNCTIONS)   # no-op if FA4 not installed

    images, img_masks, tokens, masks, state = make_raw_model_batch(policy, device, dtype)

    # Pre-generate fixed noise so both eager and sdpa/fa4 runs are exactly comparable.
    # Must be float32 to match policy.model.sample_noise() output dtype.
    cfg = policy.config
    noise_shape = (1, cfg.chunk_size, cfg.max_action_dim if hasattr(cfg, "max_action_dim") else 32)
    torch.manual_seed(1234)
    fixed_noise = torch.randn(*noise_shape, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    if args.mode in ("eager", "sdpa", "sdpa_fixed"):
        set_attn_impl(policy, args.mode)
        print(f"[*] Running single inference with attn_impl={args.mode}")
        out = run_single_inference(policy, images, img_masks, tokens, masks, state, noise=fixed_noise)
        print(f"    Output shape: {out.shape}, mean={out.float().mean():.4f}, std={out.float().std():.4f}")

    # ------------------------------------------------------------------ #
    elif args.mode == "compare":
        print("[*] Compare: eager vs sdpa_fixed (same fixed noise for fair comparison)")
        tol = 5e-2  # bfloat16 accumulation over 18 layers

        set_attn_impl(policy, "eager")
        out_eager = run_single_inference(policy, images, img_masks, tokens, masks, state,
                                         noise=fixed_noise).float()
        print(f"  eager     : shape={out_eager.shape}, mean={out_eager.mean():.5f}")

        # sdpa_fixed: our wrapper with explicit dtype cast on the mask
        set_attn_impl(policy, "sdpa_fixed")
        out_sdpa = run_single_inference(policy, images, img_masks, tokens, masks, state,
                                        noise=fixed_noise).float()
        diff = (out_eager - out_sdpa).abs()
        ok = diff.max().item() < tol
        print(f"  sdpa_fixed: mean={out_sdpa.mean():.5f}  "
              f"max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}  "
              f"{'[PASS]' if ok else '[FAIL]'}")

        # builtin sdpa for reference
        set_attn_impl(policy, "sdpa")
        try:
            out_sdpa_builtin = run_single_inference(policy, images, img_masks, tokens, masks, state,
                                                    noise=fixed_noise).float()
            diff2 = (out_eager - out_sdpa_builtin).abs()
            print(f"  sdpa(hf)  : mean={out_sdpa_builtin.mean():.5f}  "
                  f"max_diff={diff2.max():.2e}  "
                  f"{'[PASS]' if diff2.max().item() < tol else '[FAIL - dtype mismatch]'}")
        except Exception as e:
            print(f"  sdpa(hf)  : ERROR - {e}")

        # FA4
        if "flash_attention_4" in ALL_ATTENTION_FUNCTIONS:
            set_attn_impl(policy, "flash_attention_4")
            try:
                out_fa4 = run_single_inference(policy, images, img_masks, tokens, masks, state,
                                               noise=fixed_noise).float()
                diff3 = (out_eager - out_fa4).abs()
                ok3 = diff3.max().item() < tol
                print(f"  fa4       : mean={out_fa4.mean():.5f}  "
                      f"max_diff={diff3.max():.2e}  mean_diff={diff3.mean():.2e}  "
                      f"{'[PASS]' if ok3 else '[FAIL]'}")
            except Exception as e:
                print(f"  fa4       : ERROR - {e}")
        else:
            print("  fa4       : [SKIP] FA4 not installed")

    # ------------------------------------------------------------------ #
    elif args.mode == "bench":
        print(f"[*] Timing benchmark (n_iter={args.n_iter})")
        results = {}

        for impl in ("eager", "sdpa_fixed"):
            set_attn_impl(policy, impl)
            times_ms = benchmark(policy, images, img_masks, tokens, masks, state,
                                 n_warmup=3, n_iter=args.n_iter)
            results[impl] = times_ms
            print(f"  {impl:10s}: {times_ms.mean():.1f} +/- {times_ms.std():.1f} ms  "
                  f"(min={times_ms.min():.1f}, max={times_ms.max():.1f})")

        speedup = results["eager"].mean() / results["sdpa_fixed"].mean()
        print(f"\n  sdpa_fixed speedup over eager: {speedup:.2f}x")

        # FA4 (already registered at startup if installed)
        if "flash_attention_4" in ALL_ATTENTION_FUNCTIONS:
            set_attn_impl(policy, "flash_attention_4")
            times_ms = benchmark(policy, images, img_masks, tokens, masks, state,
                                 n_warmup=3, n_iter=args.n_iter)
            results["fa4"] = times_ms
            speedup_fa4 = results["eager"].mean() / times_ms.mean()
            print(f"  {'fa4':10s}: {times_ms.mean():.1f} +/- {times_ms.std():.1f} ms  "
                  f"(min={times_ms.min():.1f}, max={times_ms.max():.1f})  "
                  f"speedup={speedup_fa4:.2f}x")
        else:
            print("\n  [!] FA4 not installed - skipping FA4 benchmark")
            print("      Install: pip install flash-attn-4==4.0.0b5")

    # ------------------------------------------------------------------ #
    elif args.mode == "fa4":
        if "flash_attention_4" in ALL_ATTENTION_FUNCTIONS:
            set_attn_impl(policy, "flash_attention_4")
            print("[*] Running single inference with FA4")
            out = run_single_inference(policy, images, img_masks, tokens, masks, state,
                                       noise=fixed_noise)
            print(f"    Output shape: {out.shape}, mean={out.float().mean():.4f}")
        else:
            print("[!] FA4 not installed. Run: pip install flash-attn-4==4.0.0b5")


if __name__ == "__main__":
    main()
