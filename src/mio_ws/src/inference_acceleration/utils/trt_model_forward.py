#!/usr/bin/env python3
"""
TRT inference hooks for LeRobot PI0 / PI05 policies.

Adapted from openpi_on_thor/trt_model_forward.py for the lerobot API.

Usage:
    policy, model_id = get_policy("pi0", compile=False, amp=False)
    policy = setup_trt_engine(policy, "model_pi0.engine", model_type="pi0")
    # Now policy.select_action(batch) uses the TRT engine.
"""

import os
import sys
from functools import partial

import torch

WORKSPACE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(WORKSPACE_PATH)
from utils.trt_engine import Engine

_TARGET_DTYPE = torch.float16


# ── PI0 TRT forward ─────────────────────────────────────────────────────────────

def _pi0_trt_sample_actions(
    model_self,
    images,          # list[Tensor[B, 3, H, W]]
    img_masks,       # list[Tensor[B]]
    lang_tokens,     # Tensor[B, seq_len]  int64
    lang_masks,      # Tensor[B, seq_len]  bool
    state,           # Tensor[B, state_dim]
    noise=None,
    num_steps=None,
    **_kw,
):
    """TRT-accelerated sample_actions for PI0."""
    device = images[0].device if isinstance(images, (list, tuple)) else images.device
    B = lang_tokens.shape[0]

    images_cat = _concat_images(images, device, _TARGET_DTYPE)        # [B, N*3, H, W]
    img_masks_cat = _stack_masks(img_masks, device)                    # [B, N]

    if noise is None:
        noise = torch.randn(
            B, model_self.action_horizon, model_self.action_dim,
            dtype=_TARGET_DTYPE, device=device,
        )
    noise = _to(noise, _TARGET_DTYPE, device)
    state = _to(state, _TARGET_DTYPE, device)

    engine: Engine = model_self.trt_engine
    _set_shapes(engine, images_cat, img_masks_cat, lang_tokens, lang_masks, state, noise)

    outputs = engine(
        images_cat, img_masks_cat,
        lang_tokens.contiguous(),
        lang_masks.contiguous(),
        state, noise,
    )
    return outputs["actions"]


# ── PI05 TRT forward ────────────────────────────────────────────────────────────

def _pi05_trt_sample_actions(
    model_self,
    images,
    img_masks,
    tokens,          # lang tokens (PI05 calls them "tokens")
    masks,
    noise=None,
    num_steps=None,
    **_kw,
):
    """TRT-accelerated sample_actions for PI05 (no state input)."""
    device = images[0].device if isinstance(images, (list, tuple)) else images.device
    B = tokens.shape[0]

    images_cat = _concat_images(images, device, _TARGET_DTYPE)
    img_masks_cat = _stack_masks(img_masks, device)

    if noise is None:
        noise = torch.randn(
            B, model_self.action_horizon, model_self.action_dim,
            dtype=_TARGET_DTYPE, device=device,
        )
    noise = _to(noise, _TARGET_DTYPE, device)

    # PI05 ONNX wrapper still expects a `state` tensor (slot must be filled),
    # but the model ignores it.  Pass a zero tensor of the right shape.
    state = torch.zeros(B, model_self.state_dim, dtype=_TARGET_DTYPE, device=device)

    engine: Engine = model_self.trt_engine
    _set_shapes(engine, images_cat, img_masks_cat, tokens, masks, state, noise)

    outputs = engine(
        images_cat, img_masks_cat,
        tokens.contiguous(),
        masks.contiguous(),
        state, noise,
    )
    return outputs["actions"]


# ── setup helper ────────────────────────────────────────────────────────────────

def setup_trt_engine(policy, engine_path: str, model_type: str):
    """Attach a TRT engine to *policy* and replace sample_actions.

    Args:
        policy:      LeRobot PI0Policy or PI05Policy instance.
        engine_path: Path to the .engine file.
        model_type:  "pi0" or "pi05".

    Returns:
        The same policy object, now using TRT for inference.
    """
    assert model_type in ("pi0", "pi05"), f"Unsupported model_type: {model_type}"

    if not os.path.exists(engine_path):
        raise FileNotFoundError(
            f"TRT engine not found: {engine_path}\n"
            "Run: bash build_trt_engine.sh first."
        )

    print(f"Loading TRT engine: {engine_path}")
    inner_model = policy.model

    # Attach engine
    inner_model.trt_engine = Engine(engine_path)

    # Store dimension metadata needed for noise generation
    cfg = policy.config
    inner_model.action_horizon = cfg.chunk_size
    inner_model.action_dim = cfg.max_action_dim
    inner_model.state_dim = cfg.max_state_dim

    # Save original method (optional fallback)
    if not hasattr(inner_model, "_original_sample_actions"):
        inner_model._original_sample_actions = inner_model.sample_actions

    # Replace with TRT version
    if model_type == "pi0":
        trt_fn = partial(_pi0_trt_sample_actions, inner_model)
    else:
        trt_fn = partial(_pi05_trt_sample_actions, inner_model)

    inner_model.sample_actions = trt_fn

    # Free PyTorch weights to save VRAM.
    # Must be done AFTER storing device, because _preprocess_images uses
    # next(self.parameters()).device; we preserve a dummy buffer so that call
    # still works after the heavy sub-modules are deleted.
    device = next(inner_model.parameters()).device
    _free_pytorch_weights(inner_model, device)

    print("TRT engine attached. PyTorch weights freed.")
    return policy


# ── private helpers ─────────────────────────────────────────────────────────────

def _concat_images(
    images,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Concatenate list of [B, 3, H, W] images into [B, N*3, H, W]."""
    if isinstance(images, torch.Tensor):
        return images.to(dtype=dtype, device=device).contiguous()
    out = torch.cat([img.to(dtype=dtype, device=device) for img in images], dim=1)
    return out.contiguous()


def _stack_masks(img_masks, device: torch.device) -> torch.Tensor:
    """Stack list of [B] bool masks into [B, N]."""
    if isinstance(img_masks, torch.Tensor):
        return img_masks.to(device=device).contiguous()
    return torch.stack([m.to(device=device) for m in img_masks], dim=1).contiguous()


def _to(t: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(t)
    return t.to(dtype=dtype, device=device).contiguous()


def _set_shapes(engine: Engine, images, img_masks, lang_tokens, lang_masks, state, noise):
    """Set runtime input shapes for all dynamic-axis inputs."""
    engine.set_input_shape("images",      images.shape)
    engine.set_input_shape("img_masks",   img_masks.shape)
    engine.set_input_shape("lang_tokens", lang_tokens.shape)
    engine.set_input_shape("lang_masks",  lang_masks.shape)
    engine.set_input_shape("state",       state.shape)
    engine.set_input_shape("noise",       noise.shape)


def _free_pytorch_weights(model, device: torch.device):
    """Delete heavy PyTorch sub-modules to reclaim VRAM.

    Registers a tiny buffer so that next(model.parameters()) still works after
    deletion — required because PI0Policy._preprocess_images uses it to resolve
    the inference device.
    """
    to_delete = [
        "paligemma_with_expert",
        "action_in_proj",
        "action_out_proj",
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "time_mlp_in",
        "time_mlp_out",
    ]
    for attr in to_delete:
        if hasattr(model, attr):
            delattr(model, attr)

    import torch.nn as nn
    model.register_parameter("_trt_sentinel", nn.Parameter(torch.zeros(1, device=device), requires_grad=False))
    torch.cuda.empty_cache()
