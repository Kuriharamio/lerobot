#!/usr/bin/env python3
"""
Convert LeRobot PI0 / PI05 policies to ONNX for TensorRT deployment.

Supported precisions
--------------------
  fp16  (default)   All NVIDIA GPUs ≥ Pascal.
  fp8               Ada Lovelace (RTX 4090), Hopper (H100), Blackwell (RTX PRO 6000, Thor/GB200).
                    Requires:  pip install nvidia-modelopt[torch]
  int8              All modern NVIDIA GPUs and Jetson Orin.
                    ONNX is exported in FP16; TRT handles INT8 PTQ at engine-build time.

Device quick-reference
----------------------
  RTX 3080          --precision fp16  (or int8 for ~15 % extra throughput)
  A100              --precision fp16  (A100 has NO FP8 hardware; int8 works)
  RTX 4090          --precision fp8   (Ada FP8 Tensor Cores)
  H100              --precision fp8   (Hopper native FP8)
  Thor / GB200      --precision fp8 --enable-llm-nvfp4   (Blackwell FP8 + NVFP4 LLM)
  RTX PRO 6000 BW   --precision fp8   (Blackwell FP8 Tensor Cores)
  Orin Nano         --precision int8  (embedded Ampere; FP8 not in TRT-Jetson)

Usage
-----
  # FP16 (universal)
  python pytorch_to_onnx.py -m pi0

  # FP8 with dummy-input calibration
  python pytorch_to_onnx.py -m pi05 --precision fp8

  # FP8 with real-data calibration (higher quality)
  python pytorch_to_onnx.py -m pi0 --precision fp8 --calib-samples 64

  # Blackwell: FP8 + NVFP4 for LLM layers
  python pytorch_to_onnx.py -m pi05 --precision fp8 --enable-llm-nvfp4

  # INT8 (TRT handles PTQ at build time; just export ONNX in FP16)
  python pytorch_to_onnx.py -m pi0 --precision int8
"""

import argparse
import os
import sys
import types
from pathlib import Path

import torch
import torch.onnx
import onnx
from onnx.external_data_helper import convert_model_to_external_data

# ── path setup ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../inference_acceleration
_SRC_DIR     = os.path.dirname(_SCRIPT_DIR)                 # .../mio_ws/src
_MIO_WS_DIR  = os.path.dirname(_SRC_DIR)                    # .../mio_ws
DEFAULT_OUTPUT_BASE = os.path.join(_MIO_WS_DIR, "outputs")  # .../mio_ws/outputs

sys.path.append(_SRC_DIR)
from utils.inference_utils import get_policy

COMPUTE_DTYPE    = torch.float16
DATASET_REPO_ID  = "lerobot/libero"


# ── device / precision recommendations ─────────────────────────────────────────

DEVICE_RECS = {
    "RTX 3080":           ("fp16",     "INT8 gives ~15 % extra throughput; no FP8 HW on Ampere"),
    "A100":               ("fp16",     "INT8 available; A100 has no FP8 HW (Ampere GA100)"),
    "RTX 4090":           ("fp8",      "Ada Lovelace FP8 Tensor Cores — requires modelopt"),
    "H100":               ("fp8",      "Hopper native FP8 — requires modelopt"),
    "Thor / GB200":       ("fp8",      "Blackwell FP8 + optional --enable-llm-nvfp4 for NVFP4"),
    "RTX PRO 6000 BW":    ("fp8",      "Blackwell FP8 Tensor Cores — requires modelopt"),
    "Orin Nano":          ("int8",     "Jetson Orin embedded; TRT Jetson FP8 support limited"),
}


def print_device_recommendations():
    print("\n  Device precision recommendations:")
    print(f"  {'Device':<22} {'Best precision':<16} Notes")
    print("  " + "─" * 72)
    for dev, (prec, note) in DEVICE_RECS.items():
        print(f"  {dev:<22} {prec:<16} {note}")
    print()


# ── TRT-compatible attention helper ────────────────────────────────────────────

def _att_2d_masks_trt(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Explicit int64 cast before cumsum (required for TensorRT compatibility)."""
    cumsum = torch.cumsum(att_masks.to(torch.int64), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


# ── PI0 model patching ──────────────────────────────────────────────────────────

def _patch_pi0(model: torch.nn.Module, num_steps: int) -> torch.nn.Module:
    """Patch PI0Pytorch for ONNX export (fixed loop, int64 cumsum, per-step KV clone)."""
    model.compute_dtype = COMPUTE_DTYPE
    model.to(COMPUTE_DTYPE).eval()

    def _denoise(self, state, prefix_pad, past_kv, x_t, timestep):
        suffix_embs, suffix_pad, suffix_att, adarms = self.embed_suffix(state, x_t, timestep)
        B, S, P = suffix_pad.shape[0], suffix_pad.shape[1], prefix_pad.shape[1]
        prefix_2d = prefix_pad[:, None, :].expand(B, S, P)
        suffix_2d = _att_2d_masks_trt(suffix_pad, suffix_att)
        full_att = torch.cat([prefix_2d, suffix_2d], dim=2)
        pos_ids = (
            torch.sum(prefix_pad, dim=-1)[:, None]
            + torch.cumsum(suffix_pad.to(torch.int64), dim=1) - 1
        )
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        # Clone the prefix KV cache for this step so GemmaAttention.update() appends suffix
        # keys into a fresh copy — without this the cache grows across steps and the
        # attention-mask shape diverges from the key-state shape on step 2+.
        from transformers.cache_utils import DynamicCache as _DC
        step_kv = _DC(ddp_cache_data=[(k.clone(), v.clone(), sw) for k, v, sw in past_kv])
        out_embs, _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(full_att),
            position_ids=pos_ids,
            past_key_values=step_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms],
        )
        return self.action_out_proj(out_embs[1][:, -self.config.chunk_size:].to(COMPUTE_DTYPE))

    def _sample(self, images, img_masks, lang_tokens, lang_masks, state,
                noise=None, num_steps=None, **_kw):
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        B, dev = state.shape[0], state.device
        if noise is None:
            noise = torch.randn(B, self.config.chunk_size, self.config.max_action_dim,
                                dtype=COMPUTE_DTYPE, device=dev)
        pre_embs, pre_pad, pre_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        pre_pos = torch.cumsum(pre_pad.to(torch.int64), dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_kv = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(_att_2d_masks_trt(pre_pad, pre_att)),
            position_ids=pre_pos,
            past_key_values=None,
            inputs_embeds=[pre_embs, None],
            use_cache=True,
        )
        dt = torch.tensor(-1.0 / num_steps, dtype=COMPUTE_DTYPE, device=dev)
        x_t = noise.to(COMPUTE_DTYPE)
        t = torch.ones(1, dtype=COMPUTE_DTYPE, device=dev)
        for _ in range(num_steps):
            x_t = x_t + dt * self.denoise_step(state, pre_pad, past_kv, x_t, t.expand(B))
            t = t + dt
        return x_t

    model.denoise_step  = types.MethodType(_denoise, model)
    model.sample_actions = types.MethodType(_sample, model)
    return model


# ── PI05 model patching ─────────────────────────────────────────────────────────

def _patch_pi05(model: torch.nn.Module, num_steps: int) -> torch.nn.Module:
    """Patch PI05Pytorch for ONNX export (same as PI0 but no state in denoise_step)."""
    model.compute_dtype = COMPUTE_DTYPE
    model.to(COMPUTE_DTYPE).eval()

    def _denoise(self, prefix_pad, past_kv, x_t, timestep):
        suffix_embs, suffix_pad, suffix_att, adarms = self.embed_suffix(x_t, timestep)
        B, S, P = suffix_pad.shape[0], suffix_pad.shape[1], prefix_pad.shape[1]
        prefix_2d = prefix_pad[:, None, :].expand(B, S, P)
        suffix_2d = _att_2d_masks_trt(suffix_pad, suffix_att)
        full_att = torch.cat([prefix_2d, suffix_2d], dim=2)
        pos_ids = (
            torch.sum(prefix_pad, dim=-1)[:, None]
            + torch.cumsum(suffix_pad.to(torch.int64), dim=1) - 1
        )
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        from transformers.cache_utils import DynamicCache as _DC
        step_kv = _DC(ddp_cache_data=[(k.clone(), v.clone(), sw) for k, v, sw in past_kv])
        out_embs, _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(full_att),
            position_ids=pos_ids,
            past_key_values=step_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms],
        )
        return self.action_out_proj(out_embs[1][:, -self.config.chunk_size:].to(COMPUTE_DTYPE))

    def _sample(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **_kw):
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        B, dev = tokens.shape[0], tokens.device
        if noise is None:
            noise = torch.randn(B, self.config.chunk_size, self.config.max_action_dim,
                                dtype=COMPUTE_DTYPE, device=dev)
        pre_embs, pre_pad, pre_att = self.embed_prefix(images, img_masks, tokens, masks)
        pre_pos = torch.cumsum(pre_pad.to(torch.int64), dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_kv = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(_att_2d_masks_trt(pre_pad, pre_att)),
            position_ids=pre_pos,
            past_key_values=None,
            inputs_embeds=[pre_embs, None],
            use_cache=True,
        )
        dt = torch.tensor(-1.0 / num_steps, dtype=COMPUTE_DTYPE, device=dev)
        x_t = noise.to(COMPUTE_DTYPE)
        t = torch.ones(1, dtype=COMPUTE_DTYPE, device=dev)
        for _ in range(num_steps):
            x_t = x_t + dt * self.denoise_step(pre_pad, past_kv, x_t, t.expand(B))
            t = t + dt
        return x_t

    model.denoise_step   = types.MethodType(_denoise, model)
    model.sample_actions = types.MethodType(_sample, model)
    return model


# ── ONNX wrapper ────────────────────────────────────────────────────────────────

class PI0ONNXWrapper(torch.nn.Module):
    """Accept flat tensors; split images/masks back into per-camera lists.

    Inputs:
      images      [B, N_cam*3, H, W]   float16
      img_masks   [B, N_cam]           bool
      lang_tokens [B, seq_len]         int64
      lang_masks  [B, seq_len]         bool
      state       [B, state_dim]       float16   (PI05: unused placeholder)
      noise       [B, T, action_dim]   float16
    Output:
      actions     [B, T, action_dim]   float16
    """

    def __init__(self, model, num_steps: int, num_cameras: int, model_type: str):
        super().__init__()
        self.model       = model
        self.num_steps   = num_steps
        self.num_cameras = num_cameras
        self.model_type  = model_type

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        imgs  = [images[:, i*3:(i+1)*3] for i in range(self.num_cameras)]
        masks = [img_masks[:, i] for i in range(self.num_cameras)]
        if self.model_type == "pi05":
            return self.model.sample_actions(imgs, masks, lang_tokens, lang_masks,
                                             noise=noise, num_steps=self.num_steps)
        else:
            return self.model.sample_actions(imgs, masks, lang_tokens, lang_masks, state,
                                             noise=noise, num_steps=self.num_steps)


# ── data loading (real samples from lerobot/libero) ─────────────────────────────

def _load_real_inputs(policy, model_id: str, device: torch.device, num_samples: int = 1):
    """Load `num_samples` real frames from DATASET_REPO_ID as flattened model inputs."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    preprocess, _ = make_pre_post_processors(
        policy.config, model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    dataset = LeRobotDataset(DATASET_REPO_ID, video_backend="pyav")
    samples = []

    for i in range(min(num_samples, len(dataset))):
        frame   = dict(dataset[i])
        batch   = preprocess(frame)
        images_list, masks_list = policy._preprocess_images(batch)

        lang_tokens   = batch[OBS_LANGUAGE_TOKENS].to(device)
        lang_masks    = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
        state         = policy.prepare_state(batch).to(COMPUTE_DTYPE).to(device)
        noise         = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim,
                                    dtype=COMPUTE_DTYPE, device=device)
        images_cat    = torch.cat([img.to(COMPUTE_DTYPE) for img in images_list], dim=1)
        img_masks_cat = torch.stack(masks_list, dim=1)
        samples.append((images_cat, img_masks_cat, lang_tokens, lang_masks, state, noise))

    return samples


def _make_dummy(policy, model_id: str, device: torch.device):
    """Return one real sample from lerobot/libero; fall back to synthetic if loading fails."""
    try:
        samples = _load_real_inputs(policy, model_id, device, num_samples=1)
        print("  Dummy inputs loaded from real dataset.")
        return samples[0]
    except Exception as exc:
        print(f"  Warning: could not load real data ({exc}). Using synthetic dummy inputs.")
        cfg = policy.config
        nc   = len(cfg.image_features)
        H, W = cfg.image_resolution
        images      = torch.randn(1, nc*3, H, W,           dtype=COMPUTE_DTYPE, device=device)
        img_masks   = torch.ones(1, nc,                    dtype=torch.bool,    device=device)
        lang_tokens = torch.randint(0, 256_000, (1, cfg.tokenizer_max_length),
                                    dtype=torch.long, device=device)
        lang_masks  = torch.ones(1, cfg.tokenizer_max_length, dtype=torch.bool, device=device)
        state       = torch.randn(1, cfg.max_state_dim,    dtype=COMPUTE_DTYPE, device=device)
        noise       = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                                  dtype=COMPUTE_DTYPE, device=device)
        return images, img_masks, lang_tokens, lang_masks, state, noise


# ── FP8 quantization via NVIDIA ModelOpt ────────────────────────────────────────

def _quantize_fp8(
    wrapped_model: PI0ONNXWrapper,
    dummy_inputs,
    calib_samples: int,
    enable_llm_nvfp4: bool,
    policy,
    model_id: str,
    device: torch.device,
):
    """Apply FP8 PTQ using NVIDIA ModelOpt.

    Falls back to dummy-input calibration when calib_samples == 0.
    Real-data calibration (calib_samples > 0) gives better quantization quality.
    """
    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        raise RuntimeError(
            "FP8 quantization requires NVIDIA ModelOpt.\n"
            "Install:  pip install nvidia-modelopt[torch]\n"
            "Then re-run with --precision fp8."
        )

    print("  Configuring FP8 quantization (NVIDIA ModelOpt)...")
    quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
    # Disable quantization on Conv2d (SigLIP vision encoder) — poor FP8 accuracy
    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    if enable_llm_nvfp4:
        print("  Adding NVFP4 for LLM layers (Blackwell / Thor only)...")
        llm_prefix = "model.paligemma_with_expert.paligemma.model.language_model.layers.*"
        quant_cfg["quant_cfg"][llm_prefix] = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None, "enable": True,
        }
        quant_cfg["quant_cfg"][llm_prefix + ".output_quantizer"] = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None, "enable": False,
        }

    # Build calibration forward loop
    if calib_samples > 0:
        print(f"  Loading {calib_samples} calibration samples from {DATASET_REPO_ID}...")
        calib_data = _build_calib_data(policy, model_id, device, calib_samples)

        def forward_loop(mdl):
            mdl.eval()
            for inputs in calib_data:
                with torch.no_grad():
                    try:
                        wrapped_model.model = mdl
                        wrapped_model(*inputs)
                    except Exception as e:
                        print(f"    Warning: calibration step failed: {e}")
    else:
        print("  Using dummy-input calibration (set --calib-samples > 0 for real data)...")

        def forward_loop(mdl):
            wrapped_model.model = mdl
            with torch.no_grad():
                wrapped_model(*dummy_inputs)

    print("  Running quantization calibration...")
    quantized_model = mtq.quantize(wrapped_model.model, quant_cfg, forward_loop=forward_loop)
    print("  FP8 quantization complete.")
    mtq.print_quant_summary(quantized_model)

    if enable_llm_nvfp4:
        # Mark linear layers for NVFP4 dynamic quantization
        from modelopt.torch.quantization.utils import is_quantized_linear
        for m in quantized_model.modules():
            if isinstance(m, torch.nn.Linear) and is_quantized_linear(m):
                m.input_quantizer._trt_high_precision_dtype = "Half"
                m.input_quantizer._onnx_quantizer_type = "dynamic"
                m.output_quantizer._onnx_quantizer_type = "dynamic"
                m.weight_quantizer._onnx_quantizer_type = "static"

    wrapped_model.model = quantized_model
    return wrapped_model


def _build_calib_data(policy, model_id, device, num_samples: int):
    """Load calibration batches from lerobot dataset (delegates to _load_real_inputs)."""
    return _load_real_inputs(policy, model_id, device, num_samples=num_samples)


# ── ONNX post-process ───────────────────────────────────────────────────────────

def _postprocess_onnx(onnx_path: Path, precision: str, enable_llm_nvfp4: bool):
    """Convert large weights to external data; optionally run fp4qdq→2dq."""
    model = onnx.load(str(onnx_path), load_external_data=True)

    if precision == "fp8" and enable_llm_nvfp4:
        try:
            from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq
            print("  Converting NVFP4 QDQ → 2DQ format...")
            model = fp4qdq_to_2dq(model, verbose=True)
        except ImportError:
            print("  Warning: fp4qdq_to_2dq not available — skipping 2DQ conversion.")

    data_file = onnx_path.stem + ".data"
    convert_model_to_external_data(model, all_tensors_to_one_file=True, location=data_file)
    onnx.save(model, str(onnx_path))
    print(f"  ONNX saved → {onnx_path}")
    print(f"  Data saved → {onnx_path.parent / data_file}")


# ── main export ─────────────────────────────────────────────────────────────────

def export(
    model_name:       str,
    output_dir:       str,
    num_steps:        int,
    precision:        str,
    calib_samples:    int,
    enable_llm_nvfp4: bool,
    opset:            int = 19,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. load ──────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {model_name} (compile=False, amp=False)...")
    policy, model_id = get_policy(model=model_name, compile=False, amp=False)
    policy = policy.to(device).eval()
    inner  = policy.model
    cfg    = policy.config
    num_cameras = len(cfg.image_features)
    print(f"  model_id={model_id}  cameras={num_cameras}  "
          f"seq_len={cfg.tokenizer_max_length}  chunk={cfg.chunk_size}  steps={num_steps}")

    # 2. patch ─────────────────────────────────────────────────────────────────
    print(f"\n[2/5] Patching for ONNX ({precision.upper()})...")
    if model_name == "pi0":
        inner = _patch_pi0(inner, num_steps)
    else:
        inner = _patch_pi05(inner, num_steps)

    # 3. wrap + dummy ──────────────────────────────────────────────────────────
    print(f"\n[3/5] Building wrapper and dummy inputs...")
    dummy   = _make_dummy(policy, model_id, device)
    wrapped = PI0ONNXWrapper(inner, num_steps, num_cameras, model_name)

    # dry-run
    with torch.no_grad():
        out_test = wrapped(*dummy)
    print(f"  Dry-run OK  →  output {tuple(out_test.shape)}")

    # 4. quantize (fp8 only) ───────────────────────────────────────────────────
    if precision == "fp8":
        print(f"\n[4/5] FP8 quantization...")
        wrapped = _quantize_fp8(
            wrapped, dummy, calib_samples, enable_llm_nvfp4,
            policy, model_id, device,
        )
    elif precision == "int8":
        print(f"\n[4/5] INT8 mode: ONNX exported in FP16; TRT handles INT8 PTQ at build time.")
    else:
        print(f"\n[4/5] FP16 mode: no quantization step.")

    # 5. export ────────────────────────────────────────────────────────────────
    suffix = f"_{precision}"
    if precision == "fp8" and enable_llm_nvfp4:
        suffix += "_nvfp4"
    onnx_path = out / f"model{suffix}.onnx"

    print(f"\n[5/5] Exporting ONNX (opset={opset}) → {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped, dummy, str(onnx_path),
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
            input_names=["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"],
            output_names=["actions"],
            dynamic_axes={
                "images":      {0: "batch"},
                "img_masks":   {0: "batch"},
                "lang_tokens": {0: "batch", 1: "seq_len"},
                "lang_masks":  {0: "batch", 1: "seq_len"},
                "state":       {0: "batch"},
                "noise":       {0: "batch"},
                "actions":     {0: "batch"},
            },
        )

    _postprocess_onnx(onnx_path, precision, enable_llm_nvfp4)
    print(f"\n✓ Export complete!")
    print(f"  Next:  bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m {model_name} --onnx {onnx_path} --precision {precision}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export LeRobot PI0/PI05 to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-m", "--model", required=True, choices=["pi0", "pi05"])
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_BASE}/<model>/onnx/)",
    )
    parser.add_argument(
        "--precision", default="fp16", choices=["fp16", "fp8", "int8"],
        help="Export precision (default: fp16)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=None,
        help="Denoising steps baked into graph (default: from model config)",
    )
    parser.add_argument(
        "--calib-samples", type=int, default=32,
        help="Real calibration samples for FP8 PTQ; 0 = use dummy inputs (default: 32)",
    )
    parser.add_argument(
        "--enable-llm-nvfp4", action="store_true",
        help="Add NVFP4 for LLM layers on top of FP8 (Blackwell / Thor only)",
    )
    parser.add_argument("--opset", type=int, default=19)
    args = parser.parse_args()

    if args.enable_llm_nvfp4 and args.precision != "fp8":
        parser.error("--enable-llm-nvfp4 requires --precision fp8")

    # resolve output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(DEFAULT_OUTPUT_BASE, args.model, "onnx")

    # resolve num_steps
    if args.num_steps is None:
        tmp_policy, _ = get_policy(model=args.model, compile=False, amp=False)
        args.num_steps = tmp_policy.config.num_inference_steps
        del tmp_policy

    print_device_recommendations()
    print(f"Model:       {args.model}")
    print(f"Precision:   {args.precision}" +
          (" + NVFP4" if args.enable_llm_nvfp4 else ""))
    print(f"Steps:       {args.num_steps}")
    print(f"Output dir:  {args.output_dir}")

    export(
        model_name       = args.model,
        output_dir       = args.output_dir,
        num_steps        = args.num_steps,
        precision        = args.precision,
        calib_samples    = args.calib_samples,
        enable_llm_nvfp4 = args.enable_llm_nvfp4,
        opset            = args.opset,
    )


if __name__ == "__main__":
    main()
