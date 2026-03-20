"""
Offline inference benchmark for pi-series policies (pi0 / pi05 / pi0_fast).

Usage
─────
# use the default pi05 checkpoint in src/mio_ws/checkpoint/
python src/mio_ws/src/inference_acceleration/run_inference.py

# explicit checkpoint path (pi0, pi05, or pi0_fast all work the same way)
python src/mio_ws/src/inference_acceleration/run_inference.py \\
    --checkpoint /path/to/pi0/checkpoint

# full options
python src/mio_ws/src/inference_acceleration/run_inference.py \\
    --checkpoint /path/to/pi0_fast/checkpoint \\
    --data-dir   src/mio_ws/data/episode_000 \\
    --output-dir src/mio_ws/src/inference_acceleration/results/pi0_fast \\
    --n-warmup   5 \\
    --device     cuda

Why all three policies work without policy-specific code
────────────────────────────────────────────────────────
• pi0 / pi05 / pi0_fast all carry an `rtc_config` field; we disable it
  so that `select_action` (the code path used by lerobot_record.py) can run.
• All three use `_action_queue` (a collections.deque) for action chunking.
• `make_policy` and `make_pre_post_processors` dispatch on `config.type`
  automatically; we pass the checkpoint path and dataset metadata.
"""

import argparse
import sys
import time
from contextlib import nullcontext
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[4]  # …/lerobot
sys.path.insert(0, str(WORKSPACE / "src"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.processor.rename_processor import rename_stats  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device  # noqa: E402

# ── Default paths (overridable via CLI) ────────────────────────────────────────
_DEFAULT_CHECKPOINT  = WORKSPACE / "src" / "mio_ws" / "checkpoint"
_DEFAULT_DATASET_ROOT = WORKSPACE / "datas" / "cube2basket" / "20260307203505_100"
_DEFAULT_DATA_DIR    = WORKSPACE / "src" / "mio_ws" / "data" / "episode_000"
_DEFAULT_OUTPUT_DIR  = Path(__file__).parent / "results"

RUNNING_MEAN_WINDOW = 10
N_WARMUP_DEFAULT = 5


# ── CLI argument parsing ────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline inference benchmark for pi0 / pi05 / pi0_fast policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-c", "--checkpoint",
        type=Path, default=_DEFAULT_CHECKPOINT,
        help="Path to the policy checkpoint directory (pi0, pi05, or pi0_fast).",
    )
    p.add_argument(
        "--dataset-root",
        type=Path, default=_DEFAULT_DATASET_ROOT,
        help="Root directory of the LeRobot dataset.",
    )
    p.add_argument(
        "--data-dir",
        type=Path, default=_DEFAULT_DATA_DIR,
        help="Directory containing the pre-extracted episode numpy arrays.",
    )
    p.add_argument(
        "--output-dir",
        type=Path, default=None,
        help=(
            "Where to save plots.  Defaults to "
            "<script_dir>/results/<policy_type>."
        ),
    )
    p.add_argument(
        "--n-warmup",
        type=int, default=N_WARMUP_DEFAULT,
        help="Number of warmup inference passes before recording timing.",
    )
    p.add_argument(
        "--device",
        type=str, default=None,
        help="Override the device in the checkpoint config (e.g. 'cpu', 'cuda').",
    )
    return p.parse_args()


# Module-level variables populated by main() before the rest of the code runs.
CHECKPOINT: Path
DATASET_ROOT: Path
DATA_DIR: Path
RESULTS_DIR: Path
N_WARMUP: int
CAM_KEYS: dict[str, Path]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_episode() -> tuple[np.ndarray, np.ndarray, np.ndarray,
                            dict[str, np.ndarray], str, int]:
    """Load the pre-extracted episode-0 numpy arrays from DATA_DIR."""
    print("=" * 60)
    print("Loading episode data …")
    print(f"  Data dir   : {DATA_DIR}")
    states     = np.load(DATA_DIR / "observation.state.npy")  # (N, 16)
    actions    = np.load(DATA_DIR / "action.npy")             # (N, 16)
    timestamps = np.load(DATA_DIR / "timestamps.npy")         # (N,)
    task       = (DATA_DIR / "task.txt").read_text().strip()

    cameras: dict[str, np.ndarray] = {}
    for key, path in CAM_KEYS.items():
        cameras[key] = np.load(path)   # (N, C, H, W)

    n_frames = states.shape[0]
    print(f"  Frames     : {n_frames}")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Task       : '{task}'")
    return states, actions, timestamps, cameras, task, n_frames


# ── Policy loading (mirrors lerobot_record.py lines 506-519) ──────────────────

def load_policy(device_override: str | None = None):
    """
    Load any pi-series policy (pi0 / pi05 / pi0_fast) from CHECKPOINT.

    All three policy types work identically here because:
      • rtc_config is present on all three — disabled so select_action runs
      • _action_queue is present on all three — used for is_model_call detection
      • make_policy / make_pre_post_processors dispatch on config.type internally
    """
    print("\nLoading policy …")
    print(f"  Checkpoint : {CHECKPOINT}")

    # Step 1-2 (lerobot_record.py): load config, override pretrained_path
    policy_cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
    policy_cfg.pretrained_path = CHECKPOINT

    # Apply optional device override (e.g. --device cpu for testing)
    if device_override is not None:
        policy_cfg.device = device_override

    # Disable AOT compilation — irrelevant for offline benchmarking
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False

    # Disable RTC: pi0, pi05, and pi0_fast all assert RTC is off in select_action
    if getattr(policy_cfg, "rtc_config", None) is not None:
        policy_cfg.rtc_config.enabled = False

    # Step 3 (lerobot_record.py): build features from dataset metadata
    dataset = LeRobotDataset(
        repo_id=DATASET_ROOT.name,
        root=DATASET_ROOT,
        episodes=[0],
    )

    # make_policy — mirrors lerobot_record.py line 507
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)

    # make_pre_post_processors — mirrors lerobot_record.py lines 511-519
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=CHECKPOINT,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    device = get_safe_torch_device(policy_cfg.device)
    print(f"  Policy type: {policy_cfg.type}")
    print(f"  Device     : {device}")
    print(f"  n_action_steps / chunk_size: "
          f"{getattr(policy_cfg, 'n_action_steps', '?')} / "
          f"{getattr(policy_cfg, 'chunk_size', '?')}")

    task_from_meta: str = dataset.meta.tasks.iloc[0].name
    return policy, preprocessor, postprocessor, device, policy_cfg.use_amp, task_from_meta


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_obs_np(
    states: np.ndarray,
    cameras: dict[str, np.ndarray],
    i: int,
) -> dict[str, np.ndarray]:
    """Build the per-frame numpy observation dict consumed by predict_action."""
    return copy({
        "observation.state": states[i],
        "observation.wrist_cam": cameras["observation.wrist_cam"][i],
        "observation.front_cam": cameras["observation.front_cam"][i],
        "observation.side_cam": cameras["observation.side_cam"][i],
    })


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# ── Warmup ─────────────────────────────────────────────────────────────────────

def warmup(
    states: np.ndarray,
    cameras: dict[str, np.ndarray],
    task: str,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    use_amp: bool,
    n_warmup: int = N_WARMUP,
) -> None:
    """
    Run n_warmup full inference passes without recording timing.

    Purpose: warm up the CUDA context, memory allocator, cuDNN auto-tuner,
    and the tokenizer so that none of their one-time costs appear in the
    benchmark numbers.

    Policy/processor state is reset afterwards so the benchmark starts clean.
    """
    if n_warmup <= 0:
        return

    print(f"\nWarmup: running {n_warmup} inference pass(es) …")
    amp_ctx = (
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext()
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    for i in range(min(n_warmup, states.shape[0])):
        obs_np = _build_obs_np(states, cameras, i)
        with torch.inference_mode(), amp_ctx:
            obs = prepare_observation_for_inference(obs_np, device, task)
            obs = preprocessor(obs)
            _sync(device)
            action = policy.select_action(obs)
            _sync(device)
            _ = postprocessor(action)
        _sync(device)

    # Reset so benchmark starts from a clean state
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    print("  Warmup done — policy/processor state reset.\n")


# ── Inference loop with fine-grained timing ────────────────────────────────────

def run_inference(
    states: np.ndarray,
    actions_gt: np.ndarray,
    cameras: dict[str, np.ndarray],
    task: str,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
    """
    Run offline inference step by step, collecting per-step timing.

    Timing breakdown (all values in milliseconds):
      t_prep        :  prepare_observation_for_inference  (numpy→tensor, to device)
      t_preprocess  :  preprocessor pipeline  (normalize + tokenize + device copy)
      t_inference   :  policy.select_action   (model forward OR queue pop)
      t_postprocess :  postprocessor pipeline (unnormalize, to CPU)
      t_total       :  wall time for the full step

    CUDA synchronisation strategy
    ──────────────────────────────
    sync() is called *after* each sub-step so that the GPU time for step N is
    attributed to step N, not to the next step that happens to flush the queue.

      prepare → (no GPU work)
      preprocessor → sync()   ← device transfer latency counted here
      select_action → sync()  ← model forward or queue pop counted here
      postprocessor           ← CPU-only; no sync needed

    is_model_call[i] is True when the action queue was empty and the network
    actually ran, False for cache-hit / queue-pop steps.

    Returns predicted_actions plus five timing arrays and is_model_call.
    """
    n_frames = states.shape[0]

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    t_prep = np.zeros(n_frames)
    t_preprocess = np.zeros(n_frames)
    t_inference = np.zeros(n_frames)
    t_postprocess = np.zeros(n_frames)
    t_total = np.zeros(n_frames)
    is_model_call = np.zeros(n_frames, dtype=bool)
    predicted_actions = np.zeros_like(actions_gt)

    amp_ctx = (
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext()
    )
    chunk_size = getattr(policy.config, "n_action_steps", 1)

    print(f"Running inference on {n_frames} frames (chunk_size={chunk_size}) …")

    for i in range(n_frames):
        t_step_start = time.perf_counter()

        obs_np = _build_obs_np(states, cameras, i)

        with torch.inference_mode(), amp_ctx:
            # ── Sub-step 1: prepare_observation_for_inference ─────────────────
            # Pure CPU work; no sync needed.
            t0 = time.perf_counter()
            obs = prepare_observation_for_inference(obs_np, device, task)
            t_prep[i] = (time.perf_counter() - t0) * 1e3

            # ── Sub-step 2: preprocessor pipeline ────────────────────────────
            # Includes async CUDA data-transfer (device_processor).
            # Sync *after* so transfer latency is counted here, not in step 3.
            t0 = time.perf_counter()
            obs = preprocessor(obs)
            _sync(device)
            t_preprocess[i] = (time.perf_counter() - t0) * 1e3

            # ── Sub-step 3: policy.select_action ─────────────────────────────
            # Detect whether the model will actually run this step.
            # GPU is idle after the sync above, so t0 starts cleanly.
            is_model_call[i] = len(policy._action_queue) == 0
            t0 = time.perf_counter()
            action = policy.select_action(obs)
            _sync(device)
            t_inference[i] = (time.perf_counter() - t0) * 1e3

            # ── Sub-step 4: postprocessor pipeline ───────────────────────────
            # CPU-only: unnormalise + move to CPU.
            t0 = time.perf_counter()
            action = postprocessor(action)
            t_postprocess[i] = (time.perf_counter() - t0) * 1e3

        t_total[i] = (time.perf_counter() - t_step_start) * 1e3
        predicted_actions[i] = action.squeeze(0).cpu().float().numpy()

        call_flag = "MODEL" if is_model_call[i] else "cache"
        if is_model_call[i] or i == 0 or i == n_frames - 1:
            print(
                f"  [{call_flag}] step {i+1:4d}/{n_frames}  "
                f"total={t_total[i]:7.1f} ms  "
                f"prep={t_prep[i]:5.2f}  "
                f"pre={t_preprocess[i]:6.2f}  "
                f"infer={t_inference[i]:7.2f}  "
                f"post={t_postprocess[i]:5.2f}  "
                f"freq={1e3/t_total[i]:6.1f} Hz"
            )

    return (predicted_actions,
            t_prep, t_preprocess, t_inference, t_postprocess, t_total,
            is_model_call)


# ── Summary statistics ─────────────────────────────────────────────────────────

def _row(name: str, arr: np.ndarray) -> str:
    return (
        f"{name:<32} {arr.mean():>9.3f} {arr.std():>9.3f}"
        f" {arr.min():>9.3f} {arr.max():>9.3f}"
    )


def print_summary(
    predicted_actions: np.ndarray,
    actions_gt: np.ndarray,
    t_prep: np.ndarray,
    t_pre: np.ndarray,
    t_infer: np.ndarray,
    t_post: np.ndarray,
    t_total: np.ndarray,
    is_model_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.linalg.norm(predicted_actions - actions_gt, axis=1)
    freq = 1e3 / t_total

    n_model = is_model_call.sum()
    n_cache = (~is_model_call).sum()

    W = 72
    print("\n" + "=" * W)
    print("INFERENCE BENCHMARK SUMMARY")
    print("=" * W)
    print(f"  Steps: {len(t_total)}  |  model calls: {n_model}  |  queue pops: {n_cache}")
    print("─" * W)
    hdr = f"{'Metric':<32} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}"
    print(hdr)
    print("─" * W)

    # ── All-steps rows ────────────────────────────────────────────────────────
    print("All steps:")
    for name, arr in [
        ("  Data prep (ms)",        t_prep),
        ("  Preprocessor (ms)",     t_pre),
        ("  select_action (ms)",    t_infer),
        ("  Postprocessor (ms)",    t_post),
        ("  Total per step (ms)",   t_total),
        ("  Step freq (Hz)",        freq),
        ("  Action L2 error",       errors),
    ]:
        print(_row(name, arr))

    # ── Model-call steps only ─────────────────────────────────────────────────
    if n_model > 0:
        model_freq = 1e3 / t_infer[is_model_call]
        print(f"\nModel-call steps only  (n={n_model}):")
        for name, arr in [
            ("  select_action (ms)",   t_infer[is_model_call]),
            ("  Total step (ms)",      t_total[is_model_call]),
            ("  Model call freq (Hz)", model_freq),
        ]:
            print(_row(name, arr))

    # ── Queue-pop steps only ──────────────────────────────────────────────────
    if n_cache > 0:
        cache_freq = 1e3 / t_infer[~is_model_call]
        print(f"\nQueue-pop steps only  (n={n_cache}):")
        for name, arr in [
            ("  select_action (ms)",  t_infer[~is_model_call]),
            ("  Total step (ms)",     t_total[~is_model_call]),
            ("  Cache pop freq (Hz)", cache_freq),
        ]:
            print(_row(name, arr))

    print("=" * W)
    return errors, freq


# ── Visualisation ──────────────────────────────────────────────────────────────

def _running_mean(arr: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_indices, smoothed_values) for a running mean of window w."""
    kernel = np.ones(w) / w
    return np.arange(w - 1, len(arr)), np.convolve(arr, kernel, mode="valid")


def _mark_model_calls(ax, frames: np.ndarray, is_model_call: np.ndarray,
                      ymin: float = 0, ymax: float = 1) -> None:
    """Draw faint vertical lines at every model-call step."""
    for x in frames[is_model_call]:
        ax.axvline(x, color="crimson", alpha=0.25, linewidth=0.8, zorder=0)


def plot_results(
    predicted_actions: np.ndarray,
    actions_gt: np.ndarray,
    t_prep: np.ndarray,
    t_pre: np.ndarray,
    t_infer: np.ndarray,
    t_post: np.ndarray,
    t_total: np.ndarray,
    errors: np.ndarray,
    freq: np.ndarray,
    is_model_call: np.ndarray,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(t_total)
    frames = np.arange(n)
    w = RUNNING_MEAN_WINDOW
    model_frames = frames[is_model_call]
    cache_frames = frames[~is_model_call]

    # ── Plot 1: Stacked area — timing breakdown ────────────────────────────────
    # Separate model-call steps and queue-pop steps on different y-axes so the
    # large model-call spikes don't visually crush the cache-pop detail.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    for ax in (ax_top, ax_bot):
        ax.stackplot(
            frames,
            t_prep, t_pre, t_infer, t_post,
            labels=["Data prep", "Preprocessor", "select_action", "Postprocessor"],
            alpha=0.82,
        )
        _mark_model_calls(ax, frames, is_model_call)
        ax.set_xlim(0, n - 1)
        ax.grid(True, alpha=0.2)

    # top panel: show model-call spikes; bottom: zoom into queue-pop baseline
    q99_cache = np.percentile(t_total[~is_model_call], 99) if (~is_model_call).any() else 1
    ax_top.set_title("Timing Breakdown — model-call spikes (top) vs queue-pop baseline (bottom)")
    ax_bot.set_ylim(0, q99_cache * 1.3)
    ax_bot.set_ylabel("Time (ms)")
    ax_bot.set_xlabel("Frame index")
    # shared legend on top panel
    handles, lbls = ax_top.get_legend_handles_labels()
    ax_top.legend(handles[:4], lbls[:4], loc="upper right", fontsize=8)
    # invisible legend entry for model-call marker
    from matplotlib.lines import Line2D
    ax_top.legend(
        handles[:4] + [Line2D([0], [0], color="crimson", alpha=0.5, lw=1.5)],
        lbls[:4] + ["model call"],
        loc="upper right", fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "01_timing_breakdown.png", dpi=150)
    plt.close(fig)
    print("  Saved 01_timing_breakdown.png")

    # ── Plot 2: Per-step timing lines with running mean ────────────────────────
    step_labels = [
        "Data prep (ms)", "Preprocessor (ms)",
        "select_action (ms)", "Postprocessor (ms)",
    ]
    data = [t_prep, t_pre, t_infer, t_post]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax, label, arr, color in zip(axes, step_labels, data, colors):
        ax.plot(frames, arr, alpha=0.30, color=color, linewidth=0.7, label="raw")
        xs, ys = _running_mean(arr, w)
        ax.plot(xs, ys, color=color, linewidth=2.0, label=f"mean({w})")
        # annotate model-call spikes
        if len(model_frames):
            ax.scatter(model_frames, arr[model_frames], color="crimson",
                       s=20, zorder=5, label="model call", linewidths=0)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Per-step Timing Over Episode  (red dots = model call)", fontsize=13)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "02_timing_lines.png", dpi=150)
    plt.close(fig)
    print("  Saved 02_timing_lines.png")

    # ── Plot 3: Inference frequency — split model-call vs queue-pop ───────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    # top: step frequency for ALL steps (includes chunking effect)
    ax = axes[0]
    ax.plot(frames, freq, alpha=0.35, linewidth=0.7, color="tab:purple", label="all steps")
    xs, ys = _running_mean(freq, w)
    ax.plot(xs, ys, color="tab:purple", linewidth=2.0, label=f"mean({w})")
    ax.axhline(freq.mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"mean={freq.mean():.1f} Hz")
    if len(cache_frames):
        ax.axhline(freq[~is_model_call].mean(), color="tab:cyan", linestyle=":",
                   linewidth=1.5, label=f"queue-pop mean={freq[~is_model_call].mean():.1f} Hz")
    _mark_model_calls(ax, frames, is_model_call)
    ax.set_ylabel("Freq (Hz)")
    ax.set_title("Step frequency — all steps (red lines = model call)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # bottom: model-call frequency only (true neural-network throughput)
    ax = axes[1]
    if len(model_frames) > 1:
        model_freq = 1e3 / t_infer[is_model_call]
        ax.bar(model_frames, model_freq, width=3, color="crimson", alpha=0.7,
               label="model call freq")
        ax.axhline(model_freq.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean={model_freq.mean():.2f} Hz")
        ax.set_ylabel("Freq (Hz)")
        ax.set_title("Neural-network throughput  (model-call steps only, excludes action chunking)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    ax.set_xlabel("Frame index")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "03_inference_frequency.png", dpi=150)
    plt.close(fig)
    print("  Saved 03_inference_frequency.png")

    # ── Plot 4: Action L2 error ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(frames, errors, alpha=0.45, linewidth=0.8, color="tab:red", label="L2 error")
    xs, ys = _running_mean(errors, w)
    ax.plot(xs, ys, color="darkred", linewidth=2.0, label=f"mean({w})")
    ax.axhline(errors.mean(), color="black", linestyle="--", linewidth=1.3,
               label=f"overall mean={errors.mean():.4f}")
    _mark_model_calls(ax, frames, is_model_call)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("L2 error")
    ax.set_title("Action Prediction Error  ‖predicted − ground_truth‖₂  (red lines = model call)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "04_action_error.png", dpi=150)
    plt.close(fig)
    print("  Saved 04_action_error.png")

    # ── Plot 5: Per-joint action comparison (first 6 joints) ──────────────────
    n_joints = min(6, predicted_actions.shape[1])
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 2.8 * n_joints), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(frames, actions_gt[:, j],
                color="tab:blue", alpha=0.9, linewidth=1.0, label="Ground truth")
        ax.plot(frames, predicted_actions[:, j],
                color="tab:orange", alpha=0.9, linewidth=1.0,
                linestyle="--", label="Predicted")
        _mark_model_calls(ax, frames, is_model_call)
        joint_err = np.abs(predicted_actions[:, j] - actions_gt[:, j]).mean()
        ax.set_ylabel(f"Joint {j}", fontsize=9)
        ax.set_title(f"Joint {j}  (MAE = {joint_err:.4f})", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Per-joint Action: Predicted vs Ground Truth  (joints 0–5, red lines = model call)",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "05_joint_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved 05_joint_comparison.png")

    print(f"\nAll plots saved to {RESULTS_DIR}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global CHECKPOINT, DATASET_ROOT, DATA_DIR, RESULTS_DIR, N_WARMUP, CAM_KEYS

    args = _parse_args()

    # Populate module-level path variables from CLI args
    CHECKPOINT   = args.checkpoint.resolve()
    DATASET_ROOT = args.dataset_root.resolve()
    DATA_DIR     = args.data_dir.resolve()
    N_WARMUP     = args.n_warmup

    # Build camera-key → file-path mapping from DATA_DIR
    CAM_KEYS = {
        "observation.wrist_cam": DATA_DIR / "wrist_cam.npy",
        "observation.front_cam": DATA_DIR / "front_cam.npy",
        "observation.side_cam":  DATA_DIR / "side_cam.npy",
    }

    # 1. Load pre-extracted episode data
    states, actions_gt, timestamps, cameras, task, n_frames = load_episode()

    # 2. Load policy (pi0 / pi05 / pi0_fast all follow the same path)
    policy, preprocessor, postprocessor, device, use_amp, task_from_meta = load_policy(
        device_override=args.device,
    )
    task = task or task_from_meta

    # Resolve output directory: default to results/<policy_type>/
    if args.output_dir is not None:
        RESULTS_DIR = args.output_dir.resolve()
    else:
        policy_type = getattr(policy.config, "type", "unknown")
        RESULTS_DIR = Path(__file__).parent / "results" / policy_type
    print(f"\n  Results dir: {RESULTS_DIR}")

    # 3. Warmup
    warmup(states, cameras, task, policy, preprocessor, postprocessor, device, use_amp,
           n_warmup=N_WARMUP)

    # 4. Benchmark
    (predicted_actions,
     t_prep, t_pre, t_infer, t_post, t_total,
     is_model_call) = run_inference(
        states, actions_gt, cameras, task,
        policy, preprocessor, postprocessor, device, use_amp,
    )

    # 5. Summary
    errors, freq = print_summary(
        predicted_actions, actions_gt,
        t_prep, t_pre, t_infer, t_post, t_total,
        is_model_call,
    )

    # 6. Plots
    print("\nGenerating plots …")
    plot_results(
        predicted_actions, actions_gt,
        t_prep, t_pre, t_infer, t_post, t_total,
        errors, freq, is_model_call,
    )


if __name__ == "__main__":
    main()
