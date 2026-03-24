#!/bin/bash
# Build a TensorRT engine from an ONNX model (PI0 / PI05).
#
# Precision quick-reference
# ─────────────────────────
#   --precision fp16   All NVIDIA GPUs ≥ Pascal.
#   --precision fp8    Ada Lovelace (RTX 4090), Hopper (H100), Blackwell (RTX PRO 6000, Thor/GB200).
#                      ONNX must have been exported with --precision fp8.
#   --precision int8   All modern NVIDIA GPUs and Jetson Orin.
#                      TRT applies INT8 PTQ internally; ONNX can be any precision.
#
# Device recommendations
# ──────────────────────
#   RTX 3080           --precision fp16   (or int8 for ~15 % extra throughput)
#   A100               --precision fp16   (no FP8 HW; int8 works)
#   RTX 4090           --precision fp8    (Ada Lovelace FP8 Tensor Cores)
#   H100               --precision fp8    (Hopper native FP8)
#   Thor / GB200       --precision fp8    (Blackwell; use nvfp4 ONNX for max perf)
#   RTX PRO 6000 BW    --precision fp8    (Blackwell FP8 Tensor Cores)
#   Orin Nano          --precision int8   (Jetson Orin embedded)
#
# Usage
# ─────
#   bash build_trt_engine.sh -m pi0  --onnx /path/model_fp16.onnx
#   bash build_trt_engine.sh -m pi05 --onnx /path/model_fp8.onnx  --precision fp8
#   bash build_trt_engine.sh -m pi0  --onnx /path/model_fp16.onnx --precision int8
#
# Environment overrides (optional):
#   TRTEXEC=/custom/path/trtexec
#   NUM_CAMERAS=3  CHUNK_SIZE=50  MIN_SEQ=64  OPT_SEQ=128  MAX_SEQ=256

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────────
MODEL=""
ONNX_PATH=""
ENGINE_PATH=""
PRECISION="fp16"

# ── argument parsing ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)      MODEL="$2";      shift 2 ;;
        --onnx)          ONNX_PATH="$2";  shift 2 ;;
        --engine)        ENGINE_PATH="$2"; shift 2 ;;
        --precision)     PRECISION="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$ONNX_PATH" ]]; then
    echo "Usage: bash build_trt_engine.sh -m <pi0|pi05> --onnx <model.onnx> [--precision fp16|fp8|int8]"
    exit 1
fi
if [[ ! -f "$ONNX_PATH" ]]; then
    echo "Error: ONNX file not found: $ONNX_PATH"
    exit 1
fi
if [[ "$MODEL" != "pi0" && "$MODEL" != "pi05" ]]; then
    echo "Error: --model must be 'pi0' or 'pi05'"; exit 1
fi
if [[ "$PRECISION" != "fp16" && "$PRECISION" != "fp8" && "$PRECISION" != "int8" ]]; then
    echo "Error: --precision must be fp16, fp8, or int8"; exit 1
fi

# ── locate trtexec ───────────────────────────────────────────────────────────────
find_trtexec() {
    [[ -n "${TRTEXEC:-}" && -x "$TRTEXEC" ]] && echo "$TRTEXEC" && return
    for p in \
        "/usr/src/tensorrt/bin/trtexec" \
        "/usr/local/tensorrt/bin/trtexec" \
        "/opt/tensorrt/bin/trtexec" \
        "/usr/bin/trtexec" \
        "$(command -v trtexec 2>/dev/null || true)"
    do
        [[ -x "$p" ]] && echo "$p" && return
    done
    echo ""
}

TRTEXEC_BIN="$(find_trtexec)"
if [[ -z "$TRTEXEC_BIN" ]]; then
    echo "Error: trtexec not found. Set TRTEXEC=/path/to/trtexec or install TensorRT."
    exit 1
fi
echo "trtexec: $TRTEXEC_BIN"

# ── precision flags ──────────────────────────────────────────────────────────────
PRECISION_FLAGS="--fp16"
case "$PRECISION" in
    fp8)  PRECISION_FLAGS="--fp16 --fp8"  ;;
    int8) PRECISION_FLAGS="--fp16 --int8" ;;
esac

# ── per-model dimensions ─────────────────────────────────────────────────────────
IMAGE_SIZE=224
ACTION_DIM=32
STATE_DIM=32

if [[ "$MODEL" == "pi0" ]]; then
    # OPT_SEQ = tokenizer_max_length; lang_tokens is a dynamic axis so cannot be
    # read from the ONNX graph — default to pi0_libero's 48, override via env var.
    MIN_SEQ="${MIN_SEQ:-32}";  OPT_SEQ="${OPT_SEQ:-48}";   MAX_SEQ="${MAX_SEQ:-96}"
    CHUNK_SIZE="${CHUNK_SIZE:-50}"
else  # pi05
    MIN_SEQ="${MIN_SEQ:-64}";  OPT_SEQ="${OPT_SEQ:-128}";  MAX_SEQ="${MAX_SEQ:-256}"
    CHUNK_SIZE="${CHUNK_SIZE:-50}"
fi

# Auto-detect number of cameras from ONNX 'images' input shape (channel_dim / 3).
# Override at runtime with: NUM_CAMERAS=3 bash build_trt_engine.sh ...
if [[ -z "${NUM_CAMERAS:-}" ]]; then
    _detected=$(ONNX_FILE="$ONNX_PATH" python3 -c "
import onnx, os
m = onnx.load(os.environ['ONNX_FILE'], load_external_data=False)
for inp in m.graph.input:
    if inp.name == 'images':
        ch = inp.type.tensor_type.shape.dim[1].dim_value
        if ch and ch > 0:
            print(ch // 3)
        break
" 2>/dev/null)
    NUM_CAMERAS="${_detected:-2}"
    if [[ -n "$_detected" ]]; then
        echo "  Auto-detected cameras: $NUM_CAMERAS  (from ONNX images channel dim)"
    else
        echo "  Warning: could not detect cameras from ONNX, defaulting to $NUM_CAMERAS"
    fi
fi

IMAGE_CH=$((NUM_CAMERAS * 3))
MIN_BATCH="${MIN_BATCH:-1}"; OPT_BATCH="${OPT_BATCH:-1}"; MAX_BATCH="${MAX_BATCH:-1}"

# ── output path ──────────────────────────────────────────────────────────────────
if [[ -z "$ENGINE_PATH" ]]; then
    ENGINE_DIR="$(dirname "$ONNX_PATH")"
    ENGINE_PATH="${ENGINE_DIR}/model_${PRECISION}.engine"
fi
mkdir -p "$(dirname "$ENGINE_PATH")"

# ── summary ──────────────────────────────────────────────────────────────────────
echo ""
echo "Building TensorRT engine"
echo "  Model:      $MODEL"
echo "  Precision:  $PRECISION  ($PRECISION_FLAGS)"
echo "  ONNX:       $ONNX_PATH"
echo "  Engine:     $ENGINE_PATH"
echo "  Batch:      min=$MIN_BATCH  opt=$OPT_BATCH  max=$MAX_BATCH"
echo "  Cameras:    $NUM_CAMERAS  (channels=$IMAGE_CH)"
echo "  Seq:        min=$MIN_SEQ  opt=$OPT_SEQ  max=$MAX_SEQ"
echo "  Chunk/act:  ${CHUNK_SIZE}×${ACTION_DIM}"
echo ""

# ── timing cache (speeds up subsequent builds on the same machine) ────────────────
TIMING_CACHE_DIR="${ENGINE_DIR}"
TIMING_CACHE_FILE="${TIMING_CACHE_DIR}/trt_timing_${MODEL}.cache"
echo "  Timing cache: $TIMING_CACHE_FILE"
echo ""

# ── build ─────────────────────────────────────────────────────────────────────────
"$TRTEXEC_BIN" \
    --verbose \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    $PRECISION_FLAGS \
    --timingCacheFile="$TIMING_CACHE_FILE" \
    --useCudaGraph \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --noDataTransfers \
    --minShapes=images:${MIN_BATCH}x${IMAGE_CH}x${IMAGE_SIZE}x${IMAGE_SIZE},img_masks:${MIN_BATCH}x${NUM_CAMERAS},lang_tokens:${MIN_BATCH}x${MIN_SEQ},lang_masks:${MIN_BATCH}x${MIN_SEQ},state:${MIN_BATCH}x${STATE_DIM},noise:${MIN_BATCH}x${CHUNK_SIZE}x${ACTION_DIM} \
    --optShapes=images:${OPT_BATCH}x${IMAGE_CH}x${IMAGE_SIZE}x${IMAGE_SIZE},img_masks:${OPT_BATCH}x${NUM_CAMERAS},lang_tokens:${OPT_BATCH}x${OPT_SEQ},lang_masks:${OPT_BATCH}x${OPT_SEQ},state:${OPT_BATCH}x${STATE_DIM},noise:${OPT_BATCH}x${CHUNK_SIZE}x${ACTION_DIM} \
    --maxShapes=images:${MAX_BATCH}x${IMAGE_CH}x${IMAGE_SIZE}x${IMAGE_SIZE},img_masks:${MAX_BATCH}x${NUM_CAMERAS},lang_tokens:${MAX_BATCH}x${MAX_SEQ},lang_masks:${MAX_BATCH}x${MAX_SEQ},state:${MAX_BATCH}x${STATE_DIM},noise:${MAX_BATCH}x${CHUNK_SIZE}x${ACTION_DIM} \
    2>&1 | tee "${ENGINE_PATH}.build.log"

echo ""
echo "✓ Engine built: $ENGINE_PATH"
echo "  Log:          ${ENGINE_PATH}.build.log"
echo ""
echo "Next step:"
echo "  python src/mio_ws/src/inference_acceleration/run_trt_inference.py -m $MODEL --engine $ENGINE_PATH --mode compare"
