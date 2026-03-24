# PI0 / PI05 → ONNX → TRT Export Guide

输出目录统一在 `src/mio_ws/src/mio_ws/outputs/<model>/onnx/`。

---

## 设备精度速查

| 设备 | 架构 | 推荐精度 | 备注 |
|------|------|---------|------|
| RTX 3080 | Ampere (GA102) | `fp16` | `int8` 可再提升 ~15% 吞吐 |
| A100 | Ampere (GA100) | `fp16` | 无 FP8 硬件；`int8` 可用 |
| RTX 4090 | Ada Lovelace | `fp8` | Ada FP8 Tensor Cores；需 modelopt |
| H100 | Hopper | `fp8` | Hopper 原生 FP8；需 modelopt |
| Thor / GB200 | Blackwell | `fp8 --enable-llm-nvfp4` | FP8 + NVFP4 双重加速 |
| RTX PRO 6000 Blackwell | Blackwell | `fp8` | 同 GB200；需 modelopt |
| Jetson Orin Nano | 嵌入式 Ampere | `int8` | Jetson TRT 对 FP8 支持有限 |

---

## 完整流程

### Step 1 — 导出 ONNX

```bash
# FP16（通用，所有 GPU）
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi0
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi05

# FP8（Ada / Hopper / Blackwell，需 pip install nvidia-modelopt[torch]）
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi0  --precision fp8
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi05 --precision fp8 --calib-samples 64

# FP8 + NVFP4（Thor / GB200 / RTX PRO 6000 Blackwell）
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi0 --precision fp8 --enable-llm-nvfp4

# INT8（ONNX 仍以 FP16 导出；INT8 PTQ 在 Step 2 由 TRT 完成）
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi0 --precision int8
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--precision` | `fp16` | `fp16` / `fp8` / `int8` |
| `--num-steps` | 模型配置 | 去噪步数（烘焙进图） |
| `--calib-samples` | `32` | FP8 PTQ 真实标定样本数；`0` 用虚拟输入 |
| `--output-dir` | `src/mio_ws/outputs/<model>/onnx/` | 输出目录 |

---

### Step 2 — 构建 TRT Engine

```bash
# FP16
bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m pi0  --onnx src/mio_ws/outputs/pi0/onnx/model_fp16.onnx
bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m pi05 --onnx src/mio_ws/outputs/pi05/onnx/model_fp16.onnx

# FP8（RTX 4090 / H100 / Blackwell）
bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m pi0 --onnx src/mio_ws/outputs/pi0/onnx/model_fp8.onnx --precision fp8

# INT8（任何现代 GPU）
bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m pi0 --onnx src/mio_ws/outputs/pi0/onnx/model_fp16.onnx --precision int8

# 指定相机数或序列长度（当 checkpoint 不同时）
NUM_CAMERAS=3 OPT_SEQ=200 bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m pi05 --onnx ...
```

Engine 默认保存在与 ONNX 同目录：`model_<precision>.engine`。

---

### Step 3 — 推理测试

```bash
# PyTorch baseline（无 TRT）
python run_trt_inference.py -m pi0 --mode pytorch

# TRT 推理
python run_trt_inference.py -m pi0 --engine src/mio_ws/outputs/pi0/onnx/model_fp16.engine

# PyTorch vs TRT 对比（延迟 + cosine similarity）
python run_trt_inference.py -m pi0  --engine src/mio_ws/outputs/pi0/onnx/model_fp16.engine  --mode compare
python run_trt_inference.py -m pi05 --engine src/mio_ws/outputs/pi05/onnx/model_fp8.engine   --mode compare
```

---

## 典型命令（各设备）

### RTX 3080 / A100（FP16）
```bash
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi0
bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m pi0 --onnx src/mio_ws/outputs/pi0/onnx/model_fp16.onnx
python run_trt_inference.py -m pi0 --engine src/mio_ws/outputs/pi0/onnx/model_fp16.engine --mode compare
```

### RTX 4090 / H100（FP8）
```bash
pip install nvidia-modelopt[torch]
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi0 --precision fp8 --calib-samples 32
bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m pi0 --onnx src/mio_ws/outputs/pi0/onnx/model_fp8.onnx --precision fp8
python run_trt_inference.py -m pi0 --engine src/mio_ws/outputs/pi0/onnx/model_fp8.engine --mode compare
```

### Thor / GB200 / RTX PRO 6000 Blackwell（FP8 + NVFP4）
```bash
pip install nvidia-modelopt[torch]
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi0 --precision fp8 --enable-llm-nvfp4 --calib-samples 64
bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m pi0 --onnx src/mio_ws/outputs/pi0/onnx/model_fp8_nvfp4.onnx --precision fp8
python run_trt_inference.py -m pi0 --engine src/mio_ws/outputs/pi0/onnx/model_fp8_nvfp4.engine --mode compare
```

### Jetson Orin Nano（INT8）
```bash
python src/mio_ws/src/inference_acceleration/pytorch_to_onnx.py -m pi0 --precision int8
bash src/mio_ws/src/inference_acceleration/build_trt_engine.sh -m pi0 --onnx src/mio_ws/outputs/pi0/onnx/model_int8.onnx --precision int8
python run_trt_inference.py -m pi0 --engine src/mio_ws/outputs/pi0/onnx/model_int8.engine --mode compare
```

---

## 注意事项

- **FP8 需要 modelopt**：`pip install nvidia-modelopt[torch]`，并且 GPU 必须是 Ada Lovelace 及以上。
- **INT8 引擎构建较慢**：TRT 在 `src/mio_ws/src/inference_acceleration/build_trt_engine.sh` 阶段进行 PTQ 校准，耗时比 FP16 长。
- **相机数不匹配**：若 checkpoint 的相机数不是 2，需用 `NUM_CAMERAS=<n>` 环境变量覆盖。
- **PI0Fast 已移除**：其自回归解码的动态序列长度无法导出为静态 ONNX；若需加速请使用 `run_test.py` 的 `torch.compile` 路径。
