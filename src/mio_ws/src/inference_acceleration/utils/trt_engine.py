#!/usr/bin/env python3
"""
Minimal TensorRT engine wrapper.

Adapted from openpi_on_thor/trt_torch.py for use with lerobot policies.
Supports named-tensor I/O (kwargs) as well as positional I/O (args).
"""

import atexit
import ctypes
import os

import torch
import tensorrt as trt


def _torch_dtype(trt_dtype):
    """Map a TensorRT dtype to the equivalent torch dtype."""
    _MAP = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8:    torch.int8,
        trt.int32:   torch.int32,
        trt.int64:   torch.int64,
        trt.bool:    torch.bool,
        trt.uint8:   torch.uint8,
    }
    if trt_dtype in _MAP:
        return _MAP[trt_dtype]
    raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")


class Engine:
    """Load and run a serialized TensorRT engine.

    Example:
        engine = Engine("model_pi0.engine")
        outputs = engine(images, img_masks, lang_tokens, lang_masks, state, noise)
        actions = outputs["actions"]
    """

    def __init__(self, engine_path: str, plugins: list[str] | None = None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        # Load optional plugin libraries (e.g. custom FP8 ops)
        self._plugins = [
            ctypes.CDLL(p, ctypes.RTLD_GLOBAL) for p in (plugins or [])
        ]

        self.engine_path = engine_path
        self._load(engine_path)
        atexit.register(self._destroy)
        self._print_info()

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def _load(self, path: str):
        runtime = trt.Runtime(self.logger)
        with open(path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        assert self._engine is not None, f"Failed to deserialize engine: {path}"

        self._ctx = self._engine.create_execution_context()

        self._in_meta: list[tuple[str, tuple, torch.dtype]] = []
        self._out_meta: list[tuple[str, tuple, torch.dtype]] = []
        for name in self._engine:
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = _torch_dtype(self._engine.get_tensor_dtype(name))
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._in_meta.append((name, shape, dtype))
            else:
                self._out_meta.append((name, shape, dtype))

    def _destroy(self):
        del self._ctx
        del self._engine

    # ── introspection ────────────────────────────────────────────────────────────

    def _print_info(self):
        if int(os.getenv("LOCAL_RANK", -1)) not in (0, -1):
            return
        print("═" * 52)
        print(f"  TRT Engine: {self.engine_path}")
        print(f"  Inputs  ({len(self._in_meta)}):")
        for name, shape, dtype in self._in_meta:
            print(f"    {name}: {'×'.join(map(str, shape))}  [{dtype}]")
        print(f"  Outputs ({len(self._out_meta)}):")
        for name, shape, dtype in self._out_meta:
            print(f"    {name}: {'×'.join(map(str, shape))}  [{dtype}]")
        print("═" * 52)

    # ── dynamic shape ────────────────────────────────────────────────────────────

    def set_input_shape(self, name: str, shape: tuple | torch.Size):
        """Set runtime shape for a dynamic-axis input tensor."""
        self._ctx.set_input_shape(name, shape)

    # ── forward ─────────────────────────────────────────────────────────────────

    def __call__(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Run one inference step.

        Accepts positional tensors (matched to inputs by order) or keyword
        tensors (matched by name).  Returns a dict of output name → tensor.

        The caller is responsible for calling set_input_shape() for any
        input that has dynamic axes before calling forward().
        """
        stream = torch.cuda.current_stream()
        pinned: list[torch.Tensor] = []  # keep alive until execute finishes

        # Bind positional inputs
        for i, x in enumerate(args):
            name, _static_shape, dtype = self._in_meta[i]
            x = _validate_and_prepare(x, name, dtype, self._ctx)
            self._ctx.set_tensor_address(name, x.data_ptr())
            pinned.append(x)

        # Bind keyword inputs
        in_names = {m[0] for m in self._in_meta}
        for name, x in kwargs.items():
            if name not in in_names:
                continue
            _, _static_shape, dtype = next(m for m in self._in_meta if m[0] == name)
            x = _validate_and_prepare(x, name, dtype, self._ctx)
            self._ctx.set_tensor_address(name, x.data_ptr())
            pinned.append(x)

        # Allocate and bind outputs
        outputs: dict[str, torch.Tensor] = {}
        for name, _static_shape, dtype in self._out_meta:
            runtime_shape = self._ctx.get_tensor_shape(name)
            out = torch.empty(
                *runtime_shape, dtype=dtype, device=pinned[0].device
            )
            self._ctx.set_tensor_address(name, out.data_ptr())
            outputs[name] = out
            pinned.append(out)

        self._ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        return outputs


# ── private helpers ─────────────────────────────────────────────────────────────

def _validate_and_prepare(
    x: torch.Tensor, name: str, expected_dtype: torch.dtype, ctx
) -> torch.Tensor:
    """Validate dtype & device, make contiguous, and return the prepared tensor."""
    assert isinstance(x, torch.Tensor), (
        f"Input '{name}': expected torch.Tensor, got {type(x)}"
    )
    assert x.is_cuda, (
        f"Input '{name}': tensor must be on CUDA"
    )
    assert x.dtype == expected_dtype, (
        f"Input '{name}': dtype mismatch — expected {expected_dtype}, got {x.dtype}"
    )
    # Validate against runtime shape (if already set)
    runtime_shape = ctx.get_tensor_shape(name)
    if all(d >= 0 for d in runtime_shape):
        assert tuple(runtime_shape) == tuple(x.shape), (
            f"Input '{name}': shape mismatch — engine expects {tuple(runtime_shape)}, got {tuple(x.shape)}"
        )
    return x.contiguous()
