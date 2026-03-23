from __future__ import annotations
import json
import platform
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from threading import Event, Thread
from typing import Any
import time

import numpy as np
import pandas as pd

from ..config import dump_yaml, load_yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit_or_na() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True).strip()
        return out
    except Exception:
        return "not_git_repo"


def _safe_module_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _nvidia_smi_line(gpu_index: int) -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=name,driver_version,memory.total,pstate",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
        line = [x.strip() for x in out.splitlines() if x.strip()]
        return line[0] if line else "unavailable"
    except Exception as exc:
        return f"unavailable ({exc.__class__.__name__}: {exc})"


@dataclass(frozen=True)
class ModeSpec:
    mode_id: str
    precision: str
    capacity_k: int
    backend: str = "pytorch"
    model_name: str | None = None
    checkpoint: str | None = None
    calibration_required: bool = False


class EnergyMeter:
    def read_mj(self) -> float:
        raise NotImplementedError

    def read_power_w(self) -> float:
        return float("nan")


class NVMLEnergyMeter(EnergyMeter):
    def __init__(self, gpu_index: int = 0):
        try:
            import pynvml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "pynvml is required for nvml_total_energy backend. Install: pip install nvidia-ml-py3"
            ) from exc

        self._pynvml = pynvml
        self._pynvml.nvmlInit()
        self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))

        _ = self.read_mj()

    def read_mj(self) -> float:
        return float(self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))

    def read_power_w(self) -> float:
        return float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0


class NullEnergyMeter(EnergyMeter):
    def read_mj(self) -> float:
        return float("nan")

    def read_power_w(self) -> float:
        return float("nan")


class PowerIntegrationSampler:
    def __init__(self, meter: EnergyMeter, interval_ms: int = 10):
        self._meter = meter
        self._interval_s = max(1, int(interval_ms)) / 1000.0
        self._stop = Event()
        self._thread: Thread | None = None
        self._samples: list[tuple[int, float]] = []

    def _sample_once(self) -> None:
        t_ns = perf_counter_ns()
        p_w = float(self._meter.read_power_w())
        if np.isfinite(p_w) and p_w >= 0:
            self._samples.append((t_ns, p_w))

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(self._interval_s)
            self._sample_once()

    def start(self) -> None:
        self._samples = []
        self._stop.clear()
        self._sample_once()
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._sample_once()

        if len(self._samples) < 2:
            return float("nan"), int(len(self._samples))

        energy_j = 0.0
        prev_t, prev_p = self._samples[0]
        for t_ns, p_w in self._samples[1:]:
            if t_ns <= prev_t:
                continue
            dt_s = (t_ns - prev_t) / 1e9
            energy_j += 0.5 * (prev_p + p_w) * dt_s
            prev_t, prev_p = t_ns, p_w

        return float(energy_j), int(len(self._samples))


class BaseRuntime:
    def predict(self, row: pd.Series) -> int:
        raise NotImplementedError


def _torch_dtype_from_trt(trt_module: Any, trt_dtype: Any) -> Any:
    import torch

    mapping = {
        trt_module.float32: torch.float32,
        trt_module.float16: torch.float16,
        trt_module.int32: torch.int32,
        trt_module.int8: torch.int8,
        trt_module.bool: torch.bool,
    }
    if trt_dtype not in mapping:
        raise RuntimeError(f"Unsupported TensorRT dtype: {trt_dtype}")
    return mapping[trt_dtype]


class VisionRuntime(BaseRuntime):
    def __init__(
        self,
        model_name: str,
        checkpoint: str | None,
        precision: str,
        device: str,
        image_col: str,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from torchvision import models
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        except Exception as exc:
            raise RuntimeError(
                "Vision runtime requires torch/torchvision/Pillow."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        self._image_col = image_col
        self._device = torch.device(device)

        if precision.lower() not in {"fp32", "fp16", "bf16"}:
            raise RuntimeError(
                f"Unsupported precision for built-in vision runtime: {precision}. "
                "Use a real backend and/or external mode implementation for int8/int4."
            )

        if model_name.lower() == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise RuntimeError(f"Unsupported vision model_name: {model_name}")

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

        model.eval().to(self._device)

        if precision.lower() == "fp16":
            model = model.half()
            self._dtype = torch.float16
        elif precision.lower() == "bf16":
            model = model.to(dtype=torch.bfloat16)
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float32

        self._model = model

        self._transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, row: pd.Series) -> int:
        image_path = str(row[self._image_col])
        img = self._image_cls.open(image_path).convert("RGB")
        x = self._transform(img).unsqueeze(0).to(self._device)
        if self._dtype != self._torch.float32:
            x = x.to(dtype=self._dtype)

        with self._torch.no_grad():
            logits = self._model(x)
            pred = int(logits.argmax(dim=-1).item())
        return pred


class TensorRTImageCalibrator:
    def __init__(
        self,
        trt_module: Any,
        calibration_csv: str,
        image_col: str,
        device: str,
        cache_file: Path,
        max_samples: int | None = None,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        except Exception as exc:
            raise RuntimeError(
                "TensorRT INT8 calibration requires torch/torchvision/Pillow."
            ) from exc

        self._trt = trt_module
        self._torch = torch
        self._cache_file = cache_file
        self._image_col = image_col
        self._device = torch.device(device)
        self._image_cls = Image
        self._transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        df = pd.read_csv(calibration_csv)
        if image_col not in df.columns:
            raise RuntimeError(
                f"Calibration CSV missing image column '{image_col}': {calibration_csv}"
            )

        paths = [str(p) for p in df[image_col].tolist() if isinstance(p, str) and len(str(p)) > 0]
        if max_samples is not None and max_samples > 0:
            paths = paths[: int(max_samples)]
        if len(paths) == 0:
            raise RuntimeError("No valid calibration image paths found")

        self._paths = paths
        self._index = 0
        self._batch_size = 1
        self._device_input = torch.empty((1, 3, 224, 224), device=self._device, dtype=torch.float32)

        class _Calibrator(trt_module.IInt8EntropyCalibrator2):  # type: ignore[misc]
            def __init__(cal_self, outer: "TensorRTImageCalibrator"):
                super().__init__()
                cal_self._outer = outer

            def get_batch_size(cal_self) -> int:
                return int(cal_self._outer._batch_size)

            def get_batch(cal_self, names: list[str]) -> list[int] | None:  # noqa: ARG002
                o = cal_self._outer
                if o._index >= len(o._paths):
                    return None

                image_path = o._paths[o._index]
                o._index += 1

                img = o._image_cls.open(image_path).convert("RGB")
                x = o._transform(img).unsqueeze(0).to(device=o._device, dtype=torch.float32)
                o._device_input.copy_(x)
                return [int(o._device_input.data_ptr())]

            def read_calibration_cache(cal_self) -> bytes | None:
                p = cal_self._outer._cache_file
                if p.exists():
                    return p.read_bytes()
                return None

            def write_calibration_cache(cal_self, cache: bytes) -> None:
                p = cal_self._outer._cache_file
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(cache)

        self.calibrator = _Calibrator(self)


class TensorRTVisionRuntime(BaseRuntime):
    def __init__(
        self,
        cfg: dict[str, Any],
        mode: ModeSpec,
        model_name: str,
        checkpoint: str | None,
        device: str,
        image_col: str,
    ) -> None:
        try:
            import tensorrt as trt
            import torch
            from PIL import Image
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        except Exception as exc:
            raise RuntimeError(
                "TensorRT vision runtime requires tensorrt/torch/torchvision/Pillow."
            ) from exc

        if not device.lower().startswith("cuda"):
            raise RuntimeError("TensorRT backend requires CUDA device")
        if model_name.lower() != "resnet50":
            raise RuntimeError(f"TensorRT vision runtime supports model_name=resnet50 only, got: {model_name}")
        if mode.precision not in {"fp16", "int8", "fp32"}:
            raise RuntimeError(f"Unsupported TensorRT precision: {mode.precision}")

        self._trt = trt
        self._torch = torch
        self._device = torch.device(device)
        self._image_col = image_col
        self._image_cls = Image
        self._mode = mode

        self._transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        run_cfg = cfg.get("run", {})
        workload = cfg.get("workload", {})

        cache_root = run_cfg.get("trt_cache_root")
        if cache_root:
            cache_dir = Path(str(cache_root))
        else:
            cache_dir = Path(str(run_cfg.get("output_root", "runs"))) / "__trt_cache__"
        cache_dir = cache_dir / str(workload.get("id", "workload")) / str(run_cfg.get("gpu_name", "gpu"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = cache_dir / f"{model_name}.onnx"
        engine_path = cache_dir / f"{model_name}_{mode.mode_id}.plan"
        calib_cache = cache_dir / f"{model_name}_{mode.mode_id}.calib"

        self._ensure_onnx(onnx_path=onnx_path, model_name=model_name, checkpoint=checkpoint)
        if not engine_path.exists():
            self._build_engine(
                cfg=cfg,
                mode=mode,
                onnx_path=onnx_path,
                engine_path=engine_path,
                calib_cache=calib_cache,
                image_col=image_col,
                device=device,
            )

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        input_names: list[str] = []
        output_names: list[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode_io = engine.get_tensor_mode(name)
            if mode_io == trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif mode_io == trt.TensorIOMode.OUTPUT:
                output_names.append(name)

        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"Expected 1 input/1 output tensor, got {len(input_names)} input and {len(output_names)} output"
            )

        self._input_name = input_names[0]
        self._output_name = output_names[0]

        input_shape = tuple(int(x) for x in engine.get_tensor_shape(self._input_name))
        if any(d <= 0 for d in input_shape):
            input_shape = (1, 3, 224, 224)
            if not context.set_input_shape(self._input_name, input_shape):
                raise RuntimeError(f"Failed to set TensorRT input shape for {self._input_name}: {input_shape}")

        output_shape = tuple(int(x) for x in context.get_tensor_shape(self._output_name))
        if any(d <= 0 for d in output_shape):
            output_shape = tuple(int(x) for x in engine.get_tensor_shape(self._output_name))
        if any(d <= 0 for d in output_shape):
            output_shape = (1, 1000)

        input_dtype = _torch_dtype_from_trt(trt, engine.get_tensor_dtype(self._input_name))
        output_dtype = _torch_dtype_from_trt(trt, engine.get_tensor_dtype(self._output_name))

        self._input_tensor = torch.empty(input_shape, device=self._device, dtype=input_dtype)
        self._output_tensor = torch.empty(output_shape, device=self._device, dtype=output_dtype)

        if not context.set_tensor_address(self._input_name, int(self._input_tensor.data_ptr())):
            raise RuntimeError(f"Failed to bind TensorRT input tensor: {self._input_name}")
        if not context.set_tensor_address(self._output_name, int(self._output_tensor.data_ptr())):
            raise RuntimeError(f"Failed to bind TensorRT output tensor: {self._output_name}")

        self._engine = engine
        self._context = context
        self._runtime = runtime

    def _ensure_onnx(self, onnx_path: Path, model_name: str, checkpoint: str | None) -> None:
        if onnx_path.exists():
            return

        torch = self._torch
        from torchvision import models

        if model_name.lower() == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise RuntimeError(f"Unsupported vision model_name: {model_name}")

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

        model.eval().cpu()
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            x,
            onnx_path.as_posix(),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=17,
            do_constant_folding=True,
        )

    def _build_engine(
        self,
        cfg: dict[str, Any],
        mode: ModeSpec,
        onnx_path: Path,
        engine_path: Path,
        calib_cache: Path,
        image_col: str,
        device: str,
    ) -> None:
        trt = self._trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)

        if not parser.parse(onnx_path.read_bytes()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"TensorRT ONNX parse failed: {errs}")

        config = builder.create_builder_config()
        workspace = int(cfg.get("run", {}).get("trt_workspace_bytes", 2 << 30))
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)

        if mode.precision == "fp16":
            if not builder.platform_has_fast_fp16:
                raise RuntimeError("TensorRT FP16 requested but platform_has_fast_fp16 is false")
            config.set_flag(trt.BuilderFlag.FP16)
        elif mode.precision == "int8":
            if not builder.platform_has_fast_int8:
                raise RuntimeError("TensorRT INT8 requested but platform_has_fast_int8 is false")
            config.set_flag(trt.BuilderFlag.INT8)
            if bool(cfg.get("run", {}).get("trt_int8_allow_fp16_fallback", True)) and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            cal_csv = cfg.get("data", {}).get("calibration_csv")
            if not cal_csv:
                raise RuntimeError(f"Mode {mode.mode_id} requires INT8 calibration but data.calibration_csv is missing")
            max_cal_samples = cfg.get("measurement", {}).get("trt_calibration_max_samples")
            calibrator = TensorRTImageCalibrator(
                trt_module=trt,
                calibration_csv=str(cal_csv),
                image_col=image_col,
                device=device,
                cache_file=calib_cache,
                max_samples=int(max_cal_samples) if max_cal_samples is not None else None,
            )
            config.int8_calibrator = calibrator.calibrator
        elif mode.precision == "fp32":
            pass
        else:
            raise RuntimeError(f"Unsupported TensorRT precision: {mode.precision}")

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                f"Failed to build TensorRT engine for mode={mode.mode_id} precision={mode.precision}. "
                "For INT8, verify calibration CSV/image paths and quantization support."
            )

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(bytes(serialized))

    def predict(self, row: pd.Series) -> int:
        torch = self._torch

        image_path = str(row[self._image_col])
        img = self._image_cls.open(image_path).convert("RGB")
        x = self._transform(img).unsqueeze(0).to(device=self._device)
        if x.dtype != self._input_tensor.dtype:
            x = x.to(dtype=self._input_tensor.dtype)
        self._input_tensor.copy_(x)

        stream = torch.cuda.current_stream(device=self._device)
        ok = self._context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"TensorRT execute_async_v3 failed for mode={self._mode.mode_id}")

        logits = self._output_tensor
        if not logits.dtype.is_floating_point:
            logits = logits.to(dtype=torch.float32)
        pred = int(logits.argmax(dim=-1).item())
        return pred


class NLPRuntime(BaseRuntime):
    def __init__(
        self,
        model_name: str,
        checkpoint: str | None,
        precision: str,
        device: str,
        text_col: str,
        num_labels: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("NLP runtime requires torch and transformers.") from exc

        self._torch = torch
        self._device = torch.device(device)
        self._text_col = text_col

        if precision.lower() not in {"fp32", "fp16", "bf16"}:
            raise RuntimeError(
                f"Unsupported precision for built-in NLP runtime: {precision}. "
                "Use a real backend and/or external mode implementation for int8/int4."
            )

        source = checkpoint or model_name
        if not source:
            raise RuntimeError("NLP runtime requires checkpoint or model_name")

        tokenizer = AutoTokenizer.from_pretrained(source)

        kwargs: dict[str, Any] = {}
        if num_labels is not None:
            kwargs["num_labels"] = int(num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
        model.eval().to(self._device)

        if precision.lower() == "fp16":
            model = model.half()
            self._dtype = torch.float16
        elif precision.lower() == "bf16":
            model = model.to(dtype=torch.bfloat16)
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float32

        self._tokenizer = tokenizer
        self._model = model

    def predict(self, row: pd.Series) -> int:
        text = str(row[self._text_col])
        tok = self._tokenizer(text, return_tensors="pt", truncation=True, padding=False)
        tok = {k: v.to(self._device) for k, v in tok.items()}
        if self._dtype != self._torch.float32:
            for k in list(tok.keys()):
                if tok[k].dtype.is_floating_point:
                    tok[k] = tok[k].to(dtype=self._dtype)

        with self._torch.no_grad():
            out = self._model(**tok)
            pred = int(out.logits.argmax(dim=-1).item())
        return pred



def _cuda_sync_if_needed(device: str) -> None:
    if not device.lower().startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass


def _parse_mode_specs(cfg: dict[str, Any]) -> list[ModeSpec]:
    modes_cfg = cfg.get("modes")
    if modes_cfg is None:
        raise ValueError("profiling config must contain top-level 'modes' list")
    if not isinstance(modes_cfg, list) or len(modes_cfg) == 0:
        raise ValueError("'modes' must be a non-empty list")

    out: list[ModeSpec] = []
    for m in modes_cfg:
        out.append(
            ModeSpec(
                mode_id=str(m["mode_id"]),
                precision=str(m["precision"]).lower(),
                capacity_k=int(m["capacity_k"]),
                backend=str(m.get("backend", "pytorch")).lower(),
                model_name=m.get("model_name"),
                checkpoint=m.get("checkpoint"),
                calibration_required=bool(m.get("calibration_required", False)),
            )
        )
    return out


def _load_profile_csv(cfg: dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg["data"]
    profile_csv = Path(data_cfg["profile_csv"])
    if not profile_csv.exists():
        raise FileNotFoundError(f"profile_csv not found: {profile_csv}")
    df = pd.read_csv(profile_csv)

    req_cols = [
        data_cfg.get("sample_id_col", "sample_id"),
        data_cfg.get("label_col", "label"),
    ]
    workload_type = str(cfg["workload"]["type"])
    if workload_type == "vision_classification":
        req_cols.append(data_cfg.get("image_col", "image_path"))
    elif workload_type == "text_classification":
        req_cols.append(data_cfg.get("text_col", "text"))
    else:
        raise ValueError(f"Unsupported workload.type: {workload_type}")

    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"profile_csv missing required columns: {missing}")

    return df


def _resolve_mode_output_dir(cfg: dict[str, Any], mode: ModeSpec) -> Path:
    out_root = Path(cfg["run"]["output_root"])
    workload_id = str(cfg["workload"]["id"])
    gpu_name = str(cfg["run"].get("gpu_name", "gpu"))
    bs = int(cfg["run"].get("batch_size", 1))
    return out_root / workload_id / gpu_name / f"bs{bs}" / mode.mode_id


def _resolve_workload_output_dir(cfg: dict[str, Any]) -> Path:
    out_root = Path(cfg["run"]["output_root"])
    workload_id = str(cfg["workload"]["id"])
    gpu_name = str(cfg["run"].get("gpu_name", "gpu"))
    bs = int(cfg["run"].get("batch_size", 1))
    return out_root / workload_id / gpu_name / f"bs{bs}"


def _build_runtime(cfg: dict[str, Any], mode: ModeSpec) -> BaseRuntime:
    workload = cfg["workload"]
    data_cfg = cfg["data"]
    device = str(cfg["run"].get("device", "cuda:0"))

    model_name = mode.model_name or workload.get("model_name")
    checkpoint = mode.checkpoint or workload.get("checkpoint")

    if mode.backend == "pytorch":
        if workload["type"] == "vision_classification":
            return VisionRuntime(
                model_name=str(model_name),
                checkpoint=str(checkpoint) if checkpoint else None,
                precision=mode.precision,
                device=device,
                image_col=str(data_cfg.get("image_col", "image_path")),
            )

        if workload["type"] == "text_classification":
            return NLPRuntime(
                model_name=str(model_name),
                checkpoint=str(checkpoint) if checkpoint else None,
                precision=mode.precision,
                device=device,
                text_col=str(data_cfg.get("text_col", "text")),
                num_labels=workload.get("num_labels"),
            )

        raise RuntimeError(f"Unsupported workload type: {workload['type']}")

    if mode.backend == "tensorrt":
        if workload["type"] != "vision_classification":
            raise RuntimeError(
                f"Mode {mode.mode_id} uses backend=tensorrt but workload.type={workload['type']}. "
                "TensorRT backend is implemented for vision_classification only."
            )
        return TensorRTVisionRuntime(
            cfg=cfg,
            mode=mode,
            model_name=str(model_name),
            checkpoint=str(checkpoint) if checkpoint else None,
            device=device,
            image_col=str(data_cfg.get("image_col", "image_path")),
        )

    raise RuntimeError(f"Unsupported mode backend: {mode.backend}")


def _build_energy_meter(cfg: dict[str, Any]) -> EnergyMeter:
    measure_cfg = cfg.get("measurement", {})
    backend = str(measure_cfg.get("energy_backend", "nvml_total_energy")).lower()
    gpu_index = int(measure_cfg.get("nvml_gpu_index", 0))
    if backend == "nvml_total_energy":
        return NVMLEnergyMeter(gpu_index=gpu_index)
    if backend == "none":
        return NullEnergyMeter()
    raise RuntimeError(f"Unsupported energy backend: {backend}")


def _pick_sample_rows(df: pd.DataFrame, warmup_n: int, measure_n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < measure_n:
        raise ValueError(f"profile_csv has {len(df)} rows, need at least measure_requests={measure_n}")
    rng = np.random.default_rng(seed)
    warm_idx = rng.integers(0, len(df), size=warmup_n)
    warm = df.iloc[warm_idx].reset_index(drop=True)
    measure = df.iloc[:measure_n].reset_index(drop=True)
    return warm, measure


def _measure_mode(
    cfg: dict[str, Any],
    mode: ModeSpec,
    runtime: BaseRuntime,
    meter: EnergyMeter,
    warm_df: pd.DataFrame,
    measure_df: pd.DataFrame,
) -> pd.DataFrame:
    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    measure_cfg = cfg.get("measurement", {})

    sample_id_col = str(data_cfg.get("sample_id_col", "sample_id"))
    label_col = str(data_cfg.get("label_col", "label"))
    group_col = str(data_cfg.get("group_id_col", "group_id"))

    device = str(run_cfg.get("device", "cuda:0"))
    repeat = int(measure_cfg.get("repeat_per_request", 1))
    measure_backend = str(measure_cfg.get("energy_backend", "nvml_total_energy"))
    powerint_enabled = bool(measure_cfg.get("power_integration_enabled", False))
    power_sample_interval_ms = int(measure_cfg.get("power_sample_interval_ms", 10))

    for row in warm_df.itertuples(index=False):
        s = pd.Series(row._asdict())
        _ = runtime.predict(s)

    rows: list[dict[str, Any]] = []
    for req_id, row in enumerate(measure_df.itertuples(index=False)):
        s = pd.Series(row._asdict())

        pred_last = None
        t_start_wall = _utc_now()

        _cuda_sync_if_needed(device)
        e0_window = meter.read_mj()
        t0_window = perf_counter_ns()

        power_sampler: PowerIntegrationSampler | None = None
        try:
            if powerint_enabled:
                power_sampler = PowerIntegrationSampler(meter, interval_ms=power_sample_interval_ms)
                power_sampler.start()

            for _ in range(repeat):
                pred = int(runtime.predict(s))
                _cuda_sync_if_needed(device)
                pred_last = pred
        finally:
            if power_sampler is not None:
                energy_ws_powerint, power_samples = power_sampler.stop()
            else:
                energy_ws_powerint = float("nan")
                power_samples = 0

        t1_window = perf_counter_ns()
        e1_window = meter.read_mj()
        t_end_wall = _utc_now()

        latency_window_ms = (t1_window - t0_window) / 1e6
        latency_per_infer_ms = latency_window_ms / repeat if repeat > 0 else float("nan")

        if np.isfinite(e0_window) and np.isfinite(e1_window):
            energy_window_wh = max(0.0, (e1_window - e0_window) / 3_600_000.0)
            energy_per_infer_wh = energy_window_wh / repeat
        else:
            energy_window_wh = float("nan")
            energy_per_infer_wh = float("nan")

        if powerint_enabled and power_samples > 0:
            energy_window_wh_powerint = energy_ws_powerint / 3600.0
        else:
            energy_window_wh_powerint = float("nan")

        if np.isfinite(energy_window_wh) and latency_window_ms > 0:
            implied_power_w = energy_window_wh * 3_600_000.0 / latency_window_ms
        else:
            implied_power_w = float("nan")

        eps = 1e-12
        if np.isfinite(energy_window_wh) and np.isfinite(energy_window_wh_powerint) and energy_window_wh_powerint > 0:
            energy_backend_rel_err = abs(energy_window_wh - energy_window_wh_powerint) / max(energy_window_wh_powerint, eps)
            energy_backend_signed_err = (energy_window_wh - energy_window_wh_powerint) / max(energy_window_wh_powerint, eps)
        else:
            energy_backend_rel_err = float("nan")
            energy_backend_signed_err = float("nan")

        label = int(s[label_col])
        correct = int(pred_last == label)

        rows.append(
            {
                "run_id": f"{cfg['workload']['id']}__{mode.mode_id}__{run_cfg.get('seed', 0)}",
                "request_id": req_id,
                "workload": str(cfg["workload"]["id"]),
                "gpu_name": str(run_cfg.get("gpu_name", "gpu")),
                "sample_id": int(s[sample_id_col]),
                "label": label,
                "group_id": int(s[group_col]) if group_col in s and not pd.isna(s[group_col]) else -1,
                "mode_id": mode.mode_id,
                "precision": mode.precision,
                "capacity_k": mode.capacity_k,
                "batch_size": int(run_cfg.get("batch_size", 1)),
                "repeat_per_request": int(repeat),
                "power_sample_interval_ms": int(power_sample_interval_ms) if powerint_enabled else 0,
                "latency_window_ms": float(latency_window_ms),
                "latency_per_infer_ms": float(latency_per_infer_ms),
                "latency_ms": float(latency_per_infer_ms),  # legacy alias
                "energy_counter_start": float(e0_window),
                "energy_counter_end": float(e1_window),
                "energy_window_Wh": float(energy_window_wh),
                "energy_per_infer_Wh": float(energy_per_infer_wh),
                "energy_Wh": float(energy_per_infer_wh),  # legacy alias
                "energy_window_Wh_powerint": float(energy_window_wh_powerint),
                "power_samples": int(power_samples),
                "energy_backend_rel_err": float(energy_backend_rel_err),
                "energy_backend_signed_err": float(energy_backend_signed_err),
                "implied_power_W": float(implied_power_w),
                "pred": int(pred_last),
                "correct": int(correct),
                "start_ts": t_start_wall,
                "end_ts": t_end_wall,
                "seed": int(run_cfg.get("seed", 0)),
                "measurement_backend": measure_backend,
            }
        )

    return pd.DataFrame(rows)



def _mode_summary(df: pd.DataFrame) -> dict[str, Any]:
    lat_col = "latency_per_infer_ms" if "latency_per_infer_ms" in df.columns else "latency_ms"
    ene_col = "energy_per_infer_Wh" if "energy_per_infer_Wh" in df.columns else "energy_Wh"
    return {
        "mode_id": str(df["mode_id"].iloc[0]),
        "precision": str(df["precision"].iloc[0]),
        "capacity_k": int(df["capacity_k"].iloc[0]),
        "n_samples": int(len(df)),
        "accuracy": float(df["correct"].mean()),
        "p50_latency_ms": float(np.percentile(df[lat_col], 50)),
        "p95_latency_ms": float(np.percentile(df[lat_col], 95)),
        "p99_latency_ms": float(np.percentile(df[lat_col], 99)),
        "mean_energy_Wh": float(df[ene_col].mean()),
        "measurement_backend": str(df["measurement_backend"].iloc[0]),
    }



def _write_env_json(cfg: dict[str, Any], path: Path) -> None:
    env = {
        "time_utc": _utc_now(),
        "git_commit": _git_commit_or_na(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _safe_module_version("torch"),
        "torchvision": _safe_module_version("torchvision"),
        "transformers": _safe_module_version("transformers"),
        "tensorrt": _safe_module_version("tensorrt"),
        "numpy": _safe_module_version("numpy"),
        "pandas": _safe_module_version("pandas"),
        "nvidia_smi": _nvidia_smi_line(int(cfg.get("measurement", {}).get("nvml_gpu_index", 0))),
        "run": cfg.get("run", {}),
        "workload": cfg.get("workload", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2), encoding="utf-8")


def _select_admitted_modes(mode_summary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    ad = cfg.get("admission", {})
    hq_mode = str(ad.get("static_hq_mode_id", "fp16_k4"))
    max_drop_pp = float(ad.get("max_accuracy_drop_pp", 1.0))
    max_latency_ratio = float(ad.get("max_latency_multiplier_vs_hq_p95", 1.5))
    max_modes = int(ad.get("max_modes", 4))
    collapse_equivalent = bool(ad.get("collapse_equivalent_modes", True))
    eq_acc_pp = float(ad.get("equiv_accuracy_pp", 0.1))
    eq_p95_rel = float(ad.get("equiv_p95_rel", 0.02))
    eq_energy_rel = float(ad.get("equiv_energy_rel", 0.02))

    hq = mode_summary[mode_summary["mode_id"] == hq_mode]
    if hq.empty:
        raise RuntimeError(f"static_hq_mode_id={hq_mode} not found in mode_summary")

    hq_acc = float(hq.iloc[0]["accuracy"])
    hq_p95 = float(hq.iloc[0]["p95_latency_ms"])

    df = mode_summary.copy()
    df["accuracy_drop_pp"] = (hq_acc - df["accuracy"]) * 100.0
    df["latency_ratio_vs_hq_p95"] = df["p95_latency_ms"] / max(hq_p95, 1e-9)

    admitted = df[
        (df["accuracy_drop_pp"] <= max_drop_pp)
        & (df["latency_ratio_vs_hq_p95"] <= max_latency_ratio)
    ].copy()

    if admitted.empty:
        raise RuntimeError("No admitted mode under admission constraints")

    admitted = admitted.sort_values(["mean_energy_Wh", "p95_latency_ms", "accuracy"], ascending=[True, True, False])

    if collapse_equivalent and len(admitted) > 1:
        kept_rows: list[pd.Series] = []
        for _, cand in admitted.iterrows():
            equivalent = False
            for kept in kept_rows:
                acc_close = abs(float(cand["accuracy"]) - float(kept["accuracy"])) * 100.0 <= eq_acc_pp
                p95_base = max(float(kept["p95_latency_ms"]), 1e-9)
                ene_base = max(float(kept["mean_energy_Wh"]), 1e-12)
                p95_close = abs(float(cand["p95_latency_ms"]) - float(kept["p95_latency_ms"])) / p95_base <= eq_p95_rel
                ene_close = abs(float(cand["mean_energy_Wh"]) - float(kept["mean_energy_Wh"])) / ene_base <= eq_energy_rel
                if acc_close and p95_close and ene_close:
                    equivalent = True
                    break
            if not equivalent:
                kept_rows.append(cand)

        admitted = pd.DataFrame(kept_rows)

    admitted = admitted.head(max_modes).copy()
    admitted["admitted"] = True
    return admitted



def _save_mode_artifacts(cfg: dict[str, Any], mode: ModeSpec, measured: pd.DataFrame, warm_df: pd.DataFrame, measure_df: pd.DataFrame) -> None:
    out_dir = _resolve_mode_output_dir(cfg, mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    measured.to_csv(out_dir / "request_metrics.csv", index=False)

    summary = _mode_summary(measured)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sid_col = cfg["data"].get("sample_id_col", "sample_id")
    sample_ids = pd.DataFrame(
        {
            "phase": ["warmup"] * len(warm_df) + ["measure"] * len(measure_df),
            "sample_id": pd.concat(
                [
                    warm_df[[sid_col]].reset_index(drop=True),
                    measure_df[[sid_col]].reset_index(drop=True),
                ],
                ignore_index=True,
            ).iloc[:, 0].astype(int),
        }
    )
    sample_ids.to_csv(out_dir / "sample_ids.csv", index=False)

    _write_env_json(cfg, out_dir / "env.json")


def run_real_profiling(
    config_path: str | Path,
    mode_ids: list[str] | None = None,
    allow_partial_modes: bool = False,
) -> tuple[Path, Path, Path]:
    cfg = load_yaml(config_path)
    modes = _parse_mode_specs(cfg)
    if mode_ids:
        wanted = set(mode_ids)
        modes = [m for m in modes if m.mode_id in wanted]
        if not modes:
            raise RuntimeError(f"No modes selected by mode_ids={sorted(wanted)}")

    workload_dir = _resolve_workload_output_dir(cfg)
    workload_dir.mkdir(parents=True, exist_ok=True)

    measure_cfg = cfg.get("measurement", {})
    warmup_n = int(measure_cfg.get("warmup_requests", cfg["run"].get("warmup_requests", 500)))
    measure_n = int(measure_cfg.get("measure_requests", cfg["run"].get("measure_requests", 5000)))
    seed = int(cfg["run"].get("seed", 0))

    profile_df = _load_profile_csv(cfg)
    warm_df, measure_df = _pick_sample_rows(profile_df, warmup_n=warmup_n, measure_n=measure_n, seed=seed)

    mode_status: list[dict[str, Any]] = []
    mode_summaries: list[dict[str, Any]] = []

    for mode in modes:
        try:
            if mode.calibration_required:
                cal_csv = cfg.get("data", {}).get("calibration_csv")
                if not cal_csv:
                    raise RuntimeError(f"Mode {mode.mode_id} requires calibration but data.calibration_csv missing")
                if not Path(cal_csv).exists():
                    raise FileNotFoundError(f"Calibration CSV missing for mode {mode.mode_id}: {cal_csv}")

            runtime = _build_runtime(cfg, mode)
            meter = _build_energy_meter(cfg)
            measured = _measure_mode(cfg, mode, runtime, meter, warm_df=warm_df, measure_df=measure_df)

            if measured["latency_ms"].isnull().any() or (measured["latency_ms"] <= 0).any():
                raise RuntimeError(f"Invalid latency values for mode {mode.mode_id}")
            if measured["energy_Wh"].isnull().any() or (measured["energy_Wh"] <= 0).any():
                raise RuntimeError(
                    f"Invalid energy values for mode {mode.mode_id}. "
                    "Use nvml_total_energy backend or increase repeat_per_request."
                )

            _save_mode_artifacts(cfg, mode, measured, warm_df, measure_df)
            mode_summaries.append(_mode_summary(measured))
            mode_status.append({"mode_id": mode.mode_id, "status": "ok", "error": ""})
        except Exception as exc:
            mode_status.append(
                {
                    "mode_id": mode.mode_id,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            if not allow_partial_modes:
                raise

    status_df = pd.DataFrame(mode_status)
    status_csv = workload_dir / "mode_status.csv"
    status_df.to_csv(status_csv, index=False)

    if len(mode_summaries) == 0:
        raise RuntimeError("No successful mode profiling runs")

    mode_summary_df = pd.DataFrame(mode_summaries).sort_values(["mean_energy_Wh", "p95_latency_ms"])
    mode_summary_csv = workload_dir / "mode_summary.csv"
    mode_summary_df.to_csv(mode_summary_csv, index=False)

    admission_note = ""
    try:
        admitted_df = _select_admitted_modes(mode_summary_df, cfg)
    except RuntimeError as exc:
        hq_mode = str(cfg.get("admission", {}).get("static_hq_mode_id", "fp16_k4"))
        if mode_ids and "not found in mode_summary" in str(exc):
            admitted_df = mode_summary_df.copy()
            admitted_df["accuracy_drop_pp"] = float("nan")
            admitted_df["latency_ratio_vs_hq_p95"] = float("nan")
            admitted_df["admitted"] = True
            admission_note = (
                f"partial_mode_run: static_hq_mode_id={hq_mode} missing; "
                "admitted_modes.csv contains profiled modes only"
            )
        else:
            raise

    admitted_csv = workload_dir / "admitted_modes.csv"
    admitted_df.to_csv(admitted_csv, index=False)

    manifest = {
        "created_at_utc": _utc_now(),
        "config_path": str(config_path),
        "workload_output": str(workload_dir),
        "n_modes_requested": len(modes),
        "n_modes_success": int((status_df["status"] == "ok").sum()),
        "mode_status_csv": str(status_csv),
        "mode_summary_csv": str(mode_summary_csv),
        "admitted_modes_csv": str(admitted_csv),
        "admission_note": admission_note,
    }
    dump_yaml(workload_dir / "manifest.yaml", manifest)

    return mode_summary_csv, admitted_csv, status_csv


def _build_runtime_cache(cfg: dict[str, Any], mode_specs: list[ModeSpec]) -> dict[str, BaseRuntime]:
    cache: dict[str, BaseRuntime] = {}
    for m in mode_specs:
        cache[m.mode_id] = _build_runtime(cfg, m)
    return cache


def _measure_sequence(
    cfg: dict[str, Any],
    runtimes: dict[str, BaseRuntime],
    mode_sequence: list[str],
    sample_rows: pd.DataFrame,
    meter: EnergyMeter,
) -> pd.DataFrame:
    data_cfg = cfg["data"]
    sample_id_col = str(data_cfg.get("sample_id_col", "sample_id"))
    label_col = str(data_cfg.get("label_col", "label"))
    group_col = str(data_cfg.get("group_id_col", "group_id"))

    measure_cfg = cfg.get("measurement", {})
    repeat = int(measure_cfg.get("repeat_per_request", 1))
    backend = str(measure_cfg.get("energy_backend", "nvml_total_energy"))
    device = str(cfg["run"].get("device", "cuda:0"))
    powerint_enabled = bool(measure_cfg.get("power_integration_enabled", False))
    power_sample_interval_ms = int(measure_cfg.get("power_sample_interval_ms", 10))

    rows: list[dict[str, Any]] = []
    prev_mode = None
    for i, mode_id in enumerate(mode_sequence):
        row = sample_rows.iloc[i % len(sample_rows)]
        runtime = runtimes[mode_id]

        pred_last = None

        _cuda_sync_if_needed(device)
        e0_window = meter.read_mj()
        t0_window = perf_counter_ns()

        power_sampler: PowerIntegrationSampler | None = None
        try:
            if powerint_enabled:
                power_sampler = PowerIntegrationSampler(meter, interval_ms=power_sample_interval_ms)
                power_sampler.start()

            for _ in range(repeat):
                pred_last = int(runtime.predict(row))
                _cuda_sync_if_needed(device)
        finally:
            if power_sampler is not None:
                energy_ws_powerint, power_samples = power_sampler.stop()
            else:
                energy_ws_powerint = float("nan")
                power_samples = 0

        t1_window = perf_counter_ns()
        e1_window = meter.read_mj()

        latency_window_ms = (t1_window - t0_window) / 1e6
        latency_per_infer_ms = latency_window_ms / repeat if repeat > 0 else float("nan")
        if np.isfinite(e0_window) and np.isfinite(e1_window):
            energy_window_wh = max(0.0, (e1_window - e0_window) / 3_600_000.0)
            energy_per_infer_wh = energy_window_wh / repeat
        else:
            energy_window_wh = float("nan")
            energy_per_infer_wh = float("nan")

        if powerint_enabled and power_samples > 0:
            energy_window_wh_powerint = energy_ws_powerint / 3600.0
        else:
            energy_window_wh_powerint = float("nan")

        if np.isfinite(energy_window_wh) and latency_window_ms > 0:
            implied_power_w = energy_window_wh * 3_600_000.0 / latency_window_ms
        else:
            implied_power_w = float("nan")

        eps = 1e-12
        if np.isfinite(energy_window_wh) and np.isfinite(energy_window_wh_powerint) and energy_window_wh_powerint > 0:
            energy_backend_rel_err = abs(energy_window_wh - energy_window_wh_powerint) / max(energy_window_wh_powerint, eps)
            energy_backend_signed_err = (energy_window_wh - energy_window_wh_powerint) / max(energy_window_wh_powerint, eps)
        else:
            energy_backend_rel_err = float("nan")
            energy_backend_signed_err = float("nan")

        rows.append(
            {
                "seq_idx": i,
                "sample_id": int(row[sample_id_col]),
                "label": int(row[label_col]),
                "group_id": int(row[group_col]) if group_col in row and not pd.isna(row[group_col]) else -1,
                "mode_id": mode_id,
                "prev_mode_id": prev_mode,
                "switch_flag": int(prev_mode is not None and prev_mode != mode_id),
                "repeat_per_request": int(repeat),
                "power_sample_interval_ms": int(power_sample_interval_ms) if powerint_enabled else 0,
                "latency_window_ms": float(latency_window_ms),
                "latency_per_infer_ms": float(latency_per_infer_ms),
                "latency_ms": float(latency_per_infer_ms),  # legacy alias
                "energy_counter_start": float(e0_window),
                "energy_counter_end": float(e1_window),
                "energy_window_Wh": float(energy_window_wh),
                "energy_per_infer_Wh": float(energy_per_infer_wh),
                "energy_Wh": float(energy_per_infer_wh),  # legacy alias
                "energy_window_Wh_powerint": float(energy_window_wh_powerint),
                "power_samples": int(power_samples),
                "energy_backend_rel_err": float(energy_backend_rel_err),
                "energy_backend_signed_err": float(energy_backend_signed_err),
                "implied_power_W": float(implied_power_w),
                "pred": int(pred_last),
                "correct": int(pred_last == int(row[label_col])),
                "measurement_backend": backend,
            }
        )
        prev_mode = mode_id

    return pd.DataFrame(rows)



def run_real_switch_penalty(
    config_path: str | Path,
    pairs: list[tuple[str, str]] | None = None,
    n_requests: int = 2000,
) -> tuple[Path, Path]:
    cfg = load_yaml(config_path)
    modes = _parse_mode_specs(cfg)
    by_id = {m.mode_id: m for m in modes}

    if pairs is None or len(pairs) == 0:
        workload_dir = _resolve_workload_output_dir(cfg)
        mode_summary_csv = workload_dir / "mode_summary.csv"
        if mode_summary_csv.exists():
            ms = pd.read_csv(mode_summary_csv)
            mode_ids = ms.sort_values(["mean_energy_Wh", "p95_latency_ms"])["mode_id"].tolist()
        else:
            mode_ids = [m.mode_id for m in modes]
        pairs = [(mode_ids[i], mode_ids[i + 1]) for i in range(len(mode_ids) - 1)]

    profile_df = _load_profile_csv(cfg)
    if len(profile_df) == 0:
        raise RuntimeError("profile_csv is empty")

    sample_rows = profile_df.iloc[: min(len(profile_df), max(100, n_requests))].reset_index(drop=True)

    out_dir = _resolve_workload_output_dir(cfg) / "switch_penalty"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for a, b in pairs:
        if a not in by_id or b not in by_id:
            raise RuntimeError(f"Switch pair mode not found: {a}, {b}")

        runtimes = _build_runtime_cache(cfg, [by_id[a], by_id[b]])
        meter = _build_energy_meter(cfg)

        seq_a = [a] * n_requests
        seq_b = [b] * n_requests
        seq_ab = [a if i % 2 == 0 else b for i in range(n_requests)]

        df_a = _measure_sequence(cfg, runtimes, seq_a, sample_rows, meter)
        df_a["pattern"] = "AAAA"
        df_b = _measure_sequence(cfg, runtimes, seq_b, sample_rows, meter)
        df_b["pattern"] = "BBBB"
        df_ab = _measure_sequence(cfg, runtimes, seq_ab, sample_rows, meter)
        df_ab["pattern"] = "ABAB"

        pair_df = pd.concat([df_a, df_b, df_ab], ignore_index=True)
        pair_csv = out_dir / f"{a}__{b}.csv"
        pair_df.to_csv(pair_csv, index=False)

        mean_a_lat = float(df_a["latency_ms"].mean())
        mean_b_lat = float(df_b["latency_ms"].mean())
        mean_ab_lat = float(df_ab["latency_ms"].mean())
        base_lat = 0.5 * (mean_a_lat + mean_b_lat)

        mean_a_ene = float(df_a["energy_Wh"].mean())
        mean_b_ene = float(df_b["energy_Wh"].mean())
        mean_ab_ene = float(df_ab["energy_Wh"].mean())
        base_ene = 0.5 * (mean_a_ene + mean_b_ene)

        summary_rows.append(
            {
                "prev_mode_id": a,
                "next_mode_id": b,
                "delta_latency_ms": float(mean_ab_lat - base_lat),
                "delta_energy_Wh": float(mean_ab_ene - base_ene),
                "n_repeats": int(n_requests),
                "measurement_backend": str(cfg.get("measurement", {}).get("energy_backend", "nvml_total_energy")),
                "pair_csv": str(pair_csv),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    workload_alias = str(cfg["workload"].get("e0_alias", cfg["workload"]["id"]))
    penalties = []
    for r in summary.itertuples(index=False):
        penalties.append(
            {
                "workload": workload_alias,
                "mode_from": r.prev_mode_id,
                "mode_to": r.next_mode_id,
                "latency_penalty_ms": float(r.delta_latency_ms),
                "energy_penalty_Wh": float(r.delta_energy_Wh),
            }
        )
        penalties.append(
            {
                "workload": workload_alias,
                "mode_from": r.next_mode_id,
                "mode_to": r.prev_mode_id,
                "latency_penalty_ms": float(r.delta_latency_ms),
                "energy_penalty_Wh": float(r.delta_energy_Wh),
            }
        )

    mode_ids = sorted({m.mode_id for m in _parse_mode_specs(cfg)})
    for m in mode_ids:
        penalties.append(
            {
                "workload": workload_alias,
                "mode_from": m,
                "mode_to": m,
                "latency_penalty_ms": 0.0,
                "energy_penalty_Wh": 0.0,
            }
        )

    switch_yaml = out_dir / "switch_penalties.yaml"
    dump_yaml(switch_yaml, {"switch_penalties": penalties})
    return summary_csv, switch_yaml


def export_real_profiling_to_precomputed(config_path: str | Path, output_root: str | Path) -> Path:
    cfg = load_yaml(config_path)
    mode_specs = _parse_mode_specs(cfg)

    workload_alias = str(cfg["workload"].get("e0_alias", cfg["workload"]["id"]))
    workload_dir = _resolve_workload_output_dir(cfg)
    out_root = Path(output_root)
    out_dir = out_root / workload_alias
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in mode_specs:
        req_csv = workload_dir / mode.mode_id / "request_metrics.csv"
        if not req_csv.exists():
            continue
        df = pd.read_csv(req_csv)
        lat_col = "latency_per_infer_ms" if "latency_per_infer_ms" in df.columns else "latency_ms"
        ene_col = "energy_per_infer_Wh" if "energy_per_infer_Wh" in df.columns else "energy_Wh"
        required = {"sample_id", "pred", "correct", "group_id", lat_col, ene_col}
        miss = required - set(df.columns)
        if miss:
            raise RuntimeError(f"Missing columns in {req_csv}: {sorted(miss)}")
        out = pd.DataFrame(
            {
                "sample_id": df["sample_id"].astype(int),
                "prediction": df["pred"].astype(int),
                "correctness": df["correct"].astype(int),
                "group_id": df["group_id"].astype(int),
                "latency_ms": df[lat_col].astype(float),
                "energy_Wh": df[ene_col].astype(float),
            }
        )
        out.to_csv(out_dir / f"{mode.mode_id}.csv", index=False)

    return out_dir


def validate_real_profiling(config_path: str | Path, output_yaml: str | Path | None = None) -> Path:
    cfg = load_yaml(config_path)
    modes = _parse_mode_specs(cfg)
    workload_dir = _resolve_workload_output_dir(cfg)
    measure_n = int(cfg.get("measurement", {}).get("measure_requests", cfg["run"].get("measure_requests", 5000)))

    report: dict[str, Any] = {
        "validated_at_utc": _utc_now(),
        "workload_dir": str(workload_dir),
        "modes": [],
        "checks": {},
        "pass": True,
    }

    for m in modes:
        mode_dir = workload_dir / m.mode_id
        req = mode_dir / "request_metrics.csv"
        env = mode_dir / "env.json"
        summ = mode_dir / "summary.json"
        sid = mode_dir / "sample_ids.csv"

        mode_rep: dict[str, Any] = {
            "mode_id": m.mode_id,
            "exists": mode_dir.exists(),
            "request_metrics": req.exists(),
            "summary_json": summ.exists(),
            "env_json": env.exists(),
            "sample_ids_csv": sid.exists(),
            "n_rows": 0,
            "errors": [],
        }

        if req.exists():
            df = pd.read_csv(req)
            mode_rep["n_rows"] = int(len(df))
            for col in [
                "sample_id",
                "repeat_per_request",
                "power_sample_interval_ms",
                "latency_window_ms",
                "latency_per_infer_ms",
                "latency_ms",
                "energy_counter_start",
                "energy_counter_end",
                "energy_window_Wh",
                "energy_per_infer_Wh",
                "energy_Wh",
                "energy_window_Wh_powerint",
                "power_samples",
                "energy_backend_rel_err",
                "energy_backend_signed_err",
                "implied_power_W",
                "pred",
                "correct",
                "mode_id",
                "precision",
                "capacity_k",
            ]:
                if col not in df.columns:
                    mode_rep["errors"].append(f"missing_col:{col}")
            if "sample_id" in df.columns and df["sample_id"].isnull().any():
                mode_rep["errors"].append("null_sample_id")
            if "latency_ms" in df.columns and ((df["latency_ms"] <= 0) | df["latency_ms"].isnull()).any():
                mode_rep["errors"].append("invalid_latency")
            if "latency_window_ms" in df.columns and ((df["latency_window_ms"] <= 0) | df["latency_window_ms"].isnull()).any():
                mode_rep["errors"].append("invalid_latency_window")
            if "energy_Wh" in df.columns and ((df["energy_Wh"] <= 0) | df["energy_Wh"].isnull()).any():
                mode_rep["errors"].append("invalid_energy")
            if "energy_window_Wh" in df.columns and ((df["energy_window_Wh"] <= 0) | df["energy_window_Wh"].isnull()).any():
                mode_rep["errors"].append("invalid_energy_window")

            powerint_enabled = bool(cfg.get("measurement", {}).get("power_integration_enabled", False))
            power_min_samples = int(cfg.get("measurement", {}).get("power_min_samples", 20))
            if powerint_enabled:
                if "energy_window_Wh_powerint" in df.columns and ((df["energy_window_Wh_powerint"] <= 0) | df["energy_window_Wh_powerint"].isnull()).any():
                    mode_rep["errors"].append("invalid_energy_window_powerint")
                if "power_samples" in df.columns and (df["power_samples"] < power_min_samples).any():
                    mode_rep["errors"].append("insufficient_power_samples")
                if "energy_backend_rel_err" in df.columns and df["energy_backend_rel_err"].isnull().any():
                    mode_rep["errors"].append("missing_energy_backend_rel_err")

            if {"repeat_per_request", "latency_window_ms", "latency_per_infer_ms"}.issubset(df.columns):
                lat_recon = (df["latency_per_infer_ms"] * df["repeat_per_request"] - df["latency_window_ms"]).abs()
                if (lat_recon > 1e-3).any():
                    mode_rep["errors"].append("latency_window_mismatch")
            if {"repeat_per_request", "energy_window_Wh", "energy_per_infer_Wh"}.issubset(df.columns):
                ene_recon = (df["energy_per_infer_Wh"] * df["repeat_per_request"] - df["energy_window_Wh"]).abs()
                if (ene_recon > 1e-9).any():
                    mode_rep["errors"].append("energy_window_mismatch")
            if {"energy_window_Wh", "latency_window_ms", "implied_power_W"}.issubset(df.columns):
                valid = (df["latency_window_ms"] > 0) & df["energy_window_Wh"].notnull() & df["implied_power_W"].notnull()
                implied_ref = df.loc[valid, "energy_window_Wh"] * 3_600_000.0 / df.loc[valid, "latency_window_ms"]
                implied_err = (implied_ref - df.loc[valid, "implied_power_W"]).abs()
                if (implied_err > 1e-6).any():
                    mode_rep["errors"].append("implied_power_mismatch")

            if len(df) != measure_n:
                mode_rep["errors"].append(f"row_count_mismatch:{len(df)}!= {measure_n}")

        if mode_rep["errors"]:
            report["pass"] = False
        report["modes"].append(mode_rep)

    report["checks"]["mode_summary_csv"] = (workload_dir / "mode_summary.csv").exists()
    report["checks"]["admitted_modes_csv"] = (workload_dir / "admitted_modes.csv").exists()
    need_switch_summary = len(modes) >= 2
    report["checks"]["switch_penalty_required"] = bool(need_switch_summary)
    report["checks"]["switch_penalty_summary_csv"] = (
        (workload_dir / "switch_penalty" / "summary.csv").exists() if need_switch_summary else True
    )

    check_values = [v for k, v in report["checks"].items() if k != "switch_penalty_required"]
    if not all(bool(v) for v in check_values):
        report["pass"] = False

    out = Path(output_yaml) if output_yaml else (workload_dir / "validation_report.yaml")
    dump_yaml(out, report)
    return out













