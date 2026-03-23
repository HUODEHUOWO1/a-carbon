from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    report: dict[str, object] = {
        "probe": "vision_precision_backend_tensorrt",
        "paper_use": False,
        "backend": "tensorrt",
        "status": "unknown",
    }

    try:
        import torch
        from torchvision import models
        import tensorrt as trt
    except Exception as exc:
        report["status"] = "failed_import"
        report["error"] = f"{exc.__class__.__name__}: {exc}"
        _write(report)
        return

    report["torch_version"] = getattr(torch, "__version__", "unknown")
    report["tensorrt_version"] = getattr(trt, "__version__", "unknown")
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    report["trt_fast_fp16"] = bool(builder.platform_has_fast_fp16)
    report["trt_fast_int8"] = bool(builder.platform_has_fast_int8)

    td_path = Path("runs/spike/tmp")
    td_path.mkdir(parents=True, exist_ok=True)
    onnx_path = td_path / "resnet50_probe.onnx"

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval().cpu()
    x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
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

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    ok = parser.parse(onnx_path.read_bytes())
    report["onnx_parse_ok"] = bool(ok)
    if not ok:
        errs = []
        for i in range(parser.num_errors):
            errs.append(str(parser.get_error(i)))
        report["onnx_parse_errors"] = errs
        report["status"] = "onnx_parse_failed"
        _write(report)
        return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config)
    report["fp16_engine_build_ok"] = engine_bytes is not None
    report["int8_candidate_ready"] = bool(builder.platform_has_fast_int8)
    report["status"] = "ok" if engine_bytes is not None else "fp16_engine_build_failed"

    _write(report)


def _write(report: dict[str, object]) -> None:
    out = Path("runs/spike")
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "vision_precision_backend_probe.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_file)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
