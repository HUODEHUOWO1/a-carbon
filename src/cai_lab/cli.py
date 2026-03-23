from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path

import pandas as pd

from .analysis.stats import run_default_significance_tests, summarize_main_metrics
from .carbon.electricity_maps import fetch_matrix_carbon_data
from .config import load_yaml
from .e0.cache import build_offline_cache
from .e0.calibration import calibrate_simulator
from .e0.profiling import run_mode_profiling, run_mode_pruning
from .e0.real_profile import (
    export_real_profiling_to_precomputed,
    run_real_profiling,
    run_real_switch_penalty,
    validate_real_profiling,
)
from .e0.switch_penalty import build_default_switch_matrix, measure_switch_penalty
from .e1.runner import run_e1
from .e2.runner import run_e2
from .e3.runner import run_e3
from .e4.runner import run_e4
from .matrix import freeze_matrix
from .traces import generate_traces_from_matrix


def cmd_freeze_matrix(args: argparse.Namespace) -> None:
    frozen = freeze_matrix(args.matrix, args.output)
    print(f"frozen_matrix={args.output or frozen['paths']['frozen_matrix']}")


def cmd_e0_profile(args: argparse.Namespace) -> None:
    out = run_mode_profiling(
        matrix_path=args.matrix,
        output_root=args.output,
        precomputed_root=args.precomputed_root,
        synthetic_fallback=args.synthetic,
        strict_precomputed=args.strict_precomputed,
    )
    print(f"profile_summary={out}")


def cmd_e0_prune(args: argparse.Namespace) -> None:
    csv_out, yaml_out = run_mode_pruning(args.matrix, args.profile_summary, args.output)
    print(f"admitted_csv={csv_out}")
    print(f"admitted_yaml={yaml_out}")


def cmd_e0_cache(args: argparse.Namespace) -> None:
    out = build_offline_cache(args.profile_root, args.admitted_csv, args.output)
    print(f"offline_cache={out}")


def cmd_e0_switch_default(args: argparse.Namespace) -> None:
    out = build_default_switch_matrix(args.admitted_csv, args.output)
    print(f"switch_penalty_yaml={out}")


def cmd_e0_switch_measure(args: argparse.Namespace) -> None:
    out = measure_switch_penalty(
        aaaa_csv=args.aaaa,
        abab_csv=args.abab,
        output_yaml=args.output,
        mode_from=args.mode_from,
        mode_to=args.mode_to,
    )
    print(f"switch_penalty_yaml={out}")


def cmd_e0_calibrate(args: argparse.Namespace) -> None:
    out = calibrate_simulator(
        matrix_path=args.matrix,
        live_log_csv=args.live_log,
        output_root=args.output,
        slo_ms=args.slo_ms,
    )
    print(f"calibration_report={out}")


def _read_api_key_file(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"API key file not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    return text or None


def _read_api_key_from_secrets(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = load_yaml(p)
    key = data.get("electricity_maps_api_key")
    if key is None:
        return None
    k = str(key).strip()
    return k or None


def _resolve_electricity_maps_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return str(args.api_key).strip()

    from_file = _read_api_key_file(args.api_key_file)
    if from_file:
        return from_file

    from_env = os.getenv("ELECTRICITY_MAPS_API_KEY")
    if from_env and from_env.strip():
        return from_env.strip()

    from_secrets = _read_api_key_from_secrets(args.secrets_file)
    if from_secrets:
        return from_secrets

    return None


def cmd_carbon_fetch(args: argparse.Namespace) -> None:
    api_key = _resolve_electricity_maps_api_key(args)
    if not api_key:
        raise RuntimeError(
            "Missing Electricity Maps API key. Use one of: "
            "--api-key, --api-key-file, env ELECTRICITY_MAPS_API_KEY, or configs/secrets.yaml(electricity_maps_api_key)."
        )

    carbon_csv, meta_yaml = fetch_matrix_carbon_data(
        matrix_path=args.matrix,
        output_root=args.output,
        api_key=api_key,
        base_url=args.base_url,
    )
    print(f"carbon_csv={carbon_csv}")
    print(f"carbon_meta={meta_yaml}")


def cmd_trace_generate(args: argparse.Namespace) -> None:
    mu_ref = {}
    for item in args.mu_ref:
        workload, val = item.split("=", 1)
        mu_ref[workload] = float(val)
    outputs = generate_traces_from_matrix(args.matrix, args.output, mu_ref_rps=mu_ref)
    print(f"generated_traces={len(outputs)}")


def _split_opt(items: list[str] | None) -> list[str] | None:
    if items is None:
        return None
    out: list[str] = []
    for i in items:
        if "," in i:
            out.extend([x.strip() for x in i.split(",") if x.strip()])
        else:
            out.append(i)
    return out


def _parse_pairs(items: list[str] | None) -> list[tuple[str, str]] | None:
    if not items:
        return None
    pairs: list[tuple[str, str]] = []
    for raw in items:
        parts = [x.strip() for x in raw.split(":")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid pair format: {raw}. Expected modeA:modeB")
        pairs.append((parts[0], parts[1]))
    return pairs


def cmd_profile_real_run(args: argparse.Namespace) -> None:
    mode_summary, admitted, status = run_real_profiling(
        config_path=args.config,
        mode_ids=_split_opt(args.mode_ids),
        allow_partial_modes=args.allow_partial_modes,
    )
    print(f"real_mode_summary={mode_summary}")
    print(f"real_admitted_modes={admitted}")
    print(f"real_mode_status={status}")


def cmd_profile_real_switch(args: argparse.Namespace) -> None:
    summary_csv, switch_yaml = run_real_switch_penalty(
        config_path=args.config,
        pairs=_parse_pairs(args.pairs),
        n_requests=args.n_requests,
    )
    print(f"real_switch_summary={summary_csv}")
    print(f"real_switch_yaml={switch_yaml}")


def cmd_profile_real_export(args: argparse.Namespace) -> None:
    out = export_real_profiling_to_precomputed(args.config, args.output_root)
    print(f"real_export_precomputed_dir={out}")


def cmd_profile_real_validate(args: argparse.Namespace) -> None:
    out = validate_real_profiling(args.config, output_yaml=args.output)
    print(f"real_validation_report={out}")


def cmd_e1_run(args: argparse.Namespace) -> None:
    req, summary = run_e1(
        matrix_path=args.matrix,
        admitted_modes_csv=args.admitted_csv,
        cache_csv=args.cache_csv,
        traces_root=args.traces_root,
        carbon_csv=args.carbon_csv,
        output_root=args.output,
        switch_penalty_yaml=args.switch_penalty_yaml,
        load_name=args.load,
        controllers_filter=_split_opt(args.controllers),
        regions_filter=_split_opt(args.regions),
        seasons_filter=_split_opt(args.seasons),
        max_traces_per_workload=args.max_traces_per_workload,
        max_requests_per_trace=args.max_requests_per_trace,
        telemetry_delay_minutes=args.telemetry_delay_minutes,
        telemetry_noise_std=args.telemetry_noise_std,
        telemetry_bias=args.telemetry_bias,
        telemetry_missing_rate=args.telemetry_missing_rate,
        telemetry_fallback=args.telemetry_fallback,
        telemetry_forecast_horizon_minutes=args.telemetry_forecast_horizon_minutes,
        telemetry_seed=args.telemetry_seed,
        scenario_name=args.scenario_name,
        request_log_name=args.request_log_name,
        summary_name=args.summary_name,
    )
    print(f"e1_request_log={req}")
    print(f"e1_summary={summary}")


def cmd_e2_run(args: argparse.Namespace) -> None:
    req, raw, ci, sig = run_e2(
        matrix_path=args.matrix,
        admitted_modes_csv=args.admitted_csv,
        cache_csv=args.cache_csv,
        traces_root=args.traces_root,
        carbon_csv=args.carbon_csv,
        output_root=args.output,
        switch_penalty_yaml=args.switch_penalty_yaml,
        controllers_filter=_split_opt(args.controllers),
        regions_filter=_split_opt(args.regions),
        seasons_filter=_split_opt(args.seasons),
        max_traces_per_workload=args.max_traces_per_workload,
        max_requests_per_trace=args.max_requests_per_trace,
    )
    print(f"e2_request_log={req}")
    print(f"e2_summary_raw={raw}")
    print(f"e2_summary_ci95={ci}")
    print(f"e2_significance={sig}")


def cmd_e3_run(args: argparse.Namespace) -> None:
    raw, ci, drift, manifest = run_e3(
        matrix_path=args.matrix,
        admitted_modes_csv=args.admitted_csv,
        cache_csv=args.cache_csv,
        traces_root=args.traces_root,
        carbon_csv=args.carbon_csv,
        output_root=args.output,
        switch_penalty_yaml=args.switch_penalty_yaml,
        controllers_filter=_split_opt(args.controllers),
        loads_filter=_split_opt(args.loads),
        regions_filter=_split_opt(args.regions),
        seasons_filter=_split_opt(args.seasons),
        max_traces_per_workload=args.max_traces_per_workload,
        max_requests_per_trace=args.max_requests_per_trace,
        telemetry_seed=args.telemetry_seed,
    )
    print(f"e3_summary_raw={raw}")
    print(f"e3_summary_ci95={ci}")
    print(f"e3_drift={drift}")
    print(f"e3_scenarios={manifest}")


def cmd_e4_run(args: argparse.Namespace) -> None:
    req, raw, ci, fair_raw, fair_ci, sig = run_e4(
        matrix_path=args.matrix,
        admitted_modes_csv=args.admitted_csv,
        cache_csv=args.cache_csv,
        traces_root=args.traces_root,
        carbon_csv=args.carbon_csv,
        output_root=args.output,
        switch_penalty_yaml=args.switch_penalty_yaml,
        regions_filter=_split_opt(args.regions),
        seasons_filter=_split_opt(args.seasons),
        max_traces_per_workload=args.max_traces_per_workload,
        max_requests_per_trace=args.max_requests_per_trace,
    )
    print(f"e4_request_log={req}")
    print(f"e4_summary_raw={raw}")
    print(f"e4_summary_ci95={ci}")
    print(f"e4_fairness_raw={fair_raw}")
    print(f"e4_fairness_ci95={fair_ci}")
    print(f"e4_significance={sig}")


def cmd_stats_summary(args: argparse.Namespace) -> None:
    out = summarize_main_metrics(args.summary_csv, args.output)
    print(f"stats_summary={out}")


def cmd_stats_significance(args: argparse.Namespace) -> None:
    out = run_default_significance_tests(args.summary_csv, args.output, seed=args.seed)
    print(f"stats_significance={out}")


def _check_module(name: str) -> str:
    try:
        __import__(name)
        return "ok"
    except Exception as exc:
        return f"missing ({exc.__class__.__name__}: {exc})"


def cmd_env_check(args: argparse.Namespace) -> None:
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": {
            "numpy": _check_module("numpy"),
            "pandas": _check_module("pandas"),
            "yaml": _check_module("yaml"),
            "requests": _check_module("requests"),
            "torch": _check_module("torch"),
            "torchvision": _check_module("torchvision"),
            "transformers": _check_module("transformers"),
            "pynvml": _check_module("pynvml"),
        },
    }

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        report["nvidia_smi"] = [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        report["nvidia_smi"] = f"unavailable ({exc.__class__.__name__}: {exc})"

    print(pd.Series(report).to_string())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cai-lab")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("freeze-matrix")
    s.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    s.add_argument("--output", default="runs/frozen_matrix.yaml")
    s.set_defaults(func=cmd_freeze_matrix)

    s = sub.add_parser("e0-profile")
    s.add_argument("--matrix", default="runs/frozen_matrix.yaml")
    s.add_argument("--output", default="runs/e0/profiling")
    s.add_argument("--precomputed-root", default=None)
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--strict-precomputed", action="store_true")
    s.set_defaults(func=cmd_e0_profile)

    s = sub.add_parser("e0-prune")
    s.add_argument("--matrix", default="runs/frozen_matrix.yaml")
    s.add_argument("--profile-summary", default="runs/e0/profiling/summary.csv")
    s.add_argument("--output", default="runs/e0/profiling")
    s.set_defaults(func=cmd_e0_prune)

    s = sub.add_parser("e0-cache")
    s.add_argument("--profile-root", default="runs/e0/profiling")
    s.add_argument("--admitted-csv", default="runs/e0/profiling/admitted_modes.csv")
    s.add_argument("--output", default="runs/e0/cache")
    s.set_defaults(func=cmd_e0_cache)

    s = sub.add_parser("e0-switch-default")
    s.add_argument("--admitted-csv", default="runs/e0/profiling/admitted_modes.csv")
    s.add_argument("--output", default="runs/e0/switch_penalty/switch_penalties.yaml")
    s.set_defaults(func=cmd_e0_switch_default)

    s = sub.add_parser("e0-switch-measure")
    s.add_argument("--aaaa", required=True)
    s.add_argument("--abab", required=True)
    s.add_argument("--mode-from", required=True)
    s.add_argument("--mode-to", required=True)
    s.add_argument("--output", default="runs/e0/switch_penalty/measured_switch_penalty.yaml")
    s.set_defaults(func=cmd_e0_switch_measure)

    s = sub.add_parser("e0-calibrate")
    s.add_argument("--matrix", default="runs/frozen_matrix.yaml")
    s.add_argument("--live-log", required=True)
    s.add_argument("--slo-ms", type=float, required=True)
    s.add_argument("--output", default="runs/e0/calibration")
    s.set_defaults(func=cmd_e0_calibrate)

    s = sub.add_parser("profile-real-run")
    s.add_argument("--config", required=True)
    s.add_argument("--mode-ids", nargs="+", default=None)
    s.add_argument("--allow-partial-modes", action="store_true")
    s.set_defaults(func=cmd_profile_real_run)

    s = sub.add_parser("profile-real-switch")
    s.add_argument("--config", required=True)
    s.add_argument("--pairs", nargs="+", default=None, help="modeA:modeB modeC:modeD")
    s.add_argument("--n-requests", type=int, default=2000)
    s.set_defaults(func=cmd_profile_real_switch)

    s = sub.add_parser("profile-real-export")
    s.add_argument("--config", required=True)
    s.add_argument("--output-root", default="runs/input_profiles")
    s.set_defaults(func=cmd_profile_real_export)

    s = sub.add_parser("profile-real-validate")
    s.add_argument("--config", required=True)
    s.add_argument("--output", default=None)
    s.set_defaults(func=cmd_profile_real_validate)

    s = sub.add_parser("carbon-fetch")
    s.add_argument("--matrix", default="runs/frozen_matrix.yaml")
    s.add_argument("--output", default="runs/carbon")
    s.add_argument("--api-key", default=None)
    s.add_argument("--api-key-file", default=None)
    s.add_argument("--secrets-file", default="configs/secrets.yaml")
    s.add_argument("--base-url", default="https://api.electricitymap.org/v3")
    s.set_defaults(func=cmd_carbon_fetch)

    s = sub.add_parser("trace-generate")
    s.add_argument("--matrix", default="runs/frozen_matrix.yaml")
    s.add_argument("--output", default="runs/traces")
    s.add_argument(
        "--mu-ref",
        nargs="+",
        default=["vision=55", "nlp=90"],
        help="workload=req_per_sec, e.g. vision=55 nlp=90",
    )
    s.set_defaults(func=cmd_trace_generate)

    s = sub.add_parser("e1-run")
    s.add_argument("--matrix", default="runs/frozen_matrix.yaml")
    s.add_argument("--admitted-csv", default="runs/e0/profiling/admitted_modes.csv")
    s.add_argument("--cache-csv", default="runs/e0/cache/offline_cache.csv")
    s.add_argument("--traces-root", default="runs/traces")
    s.add_argument("--carbon-csv", default="runs/carbon/electricity_maps_carbon.csv")
    s.add_argument("--switch-penalty-yaml", default="runs/e0/switch_penalty/switch_penalties.yaml")
    s.add_argument("--load", default="nominal", choices=["nominal", "burst"])
    s.add_argument("--controllers", nargs="+", default=None)
    s.add_argument("--regions", nargs="+", default=None)
    s.add_argument("--seasons", nargs="+", default=None)
    s.add_argument("--max-traces-per-workload", type=int, default=None)
    s.add_argument("--max-requests-per-trace", type=int, default=None)
    s.add_argument("--telemetry-delay-minutes", type=int, default=0)
    s.add_argument("--telemetry-noise-std", type=float, default=0.0)
    s.add_argument("--telemetry-bias", type=float, default=1.0)
    s.add_argument("--telemetry-missing-rate", type=float, default=0.0)
    s.add_argument(
        "--telemetry-fallback",
        type=str,
        default="locf",
        choices=["locf", "rolling_hourly_median", "conservative_default"],
    )
    s.add_argument("--telemetry-forecast-horizon-minutes", type=int, default=60)
    s.add_argument("--telemetry-seed", type=int, default=0)
    s.add_argument("--scenario-name", default="e1")
    s.add_argument("--request-log-name", default="e1_request_log.csv")
    s.add_argument("--summary-name", default="e1_summary.csv")
    s.add_argument("--output", default="runs/e1")
    s.set_defaults(func=cmd_e1_run)

    s = sub.add_parser("e2-run")
    s.add_argument("--matrix", default="runs/frozen_matrix.yaml")
    s.add_argument("--admitted-csv", default="runs/e0/profiling/admitted_modes.csv")
    s.add_argument("--cache-csv", default="runs/e0/cache/offline_cache.csv")
    s.add_argument("--traces-root", default="runs/traces")
    s.add_argument("--carbon-csv", default="runs/carbon/electricity_maps_carbon.csv")
    s.add_argument("--switch-penalty-yaml", default="runs/e0/switch_penalty/switch_penalties.yaml")
    s.add_argument("--controllers", nargs="+", default=None)
    s.add_argument("--regions", nargs="+", default=None)
    s.add_argument("--seasons", nargs="+", default=None)
    s.add_argument("--max-traces-per-workload", type=int, default=None)
    s.add_argument("--max-requests-per-trace", type=int, default=None)
    s.add_argument("--output", default="runs/e2")
    s.set_defaults(func=cmd_e2_run)

    s = sub.add_parser("e3-run")
    s.add_argument("--matrix", default="runs/frozen_matrix.yaml")
    s.add_argument("--admitted-csv", default="runs/e0/profiling/admitted_modes.csv")
    s.add_argument("--cache-csv", default="runs/e0/cache/offline_cache.csv")
    s.add_argument("--traces-root", default="runs/traces")
    s.add_argument("--carbon-csv", default="runs/carbon/electricity_maps_carbon.csv")
    s.add_argument("--switch-penalty-yaml", default="runs/e0/switch_penalty/switch_penalties.yaml")
    s.add_argument("--controllers", nargs="+", default=None)
    s.add_argument("--loads", nargs="+", default=None)
    s.add_argument("--regions", nargs="+", default=None)
    s.add_argument("--seasons", nargs="+", default=None)
    s.add_argument("--max-traces-per-workload", type=int, default=None)
    s.add_argument("--max-requests-per-trace", type=int, default=None)
    s.add_argument("--telemetry-seed", type=int, default=7)
    s.add_argument("--output", default="runs/e3")
    s.set_defaults(func=cmd_e3_run)

    s = sub.add_parser("e4-run")
    s.add_argument("--matrix", default="runs/frozen_matrix.yaml")
    s.add_argument("--admitted-csv", default="runs/e0/profiling/admitted_modes.csv")
    s.add_argument("--cache-csv", default="runs/e0/cache/offline_cache.csv")
    s.add_argument("--traces-root", default="runs/traces")
    s.add_argument("--carbon-csv", default="runs/carbon/electricity_maps_carbon.csv")
    s.add_argument("--switch-penalty-yaml", default="runs/e0/switch_penalty/switch_penalties.yaml")
    s.add_argument("--regions", nargs="+", default=None)
    s.add_argument("--seasons", nargs="+", default=None)
    s.add_argument("--max-traces-per-workload", type=int, default=None)
    s.add_argument("--max-requests-per-trace", type=int, default=None)
    s.add_argument("--output", default="runs/e4")
    s.set_defaults(func=cmd_e4_run)

    s = sub.add_parser("stats-summary")
    s.add_argument("--summary-csv", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_stats_summary)

    s = sub.add_parser("stats-significance")
    s.add_argument("--summary-csv", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_stats_significance)

    s = sub.add_parser("env-check")
    s.set_defaults(func=cmd_env_check)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()