from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..analysis.stats import summarize_main_metrics
from ..e1.runner import run_e1


def _e3_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "baseline",
            "telemetry_delay_minutes": 0,
            "telemetry_noise_std": 0.0,
            "telemetry_bias": 1.0,
            "telemetry_missing_rate": 0.0,
            "telemetry_fallback": "locf",
        }
    ]

    for d in [15, 60, 180]:
        scenarios.append(
            {
                "scenario": f"delay_{d}m",
                "telemetry_delay_minutes": d,
                "telemetry_noise_std": 0.0,
                "telemetry_bias": 1.0,
                "telemetry_missing_rate": 0.0,
                "telemetry_fallback": "locf",
            }
        )

    for n in [0.05, 0.10, 0.20]:
        scenarios.append(
            {
                "scenario": f"noise_{str(n).replace('.', 'p')}",
                "telemetry_delay_minutes": 0,
                "telemetry_noise_std": n,
                "telemetry_bias": 1.0,
                "telemetry_missing_rate": 0.0,
                "telemetry_fallback": "locf",
            }
        )

    for b in [0.9, 1.1, 0.7, 1.3]:
        scenarios.append(
            {
                "scenario": f"bias_{str(b).replace('.', 'p')}",
                "telemetry_delay_minutes": 0,
                "telemetry_noise_std": 0.0,
                "telemetry_bias": b,
                "telemetry_missing_rate": 0.0,
                "telemetry_fallback": "locf",
            }
        )

    for m in [0.10, 0.30, 0.50]:
        for fb in ["locf", "rolling_hourly_median", "conservative_default"]:
            scenarios.append(
                {
                    "scenario": f"missing_{int(m*100)}_{fb}",
                    "telemetry_delay_minutes": 0,
                    "telemetry_noise_std": 0.0,
                    "telemetry_bias": 1.0,
                    "telemetry_missing_rate": m,
                    "telemetry_fallback": fb,
                }
            )

    return scenarios


def run_e3(
    matrix_path: str | Path,
    admitted_modes_csv: str | Path,
    cache_csv: str | Path,
    traces_root: str | Path,
    carbon_csv: str | Path,
    output_root: str | Path,
    switch_penalty_yaml: str | Path | None = None,
    controllers_filter: list[str] | None = None,
    loads_filter: list[str] | None = None,
    regions_filter: list[str] | None = None,
    seasons_filter: list[str] | None = None,
    max_traces_per_workload: int | None = None,
    max_requests_per_trace: int | None = None,
    telemetry_seed: int = 7,
) -> tuple[Path, Path, Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    controllers = controllers_filter or ["reactive_joint", "forecast_budgeted_joint"]
    loads = loads_filter or ["nominal", "burst"]

    scenarios = _e3_scenarios()
    manifest = pd.DataFrame(scenarios)
    manifest_csv = output_root / "e3_scenarios.csv"
    manifest.to_csv(manifest_csv, index=False)

    summary_frames = []
    for sc in scenarios:
        for load_name in loads:
            sc_name = f"e3_{sc['scenario']}_{load_name}"
            run_dir = output_root / sc_name
            run_dir.mkdir(parents=True, exist_ok=True)
            _, sum_csv = run_e1(
                matrix_path=matrix_path,
                admitted_modes_csv=admitted_modes_csv,
                cache_csv=cache_csv,
                traces_root=traces_root,
                carbon_csv=carbon_csv,
                output_root=run_dir,
                switch_penalty_yaml=switch_penalty_yaml,
                load_name=load_name,
                controllers_filter=controllers,
                regions_filter=regions_filter,
                seasons_filter=seasons_filter,
                max_traces_per_workload=max_traces_per_workload,
                max_requests_per_trace=max_requests_per_trace,
                telemetry_delay_minutes=int(sc["telemetry_delay_minutes"]),
                telemetry_noise_std=float(sc["telemetry_noise_std"]),
                telemetry_bias=float(sc["telemetry_bias"]),
                telemetry_missing_rate=float(sc["telemetry_missing_rate"]),
                telemetry_fallback=str(sc["telemetry_fallback"]),
                telemetry_forecast_horizon_minutes=60,
                telemetry_seed=telemetry_seed,
                scenario_name=sc_name,
                request_log_name="e3_request_log.csv",
                summary_name="e3_summary_raw.csv",
            )
            s = pd.read_csv(sum_csv)
            s["scenario_short"] = sc["scenario"]
            s["telemetry_delay_minutes"] = sc["telemetry_delay_minutes"]
            s["telemetry_noise_std"] = sc["telemetry_noise_std"]
            s["telemetry_bias"] = sc["telemetry_bias"]
            s["telemetry_missing_rate"] = sc["telemetry_missing_rate"]
            s["telemetry_fallback"] = sc["telemetry_fallback"]
            summary_frames.append(s)

    all_summary = pd.concat(summary_frames, ignore_index=True)
    raw_csv = output_root / "e3_summary_raw.csv"
    all_summary.to_csv(raw_csv, index=False)

    ci_csv = output_root / "e3_summary_ci95.csv"
    summarize_main_metrics(raw_csv, ci_csv)

    base = all_summary[all_summary["scenario_short"] == "baseline"].copy()
    comp = all_summary[all_summary["scenario_short"] != "baseline"].copy()
    keys = ["controller", "workload", "region", "season", "load", "trace_file"]

    merged = comp.merge(
        base[keys + ["total_operational_carbon_g", "latency_p95_ms", "latency_p99_ms", "slo_violation_rate", "switch_rate"]],
        on=keys,
        suffixes=("", "_base"),
        how="left",
    )

    merged["carbon_drift_pct"] = (
        (merged["total_operational_carbon_g"] - merged["total_operational_carbon_g_base"])
        / merged["total_operational_carbon_g_base"].replace(0, pd.NA)
        * 100.0
    )
    merged["p95_drift_pct"] = (
        (merged["latency_p95_ms"] - merged["latency_p95_ms_base"])
        / merged["latency_p95_ms_base"].replace(0, pd.NA)
        * 100.0
    )
    merged["p99_drift_pct"] = (
        (merged["latency_p99_ms"] - merged["latency_p99_ms_base"])
        / merged["latency_p99_ms_base"].replace(0, pd.NA)
        * 100.0
    )
    merged["violation_drift_pp"] = (merged["slo_violation_rate"] - merged["slo_violation_rate_base"]) * 100.0
    merged["switch_rate_drift"] = merged["switch_rate"] - merged["switch_rate_base"]

    drift_cols = [
        "scenario_short",
        "controller",
        "workload",
        "region",
        "season",
        "load",
        "trace_file",
        "carbon_drift_pct",
        "p95_drift_pct",
        "p99_drift_pct",
        "violation_drift_pp",
        "switch_rate_drift",
        "telemetry_delay_minutes",
        "telemetry_noise_std",
        "telemetry_bias",
        "telemetry_missing_rate",
        "telemetry_fallback",
    ]
    drift = merged[drift_cols].copy()
    drift_csv = output_root / "e3_drift.csv"
    drift.to_csv(drift_csv, index=False)

    return raw_csv, ci_csv, drift_csv, manifest_csv