from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..analysis.stats import run_default_significance_tests, summarize_main_metrics
from ..e1.runner import run_e1


def run_e2(
    matrix_path: str | Path,
    admitted_modes_csv: str | Path,
    cache_csv: str | Path,
    traces_root: str | Path,
    carbon_csv: str | Path,
    output_root: str | Path,
    switch_penalty_yaml: str | Path | None = None,
    controllers_filter: list[str] | None = None,
    regions_filter: list[str] | None = None,
    seasons_filter: list[str] | None = None,
    max_traces_per_workload: int | None = None,
    max_requests_per_trace: int | None = None,
) -> tuple[Path, Path, Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    controllers = controllers_filter or ["reactive_joint", "forecast_budgeted_joint", "static_hq"]

    req_csv, raw_summary_csv = run_e1(
        matrix_path=matrix_path,
        admitted_modes_csv=admitted_modes_csv,
        cache_csv=cache_csv,
        traces_root=traces_root,
        carbon_csv=carbon_csv,
        output_root=output_root,
        switch_penalty_yaml=switch_penalty_yaml,
        load_name="burst",
        controllers_filter=controllers,
        regions_filter=regions_filter,
        seasons_filter=seasons_filter,
        max_traces_per_workload=max_traces_per_workload,
        max_requests_per_trace=max_requests_per_trace,
        scenario_name="e2_burst",
        request_log_name="e2_request_log.csv",
        summary_name="e2_summary_raw.csv",
    )

    ci_summary_csv = output_root / "e2_summary_ci95.csv"
    summarize_main_metrics(raw_summary_csv, ci_summary_csv)

    sig_csv = output_root / "e2_significance.csv"
    run_default_significance_tests(raw_summary_csv, sig_csv)

    # Keep a user-friendly flattened summary for quick inspection.
    raw_df = pd.read_csv(raw_summary_csv)
    quick = (
        raw_df.groupby(["controller", "workload", "region", "season", "load"], as_index=False)
        .agg(
            {
                "carbon_per_goodput_g": "mean",
                "latency_p95_ms": "mean",
                "latency_p99_ms": "mean",
                "slo_violation_rate": "mean",
                "switch_rate": "mean",
                "controller_overhead_us_mean": "mean",
                "forecast_mae": "mean",
            }
        )
        .sort_values(["workload", "region", "season", "controller"])
    )
    quick_csv = output_root / "e2_summary_quick.csv"
    quick.to_csv(quick_csv, index=False)

    return req_csv, raw_summary_csv, ci_summary_csv, sig_csv