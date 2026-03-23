from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedBootstrapResult:
    metric: str
    controller_a: str
    controller_b: str
    n_pairs: int
    mean_diff: float
    ci95_low: float
    ci95_high: float
    p_value_two_sided: float


def add_mean_ci(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {k: v for k, v in zip(group_cols, keys)}
        n = int(len(g))
        row["n"] = n
        for c in value_cols:
            vals = g[c].astype(float).to_numpy()
            mean = float(np.mean(vals))
            if n <= 1:
                se = 0.0
            else:
                se = float(np.std(vals, ddof=1) / np.sqrt(n))
            ci = 1.96 * se
            row[f"{c}_mean"] = mean
            row[f"{c}_ci95_low"] = mean - ci
            row[f"{c}_ci95_high"] = mean + ci
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(
    df: pd.DataFrame,
    metric_col: str,
    controller_a: str,
    controller_b: str,
    match_cols: list[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> PairedBootstrapResult:
    d = df[df["controller"].isin([controller_a, controller_b])].copy()
    if d.empty:
        return PairedBootstrapResult(metric_col, controller_a, controller_b, 0, np.nan, np.nan, np.nan, np.nan)

    pivot = d.pivot_table(index=match_cols, columns="controller", values=metric_col, aggfunc="mean")
    if controller_a not in pivot.columns or controller_b not in pivot.columns:
        return PairedBootstrapResult(metric_col, controller_a, controller_b, 0, np.nan, np.nan, np.nan, np.nan)

    paired = pivot[[controller_a, controller_b]].dropna()
    diffs = (paired[controller_a] - paired[controller_b]).to_numpy(dtype=float)
    n = len(diffs)
    if n == 0:
        return PairedBootstrapResult(metric_col, controller_a, controller_b, 0, np.nan, np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(diffs[idx]))

    mean_diff = float(np.mean(diffs))
    ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
    p_two = 2.0 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))

    return PairedBootstrapResult(
        metric=metric_col,
        controller_a=controller_a,
        controller_b=controller_b,
        n_pairs=n,
        mean_diff=mean_diff,
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        p_value_two_sided=float(min(max(p_two, 0.0), 1.0)),
    )


def run_default_significance_tests(
    summary_csv: str | Path,
    output_csv: str | Path,
    seed: int = 42,
) -> Path:
    df = pd.read_csv(summary_csv)
    tests = [
        ("reactive_joint", "static_eco"),
        ("forecast_budgeted_joint", "reactive_joint"),
        ("fair_joint_tenant", "forecast_budgeted_joint"),
    ]
    metrics = [
        "carbon_per_goodput_g",
        "latency_p95_ms",
        "latency_p99_ms",
        "slo_violation_rate",
    ]
    match_cols = ["scenario", "workload", "region", "season", "load", "trace_file"]

    rows: list[dict[str, float | str | int]] = []
    for a, b in tests:
        for m in metrics:
            r = paired_bootstrap(
                df=df,
                metric_col=m,
                controller_a=a,
                controller_b=b,
                match_cols=match_cols,
                n_boot=2000,
                seed=seed,
            )
            rows.append(
                {
                    "metric": r.metric,
                    "controller_a": r.controller_a,
                    "controller_b": r.controller_b,
                    "n_pairs": r.n_pairs,
                    "mean_diff": r.mean_diff,
                    "ci95_low": r.ci95_low,
                    "ci95_high": r.ci95_high,
                    "p_value_two_sided": r.p_value_two_sided,
                }
            )

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def summarize_main_metrics(summary_csv: str | Path, output_csv: str | Path) -> Path:
    df = pd.read_csv(summary_csv)
    group_cols = ["scenario", "controller", "workload", "region", "season", "load"]
    value_cols = [
        "total_operational_carbon_g",
        "carbon_per_successful_request_g",
        "carbon_per_goodput_g",
        "latency_p95_ms",
        "latency_p99_ms",
        "slo_violation_rate",
        "accuracy",
        "switch_rate",
        "controller_overhead_us_mean",
        "forecast_mae",
    ]
    out_df = add_mean_ci(df, group_cols=group_cols, value_cols=value_cols)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    return out