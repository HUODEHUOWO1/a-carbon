from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import dump_yaml, load_yaml
from ..sim.fcfs import Arrival, simulate_fcfs


def _quantiles(series: pd.Series) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(series, 50)),
        "p95_ms": float(np.percentile(series, 95)),
        "p99_ms": float(np.percentile(series, 99)),
    }


def calibrate_simulator(
    matrix_path: str | Path,
    live_log_csv: str | Path,
    output_root: str | Path,
    slo_ms: float,
) -> Path:
    matrix = load_yaml(matrix_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    live = pd.read_csv(live_log_csv)
    required = {
        "request_id",
        "arrival_ts",
        "start_ts",
        "finish_ts",
        "service_ms",
    }
    missing = required - set(live.columns)
    if missing:
        raise ValueError(f"Missing fields in live log: {sorted(missing)}")

    arrivals = [
        Arrival(
            request_id=int(r.request_id),
            arrival_ts_ms=float(r.arrival_ts),
            service_ms=float(r.service_ms),
        )
        for r in live.itertuples(index=False)
    ]
    sim = simulate_fcfs(arrivals)
    sim_df = pd.DataFrame([c.__dict__ for c in sim])

    live_latency = live["finish_ts"] - live["arrival_ts"]
    sim_latency = sim_df["finish_ts_ms"] - sim_df["arrival_ts_ms"]

    live_q = _quantiles(live_latency)
    sim_q = _quantiles(sim_latency)

    live_violation = float((live_latency > slo_ms).mean())
    sim_violation = float((sim_latency > slo_ms).mean())

    acceptance = matrix["e0"]["simulator_acceptance"]
    p50_rel = abs(sim_q["p50_ms"] - live_q["p50_ms"]) / max(live_q["p50_ms"], 1e-9)
    p95_rel = abs(sim_q["p95_ms"] - live_q["p95_ms"]) / max(live_q["p95_ms"], 1e-9)
    vio_abs_pp = abs(sim_violation - live_violation) * 100.0

    report: dict[str, Any] = {
        "live": {**live_q, "violation_rate": live_violation},
        "sim": {**sim_q, "violation_rate": sim_violation},
        "errors": {
            "p50_relative": p50_rel,
            "p95_relative": p95_rel,
            "violation_abs_pp": vio_abs_pp,
        },
        "acceptance": {
            "p50_relative_error_max": acceptance["p50_relative_error_max"],
            "p95_relative_error_max": acceptance["p95_relative_error_max"],
            "violation_abs_error_max_pp": acceptance["violation_abs_error_max_pp"],
        },
        "pass": bool(
            p50_rel <= float(acceptance["p50_relative_error_max"])
            and p95_rel <= float(acceptance["p95_relative_error_max"])
            and vio_abs_pp <= float(acceptance["violation_abs_error_max_pp"])
        ),
    }

    out = output_root / "simulator_calibration_report.yaml"
    dump_yaml(out, report)

    merged = live[["request_id", "arrival_ts", "start_ts", "finish_ts", "service_ms"]].copy()
    sim_aligned = sim_df.rename(
        columns={
            "start_ts_ms": "sim_start_ts",
            "finish_ts_ms": "sim_finish_ts",
            "queue_wait_ms": "sim_queue_wait_ms",
        }
    )
    merged = merged.merge(sim_aligned[["request_id", "sim_start_ts", "sim_finish_ts", "sim_queue_wait_ms"]], on="request_id", how="left")
    merged.to_csv(output_root / "sim_vs_live.csv", index=False)

    return out