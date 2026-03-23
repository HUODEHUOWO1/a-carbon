from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..carbon.electricity_maps import load_carbon_timeseries
from ..config import dump_yaml, load_yaml
from ..controllers import (
    ControllerState,
    FairJointTenant,
    ForecastBudgetedJoint,
    ReactiveJoint,
    ReactivePrecision,
    StaticEco,
    StaticHQ,
    build_mode_summary,
)


@dataclass
class RequestResult:
    request_id: int
    arrival_ts: float
    start_ts: float
    finish_ts: float
    queue_wait_ms: float
    service_ms: float
    workload: str
    sample_id: int
    mode_id: str
    precision: str
    capacity_k: int
    prev_mode: str | None
    switch_flag: int
    tenant_id: str
    group_id: int
    actual_carbon_intensity: float
    observed_carbon_intensity: float
    forecast_carbon_intensity: float
    telemetry_missing_flag: int
    forecast_summary: str
    energy_Wh: float
    carbon_g: float
    success_flag: int
    slo_met: int
    prediction: int
    correctness: int
    controller_overhead_us: float


@dataclass(frozen=True)
class TelemetryDistortion:
    delay_minutes: int = 0
    additive_noise_std: float = 0.0
    multiplicative_bias: float = 1.0
    missing_rate: float = 0.0
    fallback: str = "locf"
    forecast_horizon_minutes: int = 60


class CarbonSignalSampler:
    def __init__(self, carbon_df: pd.DataFrame, season_start_utc: pd.Timestamp) -> None:
        c = carbon_df.sort_values("timestamp_utc").copy()
        c["t_ms"] = (pd.to_datetime(c["timestamp_utc"], utc=True) - season_start_utc).dt.total_seconds() * 1000.0
        self.t_ms = c["t_ms"].to_numpy(dtype=float)
        self.ci = c["carbon_intensity_g_per_kwh"].to_numpy(dtype=float)
        self.ts = pd.to_datetime(c["timestamp_utc"], utc=True).to_numpy()
        self.std = float(np.std(self.ci)) if len(self.ci) > 1 else 0.0
        self.global_median = float(np.median(self.ci))
        self.conservative_default = float(np.quantile(self.ci, 0.9))

    def _nearest_idx(self, t_ms: float) -> int:
        if t_ms <= self.t_ms[0]:
            return 0
        if t_ms >= self.t_ms[-1]:
            return len(self.t_ms) - 1
        i = int(np.searchsorted(self.t_ms, t_ms, side="left"))
        if i <= 0:
            return 0
        if i >= len(self.t_ms):
            return len(self.t_ms) - 1
        left = i - 1
        right = i
        if abs(self.t_ms[left] - t_ms) <= abs(self.t_ms[right] - t_ms):
            return left
        return right

    def ci_at(self, t_ms: float) -> float:
        return float(self.ci[self._nearest_idx(t_ms)])

    def forecast_mean(self, t_ms: float, horizon_minutes: int) -> float:
        h_ms = max(1, int(horizon_minutes)) * 60_000.0
        s = int(np.searchsorted(self.t_ms, t_ms, side="left"))
        e = int(np.searchsorted(self.t_ms, t_ms + h_ms, side="right"))
        if s >= len(self.ci):
            return float(self.ci[-1])
        if e <= s:
            e = min(len(self.ci), s + 1)
        return float(np.mean(self.ci[s:e]))

    def rolling_hourly_median(self, t_ms: float) -> float:
        idx = self._nearest_idx(t_ms)
        ts = pd.Timestamp(self.ts[idx])
        hour = int(ts.hour)
        past = self.ts[: idx + 1]
        if len(past) == 0:
            return self.global_median
        hours = pd.DatetimeIndex(past).hour
        mask = hours == hour
        vals = self.ci[: idx + 1][mask]
        if len(vals) == 0:
            return self.global_median
        return float(np.median(vals))


def _build_controller(
    name: str,
    mode_df: pd.DataFrame,
    ci_thresholds: tuple[float, float] | None = None,
):
    modes = build_mode_summary(mode_df, workload=str(mode_df.iloc[0]["workload"]))
    if ci_thresholds is None:
        t1, t2 = 180.0, 350.0
    else:
        t1, t2 = float(ci_thresholds[0]), float(ci_thresholds[1])
        if t2 <= t1:
            t1, t2 = 180.0, 350.0

    if name == "static_hq":
        return StaticHQ(modes)
    if name == "static_eco":
        return StaticEco(modes)
    if name == "reactive_precision":
        return ReactivePrecision(modes, t1=t1, t2=t2)
    if name == "reactive_joint":
        hysteresis = max(5.0, 0.05 * abs(t2 - t1))
        return ReactiveJoint(modes, t1=t1, t2=t2, hysteresis=hysteresis)
    if name == "forecast_budgeted_joint":
        return ForecastBudgetedJoint(modes)
    if name == "fair_joint_tenant":
        low_mode = sorted(modes, key=lambda m: m.energy_mean_Wh)[0].mode_id
        return FairJointTenant(ForecastBudgetedJoint(modes), low_mode_id=low_mode)
    raise ValueError(f"Unknown controller: {name}")


def _lookup_switch_penalty(
    switch_df: pd.DataFrame,
    workload: str,
    src: str | None,
    dst: str,
) -> tuple[float, float]:
    if src is None or src == dst or switch_df.empty:
        return 0.0, 0.0
    q = switch_df[
        (switch_df["workload"] == workload)
        & (switch_df["mode_from"] == src)
        & (switch_df["mode_to"] == dst)
    ]
    if q.empty:
        return 0.0, 0.0
    row = q.iloc[0]
    return float(row["latency_penalty_ms"]), float(row["energy_penalty_Wh"])


def _apply_ci_distortion(ci: float, distortion: TelemetryDistortion, ci_std: float, rng: np.random.Generator) -> float:
    noise = rng.normal(0.0, distortion.additive_noise_std * max(ci_std, 1e-6))
    out = ci * distortion.multiplicative_bias + noise
    return float(max(out, 1e-6))


def _fallback_ci(
    sampler: CarbonSignalSampler,
    t_ms: float,
    fallback: str,
    last_observed: float | None,
) -> float:
    if fallback == "locf":
        return float(last_observed if last_observed is not None else sampler.ci_at(t_ms))
    if fallback == "rolling_hourly_median":
        return float(sampler.rolling_hourly_median(t_ms))
    if fallback == "conservative_default":
        return float(sampler.conservative_default)
    raise ValueError(f"Unknown fallback: {fallback}")


def _observe_ci(
    sampler: CarbonSignalSampler,
    arrival_t_ms: float,
    distortion: TelemetryDistortion,
    last_observed: float | None,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    delayed_t = max(0.0, arrival_t_ms - max(0, distortion.delay_minutes) * 60_000.0)
    actual_ci = sampler.ci_at(arrival_t_ms)

    observed_raw = sampler.ci_at(delayed_t)
    forecast_raw = sampler.forecast_mean(delayed_t, distortion.forecast_horizon_minutes)

    observed = _apply_ci_distortion(observed_raw, distortion, sampler.std, rng)
    forecast = _apply_ci_distortion(forecast_raw, distortion, sampler.std, rng)

    missing = int(rng.random() < max(0.0, min(1.0, distortion.missing_rate)))
    if missing:
        fci = _fallback_ci(sampler, delayed_t, distortion.fallback, last_observed)
        observed = fci
        forecast = fci

    return actual_ci, observed, forecast, missing


def _summarize(
    df: pd.DataFrame,
    controller: str,
    region: str,
    season: str,
    workload: str,
    load_name: str,
    trace_file: str,
    scenario_name: str,
    slo_ms: float,
) -> dict[str, Any]:
    latency = df["finish_ts"] - df["arrival_ts"]
    total_carbon = float(df["carbon_g"].sum())
    n_success = int(df["success_flag"].sum())
    n_good = int(df["slo_met"].sum())
    goodput_sec = n_good / max((df["arrival_ts"].max() - df["arrival_ts"].min()) / 1000.0, 1e-9)

    return {
        "scenario": scenario_name,
        "controller": controller,
        "region": region,
        "season": season,
        "workload": workload,
        "load": load_name,
        "trace_file": trace_file,
        "n_requests": int(len(df)),
        "slo_ms": float(slo_ms),
        "total_operational_carbon_g": total_carbon,
        "carbon_per_successful_request_g": total_carbon / max(n_success, 1),
        "carbon_per_goodput_g": total_carbon / max(n_good, 1),
        "n_success": n_success,
        "n_slo_good": n_good,
        "goodput_per_sec": float(goodput_sec),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "latency_p99_ms": float(np.percentile(latency, 99)),
        "slo_violation_rate": float(1.0 - n_good / max(len(df), 1)),
        "accuracy": float(df["correctness"].mean()),
        "switch_rate": float(df["switch_flag"].mean()),
        "controller_overhead_us_mean": float(df["controller_overhead_us"].mean()),
        "forecast_mae": float(np.abs(df["forecast_carbon_intensity"] - df["actual_carbon_intensity"]).mean()),
        "observed_missing_rate": float(df["telemetry_missing_flag"].mean()),
    }


def run_e1(
    matrix_path: str | Path,
    admitted_modes_csv: str | Path,
    cache_csv: str | Path,
    traces_root: str | Path,
    carbon_csv: str | Path,
    output_root: str | Path,
    switch_penalty_yaml: str | Path | None = None,
    load_name: str = "nominal",
    controllers_filter: list[str] | None = None,
    regions_filter: list[str] | None = None,
    seasons_filter: list[str] | None = None,
    max_traces_per_workload: int | None = None,
    max_requests_per_trace: int | None = None,
    telemetry_delay_minutes: int = 0,
    telemetry_noise_std: float = 0.0,
    telemetry_bias: float = 1.0,
    telemetry_missing_rate: float = 0.0,
    telemetry_fallback: str = "locf",
    telemetry_forecast_horizon_minutes: int = 60,
    telemetry_seed: int = 0,
    scenario_name: str = "e1",
    request_log_name: str = "e1_request_log.csv",
    summary_name: str = "e1_summary.csv",
) -> tuple[Path, Path]:
    matrix = load_yaml(matrix_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    distortion = TelemetryDistortion(
        delay_minutes=telemetry_delay_minutes,
        additive_noise_std=telemetry_noise_std,
        multiplicative_bias=telemetry_bias,
        missing_rate=telemetry_missing_rate,
        fallback=telemetry_fallback,
        forecast_horizon_minutes=telemetry_forecast_horizon_minutes,
    )

    admitted = pd.read_csv(admitted_modes_csv)
    cache = pd.read_csv(cache_csv)

    if switch_penalty_yaml and Path(switch_penalty_yaml).exists():
        pen = load_yaml(switch_penalty_yaml)
        switch_df = pd.DataFrame(pen.get("switch_penalties", []))
    else:
        switch_df = pd.DataFrame(columns=["workload", "mode_from", "mode_to", "latency_penalty_ms", "energy_penalty_Wh"])

    summary_rows = []
    req_csv = output_root / request_log_name
    if req_csv.exists():
        req_csv.unlink()
    request_header_written = False
    total_request_rows = 0

    controllers = controllers_filter or matrix["controllers"]
    region_keys = regions_filter or list(matrix["regions"].keys())
    season_keys = seasons_filter or list(matrix["seasons"].keys())

    for workload in matrix["workloads"].keys():
        trace_files = sorted(Path(traces_root).glob(f"trace_{workload}_{load_name}_seed*.csv"))
        if not trace_files:
            raise FileNotFoundError(f"No trace files found for workload={workload}, load={load_name}")
        if max_traces_per_workload is not None:
            trace_files = trace_files[: max(1, int(max_traces_per_workload))]

        mode_df = admitted[admitted["workload"] == workload].copy()
        if mode_df.empty:
            raise ValueError(f"No admitted modes for workload={workload}")

        hq = mode_df.sort_values("accuracy", ascending=False).iloc[0]
        slo_ms = float(1.10 * float(hq["latency_p95_ms"]))

        cache_w = cache[cache["workload"] == workload].copy()
        if cache_w.empty:
            raise ValueError(f"No cache rows for workload={workload}")

        per_mode_cache = {
            m: cache_w[cache_w["mode_id"] == m].reset_index(drop=True)
            for m in sorted(mode_df["mode_id"].unique())
        }

        for region in region_keys:
            for season in season_keys:
                carbon = load_carbon_timeseries(carbon_csv, region, season)
                season_start = pd.Timestamp(matrix["seasons"][season]["week_start_utc"], tz="UTC")
                sampler = CarbonSignalSampler(carbon, season_start)

                ci_thresholds = (
                    float(np.quantile(sampler.ci, 0.33)),
                    float(np.quantile(sampler.ci, 0.66)),
                )

                for trace_file in trace_files:
                    trace = pd.read_csv(trace_file)
                    if trace.empty:
                        continue
                    if max_requests_per_trace is not None:
                        max_n = max(1, int(max_requests_per_trace))
                        if len(trace) > max_n:
                            # Keep temporal coverage across the full trace horizon.
                            idx = np.linspace(0, len(trace) - 1, num=max_n, dtype=int)
                            trace = trace.iloc[idx].copy().reset_index(drop=True)
                        else:
                            trace = trace.copy().reset_index(drop=True)

                    for controller_name in controllers:
                        controller = _build_controller(controller_name, mode_df, ci_thresholds=ci_thresholds)
                        prev_mode: str | None = None
                        server_free = 0.0
                        in_system: deque[float] = deque()
                        rolling_carbon = 0.0
                        rng = np.random.default_rng(abs(hash((controller_name, str(trace_file), region, season, scenario_name))) % 2**31)
                        rng_tel = np.random.default_rng(
                            abs(hash((telemetry_seed, controller_name, str(trace_file), region, season, scenario_name))) % 2**31
                        )
                        request_rows_scope = []
                        last_observed_ci: float | None = None

                        for rec in trace.itertuples(index=False):
                            arrival = float(rec.arrival_ts)
                            while in_system and in_system[0] <= arrival:
                                in_system.popleft()
                            queue_len = len(in_system)

                            actual_ci, observed_ci, forecast_ci, missing = _observe_ci(
                                sampler=sampler,
                                arrival_t_ms=arrival,
                                distortion=distortion,
                                last_observed=last_observed_ci,
                                rng=rng_tel,
                            )
                            last_observed_ci = observed_ci

                            state = ControllerState(
                                workload=workload,
                                current_ci=observed_ci,
                                forecast_ci=forecast_ci,
                                queue_len=queue_len,
                                prev_mode=prev_mode,
                                tenant_id=str(rec.tenant_id),
                                rolling_carbon_g=rolling_carbon,
                                slo_ms=slo_ms,
                            )

                            t0 = perf_counter()
                            mode_id = controller.choose_mode(state)
                            overhead_us = (perf_counter() - t0) * 1e6

                            pool = per_mode_cache.get(mode_id)
                            if pool is None or pool.empty:
                                raise ValueError(f"No cache pool for mode {mode_id}")

                            sampled = pool.iloc[int(rng.integers(0, len(pool)))]
                            lat_penalty, ene_penalty = _lookup_switch_penalty(switch_df, workload, prev_mode, mode_id)
                            service_ms = float(sampled["latency_ms"]) + lat_penalty
                            energy_wh = float(sampled["energy_Wh"]) + ene_penalty

                            start = max(arrival, server_free)
                            finish = start + service_ms
                            queue_wait = start - arrival
                            server_free = finish
                            in_system.append(finish)

                            carbon_g = energy_wh / 1000.0 * actual_ci
                            rolling_carbon += carbon_g

                            row = mode_df[mode_df["mode_id"] == mode_id].iloc[0]
                            res = RequestResult(
                                request_id=int(rec.request_id),
                                arrival_ts=arrival,
                                start_ts=start,
                                finish_ts=finish,
                                queue_wait_ms=queue_wait,
                                service_ms=service_ms,
                                workload=workload,
                                sample_id=int(sampled["sample_id"]),
                                mode_id=mode_id,
                                precision=str(row["precision"]),
                                capacity_k=int(row["capacity_k"]),
                                prev_mode=prev_mode,
                                switch_flag=int(prev_mode is not None and prev_mode != mode_id),
                                tenant_id=str(rec.tenant_id),
                                group_id=int(sampled["group_id"]),
                                actual_carbon_intensity=actual_ci,
                                observed_carbon_intensity=observed_ci,
                                forecast_carbon_intensity=forecast_ci,
                                telemetry_missing_flag=missing,
                                forecast_summary=f"h{telemetry_forecast_horizon_minutes}m_mean={forecast_ci:.3f}",
                                energy_Wh=energy_wh,
                                carbon_g=carbon_g,
                                success_flag=1,
                                slo_met=int((finish - arrival) <= slo_ms),
                                prediction=int(sampled["prediction"]),
                                correctness=int(sampled["correctness"]),
                                controller_overhead_us=overhead_us,
                            )
                            request_rows_scope.append(
                                {
                                    **asdict(res),
                                    "controller": controller_name,
                                    "region": region,
                                    "season": season,
                                    "load": load_name,
                                    "scenario": scenario_name,
                                    "trace_file": trace_file.name,
                                }
                            )
                            prev_mode = mode_id

                        scoped = pd.DataFrame(request_rows_scope)
                        if scoped.empty:
                            continue

                        scoped.to_csv(req_csv, mode="a", index=False, header=not request_header_written)
                        request_header_written = True
                        total_request_rows += int(len(scoped))

                        summary_rows.append(
                            _summarize(
                                scoped,
                                controller=controller_name,
                                region=region,
                                season=season,
                                workload=workload,
                                load_name=load_name,
                                trace_file=trace_file.name,
                                scenario_name=scenario_name,
                                slo_ms=slo_ms,
                            )
                        )

    summary_df = pd.DataFrame(summary_rows)

    sum_csv = output_root / summary_name
    summary_df.to_csv(sum_csv, index=False)

    metadata = {
        "scenario": scenario_name,
        "load": load_name,
        "matrix": matrix.get("matrix_name"),
        "controllers": controllers,
        "telemetry": asdict(distortion),
        "n_request_rows": total_request_rows,
        "n_summary_rows": int(len(summary_df)),
    }
    dump_yaml(output_root / "e1_metadata.yaml", metadata)
    return req_csv, sum_csv