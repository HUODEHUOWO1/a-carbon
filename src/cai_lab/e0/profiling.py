from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import dump_yaml, load_yaml


@dataclass(frozen=True)
class Mode:
    precision: str
    capacity_k: int

    @property
    def mode_id(self) -> str:
        return f"{self.precision}_k{self.capacity_k}"


def candidate_modes(matrix: dict[str, Any]) -> list[Mode]:
    precisions = matrix["mode_library"]["precision_candidates"]
    capacities = matrix["mode_library"]["capacity_k_candidates"]
    return [Mode(p, int(k)) for p in precisions for k in capacities]


def _mode_factors(mode: Mode) -> tuple[float, float, float]:
    precision_latency = {"fp16": 1.0, "int8": 0.8, "int4": 0.7}
    precision_energy = {"fp16": 1.0, "int8": 0.75, "int4": 0.65}
    precision_acc = {"fp16": 0.0, "int8": -0.3, "int4": -1.4}

    cap_latency = {4: 1.0, 2: 0.85, 1: 0.75}
    cap_energy = {4: 1.0, 2: 0.78, 1: 0.68}
    cap_acc = {4: 0.0, 2: -0.2, 1: -0.5}

    latency_factor = precision_latency[mode.precision] * cap_latency[mode.capacity_k]
    energy_factor = precision_energy[mode.precision] * cap_energy[mode.capacity_k]
    acc_drop_pp = -(precision_acc[mode.precision] + cap_acc[mode.capacity_k])
    return latency_factor, energy_factor, acc_drop_pp


def _synthetic_profile(workload: str, mode: Mode, n_total: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latency_factor, energy_factor, acc_drop_pp = _mode_factors(mode)

    if workload == "vision":
        base_latency_ms = 18.0
        base_energy_wh = 0.0012
        base_acc = 0.764
        group_num = 1000
    else:
        base_latency_ms = 11.0
        base_energy_wh = 0.0007
        base_acc = 0.945
        group_num = 4

    latency = rng.lognormal(mean=np.log(base_latency_ms * latency_factor), sigma=0.15, size=n_total)
    energy = rng.lognormal(mean=np.log(base_energy_wh * energy_factor), sigma=0.10, size=n_total)
    acc = max(0.0, base_acc - acc_drop_pp / 100.0)
    correctness = rng.binomial(1, acc, size=n_total)

    sample_id = rng.integers(0, 1_000_000, size=n_total)
    prediction = rng.integers(0, group_num, size=n_total)
    group_id = rng.integers(0, group_num, size=n_total)

    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "prediction": prediction,
            "correctness": correctness,
            "group_id": group_id,
            "latency_ms": latency,
            "energy_Wh": energy,
        }
    )


def _load_precomputed(root: Path, workload: str, mode_id: str) -> pd.DataFrame | None:
    path = root / workload / f"{mode_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"sample_id", "prediction", "correctness", "group_id", "latency_ms", "energy_Wh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError(f"Null values found in required columns for {path}")

    numeric_cols = ["sample_id", "prediction", "correctness", "group_id", "latency_ms", "energy_Wh"]
    for c in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Column {c} must be numeric in {path}")

    if ((df["correctness"] < 0) | (df["correctness"] > 1)).any():
        raise ValueError(f"Column correctness must be in [0,1] in {path}")
    if (df["latency_ms"] <= 0).any():
        raise ValueError(f"Column latency_ms must be > 0 in {path}")
    if (df["energy_Wh"] <= 0).any():
        raise ValueError(f"Column energy_Wh must be > 0 in {path}")
    return df


def run_mode_profiling(
    matrix_path: str | Path,
    output_root: str | Path,
    precomputed_root: str | Path | None = None,
    synthetic_fallback: bool = True,
    strict_precomputed: bool = False,
) -> Path:
    matrix = load_yaml(matrix_path)
    warmup = int(matrix["e0"]["warmup_requests"])
    measure = int(matrix["e0"]["measure_requests"])
    total = warmup + measure

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pre_root = Path(precomputed_root) if precomputed_root else None

    rows: list[dict[str, Any]] = []
    for workload in matrix["workloads"].keys():
        workload_dir = output_root / workload
        workload_dir.mkdir(parents=True, exist_ok=True)
        for mode in candidate_modes(matrix):
            mode_id = mode.mode_id
            raw = None
            if pre_root:
                raw = _load_precomputed(pre_root, workload, mode_id)

            if raw is None:
                if pre_root and strict_precomputed:
                    expected_path = pre_root / workload / f"{mode_id}.csv"
                    raise FileNotFoundError(
                        f"Missing precomputed profile (strict mode) for {workload}/{mode_id} at {expected_path}"
                    )
                if not synthetic_fallback:
                    raise FileNotFoundError(
                        f"No precomputed profile for {workload}/{mode_id} and synthetic_fallback=False"
                    )
                raw = _synthetic_profile(workload, mode, total, seed=abs(hash((workload, mode_id))) % 2**31)

            if len(raw) < total:
                if strict_precomputed and pre_root:
                    raise ValueError(
                        f"Precomputed profile too short for {workload}/{mode_id}: need {total}, got {len(raw)}"
                    )
                raw = raw.sample(total, replace=True, random_state=42).reset_index(drop=True)

            measured = raw.iloc[warmup : warmup + measure].copy().reset_index(drop=True)
            measured["request_id"] = np.arange(measure)
            measured["workload"] = workload
            measured["mode_id"] = mode_id
            measured["precision"] = mode.precision
            measured["capacity_k"] = mode.capacity_k

            out_file = workload_dir / f"raw_{mode_id}.csv"
            measured.to_csv(out_file, index=False)

            latency_p95 = float(np.percentile(measured["latency_ms"], 95))
            latency_p99 = float(np.percentile(measured["latency_ms"], 99))
            accuracy = float(measured["correctness"].mean())
            rows.append(
                {
                    "workload": workload,
                    "mode_id": mode_id,
                    "precision": mode.precision,
                    "capacity_k": mode.capacity_k,
                    "n": int(len(measured)),
                    "latency_mean_ms": float(measured["latency_ms"].mean()),
                    "latency_p95_ms": latency_p95,
                    "latency_p99_ms": latency_p99,
                    "energy_mean_Wh": float(measured["energy_Wh"].mean()),
                    "accuracy": accuracy,
                }
            )

    summary = pd.DataFrame(rows).sort_values(["workload", "energy_mean_Wh", "latency_p95_ms"])
    summary_path = output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    records = df.to_dict("records")
    for i, r in enumerate(records):
        dominated = False
        for j, s in enumerate(records):
            if i == j:
                continue
            not_worse = (
                s["energy_mean_Wh"] <= r["energy_mean_Wh"]
                and s["latency_p95_ms"] <= r["latency_p95_ms"]
                and s["accuracy"] >= r["accuracy"]
            )
            strictly_better = (
                s["energy_mean_Wh"] < r["energy_mean_Wh"]
                or s["latency_p95_ms"] < r["latency_p95_ms"]
                or s["accuracy"] > r["accuracy"]
            )
            if not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(r)
    return pd.DataFrame(keep)


def run_mode_pruning(
    matrix_path: str | Path,
    profile_summary_path: str | Path,
    output_root: str | Path,
) -> tuple[Path, Path]:
    matrix = load_yaml(matrix_path)
    summary = pd.read_csv(profile_summary_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    max_drop = {
        w: float(v["max_accuracy_drop_pp"]) for w, v in matrix["workloads"].items()
    }
    latency_limit = float(matrix["mode_library"]["standalone_latency_multiplier_limit_vs_static_hq_p95"])
    max_modes = int(matrix["mode_library"]["max_modes_per_workload"])

    admitted_rows = []
    admitted_yaml: dict[str, Any] = {"workloads": {}}

    for workload in sorted(summary["workload"].unique()):
        ws = summary[summary["workload"] == workload].copy()
        hq = ws[ws["mode_id"] == "fp16_k4"]
        if hq.empty:
            raise ValueError(f"Static-HQ mode fp16_k4 missing for workload {workload}")
        hq_acc = float(hq.iloc[0]["accuracy"])
        hq_p95 = float(hq.iloc[0]["latency_p95_ms"])

        ws["accuracy_drop_pp"] = (hq_acc - ws["accuracy"]) * 100.0
        ws["latency_ratio_vs_hq_p95"] = ws["latency_p95_ms"] / hq_p95

        admitted = ws[
            (ws["accuracy_drop_pp"] <= max_drop[workload])
            & (ws["latency_ratio_vs_hq_p95"] <= latency_limit)
        ].copy()

        if admitted.empty:
            raise RuntimeError(f"No admitted mode for workload {workload}")

        hq_mode_id = "fp16_k4"
        pareto = _pareto_front(admitted)
        pareto = pareto.sort_values(["energy_mean_Wh", "latency_p95_ms", "accuracy"], ascending=[True, True, False])
        selected = pareto.head(max_modes).copy()
        if hq_mode_id in admitted["mode_id"].values and hq_mode_id not in selected["mode_id"].values:
            hq_row = admitted[admitted["mode_id"] == hq_mode_id].head(1)
            selected = pd.concat([selected, hq_row], ignore_index=True)
            selected = selected.drop_duplicates(subset=["mode_id"], keep="first")

        selected["admitted"] = True
        admitted_rows.append(selected)
        admitted_yaml["workloads"][workload] = {
            "selected_mode_ids": selected["mode_id"].tolist(),
            "static_hq_mode_id": hq_mode_id,
            "static_eco_mode_id": str(selected.sort_values("energy_mean_Wh").iloc[0]["mode_id"]),
            "n_selected": int(len(selected)),
        }

    admitted_df = pd.concat(admitted_rows, ignore_index=True)
    admitted_csv = output_root / "admitted_modes.csv"
    admitted_df.to_csv(admitted_csv, index=False)

    admitted_yaml_path = output_root / "admitted_modes.yaml"
    dump_yaml(admitted_yaml_path, admitted_yaml)
    return admitted_csv, admitted_yaml_path