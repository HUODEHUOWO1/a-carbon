from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_offline_cache(
    profile_root: str | Path,
    admitted_modes_csv: str | Path,
    output_root: str | Path,
) -> Path:
    profile_root = Path(profile_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    admitted = pd.read_csv(admitted_modes_csv)
    all_rows = []
    for _, row in admitted.iterrows():
        workload = row["workload"]
        mode_id = row["mode_id"]
        raw_path = profile_root / workload / f"raw_{mode_id}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw = pd.read_csv(raw_path)
        keep_cols = [
            "sample_id",
            "prediction",
            "correctness",
            "latency_ms",
            "energy_Wh",
            "group_id",
        ]
        data = raw[keep_cols].copy()
        data["workload"] = workload
        data["mode_id"] = mode_id
        all_rows.append(data)

    cache_df = pd.concat(all_rows, ignore_index=True)
    cache_df["tenant_id"] = "unassigned"

    out_path = output_root / "offline_cache.csv"
    cache_df.to_csv(out_path, index=False)
    return out_path