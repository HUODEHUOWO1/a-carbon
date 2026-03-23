from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import dump_yaml


def measure_switch_penalty(
    aaaa_csv: str | Path,
    abab_csv: str | Path,
    output_yaml: str | Path,
    mode_from: str,
    mode_to: str,
) -> Path:
    aaaa = pd.read_csv(aaaa_csv)
    abab = pd.read_csv(abab_csv)

    latency_col = "latency_ms"
    energy_col = "energy_Wh"
    if latency_col not in aaaa or latency_col not in abab:
        raise ValueError("Both inputs must contain latency_ms")
    if energy_col not in aaaa or energy_col not in abab:
        raise ValueError("Both inputs must contain energy_Wh")

    latency_penalty = float(max(0.0, abab[latency_col].mean() - aaaa[latency_col].mean()))
    energy_penalty = float(max(0.0, abab[energy_col].mean() - aaaa[energy_col].mean()))

    out = Path(output_yaml)
    dump_yaml(
        out,
        {
            "mode_from": mode_from,
            "mode_to": mode_to,
            "latency_penalty_ms": latency_penalty,
            "energy_penalty_Wh": energy_penalty,
            "n_aaaa": int(len(aaaa)),
            "n_abab": int(len(abab)),
        },
    )
    return out


def build_default_switch_matrix(admitted_modes_csv: str | Path, output_yaml: str | Path) -> Path:
    admitted = pd.read_csv(admitted_modes_csv)
    out = Path(output_yaml)
    penalties = []
    for workload in sorted(admitted["workload"].unique()):
        ws = admitted[admitted["workload"] == workload]
        mode_ids = ws["mode_id"].tolist()
        for src in mode_ids:
            for dst in mode_ids:
                if src == dst:
                    lat = 0.0
                    ene = 0.0
                else:
                    lat = 0.25
                    ene = 0.00002
                penalties.append(
                    {
                        "workload": workload,
                        "mode_from": src,
                        "mode_to": dst,
                        "latency_penalty_ms": lat,
                        "energy_penalty_Wh": ene,
                    }
                )

    dump_yaml(out, {"switch_penalties": penalties})
    return out