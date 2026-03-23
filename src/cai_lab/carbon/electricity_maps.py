from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..config import dump_yaml, load_yaml


@dataclass(frozen=True)
class ElectricityMapsClient:
    api_key: str
    base_url: str = "https://api.electricitymap.org/v3"

    def _headers(self) -> dict[str, str]:
        return {"auth-token": self.api_key}

    def fetch_hourly_history(
        self,
        zone: str,
        start_utc: datetime,
        end_utc: datetime,
        emission_factor: str = "direct",
    ) -> pd.DataFrame:
        endpoint = f"{self.base_url}/carbon-intensity/history"
        params = {
            "zone": zone,
            "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "emissionFactorType": emission_factor,
        }
        resp = requests.get(endpoint, params=params, headers=self._headers(), timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict) or "history" not in payload:
            raise ValueError("Unexpected Electricity Maps response format")
        rows = payload["history"]
        if not isinstance(rows, list):
            raise ValueError("Unexpected Electricity Maps history format")

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        ts_col = "datetime" if "datetime" in out.columns else "time"
        ci_col = "carbonIntensity" if "carbonIntensity" in out.columns else "carbonIntensityDirect"
        estimated_col = "isEstimated"

        if ci_col not in out.columns:
            # Try a permissive fallback for schema variations.
            candidates = [c for c in out.columns if "carbon" in c.lower() and "intensity" in c.lower()]
            if not candidates:
                raise ValueError(f"Could not find carbon intensity column in: {list(out.columns)}")
            ci_col = candidates[0]

        if ts_col not in out.columns:
            raise ValueError(f"Could not find timestamp column in: {list(out.columns)}")

        out = out.rename(columns={ts_col: "timestamp_utc", ci_col: "carbon_intensity_g_per_kwh"})
        if estimated_col not in out.columns:
            out["is_estimated"] = False
        else:
            out = out.rename(columns={estimated_col: "is_estimated"})

        out["zone"] = zone
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
        return out[["zone", "timestamp_utc", "carbon_intensity_g_per_kwh", "is_estimated"]]


def fetch_matrix_carbon_data(
    matrix_path: str | Path,
    output_root: str | Path,
    api_key: str,
    base_url: str = "https://api.electricitymap.org/v3",
) -> tuple[Path, Path]:
    matrix = load_yaml(matrix_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    client = ElectricityMapsClient(api_key=api_key, base_url=base_url)
    emission_factor = matrix["carbon"]["emission_factor"]

    all_frames = []
    meta_rows: list[dict[str, Any]] = []

    for region_key, region in matrix["regions"].items():
        zone = region["grid_zone"]
        for season_key, season in matrix["seasons"].items():
            start = datetime.fromisoformat(season["week_start_utc"]).replace(tzinfo=timezone.utc)
            end = start + timedelta(days=int(season["days"]))
            df = client.fetch_hourly_history(zone=zone, start_utc=start, end_utc=end, emission_factor=emission_factor)
            if df.empty:
                raise RuntimeError(f"No carbon data for zone={zone}, season={season_key}")

            df["region_key"] = region_key
            df["season_key"] = season_key
            all_frames.append(df)

            est_share = float(df["is_estimated"].astype(float).mean())
            meta_rows.append(
                {
                    "region_key": region_key,
                    "season_key": season_key,
                    "zone": zone,
                    "n_points": int(len(df)),
                    "estimated_share": est_share,
                    "mean_ci_g_per_kwh": float(df["carbon_intensity_g_per_kwh"].mean()),
                    "std_ci_g_per_kwh": float(df["carbon_intensity_g_per_kwh"].std(ddof=0)),
                }
            )

    carbon_df = pd.concat(all_frames, ignore_index=True)
    carbon_csv = output_root / "electricity_maps_carbon.csv"
    carbon_df.to_csv(carbon_csv, index=False)

    metadata = {
        "source": "Electricity Maps",
        "emission_factor": emission_factor,
        "pull_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "zone_season": meta_rows,
    }
    meta_yaml = output_root / "electricity_maps_metadata.yaml"
    dump_yaml(meta_yaml, metadata)
    return carbon_csv, meta_yaml


def load_carbon_timeseries(
    carbon_csv: str | Path,
    region_key: str,
    season_key: str,
) -> pd.DataFrame:
    df = pd.read_csv(carbon_csv)
    df = df[(df["region_key"] == region_key) & (df["season_key"] == season_key)].copy()
    if df.empty:
        raise ValueError(f"No carbon rows for region={region_key}, season={season_key}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc")
    return df