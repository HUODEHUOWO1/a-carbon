from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class SeasonSpec:
    key: str
    start_utc: pd.Timestamp
    days: int


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=False, sort_keys=False)


def _normalize_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.isnull().any():
        n_bad = int(ts.isnull().sum())
        raise RuntimeError(f"Found {n_bad} unparsable timestamps in NESO CSV")
    return ts


def _sanitize_ci_values(series: pd.Series, zero_policy: str) -> tuple[pd.Series, pd.Series]:
    ci = pd.to_numeric(series, errors="coerce")
    invalid = ci.isnull()

    if zero_policy == "zero_as_missing":
        invalid = invalid | (ci == 0.0)
        ci = ci.mask(ci == 0.0)
    elif zero_policy == "nonpositive_as_missing":
        invalid = invalid | (ci <= 0.0)
        ci = ci.mask(ci <= 0.0)
    elif zero_policy == "keep":
        pass
    else:
        raise RuntimeError(f"Unsupported zero_policy={zero_policy}")

    return ci, invalid


def _read_neso_csv(bundle_root: Path, source: str) -> pd.DataFrame:
    if source == "regional":
        path = bundle_root / "neso_regional_carbon_intensity_forecast.csv"
        df = pd.read_csv(path)
        df["timestamp_utc"] = _normalize_ts(df["datetime"])
        return df.drop(columns=["datetime"])
    if source == "country":
        path = bundle_root / "neso_country_carbon_intensity_forecast.csv"
        df = pd.read_csv(path)
        df["timestamp_utc"] = _normalize_ts(df["datetime"])
        return df.drop(columns=["datetime"])
    if source == "national":
        path = bundle_root / "neso_national_carbon_intensity_forecast.csv"
        df = pd.read_csv(path)
        df["timestamp_utc"] = _normalize_ts(df["datetime"])
        return df.drop(columns=["datetime"])
    if source == "historic":
        path = bundle_root / "neso_historic_gb_generation_mix_and_carbon_intensity.csv"
        df = pd.read_csv(path, usecols=["DATETIME", "CARBON_INTENSITY"])
        df["timestamp_utc"] = _normalize_ts(df["DATETIME"])
        df = df.rename(columns={"CARBON_INTENSITY": "historic_ci"})
        return df.drop(columns=["DATETIME"])
    raise RuntimeError(f"Unsupported source={source}")


def _season_specs(matrix: dict[str, Any]) -> list[SeasonSpec]:
    out: list[SeasonSpec] = []
    seasons = matrix.get("seasons", {})
    for k, v in seasons.items():
        start = pd.Timestamp(str(v["week_start_utc"]), tz="UTC")
        days = int(v.get("days", 7))
        out.append(SeasonSpec(key=str(k), start_utc=start, days=days))
    return out


def _choose_series(
    df: pd.DataFrame,
    source: str,
    region_cfg: dict[str, Any],
    national_mode: str,
    zero_policy: str,
) -> tuple[pd.Series, pd.Series]:
    # Returns (ci_series, estimated_series) indexed by timestamp_utc
    ts = df["timestamp_utc"]

    if source == "regional":
        col = region_cfg.get("neso_column")
        if not col:
            raise RuntimeError("regional source requires region.neso_column in matrix")
        if col not in df.columns:
            raise RuntimeError(f"NESO regional column not found: {col}")
        raw, est = _sanitize_ci_values(df[col], zero_policy=zero_policy)
        out = pd.DataFrame({"timestamp_utc": ts, "ci": raw, "est": est.astype(float)})
        out = out.groupby("timestamp_utc", as_index=True).mean(numeric_only=True)
        out["est"] = out["est"] > 0
        return out["ci"], out["est"]

    if source == "country":
        col = region_cfg.get("neso_column")
        if not col:
            raise RuntimeError("country source requires region.neso_column in matrix")
        if col not in df.columns:
            raise RuntimeError(f"NESO country column not found: {col}")
        raw, est = _sanitize_ci_values(df[col], zero_policy=zero_policy)
        out = pd.DataFrame({"timestamp_utc": ts, "ci": raw, "est": est.astype(float)})
        out = out.groupby("timestamp_utc", as_index=True).mean(numeric_only=True)
        out["est"] = out["est"] > 0
        return out["ci"], out["est"]

    if source == "national":
        actual, actual_invalid = _sanitize_ci_values(df["actual"], zero_policy=zero_policy)
        forecast, forecast_invalid = _sanitize_ci_values(df["forecast"], zero_policy=zero_policy)
        if national_mode == "actual":
            ci = actual
            est = actual_invalid
        elif national_mode == "forecast":
            ci = forecast
            est = forecast_invalid
        else:
            # prefer_actual
            ci = actual.copy()
            mask = ci.isnull() & forecast.notnull()
            ci.loc[mask] = forecast.loc[mask]
            est = mask | ci.isnull()

        out = pd.DataFrame({"timestamp_utc": ts, "ci": ci, "est": est.astype(float)})
        out = out.groupby("timestamp_utc", as_index=True).mean(numeric_only=True)
        out["est"] = out["est"] > 0
        return out["ci"], out["est"]

    if source == "historic":
        ci, est = _sanitize_ci_values(df["historic_ci"], zero_policy=zero_policy)
        out = pd.DataFrame({"timestamp_utc": ts, "ci": ci, "est": est.astype(float)})
        out = out.groupby("timestamp_utc", as_index=True).mean(numeric_only=True)
        out["est"] = out["est"] > 0
        return out["ci"], out["est"]

    raise RuntimeError(f"Unsupported source={source}")


def _fill_half_hourly(
    ci: pd.Series,
    est: pd.Series,
    start_utc: pd.Timestamp,
    days: int,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    end_utc = start_utc + pd.Timedelta(days=int(days))
    expected_idx = pd.date_range(start=start_utc, end=end_utc - pd.Timedelta(minutes=30), freq="30min", tz="UTC")

    ci = ci.reindex(expected_idx)
    est = est.reindex(expected_idx).fillna(True)

    missing_before = int(ci.isnull().sum())

    # Fill policy: time interpolation -> ffill -> bfill
    ci_filled = ci.interpolate(method="time", limit_direction="both")
    ci_filled = ci_filled.ffill().bfill()

    missing_after = int(ci_filled.isnull().sum())
    if missing_after > 0:
        raise RuntimeError(f"Failed to fill NESO carbon series; {missing_after} NaN remain")

    est_out = est | ci.isnull()
    fill_stats = {
        "expected_points": int(len(expected_idx)),
        "missing_before_fill": missing_before,
        "missing_after_fill": missing_after,
        "filled_points": int((est_out.astype(bool)).sum()),
    }
    return ci_filled.astype(float), est_out.astype(bool), fill_stats


def convert_neso_bundle(
    matrix_path: Path,
    bundle_root: Path,
    output_csv: Path,
    output_meta: Path,
    source: str = "regional",
    national_mode: str = "prefer_actual",
    zero_policy: str = "zero_as_missing",
) -> tuple[Path, Path]:
    matrix = _load_yaml(matrix_path)
    df_source = _read_neso_csv(bundle_root, source=source)
    seasons = _season_specs(matrix)

    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for region_key, region_cfg in matrix.get("regions", {}).items():
        ci_all, est_all = _choose_series(
            df_source,
            source=source,
            region_cfg=region_cfg,
            national_mode=national_mode,
            zero_policy=zero_policy,
        )
        for season in seasons:
            ci_s, est_s, stats = _fill_half_hourly(
                ci=ci_all,
                est=est_all,
                start_utc=season.start_utc,
                days=season.days,
            )

            for ts, ci_val, est_val in zip(ci_s.index, ci_s.values, est_s.values):
                rows.append(
                    {
                        "zone": str(region_cfg.get("grid_zone", "GB")),
                        "timestamp_utc": ts.isoformat(),
                        "carbon_intensity_g_per_kwh": float(ci_val),
                        "is_estimated": bool(est_val),
                        "region_key": str(region_key),
                        "season_key": season.key,
                    }
                )

            meta_rows.append(
                {
                    "region_key": str(region_key),
                    "season_key": season.key,
                    "source": source,
                    "neso_column": str(region_cfg.get("neso_column", "")),
                    "expected_points": int(stats["expected_points"]),
                    "missing_before_fill": int(stats["missing_before_fill"]),
                    "filled_points": int(stats["filled_points"]),
                    "mean_ci_g_per_kwh": float(ci_s.mean()),
                    "std_ci_g_per_kwh": float(ci_s.std(ddof=0)),
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["region_key", "season_key", "timestamp_utc"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    meta = {
        "source": "NESO",
        "bundle_root": str(bundle_root),
        "source_dataset": source,
        "national_mode": national_mode if source == "national" else "",
        "zero_policy": zero_policy,
        "matrix_path": str(matrix_path),
        "output_csv": str(output_csv),
        "rows": int(len(out_df)),
        "region_season": meta_rows,
    }
    _dump_yaml(output_meta, meta)
    return output_csv, output_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NESO CSV bundle into CAI carbon_csv format.")
    parser.add_argument("--matrix", required=True, help="Matrix yaml path (NESO-specific).")
    parser.add_argument("--bundle-root", required=True, help="NESO bundle directory.")
    parser.add_argument("--output-csv", required=True, help="Output carbon csv path.")
    parser.add_argument("--output-meta", required=True, help="Output metadata yaml path.")
    parser.add_argument(
        "--source",
        default="regional",
        choices=["regional", "country", "national", "historic"],
        help="Which NESO CSV source to use.",
    )
    parser.add_argument(
        "--national-mode",
        default="prefer_actual",
        choices=["prefer_actual", "actual", "forecast"],
        help="National source mode; ignored for other sources.",
    )
    parser.add_argument(
        "--zero-policy",
        default="zero_as_missing",
        choices=["keep", "zero_as_missing", "nonpositive_as_missing"],
        help="How to treat zero/negative CI values before fill.",
    )
    args = parser.parse_args()

    csv_path, meta_path = convert_neso_bundle(
        matrix_path=Path(args.matrix),
        bundle_root=Path(args.bundle_root),
        output_csv=Path(args.output_csv),
        output_meta=Path(args.output_meta),
        source=str(args.source),
        national_mode=str(args.national_mode),
        zero_policy=str(args.zero_policy),
    )
    print(f"carbon_csv={csv_path}")
    print(f"carbon_meta={meta_path}")


if __name__ == "__main__":
    main()
