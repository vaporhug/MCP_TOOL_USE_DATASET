#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Wolf1069Tools")


RV_COLUMNS = ["bjd", "rv_ms", "rv_err_ms", "instrument"]
PHOT_COLUMNS = ["bjd", "flux", "flux_err", "instrument"]
ACTIVITY_COLUMNS = [
    "bjd",
    "crx",
    "e_crx",
    "dlw",
    "e_dlw",
    "cairt1",
    "e_cairt1",
    "cairt2",
    "e_cairt2",
    "cairt3",
    "e_cairt3",
    "halpha",
    "e_halpha",
    "tio7050",
    "e_tio7050",
    "tio8430",
    "e_tio8430",
    "tio8860",
    "e_tio8860",
    "bis",
    "e_bis",
    "ctr",
    "e_ctr",
    "fwhm",
    "e_fwhm",
]


def _table_preview(df: pd.DataFrame, limit: int = 20) -> dict:
    return {
        "columns": list(df.columns),
        "total_rows": int(len(df)),
        "rows": df.head(limit).to_dict("records"),
    }


def _read_whitespace_table(path: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python")


def _infer_columns(path: str) -> list[str]:
    name = Path(path).name
    if name == "carm-rv.dat":
        return RV_COLUMNS
    if name == "carm-act.dat":
        return ACTIVITY_COLUMNS
    if name.endswith(".dat"):
        return PHOT_COLUMNS
    raise ValueError(f"Unsupported table format for {path}")


def _weighted_mean(values: np.ndarray, errors: np.ndarray | None) -> float:
    if errors is None:
        return float(np.mean(values))
    safe = np.where(errors > 0, errors, np.nan)
    weights = np.where(np.isfinite(safe), 1.0 / np.square(safe), 0.0)
    if np.allclose(weights.sum(), 0.0):
        return float(np.mean(values))
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_center(values: np.ndarray, errors: np.ndarray | None) -> np.ndarray:
    if errors is None:
        return values - np.mean(values)
    return values - _weighted_mean(values, errors)


def _sinusoid_design(time: np.ndarray, period: float) -> np.ndarray:
    omega = 2.0 * math.pi / period
    return np.column_stack([np.sin(omega * time), np.cos(omega * time), np.ones_like(time)])


def _fit_weighted_linear(design: np.ndarray, values: np.ndarray, errors: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if errors is None:
        weights = np.ones(len(values))
    else:
        safe = np.where(errors > 0, errors, np.nan)
        weights = np.where(np.isfinite(safe), 1.0 / np.square(safe), 0.0)
    Xw = design * np.sqrt(weights[:, None])
    yw = values * np.sqrt(weights)
    coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    model = design @ coeffs
    return coeffs, model


def _sinusoid_power(
    time: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray | None,
    periods: np.ndarray,
) -> np.ndarray:
    centered = _weighted_center(values, errors)
    if errors is None:
        weights = np.ones_like(centered)
    else:
        safe = np.where(errors > 0, errors, np.nan)
        weights = np.where(np.isfinite(safe), 1.0 / np.square(safe), 0.0)
    baseline = np.sum(weights * np.square(centered))
    powers = []
    for period in periods:
        design = _sinusoid_design(time, period)
        _, model = _fit_weighted_linear(design, centered, errors)
        resid = centered - model
        rss = np.sum(weights * np.square(resid))
        powers.append(float(max(0.0, 1.0 - rss / baseline))) if baseline > 0 else powers.append(0.0)
    return np.array(powers)


def _read_series(path: str, time_column: str, value_column: str, error_column: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    df = _read_whitespace_table(path, _infer_columns(path))
    time = df[time_column].to_numpy(dtype=float)
    values = df[value_column].to_numpy(dtype=float)
    errors = df[error_column].to_numpy(dtype=float) if error_column else None
    mask = np.isfinite(time) & np.isfinite(values)
    if errors is not None:
        mask &= np.isfinite(errors)
    return time[mask], values[mask], errors[mask] if errors is not None else None


@mcp.tool()
def load_rv_table(path: str) -> dict:
    return _table_preview(_read_whitespace_table(path, RV_COLUMNS))


@mcp.tool()
def load_activity_table(path: str) -> dict:
    return _table_preview(_read_whitespace_table(path, ACTIVITY_COLUMNS))


@mcp.tool()
def load_photometry_index(path: str) -> dict:
    rows = []
    for raw in Path(path).read_text().splitlines():
        line = raw.rstrip()
        if not line:
            continue
        filename = line[-33:].strip()
        body = line[:-33].rstrip()
        parts = body.split()
        rows.append(
            {
                "instrument": parts[0],
                "date_range": parts[1],
                "filter": parts[2] if len(parts) > 2 else "",
                "file_name": filename,
            }
        )
    return _table_preview(pd.DataFrame(rows))


@mcp.tool()
def load_photometry_file(path: str) -> dict:
    return _table_preview(_read_whitespace_table(path, PHOT_COLUMNS))


@mcp.tool()
def combine_photometry(directory: str) -> dict:
    frames = []
    for path in sorted(Path(directory).glob("*.dat")):
        df = _read_whitespace_table(str(path), PHOT_COLUMNS)
        df["source_file"] = path.name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return {
        "total_rows": int(len(combined)),
        "n_files": int(combined["source_file"].nunique()),
        "time_span_days": float(combined["bjd"].max() - combined["bjd"].min()),
        "rows": combined.head(20).to_dict("records"),
    }


@mcp.tool()
def nightly_bin_photometry(path: str) -> dict:
    df = _read_whitespace_table(path, PHOT_COLUMNS)
    df["night"] = np.floor(df["bjd"]).astype(int)
    grouped = (
        df.groupby("night", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "bjd_mean": g["bjd"].mean(),
                    "flux_mean": _weighted_mean(g["flux"].to_numpy(), g["flux_err"].to_numpy()),
                    "flux_std": float(g["flux"].std(ddof=1)) if len(g) > 1 else 0.0,
                    "n_points": int(len(g)),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    return _table_preview(grouped)


@mcp.tool()
def scan_periodogram(
    path: str,
    time_column: str,
    value_column: str,
    error_column: str | None = None,
    min_period: float = 1.0,
    max_period: float = 200.0,
    n_periods: int = 2000,
) -> dict:
    time, values, errors = _read_series(path, time_column, value_column, error_column)
    periods = np.linspace(min_period, max_period, n_periods)
    power = _sinusoid_power(time, values, errors, periods)
    top = np.argsort(power)[-10:][::-1]
    return {
        "best_period": float(periods[top[0]]),
        "best_power": float(power[top[0]]),
        "top_periods": [{"period": float(periods[i]), "power": float(power[i])} for i in top],
    }


@mcp.tool()
def fit_weighted_sinusoid(
    path: str,
    time_column: str,
    value_column: str,
    period: float,
    error_column: str | None = None,
) -> dict:
    time, values, errors = _read_series(path, time_column, value_column, error_column)
    design = _sinusoid_design(time, period)
    coeffs, model = _fit_weighted_linear(design, values, errors)
    amp = float(np.hypot(coeffs[0], coeffs[1]))
    phase = float(math.atan2(coeffs[1], coeffs[0]))
    residuals = values - model
    return {
        "period": float(period),
        "amplitude": amp,
        "phase_radians": phase,
        "offset": float(coeffs[2]),
        "rms_residual": float(np.sqrt(np.mean(np.square(residuals)))),
        "rows": pd.DataFrame({"bjd": time, "observed": values, "model": model, "residual": residuals})
        .head(20)
        .to_dict("records"),
    }


@mcp.tool()
def subtract_signal_and_rescan(
    path: str,
    time_column: str,
    value_column: str,
    period: float,
    error_column: str | None = None,
    min_period: float = 1.0,
    max_period: float = 200.0,
    n_periods: int = 2000,
) -> dict:
    time, values, errors = _read_series(path, time_column, value_column, error_column)
    design = _sinusoid_design(time, period)
    _, model = _fit_weighted_linear(design, values, errors)
    residual = values - model
    periods = np.linspace(min_period, max_period, n_periods)
    power = _sinusoid_power(time, residual, errors, periods)
    top = np.argsort(power)[-10:][::-1]
    return {
        "removed_period": float(period),
        "best_residual_period": float(periods[top[0]]),
        "best_residual_power": float(power[top[0]]),
        "top_residual_periods": [{"period": float(periods[i]), "power": float(power[i])} for i in top],
    }


@mcp.tool()
def phase_fold_rv(path: str, period: float, t0: float = 0.0) -> dict:
    df = _read_whitespace_table(path, RV_COLUMNS)
    phase = ((df["bjd"] - t0) / period) % 1.0
    out = pd.DataFrame({"phase": phase, "rv_ms": df["rv_ms"], "rv_err_ms": df["rv_err_ms"]}).sort_values("phase")
    return _table_preview(out)


@mcp.tool()
def correlate_rv_with_activity(
    rv_path: str,
    activity_path: str,
    activity_column: str,
    max_time_delta_days: float = 0.5,
) -> dict:
    rv = _read_whitespace_table(rv_path, RV_COLUMNS).sort_values("bjd")
    act = _read_whitespace_table(activity_path, ACTIVITY_COLUMNS).sort_values("bjd")
    if activity_column not in act.columns:
        raise ValueError(f"Unknown activity column: {activity_column}")
    merged = pd.merge_asof(rv, act[["bjd", activity_column]], on="bjd", direction="nearest", tolerance=max_time_delta_days)
    merged = merged.dropna(subset=["rv_ms", activity_column])
    if len(merged) < 3:
        return {"matched_rows": int(len(merged))}
    pearson = float(np.corrcoef(merged["rv_ms"], merged[activity_column])[0, 1])
    spearman = float(
        np.corrcoef(merged["rv_ms"].rank(method="average"), merged[activity_column].rank(method="average"))[0, 1]
    )
    return {
        "matched_rows": int(len(merged)),
        "activity_column": activity_column,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "rows": merged.head(20).to_dict("records"),
    }


@mcp.tool()
def compare_candidate_period_across_signals(
    rv_path: str,
    activity_path: str,
    candidate_period: float,
    activity_columns: list[str] | None = None,
    tolerance_days: float = 2.0,
) -> dict:
    if activity_columns is None:
        activity_columns = ["crx", "dlw", "halpha", "bis", "fwhm"]
    rv_scan = scan_periodogram(rv_path, "bjd", "rv_ms", "rv_err_ms")
    activity_scans = {}
    for col in activity_columns:
        if col not in ACTIVITY_COLUMNS:
            continue
        result = scan_periodogram(activity_path, "bjd", col, f"e_{col}" if f"e_{col}" in ACTIVITY_COLUMNS else None)
        activity_scans[col] = {
            "best_period": result["best_period"],
            "near_candidate": bool(abs(result["best_period"] - candidate_period) <= tolerance_days),
        }
    return {
        "candidate_period": float(candidate_period),
        "rv_best_period": rv_scan["best_period"],
        "rv_near_candidate": bool(abs(rv_scan["best_period"] - candidate_period) <= tolerance_days),
        "activity_scans": activity_scans,
    }


@mcp.tool()
def summarize_rotation_candidates(
    photometry_directory: str,
    min_period: float = 20.0,
    max_period: float = 250.0,
    n_periods: int = 1500,
) -> dict:
    rows = []
    for path in sorted(Path(photometry_directory).glob("*.dat")):
        df = _read_whitespace_table(str(path), PHOT_COLUMNS)
        if len(df) < 10:
            continue
        periods = np.linspace(min_period, max_period, n_periods)
        power = _sinusoid_power(df["bjd"].to_numpy(float), df["flux"].to_numpy(float), df["flux_err"].to_numpy(float), periods)
        idx = int(np.argmax(power))
        rows.append(
            {
                "file": path.name,
                "best_period": float(periods[idx]),
                "best_power": float(power[idx]),
                "time_span_days": float(df["bjd"].max() - df["bjd"].min()),
                "n_points": int(len(df)),
            }
        )
    out = pd.DataFrame(rows).sort_values("best_power", ascending=False)
    return {"n_files": int(len(out)), "rows": out.to_dict("records")}


@mcp.tool()
def descriptive_stats(path: str, column_index: int) -> dict:
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=None)
    values = df.iloc[:, column_index].to_numpy(dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
