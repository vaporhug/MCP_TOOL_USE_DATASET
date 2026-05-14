#!/usr/bin/env python3
"""Generic data-processing helpers for paper task packages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def read_excel(path: str | Path, sheet_name: str | int = 0, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def read_whitespace_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python")


def dataframe_overview(df: pd.DataFrame, preview_rows: int = 5) -> dict:
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "preview": df.head(preview_rows).to_dict("records"),
    }


def file_inventory(directory: str | Path, pattern: str = "*") -> dict:
    base = Path(directory)
    rows = []
    for path in sorted(base.rglob(pattern)):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(base)),
                    "suffix": path.suffix.lower(),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    return {"base_dir": str(base), "files": rows}
