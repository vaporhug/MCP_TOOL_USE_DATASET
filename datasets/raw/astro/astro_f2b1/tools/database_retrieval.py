#!/usr/bin/env python3
"""Lightweight local/remote resource helpers for paper task packages."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


def read_text(path: str | Path) -> dict:
    path = Path(path)
    return {"path": str(path), "text": path.read_text(encoding="utf-8", errors="ignore")}


def reference_inventory(directory: str | Path) -> dict:
    base = Path(directory)
    files = []
    for path in sorted(base.glob("*")):
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "suffix": path.suffix.lower(),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    return {"base_dir": str(base), "files": files}


def fetch_url_head(url: str, max_bytes: int = 4096) -> dict:
    with urlopen(url) as response:
        data = response.read(max_bytes)
        return {
            "url": url,
            "status": getattr(response, "status", None),
            "content_type": response.headers.get("Content-Type"),
            "preview_bytes": len(data),
        }
