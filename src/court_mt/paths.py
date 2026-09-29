"""Canonical filesystem layout, resolved relative to the repository root.

Every script imports paths from here so that no module hardcodes a location.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts"


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for path in (INTERIM, PROCESSED, REPORTS, FIGURES, ARTIFACTS):
        path.mkdir(parents=True, exist_ok=True)
