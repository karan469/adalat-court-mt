"""One-line bootstrap shared by every notebook.

Notebooks in this project are thin: they orchestrate and narrate, while the logic
lives in :mod:`court_mt`. This module removes the boilerplate that would
otherwise be copy-pasted into each one, import path, hub access, chart theme,
display options, so a notebook's first cell says what it does, not how to set
Python up.
"""

from __future__ import annotations

import sys
from pathlib import Path


def setup(*, dark: bool = False, max_rows: int = 60):
    """Prepare the interpreter for notebook use and return the chart theme.

    Args:
        dark: render figures against the dark surface instead of the light one.
        max_rows: pandas display cap.

    Returns:
        The active :class:`court_mt.viz.Theme`.
    """
    root = _repo_root()
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))

    from court_mt import viz
    from court_mt.environment import configure, setup_logging

    configure()
    setup_logging()

    import pandas as pd

    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda value: f"{value: .3f}")

    return viz.use(viz.DARK if dark else viz.LIGHT)


def _repo_root() -> Path:
    """Find the repository root from the notebook's working directory.

    Notebooks run from ``notebooks/``, but may also be executed from the repo
    root by nbconvert, so the root is located by looking for a marker rather than
    assuming a fixed number of parent hops.
    """
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"repository root not found above {Path.cwd()}")
