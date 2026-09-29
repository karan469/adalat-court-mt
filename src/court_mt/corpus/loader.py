"""Discovery and loading of the parallel High Court judgment corpus.

The Drive export lays the corpus out as::

    <root>/HC Judgments | ML Assignment Text/
        english/clean/<id>.txt      # OCR-cleaned English source
        english/original/<id>.pdf   # scanned original (not used downstream)
        hindi/clean/<id>.txt        # OCR-cleaned Hindi translation
        hindi/original/<id>.pdf

Only the ``clean`` text is used; the folder is located by searching for those
leaf directories so the (space- and pipe-laden) parent name never gets encoded
into a path literal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SIDES = {"english": "en", "hindi": "hi"}


@dataclass(frozen=True)
class Judgment:
    """One judgment as a parallel English/Hindi document pair."""

    doc_id: str
    en: str
    hi: str

    @property
    def complete(self) -> bool:
        return bool(self.en.strip()) and bool(self.hi.strip())


def find_clean_dirs(root: Path) -> dict[str, Path]:
    """Locate the ``english/clean`` and ``hindi/clean`` directories under ``root``."""
    found: dict[str, Path] = {}
    for side, code in SIDES.items():
        matches = sorted(
            p for p in root.rglob(f"{side}/clean") if p.is_dir() and any(p.glob("*.txt"))
        )
        if matches:
            found[code] = matches[0]
            if len(matches) > 1:
                logger.warning("multiple %s/clean dirs; using %s", side, matches[0])
    missing = set(SIDES.values()) - set(found)
    if missing:
        raise FileNotFoundError(
            f"no *.txt found for {sorted(missing)} under {root}. "
            "Expected <root>/**/english/clean/*.txt and <root>/**/hindi/clean/*.txt"
        )
    return found


def _read(path: Path) -> str:
    # The OCR export is UTF-8; replace rather than fail on the odd stray byte.
    return path.read_text(encoding="utf-8", errors="replace")


def load_judgments(root: Path, *, require_pairs: bool = True) -> list[Judgment]:
    """Load every judgment that has text on at least one side.

    Args:
        root: directory containing the Drive export.
        require_pairs: drop documents missing either side instead of keeping them.

    Returns:
        Judgments sorted by numeric document id.
    """
    dirs = find_clean_dirs(root)
    texts: dict[str, dict[str, str]] = {}
    for code, directory in dirs.items():
        for path in directory.glob("*.txt"):
            texts.setdefault(path.stem, {})[code] = _read(path)

    judgments = [
        Judgment(doc_id=doc_id, en=sides.get("en", ""), hi=sides.get("hi", ""))
        for doc_id, sides in sorted(texts.items(), key=lambda kv: _sort_key(kv[0]))
    ]

    if require_pairs:
        incomplete = [j.doc_id for j in judgments if not j.complete]
        if incomplete:
            logger.warning("dropping %d unpaired document(s): %s", len(incomplete), incomplete)
        judgments = [j for j in judgments if j.complete]

    logger.info("loaded %d paired judgments from %s", len(judgments), root)
    return judgments


def _sort_key(doc_id: str) -> tuple[int, str]:
    return (int(doc_id), "") if doc_id.isdigit() else (1 << 30, doc_id)
