"""Reading the processed corpus back in.

Stage 1 writes ``data/processed/corpus.jsonl``; everything downstream reads it
through here, so no notebook or script re-implements the cleaning rules or the
file layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from court_mt import paths
from court_mt.corpus.loader import Judgment


def load_corpus(path: Path | None = None) -> list[Judgment]:
    """Load the processed parallel corpus."""
    path = path or paths.PROCESSED / "corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `python scripts/01_build_corpus.py` first")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return [Judgment(doc_id=r["doc_id"], en=r["en"], hi=r["hi"]) for r in records]


def texts(judgments: list[Judgment], lang: str) -> list[str]:
    """All documents on one side of the corpus."""
    if lang not in ("en", "hi"):
        raise ValueError(f"lang must be 'en' or 'hi', got {lang!r}")
    return [getattr(j, lang) for j in judgments]


def write_pairs(pairs: list, path: Path) -> Path:
    """Write aligned sentence pairs as JSONL.

    Each record keeps ``doc_id`` and the bead shape alongside the text, so a
    training example can always be traced back to the judgment and the alignment
    decision it came from.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(
                json.dumps(
                    {
                        "doc_id": pair.doc_id,
                        "en": pair.source,
                        "hi": pair.target,
                        "bead": f"{pair.bead[0]}-{pair.bead[1]}",
                        "cost": round(pair.cost, 2),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def load_pairs(path: Path) -> list[dict]:
    """Read a sentence-pair JSONL file back."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
