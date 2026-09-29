#!/usr/bin/env python3
"""Stage 1: turn the raw Drive export into a clean, versioned corpus.

Reads ``data/raw/**/{english,hindi}/clean/*.txt``, applies the normalisation in
:mod:`court_mt.corpus.clean`, and writes:

    data/processed/corpus.jsonl     one JSON object per judgment: doc_id, en, hi
    data/processed/quarantine.jsonl pairs rejected by the quality gate
    data/interim/clean/{en,hi}/     cleaned plain text, for eyeballing diffs
    reports/corpus_stats.csv        per-document character/word counts
    reports/corpus_quality.csv      the quality verdict for every pair

Five of the thirty Hindi files have had their Devanagari destroyed at encoding
time and survive only as runs of "?". They pass a length-based alignment check,
so they are caught on script content instead, see court_mt.corpus.validate.

Run before any tokenizer script; every later stage reads only corpus.jsonl, so
the cleaning rules are applied exactly once and in one place.

Usage:
    python scripts/01_build_corpus.py [--raw-dir DIR] [--no-unwrap] [--keep-ocr-markers]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from court_mt import paths  # noqa: E402
from court_mt.corpus.clean import clean  # noqa: E402
from court_mt.corpus.loader import Judgment, load_judgments  # noqa: E402
from court_mt.corpus.validate import partition  # noqa: E402
from court_mt.environment import setup_logging  # noqa: E402
from court_mt.tokenization.metrics import count_devanagari, words  # noqa: E402

logger = logging.getLogger("build_corpus")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=paths.RAW, help="directory holding the Drive export"
    )
    parser.add_argument("--out", type=Path, default=paths.PROCESSED / "corpus.jsonl")
    parser.add_argument("--no-unwrap", action="store_true", help="keep the OCR hard line wrapping")
    parser.add_argument(
        "--keep-ocr-markers", action="store_true", help="do not repair li./lili. list markers"
    )
    parser.add_argument(
        "--no-interim", action="store_true", help="skip writing per-document cleaned text"
    )
    parser.add_argument(
        "--keep-corrupted", action="store_true", help="skip the Devanagari quality gate"
    )
    return parser.parse_args(argv)


def clean_judgment(judgment: Judgment, *, unwrap: bool, fix_markers: bool) -> Judgment:
    return Judgment(
        doc_id=judgment.doc_id,
        en=clean(judgment.en, unwrap=unwrap, fix_ocr_markers=fix_markers),
        hi=clean(judgment.hi, unwrap=unwrap, fix_ocr_markers=fix_markers),
    )


def document_stats(judgment: Judgment, raw: Judgment) -> dict[str, object]:
    return {
        "doc_id": judgment.doc_id,
        "en_chars": len(judgment.en),
        "hi_chars": len(judgment.hi),
        "en_words": len(words(judgment.en)),
        "hi_words": len(words(judgment.hi)),
        "hi_devanagari_chars": count_devanagari(judgment.hi),
        # A ratio far from ~1.1 flags a misaligned or truncated pair.
        "hi_en_char_ratio": len(judgment.hi) / len(judgment.en) if judgment.en else float("nan"),
        "en_chars_removed_by_cleaning": len(raw.en) - len(judgment.en),
        "hi_chars_removed_by_cleaning": len(raw.hi) - len(judgment.hi),
    }


def write_interim(judgments: list[Judgment]) -> None:
    for lang in ("en", "hi"):
        directory = paths.INTERIM / "clean" / lang
        directory.mkdir(parents=True, exist_ok=True)
        for judgment in judgments:
            (directory / f"{judgment.doc_id}.txt").write_text(
                getattr(judgment, lang), encoding="utf-8"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    paths.ensure_dirs()

    raw_judgments = load_judgments(args.raw_dir)
    cleaned = [
        clean_judgment(j, unwrap=not args.no_unwrap, fix_markers=not args.keep_ocr_markers)
        for j in raw_judgments
    ]

    import pandas as pd

    usable, quarantined, quality = partition(cleaned)
    if args.keep_corrupted:
        usable, quarantined = cleaned, []

    pd.DataFrame([vars(report) for report in quality]).to_csv(
        paths.REPORTS / "corpus_quality.csv", index=False
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(usable, args.out)
    logger.info("wrote %d usable judgments -> %s", len(usable), args.out)
    if quarantined:
        quarantine_path = args.out.parent / "quarantine.jsonl"
        _write_jsonl(quarantined, quarantine_path)
        logger.info("wrote %d quarantined judgments -> %s", len(quarantined), quarantine_path)

    if not args.no_interim:
        write_interim(cleaned)

    kept = {j.doc_id for j in usable}
    stats = pd.DataFrame(
        [document_stats(c, r) for c, r in zip(cleaned, raw_judgments) if c.doc_id in kept]
    )
    stats_path = paths.REPORTS / "corpus_stats.csv"
    stats.to_csv(stats_path, index=False)
    logger.info("wrote %s", stats_path)

    _report(stats, quality)
    _warn_on_suspect_pairs(stats)
    return 0


def _write_jsonl(judgments: list[Judgment], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for judgment in judgments:
            handle.write(
                json.dumps(
                    {"doc_id": judgment.doc_id, "en": judgment.en, "hi": judgment.hi},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _report(stats: "pd.DataFrame", quality: list) -> None:  # type: ignore[name-defined]
    rejected = [r for r in quality if not r.usable]
    totals = stats[["en_chars", "hi_chars", "en_words", "hi_words"]].sum()
    print(f"\ncorpus: {len(stats)} usable pairs of {len(quality)}")
    if rejected:
        print(f"  quarantined: {', '.join(r.doc_id for r in rejected)} " f"({rejected[0].reason})")
    print(f"  english : {totals.en_chars:>9,} chars  {totals.en_words:>8,} words")
    print(f"  hindi   : {totals.hi_chars:>9,} chars  {totals.hi_words:>8,} words")
    print(
        f"  hi/en char ratio: median {stats.hi_en_char_ratio.median():.2f}  "
        f"range {stats.hi_en_char_ratio.min():.2f}-{stats.hi_en_char_ratio.max():.2f}"
    )
    print(
        f"  per-doc english words: median {stats.en_words.median(): .0f}  max {stats.en_words.max(): }"
    )


def _warn_on_suspect_pairs(stats: "pd.DataFrame", tolerance: float = 2.5) -> None:  # type: ignore[name-defined]
    """Flag pairs whose length ratio is far from the corpus median.

    Hindi legal translation runs slightly longer than English in characters; a
    document several times longer or shorter usually means a truncated OCR pass
    or a mismatched file, and would quietly distort every parity number.
    """
    median = stats.hi_en_char_ratio.median()
    suspect = stats[
        (stats.hi_en_char_ratio > median * tolerance)
        | (stats.hi_en_char_ratio < median / tolerance)
    ]
    if not suspect.empty:
        logger.warning(
            "%d document(s) have an unusual hi/en length ratio (median %.2f): %s",
            len(suspect),
            median,
            ", ".join(f"{r.doc_id}={r.hi_en_char_ratio:.2f}" for r in suspect.itertuples()),
        )


if __name__ == "__main__":
    raise SystemExit(main())
