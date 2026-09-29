#!/usr/bin/env python3
"""Collect the three Task-3 stages into one comparison.

Reads whatever ``run_stage.py`` has written so far and produces the tables and
figures the report needs. Runs that used different training settings are refused
rather than compared: with 120 test sentences the difference between two
optimisers is easily as large as the difference between two vocabularies, so a
mismatched pair would measure the settings.

Usage:
    python notebooks/compare_runs.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from court_mt.environment import configure, setup_logging  # noqa: E402

configure()

import pandas as pd  # noqa: E402

from court_mt import paths, viz  # noqa: E402
from court_mt.corpus.dataset import load_pairs  # noqa: E402

logger = logging.getLogger("compare_runs")

# Canonical stages come first; any tagged variant follows in name order.
STAGE_ORDER = ("baseline", "stock", "extended")
LABELS = {
    "baseline": "zero-shot (no fine-tuning)",
    "stock": "fine-tuned, stock vocab",
    "extended": "fine-tuned, extended vocab",
}


def label_for(name: str, data: dict) -> str:
    """A readable row label, describing what a tagged variant actually varied."""
    if name in LABELS:
        return LABELS[name]
    n_added = data.get("n_added_tokens", 0)
    width = data.get("max_phrase_words")
    detail = f"{n_added} pieces"
    if width == 1:
        detail += ", words only"
    elif width:
        detail += f", up to {width} words"
    return f"fine-tuned, {detail}"


# Settings that must match for a fair comparison between the two training runs.
COMPARABLE = ("epochs", "batch_size", "gradient_accumulation", "learning_rate", "optim")


def load_stages() -> dict[str, dict]:
    """Load every stage result on disk, canonical stages first, variants after."""
    found = {
        path.stem.removeprefix("task3_"): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(paths.REPORTS.glob("task3_*.json"))
    }
    for stage in STAGE_ORDER:
        if stage not in found:
            logger.warning(
                "missing task3_%s.json, run: " "python notebooks/run_stage.py --stage %s",
                stage,
                stage,
            )
    if not found:
        raise SystemExit("no stage results found; run run_stage.py first")
    return found


def ordered_names(results: dict[str, dict]) -> list[str]:
    canonical = [s for s in STAGE_ORDER if s in results]
    variants = sorted(n for n in results if n not in STAGE_ORDER)
    return canonical + variants


def check_comparable(results: dict[str, dict]) -> None:
    """Refuse to compare training runs that used different settings."""
    configs = {
        stage: {k: v for k, v in (data["training"] or {}).items() if k in COMPARABLE}
        for stage, data in results.items()
        if data.get("training")
    }
    if len(configs) < 2:
        return

    (first, baseline), *rest = configs.items()
    for stage, config in rest:
        differences = {k: (baseline[k], config[k]) for k in baseline if baseline[k] != config[k]}
        if differences:
            raise SystemExit(
                f"'{first}' and '{stage}' used different settings: {differences}\n"
                "Re-run both with the same flags, otherwise the comparison "
                "measures the settings, not the vocabulary."
            )


def main() -> int:
    setup_logging()
    paths.ensure_dirs()
    results = load_stages()
    check_comparable(results)

    ordered = ordered_names(results)
    baseline_tokens = results[ordered[0]]["scores"]["target_tokens"]
    baseline_chrf = results[ordered[0]]["scores"]["chrf"]

    frame = pd.DataFrame(
        [
            {
                "run": label_for(stage, results[stage]),
                "BLEU": results[stage]["scores"]["bleu"],
                "chrF": results[stage]["scores"]["chrf"],
                "ΔchrF": results[stage]["scores"]["chrf"] - baseline_chrf,
                "tokens/sentence": (
                    results[stage]["scores"]["target_tokens"]
                    / results[stage]["scores"]["n_sentences"]
                ),
                "token change": results[stage]["scores"]["target_tokens"] / baseline_tokens - 1,
                "added tokens": results[stage]["n_added_tokens"],
            }
            for stage in ordered
        ]
    )

    frame.to_csv(paths.REPORTS / "task3_scores.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v: .3f}"))

    test = load_pairs(paths.PROCESSED / "pairs_test.jsonl")
    translations = pd.DataFrame(
        {
            "english": [r["en"] for r in test],
            "reference": [r["hi"] for r in test],
            **{stage: results[stage]["hypotheses"] for stage in ordered},
        }
    )
    translations.to_csv(paths.REPORTS / "task3_translations.csv", index=False)

    theme = viz.use(viz.LIGHT)
    viz.save(
        viz.grouped_bar(
            frame.set_index("run")[["BLEU", "chrF"]],
            title="Translation quality on held-out judgments",
            subtitle=f"{len(test)} sentence pairs · chrF is the steadier metric at this size",
            theme=theme,
        ),
        paths.FIGURES / "task3_quality.png",
    )
    viz.save(
        viz.ranked_bar(
            frame.set_index("run")["tokens/sentence"],
            title="Hindi tokens generated per sentence",
            subtitle="Decoder steps, one forward pass each, so this is generation latency",
            value_format="{:.1f}",
            theme=theme,
        ),
        paths.FIGURES / "task3_tokens.png",
    )

    print(f"\nwrote {paths.REPORTS / 'task3_scores.csv'}")
    print(f"wrote {paths.REPORTS / 'task3_translations.csv'}")
    print(f"wrote 2 figures to {paths.FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
