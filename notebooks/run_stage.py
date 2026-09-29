#!/usr/bin/env python3
"""Run one Task-3 stage in its own process, and write its results to disk.

Each stage is a separate invocation on purpose. The three stages together need
more memory than a 24GB machine has, but no single one does, and a process that
exits returns everything it held, which a notebook kernel does not. Running them
separately is what makes this fit at all, and it means an out-of-memory failure
in one stage costs nothing already computed.

Stages:

    baseline    the un-fine-tuned model, translating zero-shot. The "before".
    stock       LoRA fine-tune on the court pairs, stock vocabulary.
    extended    the same fine-tune with 507 legal pieces added to the vocabulary.

Each writes ``reports/task3_<stage>.json`` with its scores, its translations and
the configuration that produced them. ``compare_runs.py`` reads those and builds
the tables and figures.

Usage:
    python notebooks/run_stage.py --stage baseline
    python notebooks/run_stage.py --stage stock
    python notebooks/run_stage.py --stage extended

The training stages must share hyperparameters or the comparison measures the
settings rather than the vocabulary. If one stage needs ``--optim adafactor`` to
fit in memory, the other needs it too; the script records what it used and
``compare_runs.py`` refuses to compare runs that disagree.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from court_mt.environment import configure, setup_logging  # noqa: E402

configure()

from court_mt import evaluation, finetune, modeling, paths  # noqa: E402
from court_mt.corpus.dataset import load_corpus, load_pairs, texts  # noqa: E402
from court_mt.corpus.split import split_corpus  # noqa: E402
from court_mt.tokenization import extend  # noqa: E402
from court_mt.tokenization.registry import load_adapters, load_specs  # noqa: E402

logger = logging.getLogger("run_stage")

STAGES = ("baseline", "stock", "extended")
N_PIECES = 507


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        help="adafactor stores ~1MB of embedding state instead of ~2GB; "
        "use it for BOTH training stages or not at all",
    )
    parser.add_argument(
        "--no-checkpointing",
        action="store_true",
        help="disable gradient checkpointing (faster, uses more memory)",
    )
    parser.add_argument("--pieces", type=int, default=N_PIECES)
    parser.add_argument(
        "--max-phrase-words",
        type=int,
        default=3,
        help="longest n-gram mined as a single vocabulary piece. 1 restricts the "
        "extension to whole words, which tests whether multi-word pieces are "
        "what cost the decoder its compositional flexibility",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for the output filename, so a variant does not overwrite "
        "an earlier run of the same stage (e.g. --tag words-only)",
    )
    return parser.parse_args(argv)


def mine_surfaces(n_pieces: int, max_phrase_words: int = 3) -> list[str]:
    """The domain pieces to add, mined against mBART's own segmentation.

    Re-mined rather than reused from Task 1: a candidate's value is how many
    tokens it saves, which depends on how the base tokenizer splits it, so the
    ranking does not carry over from IndicBART.
    """
    split = split_corpus(load_corpus())
    mbart = load_adapters(load_specs(paths.CONFIGS / "tokenizers.yaml", only=["mbart-50"]))[0]
    candidates = extend.mine_candidates(
        texts(split.train, "hi"),
        mbart,
        require_script="Devanagari",
        max_phrase_words=max_phrase_words,
    )
    surfaces = [c.surface for c in candidates[:n_pieces]]
    logger.info(
        "mined %d pieces (max_phrase_words=%d); longest is %d words",
        len(surfaces),
        max_phrase_words,
        max((s.count(" ") + 1 for s in surfaces), default=0),
    )
    return surfaces


def build_stage(args: argparse.Namespace):
    """Return ``(model, tokenizer, n_added)`` for the requested stage."""
    if args.stage == "baseline":
        from transformers import AutoModelForSeq2SeqLM

        tokenizer, _ = modeling.load_tokenizer()
        return AutoModelForSeq2SeqLM.from_pretrained(modeling.BASE_MODEL), tokenizer, 0

    extended = args.stage == "extended"
    config = modeling.AdaptationConfig(
        name=args.stage,
        extended_vocab=extended,
        added_surfaces=mine_surfaces(args.pieces, args.max_phrase_words) if extended else [],
    )
    return modeling.build(config)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    paths.ensure_dirs()

    test = load_pairs(paths.PROCESSED / "pairs_test.jsonl")
    sources = [r["en"] for r in test]
    references = [r["hi"] for r in test]
    device = modeling.device()

    logger.info("stage=%s  device=%s  test=%d pairs", args.stage, device, len(test))
    model, tokenizer, n_added = build_stage(args)

    training_config = None
    if args.stage != "baseline":
        training_config = finetune.TrainingConfig(
            output_dir=paths.ARTIFACTS / "runs" / args.stage,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            optim=args.optim,
            gradient_checkpointing=not args.no_checkpointing,
        )
        logger.info(
            "training: optim=%s checkpointing=%s batch=%d",
            training_config.optim,
            training_config.gradient_checkpointing,
            training_config.batch_size,
        )
        model, _ = finetune.train(
            model,
            tokenizer,
            training_config,
            finetune.load_split("train", tokenizer),
            finetune.load_split("dev", tokenizer),
        )

    hypotheses = evaluation.translate(model, tokenizer, sources, device=device)
    scores = evaluation.score(args.stage, hypotheses, references, sources, tokenizer)

    name = f"{args.stage}-{args.tag}" if args.tag else args.stage
    output = paths.REPORTS / f"task3_{name}.json"
    output.write_text(
        json.dumps(
            {
                "stage": name,
                "scores": asdict(scores),
                "n_added_tokens": n_added,
                "max_phrase_words": args.max_phrase_words,
                "hypotheses": hypotheses,
                "training": asdict(training_config)
                | {"output_dir": str(training_config.output_dir)}
                if training_config
                else None,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{name}: BLEU {scores.bleu:.2f}   chrF {scores.chrf:.2f}   "
        f"{scores.tokens_per_sentence:.1f} Hindi tokens/sentence"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
