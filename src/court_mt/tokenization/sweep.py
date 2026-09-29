"""The vocabulary-budget experiment.

:mod:`court_mt.tokenization.extend` supplies the mechanism, mine candidates,
inject pieces into a SentencePiece proto. This module runs the experiment around
it: sweep a range of budgets, and for each one measure what changed.

Two reporting decisions carry the weight here.

**Both splits.** Pieces are mined from the training split, so training-set
fertility improves partly because the vocabulary has memorised that split's
terminology. The gap between the two curves is the optimism, and on a corpus of a
few dozen judgments it is large enough that quoting the in-domain number alone
would materially overstate the method.

**Both languages.** IndicBART uses one shared vocabulary for the English it reads
and the Hindi it writes, so a target-side extension has to be shown not to have
disturbed the source side. The two do not cost the same, either: the encoder
reads the English in a single parallel pass, while the decoder emits Hindi one
token at a time. Target tokens are therefore what generation latency tracks, and
the pair total is what a packed training example costs, different questions with
different right answers, so the frame reports all three rather than choosing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from court_mt.tokenization import extend, metrics
from court_mt.tokenization.adapters import TokenizerAdapter, TokenizerSpec, build

logger = logging.getLogger(__name__)

DEFAULT_BUDGETS = (250, 500, 1000, 2000, 4000, 8000)


def aggregate(adapter: TokenizerAdapter, texts: list[str]) -> metrics.TextStats:
    """Pool token statistics over a set of documents."""
    if not texts:
        raise ValueError("no texts to aggregate")
    total = metrics.measure(adapter, texts[0])
    for text in texts[1:]:
        total = total + metrics.measure(adapter, text)
    return total


def run_extension_sweep(
    base: TokenizerAdapter,
    candidates: list[extend.Candidate],
    *,
    train_target: list[str],
    held_out_target: list[str],
    held_out_source: list[str] | None = None,
    output_dir: Path,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    hidden_size: int,
    tied_embeddings: bool = True,
) -> pd.DataFrame:
    """Extend ``base`` at each budget and measure the result.

    Args:
        train_target: target-language documents the candidates were mined from.
        held_out_target: target-language documents the vocabulary has not seen.
        held_out_source: the matching source-language documents. Supplying them
            adds the source and pair-total columns, which is what makes the
            result a translation number rather than a monolingual one.

    Returns:
        One row per budget, plus a zero-budget baseline row.
    """
    base_model = extend.base_model_path(base)
    base_train = aggregate(base, train_target)
    base_held = aggregate(base, held_out_target)
    base_source = aggregate(base, held_out_source) if held_out_source else None

    # Budgets above the candidate pool are clamped rather than dropped: on a small
    # corpus that final point is the one that shows whether the curve has actually
    # flattened, so losing it would leave the knee indistinguishable from the end.
    planned = sorted({min(b, len(candidates)) for b in budgets})
    if planned[-1] < max(budgets):
        logger.info("budgets above %d clamped to the candidate pool", planned[-1])

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _row(
            base,
            0,
            base_train,
            base_held,
            base_source,
            base_train,
            base_held,
            base_source,
            0,
            hidden_size,
            tied_embeddings,
        )
    ]

    for budget in planned:
        model_path, n_added = extend.extend_sentencepiece(
            base_model, candidates[:budget], output_dir / f"{base.key}_plus{budget}.model"
        )
        adapter = build(TokenizerSpec(key=f"{base.key}+{budget}", spec=f"sp:{model_path}"))
        rows.append(
            _row(
                adapter,
                n_added,
                aggregate(adapter, train_target),
                aggregate(adapter, held_out_target),
                aggregate(adapter, held_out_source) if held_out_source else None,
                base_train,
                base_held,
                base_source,
                n_added,
                hidden_size,
                tied_embeddings,
            )
        )

    return pd.DataFrame(rows)


def _row(
    adapter: TokenizerAdapter,
    n_added: int,
    train: metrics.TextStats,
    target: metrics.TextStats,
    source: metrics.TextStats | None,
    base_train: metrics.TextStats,
    base_target: metrics.TextStats,
    base_source: metrics.TextStats | None,
    added: int,
    hidden_size: int,
    tied_embeddings: bool,
) -> dict[str, object]:
    train_change = train.fertility / base_train.fertility - 1
    target_change = target.n_tokens / base_target.n_tokens - 1

    row: dict[str, object] = {
        "pieces_added": n_added,
        "vocab_size": adapter.vocab_size,
        "train_fertility": train.fertility,
        "held_out_fertility": target.fertility,
        "train_change": train_change,
        "held_out_change": target.fertility / base_target.fertility - 1,
        "optimism_gap": train_change - (target.fertility / base_target.fertility - 1),
        "target_tokens": target.n_tokens,
        "target_change": target_change,
        "added_parameters": extend.embedding_cost(
            added, hidden_size, tied_embeddings=tied_embeddings
        ),
        "lossless": target.roundtrip_ok,
    }

    if source is not None and base_source is not None:
        pair = source.n_tokens + target.n_tokens
        base_pair = base_source.n_tokens + base_target.n_tokens
        row.update(
            {
                "source_tokens": source.n_tokens,
                "source_change": source.n_tokens / base_source.n_tokens - 1,
                "pair_tokens": pair,
                "pair_change": pair / base_pair - 1,
            }
        )
    return row


def budget_by_trainability(candidates: list[extend.Candidate], *, min_occurrences: int = 10) -> int:
    """Largest budget in which every added piece is frequent enough to be learned.

    Held-out fertility on this corpus keeps improving as the budget grows, so the
    curve offers no knee to stop at. The real constraint is elsewhere: each added
    piece creates a **randomly initialised** embedding row, and that row is only
    useful once training has seen the piece enough times to move it away from
    noise. A piece appearing twice in 21 judgments adds a parameter vector the
    fine-tune cannot train, while still costing memory and softmax mass.

    Candidates are ranked by tokens saved, not frequency, so this is a filter over
    the ranked list rather than a simple prefix length, it returns how many of
    the top candidates clear the bar.
    """
    return sum(1 for candidate in candidates if candidate.frequency >= min_occurrences)


def marginal_gains(sweep: pd.DataFrame) -> pd.Series:
    """Held-out fertility improvement added by each successive budget, in points.

    Indexed by ``pieces_added`` so each gain is attributable to the budget that
    bought it rather than to a row number.
    """
    ordered = sweep.sort_values("pieces_added").set_index("pieces_added")
    return (-ordered["held_out_change"].diff() * 100).rename("marginal_gain_pp").dropna()
