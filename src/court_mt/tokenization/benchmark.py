"""The benchmark loop: every tokenizer against every judgment, both languages.

Produces two tidy frames rather than one, because the questions have different
shapes:

``per_document``
    One row per (tokenizer, document, language). Keeps the distribution, which is
    what context-window and chunking decisions depend on, a mean of 900 tokens
    hides that a third of judgments blow past 1024.

``summary``
    One row per tokenizer. Corpus ratios are computed as ratio-of-sums (pooling
    counts, then dividing), never as the mean of per-document ratios, which would
    weight a two-page order the same as a forty-page judgment.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm

from court_mt.corpus.loader import Judgment
from court_mt.tokenization import cost, metrics
from court_mt.tokenization.adapters import TokenizerAdapter

logger = logging.getLogger(__name__)

LANGUAGES = ("en", "hi")


@dataclass
class _LanguageAccumulator:
    """Pooled counts and per-document lengths for one tokenizer/language pair."""

    total: metrics.TextStats | None = None
    token_lengths: list[int] = field(default_factory=list)
    token_counts: Counter[int] = field(default_factory=Counter)

    def add(self, stats: metrics.TextStats, ids: list[int]) -> None:
        self.total = stats if self.total is None else self.total + stats
        self.token_lengths.append(stats.n_tokens)
        self.token_counts.update(ids)


def run(
    adapters: list[TokenizerAdapter], judgments: list[Judgment], *, show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark every adapter over every judgment.

    Returns:
        ``(summary, per_document)`` frames, summary sorted by Hindi fertility.
    """
    if not judgments:
        raise ValueError("no judgments to benchmark")

    document_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for adapter in tqdm(adapters, desc="tokenizers", disable=not show_progress):
        accumulators = {lang: _LanguageAccumulator() for lang in LANGUAGES}

        for judgment in judgments:
            per_lang_stats = {}
            for lang in LANGUAGES:
                text = getattr(judgment, lang)
                ids = adapter.encode(text)
                stats = metrics.measure(adapter, text)
                accumulators[lang].add(stats, ids)
                per_lang_stats[lang] = stats
                document_rows.append(_document_row(adapter, judgment, lang, stats))

            document_rows[-1]["parity"] = metrics.parity(per_lang_stats["hi"], per_lang_stats["en"])

        summary_rows.append(_summary_row(adapter, accumulators, len(judgments)))

    summary = pd.DataFrame(summary_rows).sort_values("hi_fertility").reset_index(drop=True)
    per_document = pd.DataFrame(document_rows)
    return _add_relative_costs(summary), per_document


def _document_row(
    adapter: TokenizerAdapter, judgment: Judgment, lang: str, stats: metrics.TextStats
) -> dict[str, object]:
    return {
        "tokenizer": adapter.key,
        "doc_id": judgment.doc_id,
        "lang": lang,
        "n_chars": stats.n_chars,
        "n_words": stats.n_words,
        "n_tokens": stats.n_tokens,
        "fertility": stats.fertility,
        "chars_per_token": stats.chars_per_token,
        "whole_word_rate": stats.whole_word_rate,
        "unk_rate": stats.unk_rate,
        "roundtrip_ok": stats.roundtrip_ok,
        "spacing_ok": stats.spacing_ok,
    }


def _summary_row(
    adapter: TokenizerAdapter, accumulators: dict[str, _LanguageAccumulator], n_documents: int
) -> dict[str, object]:
    en, hi = accumulators["en"], accumulators["hi"]
    assert en.total is not None and hi.total is not None

    hi_profile = cost.CostProfile(total_tokens=hi.total.n_tokens, n_documents=n_documents)
    fit = cost.context_fit(hi.token_lengths)

    row: dict[str, object] = {
        "tokenizer": adapter.key,
        "label": adapter.label,
        "family": adapter.spec.family or adapter.spec.scheme,
        "vocab_size": adapter.vocab_size,
        "specials_per_seq": adapter.n_special_per_sequence,
        # Verbosity, per language.
        "en_fertility": en.total.fertility,
        "hi_fertility": hi.total.fertility,
        "en_chars_per_token": en.total.chars_per_token,
        "hi_chars_per_token": hi.total.chars_per_token,
        "hi_bytes_per_token": hi.total.bytes_per_token,
        # The Indic penalty.
        "parity": metrics.parity(hi.total, en.total),
        "hi_whole_word_rate": hi.total.whole_word_rate,
        # Fidelity: a cheap tokenizer that drops characters is not cheap.
        "hi_unk_rate": hi.total.unk_rate,
        "en_unk_rate": en.total.unk_rate,
        "hi_roundtrip_ok": hi.total.roundtrip_ok,
        "en_roundtrip_ok": en.total.roundtrip_ok,
        # Character-lossless but re-spaced: a detokenisation cost, not data loss.
        "hi_spacing_ok": hi.total.spacing_ok,
        "en_spacing_ok": en.total.spacing_ok,
        # Distributional quality of the segmentation.
        "hi_renyi_efficiency": metrics.renyi_efficiency(hi.token_counts, adapter.vocab_size),
        "hi_vocab_utilization": metrics.vocab_utilization(hi.token_counts, adapter.vocab_size),
        # Deployment consequences.
        "hi_tokens_total": hi.total.n_tokens,
        "en_tokens_total": en.total.n_tokens,
        "hi_mean_tokens_per_doc": hi_profile.mean_tokens_per_document,
        "hi_p95_tokens_per_doc": float(pd.Series(hi.token_lengths).quantile(0.95)),
        "hi_max_tokens_per_doc": max(hi.token_lengths),
        "hi_usd_per_corpus_pass": hi_profile.usd_per_corpus_pass,
    }
    row.update({f"hi_fit_{limit}": share for limit, share in fit.items()})
    row["_hi_lengths"] = hi.token_lengths
    return row


def _add_relative_costs(summary: pd.DataFrame) -> pd.DataFrame:
    """Index token and attention cost against the most efficient tokenizer on Hindi."""
    baseline_index = summary["hi_tokens_total"].idxmin()
    baseline_tokens = int(summary.loc[baseline_index, "hi_tokens_total"])
    baseline_lengths = summary.loc[baseline_index, "_hi_lengths"]

    summary["hi_token_cost_index"] = summary["hi_tokens_total"].apply(
        lambda total: cost.linear_cost_index(int(total), baseline_tokens)
    )
    summary["hi_attention_cost_index"] = summary["_hi_lengths"].apply(
        lambda lengths: cost.attention_cost_index(lengths, baseline_lengths)
    )
    summary.attrs["baseline_tokenizer"] = summary.loc[baseline_index, "tokenizer"]
    return summary.drop(columns=["_hi_lengths"])
