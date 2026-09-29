"""Translating token counts into the costs a deployment actually pays.

Fertility is an abstraction; what a team feels is three concrete things, and each
scales differently with sequence length:

* **Price**, API billing is linear in tokens, so a 3x parity penalty is a 3x
  bill for the same judgment.
* **Context**, a judgment either fits the window or has to be chunked. This is
  a step function, not a gradient, so it is reported as "share of documents that
  fit".
* **Compute**, prefill cost is linear in tokens for the feed-forward path but
  quadratic for attention. At judgment length (thousands of tokens) the
  quadratic term is what bites, so both indices are reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Indicative published list prices, USD per million input tokens. Used only to
# turn token deltas into a number a reader can weigh; ratios are what matter.
USD_PER_MILLION_INPUT = 2.50

# Context windows worth checking: mBART/IndicBART encoders are hard-capped at
# 1024 positions, most small decoder LLMs sit at 4k-8k.
CONTEXT_LIMITS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class CostProfile:
    """Deployment-facing cost summary for one tokenizer on one language."""

    total_tokens: int
    n_documents: int
    usd_per_million: float = USD_PER_MILLION_INPUT

    @property
    def mean_tokens_per_document(self) -> float:
        return self.total_tokens / self.n_documents if self.n_documents else float("nan")

    @property
    def usd_per_corpus_pass(self) -> float:
        """Cost of sending the whole corpus through once as input."""
        return self.total_tokens / 1_000_000 * self.usd_per_million


def context_fit(
    token_lengths: Sequence[int], limits: Sequence[int] = CONTEXT_LIMITS
) -> dict[int, float]:
    """Share of documents that fit whole within each context limit."""
    if not token_lengths:
        return {limit: float("nan") for limit in limits}
    return {
        limit: sum(1 for n in token_lengths if n <= limit) / len(token_lengths) for limit in limits
    }


def chunks_required(token_lengths: Sequence[int], limit: int, overlap: int = 0) -> int:
    """Number of windows needed to cover every document at ``limit`` tokens.

    ``overlap`` models the stride a real chunker uses to preserve cross-sentence
    context; it makes long documents cost strictly more than ``ceil(n/limit)``.
    """
    if limit <= overlap:
        raise ValueError(f"limit ({limit}) must exceed overlap ({overlap})")
    stride = limit - overlap
    return sum(max(1, -(-(max(0, n - limit)) // stride) + 1) for n in token_lengths)


def linear_cost_index(total_tokens: int, baseline_tokens: int) -> float:
    """Relative cost of the token-linear path (billing, feed-forward, memory)."""
    return total_tokens / baseline_tokens if baseline_tokens else float("nan")


def attention_cost_index(token_lengths: Sequence[int], baseline_lengths: Sequence[int]) -> float:
    """Relative cost of the quadratic attention path over the same documents.

    Computed per document and summed, because sum of squares is not the square of
    the sum, a corpus of many short documents is far cheaper than one long one
    at equal token totals.
    """
    numerator = sum(n * n for n in token_lengths)
    denominator = sum(n * n for n in baseline_lengths)
    return numerator / denominator if denominator else float("nan")
