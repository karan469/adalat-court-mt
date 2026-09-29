"""Sentence alignment by length, after Gale & Church (1993).

Segmenting both sides of a judgment does not pair them: the Hindi translation
merges two English sentences here, splits one into three there, and occasionally
drops a clause entirely. Pairing by position would silently mis-pair everything
after the first discrepancy.

Gale-Church aligns on a single observation: a translated sentence is about a
constant multiple of its source's length in characters, and the deviation is
roughly normal. Dynamic programming then finds the cheapest sequence of *beads*
(1-1, 1-2, 2-1, 1-0, 0-1, 2-2) over the whole document, so a local merge or
deletion costs a little rather than derailing the rest.

Why this rather than embeddings: it needs no model, runs on CPU in milliseconds,
and is the standard baseline for exactly this situation, a clean parallel
document pair with occasional structural divergence. LaBSE or LASER would align
better on noisy or reordered text, and are the obvious upgrade if the quality
checks in notebook 4 come back poor.

One deviation from the paper: ``c`` and ``s2`` are calibrated on this corpus
rather than taken from the paper's English-German constants. Hindi runs *shorter*
than English in characters here (about 0.93x), so the published ``c = 1.0`` would
treat every correct pair as slightly anomalous.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import regex

from court_mt.corpus.loader import Judgment

logger = logging.getLogger(__name__)

# Bead shapes and their prior costs, as -100 * log(probability) from the paper.
# A 1-1 alignment is free; anything else has to pay for itself in length fit.
BEADS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0),
    (1, 0, 230),
    (0, 1, 230),
    (2, 1, 440),
    (1, 2, 440),
    (2, 2, 450),
)

# Guards against division by zero on insertion/deletion beads.
MIN_LENGTH = 1


@dataclass(frozen=True)
class AlignedPair:
    """One aligned unit: n source sentences against m target sentences."""

    doc_id: str
    source: str
    target: str
    bead: tuple[int, int]
    cost: float

    @property
    def is_one_to_one(self) -> bool:
        return self.bead == (1, 1)

    @property
    def length_ratio(self) -> float:
        return len(self.target) / len(self.source) if self.source else math.inf


def calibrate(judgments: list[Judgment]) -> tuple[float, float]:
    """Estimate the target/source length ratio and its variance from the corpus.

    ``s2`` is **not** the variance of the length ratio. The model assumes the
    length difference grows with the square root of the source length, so the
    quantity that must be unit-normal is ``(l2 - c*l1) / sqrt(l1*s2)``, making
    ``s2`` the mean of ``(l2 - c*l1)^2 / l1``, in characters. Estimating the ratio
    variance instead yields a number ~1000x too small, the model becomes wildly
    overconfident, and the aligner collapses onto 2-2 beads because merging is the
    only way to average away deviations it considers impossible.

    Returns:
        ``(c, s2)``, mean characters of target per character of source, and the
        length-normalised variance, both measured document-wise.
    """
    usable = [j for j in judgments if j.en]
    if not usable:
        raise ValueError("no judgments to calibrate on")

    source = [len(j.en) for j in usable]
    target = [len(j.hi) for j in usable]
    c = sum(target) / sum(source)
    s2 = sum((t - c * s) ** 2 / s for s, t in zip(source, target)) / len(usable)

    # Documents average away sentence-level variation, so a document-fitted s2 is
    # a floor. The published English-German value is ~6.8; anything far below that
    # would reintroduce the overconfidence this function exists to avoid.
    s2 = max(s2, 1.0)
    logger.info("calibrated Gale-Church: c=%.4f, s2=%.3f (n=%d)", c, s2, len(usable))
    return c, s2


def _normal_tail(z: float) -> float:
    """Two-sided normal tail probability, 2 * (1 - Phi(|z|))."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _length_cost(source_len: int, target_len: int, c: float, s2: float) -> float:
    """Cost of pairing ``source_len`` characters with ``target_len`` characters."""
    source_len = max(source_len, MIN_LENGTH)
    mean = source_len * c
    deviation = (target_len - mean) / math.sqrt(source_len * s2)
    probability = _normal_tail(deviation)
    # Clamp: an exact-zero tail would be an infinite cost and break the argmin.
    return -100.0 * math.log(max(probability, 1e-300))


def align_document(
    source: list[str], target: list[str], *, c: float, s2: float
) -> list[tuple[tuple[int, int], float, list[str], list[str]]]:
    """Align two sentence lists, returning the cheapest bead sequence.

    Returns:
        One tuple per bead: ``(shape, cost, source_sentences, target_sentences)``.
    """
    n, m = len(source), len(target)
    source_lengths = [len(s) for s in source]
    target_lengths = [len(t) for t in target]

    infinity = float("inf")
    best = [[infinity] * (m + 1) for _ in range(n + 1)]
    backpointer: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if best[i][j] == infinity:
                continue
            for di, dj, prior in BEADS:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                cost = (
                    best[i][j]
                    + prior
                    + _length_cost(sum(source_lengths[i:ni]), sum(target_lengths[j:nj]), c, s2)
                )
                if cost < best[ni][nj]:
                    best[ni][nj] = cost
                    backpointer[ni][nj] = (di, dj)

    if best[n][m] == infinity:
        logger.warning("no alignment path found for %d x %d sentences", n, m)
        return []

    # Walk the backpointers from the corner and reverse.
    beads: list[tuple[tuple[int, int], float, list[str], list[str]]] = []
    i, j = n, m
    while (i, j) != (0, 0):
        step = backpointer[i][j]
        if step is None:
            logger.warning("broken backtrace at (%d, %d)", i, j)
            break
        di, dj = step
        beads.append(
            ((di, dj), best[i][j] - best[i - di][j - dj], source[i - di : i], target[j - dj : j])
        )
        i, j = i - di, j - dj

    return list(reversed(beads))


def align_judgment(
    judgment: Judgment,
    source_sentences: list[str],
    target_sentences: list[str],
    *,
    c: float,
    s2: float,
) -> list[AlignedPair]:
    """Align one judgment into :class:`AlignedPair` records.

    Insertion and deletion beads (1-0, 0-1) are dropped: they record that a
    sentence has no counterpart, which is useful diagnostically but is not a
    training example.
    """
    pairs: list[AlignedPair] = []
    for shape, cost, sources, targets in align_document(
        source_sentences, target_sentences, c=c, s2=s2
    ):
        if not sources or not targets:
            continue
        pairs.append(
            AlignedPair(
                doc_id=judgment.doc_id,
                source=" ".join(sources),
                target=" ".join(targets),
                bead=shape,
                cost=cost,
            )
        )
    return pairs


# --- Quality filtering -------------------------------------------------------
#
# Alignment always produces an answer; some of those answers are wrong. These
# thresholds are deliberately loose, the goal is to remove pairs that would
# teach the model something false, not to curate a clean-looking corpus.

# A backstop for alignments that are structurally broken, not a quality bar.
# Set at roughly the 99th percentile of observed costs: `s2` is fitted on document
# lengths, where sentence-level variation averages out, so the model treats freely
# translated sentences as outliers and their costs run high even when the pairing
# is correct. A tighter threshold (400 was tried) discards ~240 perfectly good
# pairs, on a corpus this small, real damage. Genuinely broken alignments still
# score far above this; the observed maximum is ~4900.
MAX_COST = 1200.0

# A translation that is less than half or more than double its source in
# characters is almost always a mis-pairing rather than a verbose rendering.
MIN_RATIO, MAX_RATIO = 0.45, 2.2

# Shorter than this is a fragment (a stray marker, a page number).
MIN_CHARS = 12

DEVANAGARI = regex.compile(r"\p{Devanagari}")


def filter_pairs(
    pairs: list[AlignedPair],
    *,
    max_cost: float = MAX_COST,
    min_chars: int = MIN_CHARS,
    ratio_bounds: tuple[float, float] = (MIN_RATIO, MAX_RATIO),
) -> tuple[list[AlignedPair], list[tuple[AlignedPair, str]]]:
    """Split aligned pairs into those safe to train on and those to reject.

    Returns:
        ``(kept, rejected)`` where each rejection carries the reason, so the
        thresholds can be judged against what they actually removed rather than
        taken on trust.
    """
    low, high = ratio_bounds
    kept: list[AlignedPair] = []
    rejected: list[tuple[AlignedPair, str]] = []

    for pair in pairs:
        if len(pair.source) < min_chars or len(pair.target) < min_chars:
            reason = "too short"
        elif not DEVANAGARI.search(pair.target):
            # The Hindi side segmented into something with no Hindi in it, # a citation line or a stray number, not a translation.
            reason = "no Devanagari on target side"
        elif not (low <= pair.length_ratio <= high):
            reason = f"length ratio {pair.length_ratio:.2f} outside [{low}, {high}]"
        elif pair.cost > max_cost:
            reason = f"alignment cost {pair.cost:.0f} > {max_cost:.0f}"
        else:
            kept.append(pair)
            continue
        rejected.append((pair, reason))

    if rejected:
        logger.info("filtered %d/%d pairs", len(rejected), len(pairs))
    return kept, rejected
