"""Corpus quality gates.

Five of the thirty Hindi files in this dataset contain no Devanagari at all: the
script was destroyed at encoding time, leaving each character as a literal ``?``
(``1.???????????. 1181(??/??)???? 1998...``). The English side of those pairs is
intact, and their file sizes and hi/en length ratios look entirely normal, so a
length-based alignment check passes them, and they flow silently into the
benchmark.

They must not. A run of ``?`` is cheap and uniform to tokenise in every
vocabulary, so those documents compress the differences between tokenizers and
drag every Hindi fertility and parity number toward each other. The corruption
has to be detected on script content, not on length.

The source PDFs do still contain Devanagari, but extracting them is not a drop-in
recovery: their font CMaps are broken, so extraction yields text peppered with
control characters (``प्रति\\x03वेद्य``) and would be markedly worse than the 25
clean files. Quarantining is the honest choice, and the quarantine is written out
so the decision is auditable rather than invisible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from court_mt.corpus.loader import Judgment
from court_mt.tokenization.metrics import count_devanagari

logger = logging.getLogger(__name__)

# A genuine Hindi judgment in this corpus runs 60-77% Devanagari by character;
# the rest is digits, Latin case citations and punctuation. 30% is far below any
# real document and far above the 0% the corrupted files show.
MIN_DEVANAGARI_SHARE = 0.30

# Mojibake substitution marks. A handful is normal punctuation; over 20% of the
# document means the encoding failed.
REPLACEMENT_CHARS = "?�"
MAX_REPLACEMENT_SHARE = 0.20


@dataclass(frozen=True)
class QualityReport:
    """Verdict on one judgment pair."""

    doc_id: str
    hi_chars: int
    devanagari_share: float
    replacement_share: float
    usable: bool
    reason: str


def assess(
    judgment: Judgment,
    *,
    min_devanagari_share: float = MIN_DEVANAGARI_SHARE,
    max_replacement_share: float = MAX_REPLACEMENT_SHARE,
) -> QualityReport:
    """Check whether a pair's Hindi side is really Hindi."""
    hindi = judgment.hi
    n_chars = len(hindi)
    if n_chars == 0:
        return QualityReport(judgment.doc_id, 0, 0.0, 0.0, False, "empty hindi side")

    devanagari_share = count_devanagari(hindi) / n_chars
    replacement_share = sum(hindi.count(c) for c in REPLACEMENT_CHARS) / n_chars

    if devanagari_share < min_devanagari_share:
        reason = f"only {devanagari_share:.1%} Devanagari (expected >{min_devanagari_share:.0%})"
    elif replacement_share > max_replacement_share:
        reason = f"{replacement_share:.1%} replacement characters, encoding lost"
    else:
        reason = "ok"

    return QualityReport(
        doc_id=judgment.doc_id,
        hi_chars=n_chars,
        devanagari_share=devanagari_share,
        replacement_share=replacement_share,
        usable=reason == "ok",
        reason=reason,
    )


def partition(
    judgments: list[Judgment], **thresholds: float
) -> tuple[list[Judgment], list[Judgment], list[QualityReport]]:
    """Split judgments into usable and quarantined, with a report for every one.

    Returns:
        ``(usable, quarantined, reports)``, reports cover all inputs, in order.
    """
    reports = [assess(j, **thresholds) for j in judgments]  # type: ignore[arg-type]
    verdicts = {r.doc_id: r.usable for r in reports}

    usable = [j for j in judgments if verdicts[j.doc_id]]
    quarantined = [j for j in judgments if not verdicts[j.doc_id]]

    if quarantined:
        logger.warning(
            "quarantined %d/%d pair(s) with an unusable Hindi side: %s",
            len(quarantined),
            len(judgments),
            ", ".join(f"{r.doc_id} ({r.reason})" for r in reports if not r.usable),
        )
    return usable, quarantined, reports
