"""Deterministic train/dev/test splits over whole judgments.

Splitting is at **document** level, never sentence level. Judgments repeat their
own terminology heavily, a party's name, the statute under construction, the
phrasing of the charge, so sentences from one judgment landing on both sides of
a split would leak exactly the domain vocabulary this project is trying to learn,
and every fertility gain would be measured against text the tokenizer had already
seen.

The ratio follows the brief's 80/10/10. The seed is fixed so the split is
reproducible across runs and machines.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

from court_mt.corpus.loader import Judgment

logger = logging.getLogger(__name__)

DEFAULT_SEED = 20260928
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class Split:
    """A train/dev/test partition of the corpus."""

    train: list[Judgment]
    dev: list[Judgment]
    test: list[Judgment]

    @property
    def held_out(self) -> list[Judgment]:
        """Dev and test together, everything the tokenizer must not have seen."""
        return self.dev + self.test

    def summary(self) -> str:
        return (
            f"train {len(self.train)} ({_ids(self.train)}), "
            f"dev {len(self.dev)} ({_ids(self.dev)}), "
            f"test {len(self.test)} ({_ids(self.test)})"
        )


def _ids(judgments: list[Judgment]) -> str:
    return ",".join(j.doc_id for j in judgments)


def split_corpus(
    judgments: list[Judgment],
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> Split:
    """Partition judgments into train/dev/test.

    Dev and test are guaranteed at least one document each whenever the corpus
    has three or more, so a held-out measurement is always possible, rounding
    on 25 documents would otherwise empty them.
    """
    if not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    shuffled = list(judgments)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    # Round up, not to nearest: round() is banker's rounding, so 25 * 0.1 = 2.5
    # becomes 2 and the held-out sets end up at ~6% each instead of ~10%.
    n_dev = max(1, math.ceil(total * ratios[1])) if total >= 3 else 0
    n_test = max(1, math.ceil(total * ratios[2])) if total >= 3 else 0
    n_train = total - n_dev - n_test
    if n_train < 1:
        raise ValueError(f"corpus of {total} is too small to split {ratios}")

    split = Split(
        train=shuffled[:n_train],
        dev=shuffled[n_train : n_train + n_dev],
        test=shuffled[n_train + n_dev :],
    )
    logger.info("split %d judgments: %s", total, split.summary())
    return split
