"""Translation quality and token cost on the held-out judgments.

Reports two things side by side, because the assignment asks for a before/after
on both and they can move in opposite directions: a vocabulary that halves token
count is worthless if it also halves quality.

**chrF is the primary metric, not BLEU.** The test split is 120 sentence pairs.
BLEU is computed over 4-gram matches and is noisy at that size, especially for a
morphologically rich target where a correct translation often differs from the
reference by an inflection. chrF works on character n-grams, degrades gracefully
on small samples, and is the better-behaved choice for Hindi. BLEU is reported
too, since the brief names it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Beam search rather than greedy: the difference is large for MT, and 120
# sentences is small enough that the cost is irrelevant.
NUM_BEAMS = 4
MAX_NEW_TOKENS = 256


@dataclass
class Scores:
    """Quality and cost for one model on one split."""

    name: str
    bleu: float
    chrf: float
    target_tokens: int
    source_tokens: int
    n_sentences: int

    @property
    def tokens_per_sentence(self) -> float:
        return self.target_tokens / self.n_sentences if self.n_sentences else float("nan")


@torch.no_grad()
def translate(
    model,
    tokenizer,
    sources: list[str],
    *,
    device: torch.device,
    batch_size: int = 8,
    num_beams: int = NUM_BEAMS,
    show_progress: bool = True,
) -> list[str]:
    """Translate English sentences into Hindi."""
    model.eval()
    model.to(device)

    # mBART needs the target language forced as the first generated token,
    # otherwise it decodes into whichever language it prefers.
    forced_bos = tokenizer.lang_code_to_id[tokenizer.tgt_lang]

    outputs: list[str] = []
    batches = range(0, len(sources), batch_size)
    for start in tqdm(batches, desc="translating", disable=not show_progress):
        batch = sources[start : start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=192
        ).to(device)
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos,
            num_beams=num_beams,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def score(
    name: str, hypotheses: list[str], references: list[str], sources: list[str], tokenizer
) -> Scores:
    """Compute BLEU, chrF and the token cost of the produced translations."""
    import sacrebleu

    # Hindi has no whitespace-token convention that matches BLEU's default
    # tokenizer; 'flores101' is the SentencePiece-based tokenizer sacrebleu
    # provides for non-Latin targets.
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="flores101")
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    count = lambda texts: sum(
        len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts
    )
    return Scores(
        name=name,
        bleu=bleu.score,
        chrf=chrf.score,
        target_tokens=count(references),
        source_tokens=count(sources),
        n_sentences=len(hypotheses),
    )


def compare(scores: list[Scores]) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Tabulate several runs against the first as baseline."""
    import pandas as pd

    baseline = scores[0]
    frame = pd.DataFrame(
        [
            {
                "run": s.name,
                "BLEU": s.bleu,
                "chrF": s.chrf,
                "ΔchrF": s.chrf - baseline.chrf,
                "target tokens": s.target_tokens,
                "tokens/sentence": s.tokens_per_sentence,
                "token change": s.target_tokens / baseline.target_tokens - 1,
            }
            for s in scores
        ]
    )
    return frame
