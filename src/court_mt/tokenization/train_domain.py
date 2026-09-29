"""Training SentencePiece vocabularies on the judgment corpus.

This is the first of the two optimisation routes in Task 1. A vocabulary learned
on 30 judgments cannot compete with a pretrained multilingual one on coverage,
but it establishes the *floor*: how few tokens this domain needs when the
vocabulary is spent entirely on legal Hindi and English. Every pretrained
tokenizer's fertility is then readable as a gap against that floor.

Settings that matter for Devanagari, and why:

``character_coverage=1.0``
    The default 0.9995 drops the rare-character tail. In legal Hindi that tail is
    nukta forms, Vedic signs and stray Latin in citations, exactly the material
    that would become ``<unk>``.

``byte_fallback=True``
    Guarantees losslessness: anything outside the vocabulary decomposes to UTF-8
    bytes instead of ``<unk>``, so fertility gains can never be bought by
    silently discarding characters.

``split_digits=True``
    Judgments are dense with dates, section numbers and case numbers. Splitting
    digits individually keeps the vocabulary from memorising "2003" and friends,
    which generalise poorly.

``hard_vocab_limit=False``
    25 judgments simply do not contain enough distinct material to build a large
    vocabulary: asking for 8k pieces makes the trainer abort outright. With the
    limit soft it returns the largest vocabulary the corpus supports and reports
    it, which is the number worth knowing anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Reserved ids kept compatible with the seq2seq models used in Task 3.
CONTROL_SYMBOLS = ("<pad>", "<mask>")


@dataclass(frozen=True)
class DomainVocabConfig:
    """One SentencePiece training run."""

    name: str
    vocab_size: int
    model_type: str = "unigram"
    character_coverage: float = 1.0
    split_digits: bool = True
    byte_fallback: bool = True

    @property
    def spec(self) -> str:
        """Roster entry pointing at the trained model, for the benchmark config."""
        return f"sp:artifacts/spm/{self.name}.model"


def sentence_stream(texts: Iterable[str]) -> Iterator[str]:
    """Yield non-empty lines, which is the granularity SentencePiece trains on."""
    for text in texts:
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped:
                yield stripped


def train(
    texts: Iterable[str], config: DomainVocabConfig, output_dir: Path, *, overwrite: bool = False
) -> Path:
    """Train one vocabulary and return the path to the ``.model`` file."""
    import sentencepiece as spm

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{config.name}.model"
    if model_path.exists() and not overwrite:
        logger.info("reusing existing %s", model_path)
        return model_path

    sentences = list(sentence_stream(texts))
    longest = max((len(s.encode("utf-8")) for s in sentences), default=0)
    max_sentence_length = max(4192, longest + 1024)
    logger.info(
        "training %s: %s, vocab=%d, %d sentences (longest %d bytes)",
        config.name,
        config.model_type,
        config.vocab_size,
        len(sentences),
        longest,
    )

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        # Accept a smaller vocabulary rather than aborting when the corpus
        # cannot supply vocab_size distinct pieces.
        hard_vocab_limit=False,
        # A reflowed judgment paragraph can exceed the 4192-byte default, and the
        # trainer *silently skips* any line over the limit. On a corpus this small
        # that is real training data lost, so the cap is raised past the longest
        # paragraph present.
        max_sentence_length=max_sentence_length,
        minloglevel=1,
        model_prefix=str(output_dir / config.name),
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        split_digits=config.split_digits,
        byte_fallback=config.byte_fallback,
        # NFKC + MT-oriented whitespace handling, matching mBART/mT5 preprocessing.
        normalization_rule_name="nmt_nfkc",
        control_symbols=list(CONTROL_SYMBOLS),
        unk_id=0,
        bos_id=1,
        eos_id=2,
        # A 30-judgment corpus is small; let the trainer see all of it at once.
        input_sentence_size=0,
        train_extremely_large_corpus=False,
        num_threads=4,
    )
    import sentencepiece as spm_runtime

    achieved = spm_runtime.SentencePieceProcessor(model_file=str(model_path)).get_piece_size()
    if achieved < config.vocab_size:
        logger.warning(
            "%s: corpus supports only %d pieces of the %d requested",
            config.name,
            achieved,
            config.vocab_size,
        )
    logger.info("wrote %s (%d pieces)", model_path, achieved)
    return model_path


def achieved_vocab_size(model_path: Path) -> int:
    """Actual piece count of a trained model, which may be below the request."""
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(model_path)).get_piece_size()
