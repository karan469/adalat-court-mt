"""Extending a pretrained vocabulary with domain pieces.

The second optimisation route in Task 1, and the one that transfers to Task 3:
instead of replacing a pretrained tokenizer, keep it and buy back the Devanagari
fertility gap with a few thousand legal-domain pieces.

The surgery is done on the SentencePiece model proto rather than through
``tokenizer.add_tokens``. That distinction is the whole point:

* ``add_tokens`` registers strings in a trie that runs *before* normalisation and
  pre-tokenisation. The matched string is cut out of the text, and the space that
  preceded it becomes its own token, so on a SentencePiece model the "saving"
  can be cancelled out by a new whitespace token, and the tokenizer no longer
  round-trips cleanly.
* Injecting pieces into the proto makes them ordinary vocabulary entries,
  complete with the ``▁`` word-boundary marker, and the Viterbi segmenter weighs
  them against existing pieces exactly as it does native ones.

For a unigram model the injected score must be a real log-probability, otherwise
the segmenter never chooses the new piece and fertility does not move. Scores are
therefore estimated from the corpus, which is also why a piece's score and its
mined frequency come from the same pass.
"""

from __future__ import annotations

import logging
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import regex

from court_mt.tokenization import metrics
from court_mt.tokenization.adapters import TokenizerAdapter

logger = logging.getLogger(__name__)

# SentencePiece's word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
WORD_BOUNDARY = "▁"

# A candidate must carry at least one letter. Judgments are dense with section
# numbers and dates, and those tokenise identically in both languages, so a bare
# "2023" or "02 11 2022" is not a Hindi piece at all: it makes the English side
# cheaper too, and spends an embedding row on a string that will never recur.
HAS_LETTER = regex.compile(r"\p{L}")

# A candidate must also *begin* with a letter. The letter test alone still admits
# phrases like "11 2011 को" or "185 के", where a date or section number has picked
# up a trailing postposition, the digits are the bulk of the piece and recur no
# more often than the date does. "धारा 302" is unaffected: it opens with a word.
STARTS_WITH_LETTER = regex.compile(r"^\p{L}")

# ModelProto.trainer_spec.model_type enum values.
MODEL_TYPE_UNIGRAM, MODEL_TYPE_BPE = 1, 2

# ModelProto.SentencePiece.Type.NORMAL
PIECE_TYPE_NORMAL = 1


@dataclass(frozen=True)
class Candidate:
    """A domain string worth its own vocabulary entry."""

    surface: str
    frequency: int
    base_tokens: int
    log_prob: float

    @property
    def tokens_saved(self) -> int:
        """Tokens removed from the corpus if this piece becomes a single token."""
        return self.frequency * (self.base_tokens - 1)

    @property
    def piece(self) -> str:
        """Surface form as a word-initial SentencePiece piece.

        Internal spaces become word-boundary markers, which is how SentencePiece
        represents them, so a fixed legal collocation such as "उच्च न्यायालय"
        can occupy a single vocabulary entry.
        """
        return WORD_BOUNDARY + self.surface.replace(" ", WORD_BOUNDARY)

    @property
    def n_words(self) -> int:
        return self.surface.count(" ") + 1


def mine_candidates(
    texts: list[str],
    adapter: TokenizerAdapter,
    *,
    min_frequency: int = 2,
    min_base_tokens: int = 2,
    min_length: int = 3,
    max_phrase_words: int = 3,
    require_script: str | None = None,
    limit: int | None = None,
) -> list[Candidate]:
    """Rank corpus n-grams by how many tokens a dedicated piece would save.

    Ranking on ``frequency * (base_tokens - 1)`` rather than raw frequency is what
    makes a small budget pay: a word already encoded as one token saves nothing
    however often it occurs, while a five-token legal compound appearing 20 times
    saves 80.

    Phrases up to ``max_phrase_words`` long are mined alongside single words.
    Legal Hindi is full of fixed collocations, ``उच्च न्यायालय``, ``दण्ड
    संहिता``, that always co-occur, and SentencePiece can hold a whole
    collocation in one piece by encoding the internal space as a boundary marker.
    On a corpus this small that is also what keeps the candidate pool from
    running dry after a few hundred entries.

    Two filters keep the vocabulary honest. Candidates must contain a letter, so
    dates and bare section numbers never become pieces, they recur too rarely to
    earn an embedding row, and being script-neutral they would silently cheapen
    the *other* language too. ``require_script`` (a Unicode script name such as
    ``"Devanagari"``) additionally pins mining to one script, so a target-side
    extension provably cannot move the source side.
    """
    script_filter = regex.compile(rf"\p{{{require_script}}}") if require_script else None
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = metrics.words(text)
        for width in range(1, max_phrase_words + 1):
            counts.update(" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1))

    total = sum(counts.values())
    if not total:
        return []

    candidates = [
        Candidate(
            surface=surface,
            frequency=frequency,
            base_tokens=n_tokens,
            log_prob=math.log(frequency / total),
        )
        for surface, frequency in counts.items()
        if frequency >= min_frequency
        and len(surface) >= min_length
        and HAS_LETTER.search(surface)
        and STARTS_WITH_LETTER.match(surface)
        and (script_filter is None or script_filter.search(surface))
        and (n_tokens := len(adapter.encode(" " + surface))) >= min_base_tokens
    ]

    candidates.sort(key=lambda c: (-c.tokens_saved, -c.frequency, c.surface))
    logger.info(
        "mined %d candidates from %d distinct n-grams (%d tokens saveable)",
        len(candidates),
        len(counts),
        sum(c.tokens_saved for c in candidates),
    )
    return candidates[:limit] if limit else candidates


def base_model_path(adapter: TokenizerAdapter) -> Path:
    """Locate the ``.model`` file backing a SentencePiece-based HF tokenizer."""
    tok = adapter._tok  # noqa: SLF001 - adapters intentionally expose no setter
    for attribute in ("vocab_file", "sp_model_file"):
        value = getattr(tok, attribute, None)
        if value and Path(value).is_file():
            return Path(value)
    raise FileNotFoundError(
        f"{adapter.key}: no SentencePiece .model found; "
        "proto extension only applies to SentencePiece tokenizers"
    )


def extend_sentencepiece(
    base_model: Path,
    candidates: list[Candidate],
    output_path: Path,
    *,
    score_floor_margin: float = 1.0,
) -> tuple[Path, int]:
    """Inject ``candidates`` into a copy of ``base_model``.

    Args:
        base_model: the pretrained ``.model`` to extend (never modified).
        candidates: pieces to add, already ranked; duplicates are skipped.
        output_path: where the extended model is written.
        score_floor_margin: for BPE models, how far above the lowest existing
            merge score to place new pieces so the segmenter can reach them.

    Returns:
        ``(output_path, n_added)``.
    """
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2

    proto = sp_pb2.ModelProto()
    proto.ParseFromString(base_model.read_bytes())

    model_type = proto.trainer_spec.model_type
    existing = {piece.piece for piece in proto.pieces}
    scores = [piece.score for piece in proto.pieces if piece.type == PIECE_TYPE_NORMAL]
    min_score, max_score = (min(scores), max(scores)) if scores else (-1e4, 0.0)

    added = 0
    for candidate in candidates:
        if candidate.piece in existing:
            continue
        entry = proto.pieces.add()
        entry.piece = candidate.piece
        entry.type = PIECE_TYPE_NORMAL
        entry.score = _piece_score(candidate, model_type, min_score, max_score, score_floor_margin)
        existing.add(candidate.piece)
        added += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(proto.SerializeToString())
    logger.info(
        "extended %s (+%d pieces, %d -> %d) -> %s",
        base_model.name,
        added,
        len(proto.pieces) - added,
        len(proto.pieces),
        output_path,
    )
    return output_path, added


def _piece_score(
    candidate: Candidate, model_type: int, min_score: float, max_score: float, margin: float
) -> float:
    """Score a new piece on the scale its base model actually uses."""
    if model_type == MODEL_TYPE_UNIGRAM:
        # Unigram scores are log-probabilities: the corpus estimate is directly
        # comparable to the pretrained ones, clamped so a rare piece cannot fall
        # below the model's own floor and become unreachable.
        return max(candidate.log_prob, min_score + margin)
    if model_type == MODEL_TYPE_BPE:
        # BPE scores are negated merge ranks. Placing new pieces just above the
        # floor makes them available as a last-resort merge without reordering
        # the pretrained merge table.
        logger.debug("BPE base model: new pieces get low merge priority")
        return min_score + margin
    logger.warning("unrecognised SentencePiece model_type=%s; using log-prob score", model_type)
    return max(candidate.log_prob, min_score + margin)


def copy_tokenizer_with_model(
    adapter: TokenizerAdapter, extended_model: Path, output_dir: Path
) -> Path:
    """Materialise a full HF tokenizer directory around an extended ``.model``.

    Task 3 needs a loadable tokenizer, not just a SentencePiece file: the saved
    directory carries the base tokenizer's configuration (language codes, special
    tokens) with the extended vocabulary swapped in.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter._tok.save_pretrained(output_dir)  # noqa: SLF001
    for target in output_dir.rglob("*.model"):
        shutil.copyfile(extended_model, target)
    # The fast tokenizer's serialised vocabulary would shadow the .model file.
    for stale in output_dir.glob("tokenizer.json"):
        stale.unlink()
    logger.info("wrote extended tokenizer directory %s", output_dir)
    return output_dir


def embedding_cost(n_added: int, hidden_size: int, *, tied_embeddings: bool = True) -> int:
    """Parameters added by resizing the embedding table for ``n_added`` tokens.

    With tied input/output embeddings one matrix grows; untied models pay twice.
    This is the number that makes the trade-off concrete, the fertility win has
    to be worth the extra parameters, which start out randomly initialised and
    need training data to become useful.
    """
    return n_added * hidden_size * (1 if tied_embeddings else 2)


def hidden_size_of(model_repo: str) -> tuple[int, bool]:
    """Read ``(hidden_size, tied_embeddings)`` from a model's config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_repo)
    hidden = getattr(config, "d_model", None) or getattr(config, "hidden_size", None)
    if hidden is None:
        raise AttributeError(f"{model_repo}: config exposes neither d_model nor hidden_size")
    return int(hidden), bool(getattr(config, "tie_word_embeddings", True))
