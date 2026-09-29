"""Token-efficiency measurements.

Everything here is a pure function of (tokenizer, text) so the numbers in the
report can be regenerated from the corpus alone.

The metrics answer three separate questions the assignment asks:

*How verbose is this vocabulary?*
    ``fertility`` (tokens per word) and ``chars_per_token``. Fertility is the
    headline number, but it is only comparable within a language, Hindi words
    carry more morphology per token than English ones, so cross-language claims
    use ``parity`` instead.

*How much of that is an Indic-script penalty?*
    ``parity`` = Hindi tokens / English tokens over the *same* document. A
    vocabulary with no Devanagari coverage falls back to UTF-8 bytes, where one
    character costs three tokens, and parity blows past 3x. This is the number
    that translates directly into context and price.

*Is the vocabulary even lossless here?*
    ``unk_rate`` and ``roundtrip_ok``. SentencePiece and WordPiece models
    silently map unseen characters to ``<unk>``; a low fertility bought by
    dropping characters is not a saving.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass

import regex

from court_mt.tokenization.adapters import TokenizerAdapter

# A word is a run of letters/marks/digits, optionally hyphenated or apostrophed.
# \p{M} is essential for Devanagari: without it every matra would end a word.
WORD_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+(?:['’\-][\p{L}\p{M}\p{N}]+)*")
DEVANAGARI_RE = regex.compile(r"[\p{Devanagari}]")

# Rényi entropy order from Zouhar et al. (2023), "Tokenization and the Noiseless
# Channel" - alpha=2.5 correlates best with downstream MT quality.
RENYI_ALPHA = 2.5


def words(text: str) -> list[str]:
    """Split ``text`` into script-aware word units."""
    return WORD_RE.findall(text)


def count_devanagari(text: str) -> int:
    """Number of Devanagari codepoints in ``text``."""
    return len(DEVANAGARI_RE.findall(text))


@dataclass(frozen=True)
class TextStats:
    """Raw counts for one text under one tokenizer. Ratios are derived, not stored."""

    n_chars: int
    n_bytes: int
    n_words: int
    n_tokens: int
    n_unk: int
    n_whole_words: int
    n_devanagari: int
    roundtrip_ok: bool
    spacing_ok: bool

    @property
    def fertility(self) -> float:
        """Tokens per word, the standard verbosity measure."""
        return self.n_tokens / self.n_words if self.n_words else math.nan

    @property
    def chars_per_token(self) -> float:
        """Characters carried per token; higher is denser."""
        return self.n_chars / self.n_tokens if self.n_tokens else math.nan

    @property
    def bytes_per_token(self) -> float:
        """UTF-8 bytes per token, compression rate, script-neutral."""
        return self.n_bytes / self.n_tokens if self.n_tokens else math.nan

    @property
    def unk_rate(self) -> float:
        return self.n_unk / self.n_tokens if self.n_tokens else math.nan

    @property
    def whole_word_rate(self) -> float:
        """Share of words encoded as exactly one token (no fragmentation)."""
        return self.n_whole_words / self.n_words if self.n_words else math.nan

    def __add__(self, other: TextStats) -> TextStats:
        """Pool two texts, so corpus totals are ratio-of-sums not mean-of-ratios."""
        if not isinstance(other, TextStats):
            return NotImplemented
        return TextStats(
            n_chars=self.n_chars + other.n_chars,
            n_bytes=self.n_bytes + other.n_bytes,
            n_words=self.n_words + other.n_words,
            n_tokens=self.n_tokens + other.n_tokens,
            n_unk=self.n_unk + other.n_unk,
            n_whole_words=self.n_whole_words + other.n_whole_words,
            n_devanagari=self.n_devanagari + other.n_devanagari,
            roundtrip_ok=self.roundtrip_ok and other.roundtrip_ok,
            spacing_ok=self.spacing_ok and other.spacing_ok,
        )


def measure(adapter: TokenizerAdapter, text: str) -> TextStats:
    """Tokenize ``text`` once and collect every count derived from it."""
    ids = adapter.encode(text)
    word_list = words(text)
    unk_id = adapter.unk_id

    return TextStats(
        n_chars=len(text),
        n_bytes=len(text.encode("utf-8")),
        n_words=len(word_list),
        n_tokens=len(ids),
        n_unk=sum(1 for i in ids if i == unk_id) if unk_id is not None else 0,
        n_whole_words=_count_whole_words(adapter, word_list),
        n_devanagari=count_devanagari(text),
        roundtrip_ok=_roundtrips(adapter, text, ids),
        spacing_ok=whitespace_preserved(adapter, text, ids),
    )


def _count_whole_words(adapter: TokenizerAdapter, word_list: list[str]) -> int:
    """Token occurrences of words that survive as a single token.

    Each word is encoded with a leading space, so SentencePiece and byte-level
    BPE see the word-initial form they were trained on, the form that actually
    appears in running text.

    Encoding is done once per *distinct* word and weighted by frequency. On a
    judgment that is a ~10x saving, which matters because the slow SentencePiece
    tokenizers are on the critical path here.
    """
    counts = Counter(word_list)
    return sum(n for word, n in counts.items() if len(adapter.encode(" " + word)) == 1)


def _roundtrips(adapter: TokenizerAdapter, text: str, ids: list[int]) -> bool:
    """True if decoding loses no characters.

    Whitespace is removed entirely before comparing, not merely collapsed. The
    question this metric exists to answer is whether the vocabulary can represent
    the script, so a WordPiece model that re-spaces punctuation (``बैंक।`` ->
    ``बैंक ।``) must not be scored the same as one that drops a character. Space
    placement is a detokenisation concern and is reported separately by
    :func:`whitespace_preserved`.
    """
    return _strip_all_space(adapter.decode(ids)) == _strip_all_space(text)


def whitespace_preserved(adapter: TokenizerAdapter, text: str, ids: list[int]) -> bool:
    """True if decoding also restores the original spacing exactly."""
    return _canonical(adapter.decode(ids)) == _canonical(text)


def _canonical(text: str) -> str:
    """NFC form with whitespace runs collapsed to single spaces."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def _strip_all_space(text: str) -> str:
    """NFC form with every whitespace character removed."""
    return "".join(unicodedata.normalize("NFC", text).split())


def parity(hindi: TextStats, english: TextStats) -> float:
    """Hindi-to-English token ratio for the same content.

    1.0 means the vocabulary spends the same budget on both languages; 3.0+
    means Devanagari has decayed to byte-level fallback.
    """
    return hindi.n_tokens / english.n_tokens if english.n_tokens else math.nan


def renyi_efficiency(
    token_counts: Counter[int], vocab_size: int, alpha: float = RENYI_ALPHA
) -> float:
    """Normalised Rényi entropy of the corpus token distribution, in [0, 1].

    Penalises both a vocabulary dominated by a few frequent tokens and one whose
    mass is spread over near-unused entries; higher predicts better MT quality
    at equal fertility.
    """
    total = sum(token_counts.values())
    if total == 0 or vocab_size <= 1:
        return math.nan
    probs = [c / total for c in token_counts.values()]
    if math.isclose(alpha, 1.0):
        entropy = -sum(p * math.log(p) for p in probs)
    else:
        entropy = math.log(sum(p**alpha for p in probs)) / (1.0 - alpha)
    return entropy / math.log(vocab_size)


def vocab_utilization(token_counts: Counter[int], vocab_size: int) -> float:
    """Share of the vocabulary this corpus ever touches."""
    return len(token_counts) / vocab_size if vocab_size else math.nan
