"""A single interface over the tokenizer families we compare.

Tokenizers reach us in three incompatible shapes, HuggingFace ``AutoTokenizer``,
OpenAI ``tiktoken`` encodings, and raw SentencePiece models (including the ones
we train ourselves). The metrics layer must not care which, so each family gets
a thin adapter exposing the same four operations: ``encode``, ``decode``,
``pieces`` and ``vocab_size``.

Two decisions keep the comparison fair across families:

* **No special tokens.** Every adapter encodes content only. BOS/EOS/language
  tags are a fixed per-sequence constant that differs by model, so they are
  reported separately rather than smuggled into fertility.
* **Lossless-decode checks.** Byte-level BPE always round-trips; SentencePiece
  and WordPiece can silently drop characters outside their vocabulary. The
  ``decode`` path is what :mod:`court_mt.tokenization.metrics` uses to catch it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Prefix -> family, used both to dispatch construction and to group the report.
SCHEMES = {"hf": "huggingface", "tiktoken": "tiktoken", "sp": "sentencepiece"}


@lru_cache(maxsize=1)
def _bytelevel_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode table used by byte-level BPE.

    Byte-level BPE vocabularies store each byte as a printable stand-in
    character, so ``convert_ids_to_tokens`` returns mojibake like ``à¤ħà¤ª``
    where the text is ``अप``. This table maps those stand-ins back to the bytes
    they represent, which is the only way to show what a byte-level token
    actually covers.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    mapping = {chr(b): b for b in printable}
    overflow = 0
    for byte in range(256):
        if byte not in printable:
            mapping[chr(256 + overflow)] = byte
            overflow += 1
    return mapping


def render_bytelevel(pieces: list[str]) -> list[str]:
    """Render byte-level BPE tokens as the text they encode.

    Tokens are decoded as a group so a multi-byte character split across two
    tokens still resolves; each token then shows its own slice, with ``\ufffd``
    marking a fragment that is not a whole character on its own.
    """
    table = _bytelevel_decoder()
    if not all(character in table for piece in pieces for character in piece):
        return pieces

    rendered = []
    for piece in pieces:
        raw = bytes(table[character] for character in piece)
        rendered.append(raw.decode("utf-8", "replace"))
    return rendered


@dataclass(frozen=True)
class TokenizerSpec:
    """Declarative description of one tokenizer under test."""

    key: str
    spec: str
    label: str = ""
    family: str = ""
    note: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def scheme(self) -> str:
        scheme = self.spec.split(":", 1)[0]
        if scheme not in SCHEMES:
            raise ValueError(f"{self.spec!r}: scheme must be one of {sorted(SCHEMES)}")
        return scheme

    @property
    def target(self) -> str:
        """The part after the scheme: a HF repo id, tiktoken encoding, or path."""
        return self.spec.split(":", 1)[1]


class TokenizerAdapter(ABC):
    """Uniform, content-only view of a tokenizer."""

    def __init__(self, spec: TokenizerSpec) -> None:
        self.spec = spec

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def label(self) -> str:
        return self.spec.label or self.spec.key

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Token ids for ``text``, excluding special tokens."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Inverse of :meth:`encode`, excluding special tokens."""

    @abstractmethod
    def pieces(self, text: str) -> list[str]:
        """Human-readable surface form of each token, for fragmentation analysis."""

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    def unk_id(self) -> int | None:
        """Id of the unknown token, or ``None`` for vocabularies without one."""
        return None

    @property
    def n_special_per_sequence(self) -> int:
        """Fixed token overhead a real call adds around the content."""
        return 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.spec.key!r}, vocab={self.vocab_size})"


class HuggingFaceAdapter(TokenizerAdapter):
    """Any tokenizer loadable through ``transformers.AutoTokenizer``."""

    @cached_property
    def _tok(self):
        from transformers import AutoTokenizer

        # Some Indic SentencePiece models (IndicBART, indic-bert) ship no fast
        # converter, so the roster can force the slow path per entry.
        options: dict[str, object] = {
            "use_fast": True,
            "trust_remote_code": False,
            **dict(self.spec.extra.get("kwargs", {})),  # type: ignore[arg-type]
        }
        return AutoTokenizer.from_pretrained(self.spec.target, **options)

    def encode(self, text: str) -> list[int]:
        return self._tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def pieces(self, text: str) -> list[str]:
        # Byte-level BPE vocabularies (BLOOM, GPT-2 lineage) store tokens in an
        # escaped byte alphabet; SentencePiece and WordPiece are already legible
        # and render_bytelevel leaves them untouched.
        return render_bytelevel(self._tok.convert_ids_to_tokens(self.encode(text)))

    @property
    def vocab_size(self) -> int:
        # len() includes tokens added after pretraining; vocab_size does not.
        return max(len(self._tok), int(getattr(self._tok, "vocab_size", 0)))

    @property
    def unk_id(self) -> int | None:
        return self._tok.unk_token_id

    @property
    def n_special_per_sequence(self) -> int:
        return int(self._tok.num_special_tokens_to_add(pair=False))


class TiktokenAdapter(TokenizerAdapter):
    """OpenAI byte-level BPE encodings (``cl100k_base``, ``o200k_base``, ...)."""

    @cached_property
    def _enc(self):
        import tiktoken

        return tiktoken.get_encoding(self.spec.target)

    def encode(self, text: str) -> list[int]:
        # encode_ordinary skips special-token parsing, so "<|endoftext|>" in the
        # source text is treated as literal characters instead of raising.
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def pieces(self, text: str) -> list[str]:
        # A single token is an arbitrary byte string: a Devanagari character
        # spans up to three tokens, none of them valid UTF-8 on its own.
        return [
            self._enc.decode_single_token_bytes(i).decode("utf-8", "replace")
            for i in self.encode(text)
        ]

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab


class SentencePieceAdapter(TokenizerAdapter):
    """A raw ``.model`` file, including the domain vocabularies we train."""

    @cached_property
    def _sp(self):
        import sentencepiece as spm

        path = Path(self.spec.target).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SentencePiece model not found: {path}")
        return spm.SentencePieceProcessor(model_file=str(path))

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)

    def pieces(self, text: str) -> list[str]:
        return self._sp.encode(text, out_type=str)

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()

    @property
    def unk_id(self) -> int | None:
        return self._sp.unk_id()


_ADAPTERS: dict[str, type[TokenizerAdapter]] = {
    "hf": HuggingFaceAdapter,
    "tiktoken": TiktokenAdapter,
    "sp": SentencePieceAdapter,
}


def build(spec: TokenizerSpec) -> TokenizerAdapter:
    """Instantiate the adapter for ``spec`` (model weights load lazily)."""
    return _ADAPTERS[spec.scheme](spec)
