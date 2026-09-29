"""Light normalisation of the OCR'd judgment text.

The cleaning here is deliberately conservative: the tokenizer study measures
cost on text a real translation pipeline would send to a model, so the goal is
to undo *transport* artefacts (hard wrapping, split hyphens, mojibake-ish
punctuation) without rewriting the legal language itself.

Every transform is a standalone function so the benchmark can report the effect
of each one, and :func:`clean` composes them in the order they must run.
"""

from __future__ import annotations

import re
import unicodedata

# A block opener: "1.", "12)", "(iv)", "i.", "(a)", "A.", or a Devanagari digit.
LIST_MARKER = re.compile(
    r"""^\s*(?:
          \(?\d{1,3}[.)]            # 1.  12)  (3)
        | \(?[ivxlcdm]{1,6}[.)]     # i.  (iv)
        | \(?[a-zA-Z][.)]           # a.  (B)
        | \(?[०-९]{1,3}[.)]  # Devanagari digits
    )\s+""",
    re.VERBOSE,
)

SENTENCE_FINAL = re.compile(r"[.?!:;।॥][\"'’”)\]]*$")
STARTS_BLOCK = re.compile(r"^[A-Z(\[ऀ-ॿ\"“]")

# OCR reads Roman-numeral list markers through a Latin lens: ii -> li, iii -> lili.
OCR_ROMAN_MARKER = re.compile(r"^(\s*)(l+i+l*i*)([.)]\s+)", re.IGNORECASE)

# Keys are written as escapes rather than literal glyphs. Several of these
# characters are invisible or easily confused in an editor, so a stray edit to a
# literal key silently rewrites the corpus instead of raising.
SMART_PUNCT = {
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201a": "'",  # single low-9 quote
    "\u201b": "'",  # single high-reversed-9 quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u201e": '"',  # double low-9 quote
    "\u201f": '"',  # double high-reversed-9 quote
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign
    "\u00a0": " ",  # no-break space
    "\u2009": " ",  # thin space
    "\u200a": " ",  # hair space
    "\u202f": " ",  # narrow no-break space
    "\u2026": "...",  # ellipsis
}
_SMART_RE = re.compile("|".join(map(re.escape, SMART_PUNCT)))

# Zero-width and bidi controls survive OCR and cost real tokens downstream.
INVISIBLES = re.compile(r"[​‌‍‎‏⁠﻿]")

DEHYPHEN = re.compile(r"(\w)-\s*$")


def normalize_unicode(text: str) -> str:
    """NFC-normalise and strip invisible control characters.

    NFC matters for Devanagari: decomposed nukta/matra sequences tokenise
    differently from their composed form on byte-level BPE vocabularies.
    """
    return INVISIBLES.sub("", unicodedata.normalize("NFC", text))


def normalize_punctuation(text: str) -> str:
    """Fold typographic quotes, dashes and exotic spaces onto ASCII equivalents."""
    return _SMART_RE.sub(lambda m: SMART_PUNCT[m.group(0)], text)


def fix_ocr_roman_markers(text: str) -> str:
    """Repair ``li.``-style list markers misread from ``ii.``. Opt-in, see caveat.

    Off by default, because the glyph mapping is not one-to-one: OCR renders
    ``ii.`` as ``li.`` (two characters) but ``iii.`` as ``lili.`` (four), so
    counting i/l glyphs recovers the first and gets the second wrong. Repairing
    these reliably needs positional renumbering over an already-reflowed list,
    which is more machinery than a cosmetic marker warrants, the markers are a
    handful of tokens and affect no measurement. Left as a documented OCR
    artefact instead.
    """

    def repair(match: re.Match[str]) -> str:
        indent, marker, tail = match.groups()
        count = marker.lower().count("i") + marker.lower().count("l")
        return f"{indent}{'i' * count}{tail}" if 2 <= count <= 6 else match.group(0)

    return "\n".join(OCR_ROMAN_MARKER.sub(repair, line) for line in text.split("\n"))


def unwrap_paragraphs(text: str) -> str:
    """Reflow hard-wrapped lines back into logical paragraphs.

    The OCR export wraps at ~55 characters and sprinkles blank lines *inside*
    list items, so a blank line alone is not a paragraph boundary. A new
    paragraph starts only at an explicit list marker, or after a blank line when
    the preceding text actually ended a sentence.
    """
    paragraphs: list[list[str]] = []
    previous_blank = False

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            previous_blank = True
            continue

        current = paragraphs[-1] if paragraphs else None
        ends_sentence = bool(current and SENTENCE_FINAL.search(current[-1]))
        starts_new = (
            current is None
            or LIST_MARKER.match(line)
            or (previous_blank and ends_sentence and STARTS_BLOCK.match(line))
        )
        previous_blank = False

        if starts_new:
            paragraphs.append([line])
            continue

        assert current is not None  # starts_new covers the empty case
        joined = DEHYPHEN.sub(r"\1", current[-1]) if DEHYPHEN.search(current[-1]) else None
        if joined is not None:
            current[-1] = joined + line
        else:
            current.append(line)

    return "\n\n".join(" ".join(p) for p in paragraphs)


def collapse_spaces(text: str) -> str:
    """Squeeze runs of spaces/tabs and tidy space before punctuation."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:?!।॥)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean(text: str, *, fix_ocr_markers: bool = False, unwrap: bool = True) -> str:
    """Apply the full cleaning pipeline in dependency order."""
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    if fix_ocr_markers:
        text = fix_ocr_roman_markers(text)
    if unwrap:
        text = unwrap_paragraphs(text)
    return collapse_spaces(text)
