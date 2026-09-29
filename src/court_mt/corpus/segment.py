"""Sentence segmentation for bilingual legal text.

Off-the-shelf splitters break on judgments, because a judgment is mostly periods
that do not end sentences: ``27.05.2003``, ``Section 302``, ``AIR 1963 S.C. 1295``,
``Mr. P.S. Patwalia``, ``M/s. Tata Ltd. v. State of U.P.``, and list markers like
``(iv)``. Splitting naively on ``.`` shatters a judgment into fragments, and every
fragment becomes a bad training pair downstream.

The approach is protect-then-split: mask the periods that are known not to end a
sentence, split on what remains, then unmask. Masking rather than
pattern-matching the boundary keeps the surface text byte-identical, the
assignment asks for legal formatting to be preserved, so a segmenter that
normalises citations while cutting them is already the wrong tool.

Hindi is segmented on the danda (``।``) primarily, but judgments in this corpus
mix danda and full stop freely, so both are boundaries on the Hindi side.
"""

from __future__ import annotations

import re

# A period is masked when it belongs to one of these, never when it ends a clause.
ABBREVIATIONS = (
    # Honorifics and parties
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Shri",
    "Smt",
    "Sri",
    "Hon'ble",
    "Hon",
    "M/s",
    # Case-citation furniture
    "v",
    "vs",
    "Ors",
    "Anr",
    "Anor",
    "Co",
    "Ltd",
    "Pvt",
    "Inc",
    # Courts and reporters
    "SC",
    "HC",
    "S.C",
    "A.I.R",
    "AIR",
    "SCC",
    "SCR",
    "INSC",
    "Cr",
    "Crl",
    "LJ",
    # Structural references
    "No",
    "Nos",
    "Art",
    "Arts",
    "Sec",
    "Secs",
    "Sc",
    "Cl",
    "Ch",
    "Sch",
    "para",
    "paras",
    "pp",
    "Vol",
    "Ed",
    "r",
    "rr",
    "s",
    "ss",
    # Latin
    "i.e",
    "e.g",
    "viz",
    "etc",
    "ibid",
    "cf",
    "seq",
    # Indian state and statute short forms that appear mid-sentence
    "U.P",
    "M.P",
    "A.P",
    "H.P",
    "J.K",
    "N.C.T",
    "Govt",
    "Dept",
    "Addl",
    "Asstt",
)

# Private-use codepoint: never appears in judgment text, and unlike NUL it passes
# safely through source files, pipes and JSON on the way to being unmasked.
SENTINEL = ""

# Order matters. Dotted acronyms run first and consume all their own periods:
# left to the single-initial rule, "A.I.R." would keep its final period (nothing
# capitalised follows) and split "A.I.R. 1963 S.C. 1295" before the page number.
_PROTECTORS: tuple[re.Pattern[str], ...] = (
    # Dotted acronyms: A.I.R., S.C., U.P., P.S., two or more capital-dot pairs.
    re.compile(r"(?:[A-Z]\.){2,}"),
    # Dates and decimals: 27.05.2003, 1.5
    re.compile(r"\d\.(?=\d)"),
    # Initials in names: A.K. Sikri, a lone capital then a dot, before a capital.
    re.compile(r"(?<![A-Za-z])[A-Z]\.(?=\s?[A-Z])"),
)

# A clause number opening a paragraph: "1." "12)" "(iv)". Only ever masked at the
# very start of a paragraph, clean() already gave each numbered clause its own
# paragraph, so a number followed by a period anywhere else is ordinary prose
# ("...under Section 302. Mr. Patwalia...") and must stay a sentence boundary.
_LEADING_MARKER = re.compile(r"^\(?[0-9ivxlcIVXLC०-९]{1,6}\)?[.)]")

_ABBREV_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(a) for a in ABBREVIATIONS) + r")\.", re.IGNORECASE
)

# A boundary is sentence-final punctuation, optional closing quote/bracket, then
# whitespace, then something that can open a sentence.
_BOUNDARY_EN = re.compile(r'(?<=[.?!])(["\')\]]*)\s+(?=[A-Z(\[""ऀ-ॿ\d])')
_BOUNDARY_HI = re.compile(r'(?<=[।?!.])(["\')\]]*)\s+(?=[ऀ-ॿ(\[A-Z\d""])')

_DANGLING = re.compile(r"^[\s\W_]+$")


def _mask(text: str, *, leading_marker: bool = True) -> str:
    """Replace non-terminal periods with a sentinel character."""
    if leading_marker:
        text = _LEADING_MARKER.sub(lambda m: m.group(0).replace(".", SENTINEL), text, count=1)
    for pattern in _PROTECTORS:
        text = pattern.sub(lambda m: m.group(0).replace(".", SENTINEL), text)
    return _ABBREV_RE.sub(lambda m: m.group(0).replace(".", SENTINEL), text)


def _unmask(text: str) -> str:
    return text.replace(SENTINEL, ".")


def split_sentences(text: str, lang: str, *, min_chars: int = 2) -> list[str]:
    """Split ``text`` into sentences, preserving the original characters exactly.

    Args:
        text: cleaned document text, paragraphs separated by blank lines.
        lang: ``"en"`` or ``"hi"``, selects the boundary pattern.
        min_chars: drop fragments shorter than this (stray punctuation).

    Returns:
        Sentences in document order. Paragraph breaks are always boundaries, so a
        numbered clause never merges into the one above it.
    """
    if lang not in ("en", "hi"):
        raise ValueError(f"lang must be 'en' or 'hi', got {lang!r}")
    boundary = _BOUNDARY_EN if lang == "en" else _BOUNDARY_HI

    sentences: list[str] = []
    # Paragraphs are hard boundaries: clean() already reflowed each list item and
    # each numbered clause into its own paragraph, and that structure is a better
    # boundary signal than punctuation.
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        masked = _mask(paragraph)
        parts = boundary.sub(lambda m: m.group(1) + "\n", masked).split("\n")
        sentences.extend(_unmask(part).strip() for part in parts)

    return [s for s in sentences if len(s) >= min_chars and not _DANGLING.match(s)]


def segment_document(en: str, hi: str) -> tuple[list[str], list[str]]:
    """Segment both sides of a judgment."""
    return split_sentences(en, "en"), split_sentences(hi, "hi")
