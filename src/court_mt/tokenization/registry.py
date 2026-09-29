"""Loading the set of tokenizers under test from YAML.

Keeping the roster in config rather than code means a tokenizer that turns out to
be gated, unavailable, or simply uninteresting can be dropped without touching
the benchmark, and the domain vocabularies we train are declared the same way as
the pretrained ones.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from court_mt.tokenization.adapters import TokenizerAdapter, TokenizerSpec, build

logger = logging.getLogger(__name__)


def load_specs(config_path: Path, *, only: list[str] | None = None) -> list[TokenizerSpec]:
    """Parse the tokenizer roster, optionally filtered to a subset of keys."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    entries = raw.get("tokenizers") or []
    if not entries:
        raise ValueError(f"{config_path} declares no tokenizers")

    specs = [
        TokenizerSpec(
            key=entry["key"],
            spec=entry["spec"],
            label=entry.get("label", ""),
            family=entry.get("family", ""),
            note=entry.get("note", ""),
            extra=entry.get("extra", {}) or {},
        )
        for entry in entries
        if not entry.get("skip", False)
    ]

    duplicates = {s.key for s in specs if sum(1 for o in specs if o.key == s.key) > 1}
    if duplicates:
        raise ValueError(f"duplicate tokenizer keys in {config_path}: {sorted(duplicates)}")

    if only:
        wanted = set(only)
        unknown = wanted - {s.key for s in specs}
        if unknown:
            raise KeyError(f"unknown tokenizer key(s): {sorted(unknown)}")
        specs = [s for s in specs if s.key in wanted]

    return specs


def load_adapters(
    specs: list[TokenizerSpec], *, skip_failures: bool = True
) -> list[TokenizerAdapter]:
    """Build and eagerly validate each adapter.

    Construction is lazy, so a bad repo id or a missing local model would only
    surface deep inside the benchmark loop. A one-character encode forces the
    load here, where the failure can be reported and skipped cleanly.
    """
    adapters: list[TokenizerAdapter] = []
    for spec in specs:
        adapter = build(spec)
        try:
            adapter.encode("test")
        except Exception as exc:  # noqa: BLE001 - report and continue by design
            if not skip_failures:
                raise
            logger.warning("skipping %s (%s): %s", spec.key, spec.spec, exc)
            continue
        logger.info("loaded %-22s vocab=%7d", spec.key, adapter.vocab_size)
        adapters.append(adapter)

    if not adapters:
        raise RuntimeError("no tokenizers could be loaded")
    return adapters
