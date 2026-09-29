"""Process-level setup that must happen before the HF libraries are imported.

Call :func:`configure` as the first statement of every entry point.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# huggingface_hub attaches any locally stored token to *every* request, including
# ones for public repos. An expired OAuth token therefore turns a public model
# into a 401 that transformers reports as "not a valid model identifier". Opting
# out of the implicit token keeps public downloads anonymous and unaffected,
# while an explicitly exported HF_TOKEN still works for gated repos.
_HUB_DEFAULTS = {
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    # Progress bars interleave badly with the benchmark's own tqdm output.
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    # This is a PyTorch-only project, but transformers imports TensorFlow and
    # Flax opportunistically, `data_collator` reaches them through
    # `data/processors/glue.py`. On an image that ships TensorFlow (Kaggle does)
    # any protobuf mismatch in *its* dependency tree then crashes a run that
    # never wanted TensorFlow. Declaring the backend skips those imports.
    "USE_TF": "0",
    "USE_JAX": "0",
}


def configure(*, quiet_warnings: bool = True) -> None:
    """Apply hub defaults without overriding anything the caller set explicitly."""
    for key, value in _HUB_DEFAULTS.items():
        os.environ.setdefault(key, value)

    if os.environ.get("HF_TOKEN"):
        # An explicit token is respected; drop the opt-out so gated repos resolve.
        os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
        logger.info("HF_TOKEN present; using authenticated hub access")

    if quiet_warnings:
        import warnings

        # The fast-conversion and clean_up_tokenization_spaces warnings fire once
        # per tokenizer and say nothing actionable at this point: the roster
        # already pins byte-fallback SentencePiece models to the slow path.
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")
        warnings.filterwarnings(
            "ignore", message=".*sentencepiece tokenizer that you are converting.*"
        )


def setup_logging(level: int = logging.INFO) -> None:
    """Consistent, quiet logging for the scripts."""
    logging.basicConfig(level=level, format="%(levelname)-7s %(name)s: %(message)s")
    for noisy in ("transformers", "huggingface_hub", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
