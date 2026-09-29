"""Building the model and tokenizer for legal-domain adaptation.

The base is ``facebook/mbart-large-50-many-to-many-mmt``: it already translates
English to Hindi, so fine-tuning measures adaptation to legal register rather
than teaching translation from scratch on 907 sentence pairs.

Three decisions are encoded here.

**Vocabulary is extended with ``add_tokens``, not by editing the SentencePiece
proto.** mBART-50's vocabulary is laid out ``[specials][sp pieces][language
codes][mask]``, so injecting pieces into the proto shifts every language-code id
and silently misaligns their *pretrained* embeddings. ``add_tokens`` appends past
the end, leaving all existing ids fixed. Measured on held-out Hindi the two
approaches are within half a point (-16.3% vs -16.8%), which is a cheap price for
not having to relocate pretrained rows.

**New rows are mean-initialised.** A random row is noise, and 907 sentence pairs
is not enough to train 507 of them out of noise. Each new row instead starts as
the mean of the embeddings of the pieces it replaces, which puts it in roughly
the right region of the space before training begins.

**The new rows are trainable, and must be.** mBART-50 sets
``tie_word_embeddings=True``, so the embedding matrix is *also* the decoder's
output projection: the logit for emitting a token is ``hidden_state · emb[token]``.
A frozen new row is therefore an uncalibrated classifier row, mean
initialisation gives a usable *input* representation, but as an output vector the
mean of two constituents does not mean "emit this compound". Since the extended
run's labels contain these tokens, freezing them would train the model toward
targets it is not allowed to represent, and the run would lose to the stock one
for reasons that have nothing to do with the vocabulary.

A gradient mask keeps the update confined to the 504 new rows, 0.5M parameters.
The inefficiency to be aware of: ``requires_grad`` is a property of the whole
tensor, so the optimiser carries Adam state for all 250k rows (~3GB) even though
only 504 of them ever move. Lower ``batch_size`` and raise
``gradient_accumulation`` if that is tight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
SOURCE_LANG, TARGET_LANG = "en_XX", "hi_IN"

# mBART attention projections. Deliberately not fc1/fc2: the feed-forward layers
# more than double the adapter size for little gain at this data scale.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


@dataclass
class AdaptationConfig:
    """Everything that distinguishes one fine-tuning run from another."""

    name: str
    extended_vocab: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = LORA_TARGETS
    added_surfaces: list[str] = field(default_factory=list)
    train_new_embeddings: bool = True


def load_tokenizer(surfaces: list[str] | None = None):
    """Load the mBART-50 tokenizer, optionally extended with domain pieces.

    Returns the tokenizer and the number of tokens actually added (duplicates of
    existing vocabulary entries are skipped by ``add_tokens``).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, use_fast=False, src_lang=SOURCE_LANG, tgt_lang=TARGET_LANG
    )
    if not surfaces:
        return tokenizer, 0

    n_added = tokenizer.add_tokens(list(surfaces))
    logger.info("extended tokenizer: +%d pieces -> %d", n_added, len(tokenizer))
    return tokenizer, n_added


def mean_initialise(model, tokenizer, n_added: int) -> None:
    """Seed each newly added embedding row with the mean of its constituents.

    The surface of a new token is re-encoded with the *original* vocabulary, the
    ids it would have used before extension, and their embeddings averaged. A
    piece meaning "High Court" therefore starts halfway between "High" and
    "Court" rather than at random noise.
    """
    if n_added == 0:
        return

    embeddings = model.get_input_embeddings().weight.data
    original_size = embeddings.shape[0] - n_added
    added = tokenizer.convert_ids_to_tokens(range(original_size, embeddings.shape[0]))

    with torch.no_grad():
        for offset, surface in enumerate(added):
            # sp_model is the untouched SentencePiece model, so it segments the
            # surface exactly as the tokenizer did before the pieces were added.
            pieces = tokenizer.sp_model.encode(surface.replace("▁", " ").strip())
            ids = [
                p + tokenizer.fairseq_offset
                for p in pieces
                if p + tokenizer.fairseq_offset < original_size
            ]
            if ids:
                embeddings[original_size + offset] = embeddings[ids].mean(dim=0)

    logger.info("mean-initialised %d new embedding rows", n_added)


def restrict_embedding_gradients(model, n_added: int) -> int:
    """Make the embedding trainable, but let only the newly added rows update.

    The hook zeroes the gradient of every pretrained row, so the optimiser cannot
    move them. Note the cost: ``requires_grad`` applies to the whole tensor, so
    Adam still carries state for all 250k rows. Enable only when the extended run
    needs the new rows trained, not as a default.

    Returns the number of embedding parameters that actually receive updates.
    """
    if n_added == 0:
        return 0

    embedding = model.get_input_embeddings()
    weight = embedding.weight
    weight.requires_grad_(True)

    n_rows, width = weight.shape
    mask = torch.zeros(n_rows, 1, dtype=weight.dtype, device=weight.device)
    mask[n_rows - n_added :] = 1.0

    # Zeroing the gradient of every pretrained row keeps the optimiser from
    # touching them, including through the tied output projection.
    weight.register_hook(lambda grad: grad * mask.to(grad.device, grad.dtype))

    trainable = n_added * width
    logger.info(
        "embedding: %d of %d rows trainable (%.2fM params)", n_added, n_rows, trainable / 1e6
    )
    return trainable


def build(config: AdaptationConfig):
    """Assemble tokenizer and LoRA-wrapped model for one run."""
    from peft import LoraConfig, TaskType, get_peft_model

    surfaces = config.added_surfaces if config.extended_vocab else []
    tokenizer, n_added = load_tokenizer(surfaces)

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    if n_added:
        model.resize_token_embeddings(len(tokenizer))
        mean_initialise(model, tokenizer, n_added)

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
        ),
    )

    # get_peft_model froze everything, adapters aside. The new embedding rows are
    # re-enabled: they double as output-classifier rows under tied embeddings.
    updated_rows = (
        restrict_embedding_gradients(model, n_added) if config.train_new_embeddings else 0
    )

    adapters = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and "shared" not in name and "embed_tokens" not in name
    )
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "%s: %.2fM LoRA + %.2fM embedding = %.2fM effective, of %.1fM total (%.2f%%)",
        config.name,
        adapters / 1e6,
        updated_rows / 1e6,
        (adapters + updated_rows) / 1e6,
        total / 1e6,
        100 * (adapters + updated_rows) / total,
    )
    return model, tokenizer, n_added


def device() -> torch.device:
    """Best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free(*objects) -> None:
    """Release models and datasets, then return cached device memory.

    Fine-tuning two models in one session holds both in memory at once unless the
    first is explicitly dropped: a 611M model is ~2.4GB of parameters, and the
    extended run adds ~3GB of optimiser state for the embedding on top. PyTorch
    also caches freed blocks rather than returning them, so ``empty_cache`` is
    needed as well as ``del``, without it MPS reports memory that nothing owns.
    """
    import gc

    for obj in objects:
        del obj
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def memory_report(namespace: dict | None = None, top: int = 8) -> None:
    """Print where device memory has gone, and what is holding it.

    ``torch.mps.current_allocated_memory`` counts only live tensors; the driver
    figure includes blocks PyTorch has cached but not returned, which is usually
    the gap people mistake for a leak. Pass ``globals()`` to also list the
    notebook variables holding the most memory.
    """
    if torch.backends.mps.is_available():
        live = torch.mps.current_allocated_memory() / 1e9
        driver = torch.mps.driver_allocated_memory() / 1e9
        print(f"MPS live tensors : {live:6.2f} GB")
        print(
            f"MPS driver total : {driver:6.2f} GB   (cached but unreturned: {driver - live:.2f} GB)"
        )
    elif torch.cuda.is_available():
        print(f"CUDA allocated   : {torch.cuda.memory_allocated() / 1e9:6.2f} GB")
        print(f"CUDA reserved    : {torch.cuda.memory_reserved() / 1e9:6.2f} GB")

    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"process peak RSS : {rss:6.2f} GB")

    if namespace is None:
        return

    def size_of(obj) -> int:
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, torch.nn.Module):
            total = sum(p.numel() * p.element_size() for p in obj.parameters())
            total += sum(
                p.grad.numel() * p.grad.element_size()
                for p in obj.parameters()
                if p.grad is not None
            )
            return total
        return 0

    holders = sorted(
        ((name, size_of(value)) for name, value in namespace.items() if not name.startswith("_")),
        key=lambda pair: -pair[1],
    )
    holders = [(n, s) for n, s in holders if s > 1e7][:top]
    if holders:
        print("\nlargest objects in scope:")
        for name, size in holders:
            print(f"  {name:22s} {size / 1e9:6.2f} GB")
