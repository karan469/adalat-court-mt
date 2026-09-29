"""LoRA fine-tuning of mBART-50 on the aligned court sentence pairs.

Settings are chosen for the data size, not copied from a general recipe. 907
training pairs is small enough that the usual risks invert: the learning rate has
to be low enough not to wash out pretrained translation ability, and the run has
to be short enough not to memorise the training judgments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from court_mt import paths
from court_mt.corpus.dataset import load_pairs

logger = logging.getLogger(__name__)

# Aligned sentence pairs are short, the 99th percentile is well under this, so a
# tight cap wastes no compute on padding while truncating almost nothing.
MAX_LENGTH = 192

# Label positions that the loss must ignore.
IGNORE_INDEX = -100


@dataclass
class TrainingConfig:
    """Hyperparameters for one fine-tuning run."""

    output_dir: Path
    epochs: int = 5
    batch_size: int = 4
    gradient_accumulation: int = 2
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 20260928
    max_length: int = MAX_LENGTH
    # Recomputes activations in the backward pass instead of storing them.
    # Numerically identical, ~30% slower, and frees several GB, so it can be
    # turned on for one run without invalidating a comparison against another.
    gradient_checkpointing: bool = False
    # "adafactor" stores a factored second moment: for the embedding that is
    # (250k + 1k) values rather than 250k x 1k, ~1MB instead of ~2GB. Unlike
    # checkpointing this *does* change optimisation, so both runs must match.
    optim: str = "adamw_torch"


class PairDataset(Dataset):
    """Tokenised English->Hindi sentence pairs."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = MAX_LENGTH):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        # text_target routes the Hindi through the target-language side of the
        # tokenizer, which sets the hi_IN language code on the labels.
        encoded = self.tokenizer(
            record["en"], text_target=record["hi"], max_length=self.max_length, truncation=True
        )
        return {k: encoded[k] for k in ("input_ids", "attention_mask", "labels")}


def load_split(name: str, tokenizer, max_length: int = MAX_LENGTH) -> PairDataset:
    """Load one of the train/dev/test sentence-pair files."""
    records = load_pairs(paths.PROCESSED / f"pairs_{name}.jsonl")
    logger.info("%s: %d pairs", name, len(records))
    return PairDataset(records, tokenizer, max_length)


def train(model, tokenizer, config: TrainingConfig, train_set, dev_set):
    """Fine-tune ``model`` and return it, along with the training history."""
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

    if config.gradient_checkpointing:
        # With a PEFT-wrapped model the base weights are frozen, so no input to
        # the checkpointed blocks requires grad and the backward pass finds
        # nothing to differentiate. This re-enables gradient flow into them.
        model.enable_input_require_grads()

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=IGNORE_INDEX, padding="longest"
    )

    arguments = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to=[],
        seed=config.seed,
        # fp16 is a CUDA path; MPS and CPU run fp32.
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    result = trainer.train()
    logger.info("trained %d steps, final loss %.4f", result.global_step, result.training_loss)
    return model, trainer.state.log_history
