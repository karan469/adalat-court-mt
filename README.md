# Token-efficient English to Hindi translation for Indian court judgments

Adalat AI ML assignment. All five tasks are complete.

The full write-up with methods, figures and qualitative analysis is in
**[REPORT.md](REPORT.md)**.

## Results

**Tokenizer choice is almost entirely a Hindi decision.** Across eight
tokenizers, English fertility varies by 23% and Hindi fertility varies by 299%.

| tokenizer | vocab | HI fertility | parity | HI chars/token |
|---|---:|---:|---:|---:|
| MuRIL | 197k | 1.249 | 1.008 | 4.16 |
| **IndicBART** | **64k** | **1.279** | **0.928** | 4.06 |
| BLOOMZ | 250k | 1.308 | 1.075 | 3.97 |
| mBART-50 | 250k | 1.389 | 1.047 | 3.74 |
| Sarvam-1 | 68k | 1.496 | 0.996 | 3.47 |
| GPT-4o `o200k` | 200k | 1.627 | 1.331 | 3.19 |
| mT5 | 250k | 1.968 | 1.373 | 2.64 |
| GPT-4 `cl100k` | 100k | 4.986 | **4.066** | 1.04 |

GPT-4's `cl100k` costs 4x the tokens and 16x the attention compute of the best
option, because Devanagari falls through to byte fallback at 1.04 characters per
token. No judgment fits a 1024-token encoder under any tokenizer.

**Fine-tuning and vocabulary extension**, on 120 held-out sentence pairs:

| run | BLEU | chrF | HI tokens/sentence |
|---|---:|---:|---:|
| zero-shot, no fine-tuning | 29.91 | 49.46 | 55.1 |
| **fine-tuned, stock vocab** | **34.93** | **53.21** | 55.1 |
| fine-tuned, 504 pieces with phrases | 30.37 | 49.65 | **45.5** |
| fine-tuned, 498 pieces, words only | 34.06 | **53.28** | 49.7 |

Fine-tuning on 907 sentence pairs is worth 5.02 BLEU. Whole-word domain
vocabulary then removes 9.8% of decoder steps at no measurable quality cost.
Multi-word phrase pieces save more tokens (17.5%) and cost 3.56 chrF, which wipes
out most of what fine-tuning gained.

**A data note.** Five of the thirty Hindi files contain no Hindi: the Devanagari
was destroyed at encoding time and survives only as runs of `?`. They pass every
length-based check and are detectable only on script content. Including them
moved GPT-4 parity from 4.07 to 3.56 and MuRIL from sixth place to third. The
working corpus is 25 pairs.

## Layout

```
REPORT.md                    the write-up: decisions, results, qualitative analysis
configs/tokenizers.yaml      the tokenizer roster, with reasons for exclusions
scripts/01_build_corpus.py   raw export to cleaned, quality-gated corpus
notebooks/                   the study (see below)
notebooks/run_stage.py       one fine-tuning or evaluation stage, as a process
notebooks/compare_runs.py    collects stage results into tables and figures
src/court_mt/
  corpus/                    loading, cleaning, quality gate, segmentation,
                             alignment, splits
  tokenization/              adapters, metrics, cost, benchmark, extension, sweep
  modeling.py                model assembly, vocabulary extension, LoRA
  finetune.py                training loop
  evaluation.py              generation, BLEU, chrF, token cost
  viz.py                     chart theme and figures
reports/                     CSV results, run outputs, figures
kaggle/                      bundle and instructions for running Task 3 on a GPU
```

Notebooks orchestrate and narrate; the logic lives in `src/court_mt` so it stays
testable and reusable. Each notebook is committed with its outputs.

| notebook | contents |
|---|---|
| `01_corpus.ipynb` | OCR artefacts, cleaning, the quality gate that rejects five files |
| `02_tokenizer_benchmark.ipynb` | the eight-tokenizer comparison |
| `03_vocabulary_optimisation.ipynb` | domain vocabulary, trained from scratch and extended |
| `04_sentence_alignment.ipynb` | segmentation, Gale-Church alignment, filtering, splits |
| `05_model_adaptation.ipynb` | reads the run outputs, tables and qualitative analysis |

## Running it

```bash
uv venv --python 3.11 .venv
uv pip install -e .

python scripts/01_build_corpus.py     # writes data/processed/corpus.jsonl
jupyter lab notebooks/                # notebooks 1 to 4
```

Tasks 1 and 2 need **no GPU**. The whole study runs on CPU in about two minutes.

Task 3 needs one. Each stage runs as a separate process so that memory is
released between them:

```bash
uv pip install -e '.[train]'
python notebooks/run_stage.py --stage baseline
python notebooks/run_stage.py --stage stock     --no-checkpointing
python notebooks/run_stage.py --stage extended  --no-checkpointing
python notebooks/run_stage.py --stage extended  --max-phrase-words 1 --tag words-only --no-checkpointing
python notebooks/compare_runs.py
```

`kaggle/KAGGLE_SETUP.md` covers running these on a free Kaggle T4, which is what
produced the numbers above.

### Data

Place the Drive export anywhere under `data/raw/`. The loader finds
`**/english/clean/*.txt` and `**/hindi/clean/*.txt` by pattern, so the folder
name does not matter. The judgments are not committed.

### If HuggingFace downloads fail with "not a valid model identifier"

A stored but expired HF token gets attached to every request and turns public
repos into 401s. `court_mt.environment.configure()` disables the implicit token,
and every entry point calls it. An explicitly exported `HF_TOKEN` still works for
gated repos.

## Method notes

- **Parity, not fertility, is the cross-language number.** Fertility compares
  tokenizers within one language. Parity (Hindi tokens divided by English tokens
  for the same judgment) isolates the script penalty. The Hindi text here is
  shorter than the English in characters, so parity above 1.0 is pure overhead.
- **Corpus ratios are ratio of sums**, never the mean of per-document ratios,
  which would weight a two-page order like a forty-page judgment.
- **Fidelity is two questions.** Character loss and whitespace loss are reported
  separately. Every tokenizer here reports zero `<unk>`, so a round-trip check is
  the only thing that catches the real losses.
- **Vocabulary gains are measured on held-out judgments.** The same figure on the
  training split reads up to 12 points better.
- **Splits are document-level.** Judgments repeat their own terminology heavily,
  so sentence-level splitting would leak the vocabulary being learned.
