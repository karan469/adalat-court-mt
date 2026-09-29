# Running Task 3 on a Kaggle GPU

The local MPS runs fail because a 24GB Mac is shared between CPU and GPU and was
already swapping before the run started. A Kaggle T4 has 16GB of dedicated VRAM
with nothing competing for it, and this job, 907 sentence pairs, 5 epochs, fits
with room to spare.

There is **no supported way to attach VS Code to Kaggle**: it exposes no SSH or
remote-dev endpoint. The tunnel workaround (ngrok/cloudflared from inside a
notebook) breaks at the session limit and runs against Kaggle's terms. If you
want real remote VS Code on a GPU, use Lightning AI Studios (free tier) or
RunPod instead, but for two ten-minute jobs, a Kaggle notebook is faster to a
result than setting either of those up.

## 1. Upload the bundle

`kaggle/court-mt-bundle.zip` holds everything the run needs: the `court_mt`
package, the tokenizer roster, and the aligned sentence pairs. It is not in the
repository (it packages the corpus), so build it first from the project root:

```bash
zip -r kaggle/court-mt-bundle.zip \
    src/court_mt configs/tokenizers.yaml \
    data/processed/corpus.jsonl data/processed/pairs_*.jsonl \
    notebooks/run_stage.py notebooks/compare_runs.py \
    -x '*__pycache__*' '*.pyc'
```

1. Go to <https://www.kaggle.com/datasets> → **New Dataset**
2. Drop in `court-mt-bundle.zip`
3. Title it **court-mt-bundle** (the notebook below expects that slug)
4. Create

## 2. Create the notebook

**Code** → **New Notebook**, then in the right-hand panel:

- **Accelerator**: `GPU T4 x2` (one is used; two is what Kaggle offers)
- **Internet**: **On**, required, the run downloads mBART from HuggingFace
- **Add Input** → your `court-mt-bundle` dataset

## 3. Paste and run

Each cell below is a separate notebook cell.

```python
# 1, dependencies. Kaggle ships torch already; these are the rest.
!pip install -q transformers==4.44.2 tokenizers==0.19.1 peft==0.13.0 \
    sentencepiece==0.2.0 sacrebleu==2.4.3 regex==2024.9.11

# protobuf is deliberately NOT pinned. Kaggle's image ships TensorFlow, and
# pinning protobuf below what that TensorFlow was built against makes any
# transformers import that touches TF fail. court_mt.environment sets USE_TF=0
# so TensorFlow is never imported, but leaving the version alone is the belt to
# that braces.
```

```python
# 2, unpack. paths.py resolves the project root two levels above
# src/court_mt/, so unzipping into /kaggle/working puts data/ and reports/
# exactly where the code expects them.
!cd /kaggle/working && unzip -oq /kaggle/input/court-mt-bundle/court-mt-bundle.zip
!ls /kaggle/working/data/processed
```

```python
# 3, the un-fine-tuned baseline. ~3 minutes.
!cd /kaggle/working && python notebooks/run_stage.py --stage baseline
```

```python
# 4, fine-tune, stock vocabulary. ~8 minutes.
!cd /kaggle/working && python notebooks/run_stage.py --stage stock --no-checkpointing
```

```python
# 5, fine-tune, extended vocabulary. ~8 minutes.
!cd /kaggle/working && python notebooks/run_stage.py --stage extended --no-checkpointing
```

```python
# 6, tables and figures.
!cd /kaggle/working && python notebooks/compare_runs.py
```

`--no-checkpointing` is safe here and about 30% faster: checkpointing existed
only to save memory we no longer need to save. It must be set the same way on
both training stages, and `compare_runs.py` refuses to compare runs whose
settings disagree.

Leave `--optim` alone. The default `adamw_torch` fits comfortably in 16GB of
dedicated VRAM, and Adafactor was only ever a memory workaround.

## 4. Bring the results home

The run writes to `/kaggle/working/reports/`:

```
task3_baseline.json     scores + translations, un-fine-tuned
task3_stock.json        scores + translations, stock vocabulary
task3_extended.json     scores + translations, extended vocabulary
task3_scores.csv        the comparison table
task3_translations.csv  every test sentence, all three runs side by side
figures/task3_*.png
```

Download them from the notebook's **Output** tab, or add a cell:

```python
import shutil
shutil.make_archive("/kaggle/working/task3_results", "zip", "/kaggle/working/reports")
```

Drop the unzipped contents into `reports/` locally, and notebook 5 reads them
without needing a GPU.

## Note on runtime

Kaggle gives 30 GPU-hours a week and cuts sessions at 9 hours (12 for notebooks).
This run needs about 20 minutes, so neither limit is close. The session does
discard its environment when it ends, nothing here depends on that, since every
stage writes its results to disk as it finishes.
