# Hindi–English Code-Switched ASR

A baseline for recognising Hindi–English code-switched speech: wav2vec 2.0 with a CTC head, greedy decoding and KenLM beam search, on MUCS 2021 Subtask-2. It follows the approach of Chadha et al. (2022), Approach C2. The encoder may not be the paper's exact checkpoint, so this reproduces the method, not their number.

**Status: Stage 1B is complete (iteration 1, git tag `iteration1`).** On the MUCS blind test, with KenLM beam search: **WER 21.54, T-WER 20.22**, confirmed with the organisers' own scoring script.

- [Stage 1A report](docs/stage_1a_report.md): dataset statistics, model, tokenizer and GPU requirements
- [Stage 1B report](docs/stage_1b_report.md): training, evaluation, error analysis and the Stage 2 handoff
- [Improvement iterations](docs/iterations.md): changes made after the baseline, one at a time, with results as they finish

## Results

Measured here, from one training run (seed 42):

| Split | Decoder | WER | CER | T-WER |
|---|---|---|---|---|
| Validation (the tuning set) | Greedy | 24.07 | 15.82 | – |
| Validation (the tuning set) | KenLM | 20.08 | 15.24 | – |
| Test | Greedy | 35.64 | 26.78 | 33.02 (approximate) |
| Test | KenLM | 31.29 | 25.81 | 28.25 (approximate) |
| Blind test | Greedy | 24.96 | 14.04 | 23.99 |
| **Blind test** | **KenLM** | **21.54** | **13.65** | **20.22** |

Published blind-test results, for reference only:

| System | WER | T-WER |
|---|---|---|
| Official MUCS baseline | 25.53 | 23.80 |
| Chadha et al. (2022), Approach C2 | 21.77 | 20.75 |
| Best leaderboard entry (JHU-CLSP/GoVivace) | 16.43 | 15.51 |

Decoder settings were tuned on validation data only and frozen before test or blind were decoded. Test segments line up poorly with their transcripts, so test and blind numbers are not comparable; the Stage 1B report explains why.

### Improvement iterations

Each iteration after the baseline changes one thing, and iterations are compared on validation data only. The script-folded val WER maps English words written in Devanagari to Latin on both sides before scoring, so iterations that change the script of their targets are judged fairly. All results use KenLM:

| Iteration | Change | Val WER, script-folded | Blind WER | Blind T-WER |
|---|---|---|---|---|
| 1 | Stage 1B baseline | 17.75 | 21.54 | 20.22 |
| 2 | More training, from iteration 1's weights | 16.54 | 21.01 | 19.74 |

Details, and the iterations still running, are in [docs/iterations.md](docs/iterations.md).

## Reproducing the results

Tested on Windows 11 with Python 3.14 and an RTX 3050 6 GB. The Colab notebook, `notebooks/colab_stage1b_baseline.ipynb`, has not yet been run on Colab. Commands are for PowerShell from the repository root.

### 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r env\requirements.txt
```

KenLM decoding needs three more steps. PyPI has no Windows wheel for KenLM, and `pyctcdecode` pins an old numpy that is broken on Python 3.14. The build needs MinGW-w64 and Git Bash.

```powershell
.venv\Scripts\python -m pip install --no-deps pyctcdecode
.venv\Scripts\python -m pip install pygtrie hypothesis
bash scripts/build_kenlm.sh
```

`env/requirements.lock` records the exact versions used.

### 2. Dataset

MUCS 2021 is CC BY-SA 4.0 and is never committed here. This downloads about 8.8 GB, checks the archives, and builds manifests and the tokenizer. No manual access step is needed.

```powershell
.venv\Scripts\python -m src.data.download --only test blind train
.venv\Scripts\python -m src.data.manifest --config configs/data.yaml
.venv\Scripts\python -m src.models.tokenizer --config configs/data.yaml
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python -m src.training.smoke_test
```

### 3. Training

```powershell
.venv\Scripts\python -m src.training.train --config configs/train_baseline.yaml --resume auto
```

This took 8 h 52 min on an RTX 3050 6 GB. The same command resumes after an interruption. Output goes to `data/checkpoints/baseline_rtx3050/`: `best/` is the best checkpoint by validation WER, and `train_log.csv` and `epoch_log.csv` hold the curves. Test and blind data are refused by the training code.

### 4. Loading a checkpoint

```python
from src.models.ctc_model import load_model
model, vocab, feature_extractor = load_model("data/checkpoints/baseline_rtx3050/best")
```

### 5. Transcribing audio

The audio can be any common format at any sample rate, and `--start` and `--end` select a segment.

```powershell
.venv\Scripts\python -m src.inference --checkpoint data/checkpoints/baseline_rtx3050/best --audio path\to\audio.wav
```

### 6. Greedy decoding

Greedy decoding is the default above, and it needs no language model. For a whole split:

```powershell
.venv\Scripts\python -m src.evaluate --split val --decoder greedy --checkpoint data/checkpoints/baseline_rtx3050/best
```

### 7. KenLM decoding

Build the 5-gram model from training transcripts. It uses `lmplz` when installed, and otherwise a built-in Kneser-Ney builder. Then tune on validation data only, which saves the settings to `configs/decode.yaml`.

```powershell
.venv\Scripts\python -m src.language_model --config configs/data.yaml --order 5
.venv\Scripts\python -m src.logit_cache --split val --checkpoint data/checkpoints/baseline_rtx3050/best
.venv\Scripts\python -m src.tune_lm --checkpoint data/checkpoints/baseline_rtx3050/best --write-config
.venv\Scripts\python -m src.inference --checkpoint data/checkpoints/baseline_rtx3050/best --audio path\to\audio.wav --kenlm data/language_model/train_5gram.klm
```

### 8. Evaluation

```powershell
.venv\Scripts\python -m src.evaluate --split blind --decoder beam --checkpoint data/checkpoints/baseline_rtx3050/best
```

On blind data, WER and T-WER follow the organisers' rules, and each run is confirmed against their own `compute_wer.py`. T-WER on test is an approximation, because test references were not annotated the same way. The word map comes from the blind references, so it is used for scoring only, never for training or tuning.

### 9. Everything at once

With a trained checkpoint in place, this rebuilds the language model, tunes on validation data, evaluates every split with both decoders, and redraws the figures in `results/figures/`:

```powershell
bash scripts/evaluate_baseline.sh data/checkpoints/baseline_rtx3050/best
```

# Research Dossier

The research phase behind the project. Some of its findings were later corrected; see the corrections table in the Stage 1A report.

## Contents

| Path | What it is |
|---|---|
| `literature/papers.md` | 29 annotated papers (13 primary Hindi–English, 7 supporting, 9 other-pair with justifications), 12 comparability warnings, the official MUCS 2021 Subtask-2 leaderboard, PDF inventory, open questions |
| `literature/comparison_table.csv` | The same 29 papers, 19 columns each |
| `literature/pdfs/` | 30 open-access PDFs — **git-ignored by default**, see below |
| `datasets/datasets.md` | 7 Hindi–English speech corpora with hours, licence, access, annotation granularity, script and course-project feasibility, plus an exclusions table |
| `replication_candidates/candidates.md` | 12 candidates scored on 5 axes, top 3 with trade-offs, 8 ruled out with reasons |
| `improvement_ideas/ideas.md` | 5 improvements with mechanism, effort, expected effect and uncertainty |
| `implementation_brief.md` | Full implementation brief: data, preprocessing, specified *and* unspecified hyperparameters, metrics, targets, ablation ladder, repo structure |
| `_work/` | Fetch logs, extracted text and numeric-verification output — **git-ignored**, see `_work/README.md` |

## Headline findings

1. **MUCS 2021 Subtask-2** is the only open, licensed, benchmarked Hindi–English code-switched speech corpus: 89.86 h train / 5.18 h test / 6.24 h blind, CC BY-SA 4.0, OpenSLR SLR104.
2. **The Microsoft MSCS / WSTCSMC 2020 corpus has no Hindi–English pair** (Gujarati-, Tamil- and Telugu-English only), so there is no real Hindi–English language-diarization or frame-level LID benchmark. All published Hindi–English diarization numbers are on synthetic single-speaker speech.
3. **Recommendation:** replicate Chadha et al. (2022) "Approach C2" — wav2vec 2.0 + CTC + KenLM — using the configured Vakyansh Hindi-4200 lineage checkpoint. Its identity with the paper's weights is unverified. `ai4bharat/indicwav2vec-hindi` is gated in the Stage 1A audit. Official blind-set baseline 25.53 WER / 23.80 T-WER; Chadha et al. 21.77 / 20.75; best leaderboard entry 16.43 / 15.51.

## A note on the PDFs

`literature/pdfs/` holds 30 open-access PDFs (21.3 MB) from the ISCA Archive, ACL Anthology and arXiv. They are **excluded from version control by default**: those papers are free to read, but redistributing the files in a repository is a separate permission that varies by publisher, and four of the cited papers are paywalled and represented only by their preprints. Every paper is cited with title, authors, venue, year and DOI in `literature/papers.md`, so the dossier is fully usable without them. Remove the `literature/pdfs/` line from `.gitignore` only if you have checked the terms and the repository is private.

## Provenance

Every number in these documents was read from a fetched PDF or from the official challenge site and cross-checked against the source text; see `_work/README.md` for the retrieval log, the tools used, and the known limitations.
