# Hindi–English Code-Switched ASR

A project to replicate and then improve a Hindi–English code-switched ASR system on MUCS 2021 Subtask-2, using wav2vec 2.0 + CTC.

**Status: Stage 1A complete; Stage 1B execution preparation in progress.** The data pipeline and CTC training infrastructure are built and tested. **Full-scale training has not been run yet, so there are no accuracy results.**

- [Stage 1A report](docs/stage_1a_report.md): dataset statistics, model and tokenizer details, test results, GPU requirements and limitations
- [Stage 1B handoff](docs/stage_1b_handoff.md): what comes next

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r envequirements.txt
.venv\Scripts\python -m src.data.download --only test blind train
.venv\Scripts\python -m src.data.manifest --config configs/data.yaml
.venv\Scripts\python -m src.models.tokenizer --config configs/data.yaml
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python -m src.training.smoke_test
```

Data licence: MUCS 2021 is CC BY-SA 4.0 and is never committed here.

For baseline GPU setup, use the [Stage 1B Colab notebook](notebooks/colab_stage1b_baseline.ipynb) and [Colab requirements](env/requirements-colab.txt). The notebook requires the `stage-1b-baseline` branch to be pushed to GitHub before its fresh-runtime clone cell can run. Colab/cloud setup and training have not been executed or verified from this checkout.

## Stage 1B code paths

The Stage 1B utilities are implemented, but **full training, KenLM decoding, and
evaluation have not been run in this checkout**. The current environment has no
MUCS data and CPU-only PyTorch. See
[the code handoff](docs/stage_1b_code_report.md) for what is and is not verified.

Build the 5-gram LM from `train.jsonl` only. KenLM's `lmplz` and `build_binary`
executables must be installed and on `PATH`:

```powershell
python -m src.language_model --config configs/data.yaml --order 5
```

Transcribe one audio file greedily:

```powershell
python -m src.inference --checkpoint data/checkpoints/baseline_him4200/best --audio path\to\audio.wav
```

For KenLM beam decoding, build/install the KenLM Python binding and optional
`pyctcdecode` dependencies as described in
[the Stage 1B handoff](docs/stage_1b_handoff.md), then supply the compiled model:

```powershell
python -m src.inference --checkpoint data/checkpoints/baseline_him4200/best --audio path\to\audio.wav --kenlm data\language_model\train_5gram.klm --beam-width 3000 --lm-weight 0.5 --word-insertion 1.0
```

Generate predictions for each permitted evaluation split. The default scorer
reports diagnostic WER/CER against cleaned manifest text and writes a complete
`<utterance-id> <hypothesis>` file. It does **not** substitute those diagnostics
for official test/blind WER or T-WER:

```powershell
python -m src.evaluate --split val --checkpoint data/checkpoints/baseline_him4200/best
python -m src.evaluate --split test --checkpoint data/checkpoints/baseline_him4200/best
python -m src.evaluate --split blind --checkpoint data/checkpoints/baseline_him4200/best
python -m src.analysis.error_analysis --predictions results/evaluations/blind_greedy.jsonl --output results/evaluations/blind_error_analysis.json
```

For LM experiments, repeat evaluation on **validation only** while tuning
`--lm-weight`, `--word-insertion`, and `--beam-width`. Freeze those settings
before test/blind evaluation. The analysis tool derives script and switch-context
labels from Unicode and is diagnostic only; it cannot identify transliteration
or homophone causes without manual review.

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
