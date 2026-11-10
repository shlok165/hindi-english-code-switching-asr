# Hindi–English Code-Switched ASR

A baseline for recognising Hindi–English code-switched speech: wav2vec 2.0 with a CTC head, greedy decoding and KenLM beam search, on MUCS 2021 Subtask-2. It follows the approach of Chadha et al. (2022), Approach C2. The encoder may not be the paper's exact checkpoint, so this reproduces the method, not their number.

**Status: Stage 1B is complete (iteration 1, git tag `iteration1`).** On the MUCS blind test, with KenLM beam search: **WER 21.54, T-WER 20.22**, confirmed with the organisers' own scoring script.

The improvement iterations are finished. The final model, chosen on validation data only, is iteration 4 (git tag `iteration4`): **blind WER 20.44, T-WER 19.20**, also confirmed with the organisers' script. See [Improvement iterations](#improvement-iterations).

- [Stage 1A report](docs/stage_1a_report.md): dataset statistics, model, tokenizer and GPU requirements
- [Stage 1B report](docs/stage_1b_report.md): training, evaluation, error analysis and the Stage 2 handoff
- [Improvement iterations](docs/iterations.md): changes made after the baseline, one at a time, with results as they finish
- [Inference and evaluation backend](docs/backend.md): how transcription, validation, scoring and the language-model backends work, and what was verified
- [CPU-only setup](docs/cpu_setup.md): transcribing and scoring on a machine without a GPU

## Overview

**Task.** Transcribe Hindi-English code-switched speech (Hindi sentences with English words, often technical terms, in either script) into text, scored by word error rate. The data is MUCS 2021 Subtask-2 (OpenSLR SLR104, CC BY-SA 4.0): spoken-tutorial recordings.

| | train | val (held out from train) | test | blind |
|---|---|---|---|---|
| Recordings / utterances | 499 / 50,509 | 22 / 2,316 | 30 / 3,136 | 35 / 4,034 |
| Hours | 85.6 | 4.0 | 5.2 | 6.2 |
| Latin-script words | 25.1 % | 23.8 % | 24.3 % | 32.3 % |

Val is held out from train by whole recordings and is the only split used for choices. Test and blind are scored once per model. Blind writes English words in Latin script more often, so its official T-WER (which forgives listed transliterations) is lower than its WER.

**Model.** wav2vec 2.0 base (12 transformer layers, 768 hidden units, about 50 frames per second), initialised from the Vakyansh Hindi checkpoint `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`, with a new linear CTC head over 120 character tokens: Devanagari, lower-case Latin, digits, 16 spoken symbols, `|` as the word delimiter and `<pad>` as the CTC blank. Text is kept in the script it was transcribed in; there is no transliteration in the model.

**Decoding.** Greedy (best token per frame), or beam search (pyctcdecode) with a 5-gram KenLM model trained on the training transcripts, with the weights tuned on val. On a machine without the `kenlm` package the same model is read from its ARPA file by a pure-Python backend that was checked against KenLM (see [docs/backend.md](docs/backend.md)).

**Iterations.** Five training iterations after the data pipeline, each changing one thing and compared on val only; iteration 4 is the final model. The table under [Improvement iterations](#improvement-iterations) summarises them.

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
| 3 | As 2, with English words in the training transcripts written in Latin script | 16.75 | 19.36 | 18.82 |
| 4 | As 2, without the training utterances whose transcript does not match their audio | **16.36** | 20.44 | 19.20 |
| 5 | As 4, with the training segments re-cut where their transcripts are, instead of dropped | 17.81 | 19.56 | 18.15 |

Iteration 3's blind gain comes mostly from matching the blind references, which write English in Latin script. On the val measure used to choose, it ties with iteration 2, so iteration 4 builds on iteration 2. Iteration 4 improves on iteration 2 on every split and is the best on val, so it is the final model.

Iteration 5 is better at correctly cut speech, but val and test keep the released cuts. Their references often hold whole words that the cut has partly removed, and iteration 4, trained on the same kind of cuts, learned to complete them. So iteration 5 scores worse on val and test. It scores better on blind, whose references appear to stop where the audio does. As with iteration 3, blind played no part in the choice.

Details are in [docs/iterations.md](docs/iterations.md).

## Using the final model

The final model is iteration 4. Model weights are never committed here: `data/checkpoints/iteration4/best` comes from training it (step 10 below) or from unpacking `checkpoints.zip`, which also holds the language model (step 7) and the organisers' scoring files. No GPU is needed for anything below; [docs/cpu_setup.md](docs/cpu_setup.md) has the CPU-only installation.

`configs/inference.yaml` names the final model and its decoder settings, so transcribing needs no flags:

```powershell
.venv\Scripts\python -m src.inference --audio path\to\audio.wav                  # greedy, no language model needed
.venv\Scripts\python -m src.inference --audio path\to\audio.wav --decoder beam   # KenLM beam search, settings tuned on val
.venv\Scripts\python -m src.inference --audio path\to\audio.wav --json           # one JSON object per file
.venv\Scripts\python -m src.inference --manifest list.csv --output predictions.jsonl --decoder beam
```

The audio can be any common format at any sample rate; `--start` and `--end` select a segment, `--device cpu|cuda|auto` picks the device, `--lm-backend kenlm|arpa|auto` the language-model reader, and `--checkpoint`, `--decode-config` and `--kenlm` override the config (the form from earlier versions, `--checkpoint data/checkpoints/iteration4/best --kenlm data/language_model/train_5gram.klm --decode-config configs/decode_iteration4.yaml`, still works). A manifest is a CSV or JSONL with an `audio` column and optional `id`, `start`, `end` and `reference`.

Files are validated before the model runs: missing, empty, corrupted, silent, too short (under 0.1 s), too long (over 60 s) or non-finite audio is reported per file instead of crashing, and the command exits 1 if any file failed, 2 if an artifact is missing. With `--json`:

```json
{"audio": "a.wav", "text": "libreoffice base पर spoken tutorial में आपका स्वागत है", "decoder": "beam", "start": 0.0, "end": 4.0, "duration": 4.0, "sample_rate": 16000, "lm_backend": "kenlm", "processing_seconds": 1.9, "warnings": []}
{"audio": "b.wav", "error": "b.wav: audio is too short to transcribe (0.006 s; minimum 0.1 s)"}
```

(The text shown is a val reference, for the shape of the output; no MUCS audio was transcribed while writing this.) From Python:

```python
from src.inference import Transcriber
transcriber = Transcriber.from_config(decoder="beam")   # loads the model and LM once
result = transcriber.transcribe("a.wav", start=0.0, end=4.0)
result.text, result.warnings
```

### Dashboard

`app.py` is a Streamlit app with three sections: an overview, a page that transcribes an uploaded WAV file on the CPU with the final model (greedy or KenLM beam search), and the iteration results table read from `results/iterations.json`. It needs the checkpoint unpacked as above and `streamlit` installed (`env/requirements.txt` lists it).

```powershell
.venv\Scripts\python -m streamlit run app.py
```

Streamlit prints the local URL (normally http://localhost:8501). The model loads on the first transcription and stays loaded while the server runs. `tests/test_app.py` renders every section headlessly.

### Scoring transcripts

`src.score_files` scores reference and hypothesis text without a model, audio or the dataset: Kaldi-style `id text` files, a CSV with `reference` and `hypothesis` columns, or the JSONL written by `src.inference --manifest` (with a `reference` column) or `src.evaluate`:

```powershell
.venv\Scripts\python -m src.score_files --ref ref.txt --hyp hyp.txt --output-dir results\cpu_eval\scoring\my_run
.venv\Scripts\python -m src.score_files --predictions predictions.jsonl --split blind --output-dir results\cpu_eval\scoring\my_run
```

It writes `summary.json` (WER, CER, T-WER where the organisers' maps apply, the training-cleaned and script-folded variants where available, and substitution/deletion/insertion counts, each labelled with the normalisation used), `utterances.csv`, `word_errors.csv` and `confusions.csv`, and never overwrites existing output. Rescoring the committed iteration 4 evaluations with it reproduces their `*_metrics.json` exactly. `src.analysis.error_analysis --predictions ... --output ... --csv-dir ...` categorises the errors (by operation, script, derived language, switch point and duration). `src.evaluate` still scores a whole split from cached log-probabilities and needs the dataset.

### Limitations

- Accuracy is what the [Results](#results) tables report; it has not been re-measured on a CPU, because the dataset is not part of the repository. CPU inference was checked end to end on synthetic audio only.
- The model expects tutorial-style Hindi-English speech at conversational pace; it produces text for any audio, including noise, and reports silence rather than guessing. There is no confidence score.
- Utterances over 60 s are refused rather than split automatically; use `--start`/`--end`. Decoding one utterance at a time on a CPU costs a few seconds per clip after a one-off load of 5–40 s.
- Without the `kenlm` package, beam decoding parses the ARPA model in Python (3.5 s, 250 MB) each time a `Transcriber` is created.
- T-WER is official only on blind; on test it is an approximation, and the script-folded WER needs the train-only script map, which is built from the dataset and not shipped.

## Reproducing the results

Tested on Windows 11 with Python 3.14 and an RTX 3050 6 GB. The Colab notebook, `notebooks/colab_stage1b_baseline.ipynb`, has not yet been run on Colab. Commands are for PowerShell from the repository root.

### 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r env\requirements.txt
```

KenLM decoding needs pyctcdecode, which pins an old numpy that is broken on Python 3.14, so it is installed without dependencies:

```powershell
.venv\Scripts\python -m pip install --no-deps pyctcdecode
.venv\Scripts\python -m pip install pygtrie hypothesis
```

The `kenlm` package itself is optional: PyPI has no Windows wheel, and without it beam search reads the same model from its ARPA file (`--lm-backend auto`). To build it anyway, `bash scripts/build_kenlm.sh` compiles it with MinGW-w64 from Git Bash, or `pip install kenlm` works on Linux, macOS and WSL. For a machine without a GPU, install torch from the CPU index instead; see [docs/cpu_setup.md](docs/cpu_setup.md). `env/requirements.lock` records the exact versions used for the reported results.

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

### 10. The final model (iteration 4)

Iteration 4 starts from the baseline's best checkpoint and uses its language model, so steps 3 and 7 come first. It scores every training utterance with the baseline model and drops those whose transcript does not match the audio. Then it trains for 44,000 more updates and evaluates:

```powershell
.venv\Scripts\python -m src.analysis.score_train --config configs/train_baseline.yaml --checkpoint data/checkpoints/baseline_rtx3050/best
.venv\Scripts\python -m src.data.filter_train --manifest-dir data/processed/manifests --output-dir data/processed/manifests_clean
.venv\Scripts\python -m src.training.train --config configs/train_iteration4.yaml --resume auto
bash scripts/evaluate_iteration.sh iteration4 data/checkpoints/iteration4/best
```

Training took 7 h 2 min on the RTX 3050. The evaluation script tunes the decoder on val and commits the settings to git before it decodes test and blind. The other iterations' configs and data steps are in [docs/iterations.md](docs/iterations.md).

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
