> **Superseded.** This progress note was written before anything had been trained or evaluated. The measured Stage 1B results, and the fixes made to this code since, are in [stage_1b_report.md](stage_1b_report.md).

# Stage 1B Progress Report

**Branch:** `stage-1b-baseline`, created from `origin/stage-1a-pipeline`.

**Status:** Stage 1A fixes are present on the work branch, and initial Stage 1B
code paths are implemented and unit-tested. Full-scale training and experimental
evaluation have not been run. No model checkpoint or measured accuracy result
is claimed.

## Work completed

- Started the work branch from the Stage 1A fixes branch, preserving the
  previously missing `src/data/` pipeline and the `.gitignore` correction that
  distinguishes source package `src/data/` from ignored runtime `data/`.
- Added single-utterance inference in `src/inference.py`. Greedy decoding works
  without optional dependencies; KenLM beam decoding uses `pyctcdecode` and
  the model's vocabulary order, with the CTC blank at id 0 and `|` mapped to
  a space. The CLI accepts audio segment bounds and exposes beam width, LM
  weight, and word insertion parameters.
- Added `src/language_model.py`, which accepts only `train.jsonl`, checks every
  row's split, and invokes KenLM's `lmplz` and `build_binary`.
- Added `src/evaluate.py`, which preserves every row (including empty
  references), writes prediction JSONL and official-scorer-compatible
  hypothesis files, and labels its local WER/CER as cleaned-text diagnostics.
- Added cleaned-text CER with Unicode code points and spaces counted as
  characters; punctuation and whitespace normalization follows the training
  transcript cleaner.
- Added `src/analysis/error_analysis.py` for word-level edit categories,
  Unicode-script diagnostics, switch-context heuristics, long-utterance
  summaries, and representative examples. Language and switch labels are
  explicitly heuristic rather than ground-truth annotations.
- Added initial decoder settings in `configs/decode.yaml`. LM weight and word
  insertion values are explicitly guessed starting points; they have not been
  tuned.
- Updated `README.md` with commands and the current completion limitations.
- Added tests for CER, beam decoder vocabulary labels, the training-only LM
  input guard, and error-analysis output.

## Verification

- `python -m pytest tests -q` — **39 passed, 1 skipped**.
- `python -m compileall -q src tests` — passed.
- `git diff --check` — passed.
- `--help` CLI checks passed for inference, LM building, evaluation, and error
  analysis.
- `notebooks/colab_stage1b_baseline.ipynb` parses as valid notebook JSON; all
  10 Python cells pass Python syntax parsing. The notebook has not been run on
  Google Colab.
- Pylance import analysis resolves the installed project imports; `pyctcdecode`
  is unresolved in the selected local interpreter, as expected for the optional
  beam decoder. KenLM's Python module and command-line tools are also absent
  locally.

These checks do not execute model inference or produce experimental results.

## Colab preparation

- Added `notebooks/colab_stage1b_baseline.ipynb` with GPU/CUDA and memory checks,
  exact-branch clone guard, Colab dependency installation that checks PyTorch
  was retained, Drive artifact paths, local ephemeral dataset handling,
  archive/download/manifest/tokenizer steps, split and audio integrity checks,
  encoder verification, tests, smoke test, checkpoint-resume preparation,
  opt-in full training, and optional KenLM/decoding instructions.
- Large training audio remains on `/content` for I/O performance. The notebook
  directs training checkpoints, Hugging Face model cache, configs, and reports
  to Google Drive. Evaluation artifacts are written directly to Drive.
- Dataset validation checks split counts, duplicate and cross-split utterance
  IDs, train/validation recording disjointness, official test/blind recording
  separation, audio existence, sample rate, segment bounds, transcript
  normalization, and full-recording audit results. Train/test and train/blind
  sentence repetition is reported rather than silently removed.
- Added `env/requirements-colab.txt`; it uses the standard project
  requirements without pinning a Windows CUDA wheel. `pyctcdecode` and the
  compiled KenLM tools remain optional and are installed/built only for LM
  decoding.
- Exact full-training command (with Drive persistence):

  ```bash
  python -m src.training.train --config configs/train_baseline.yaml --resume auto \
    --set training.output_dir=/content/drive/MyDrive/hindi-english-asr-stage1b/checkpoints
  ```

### External blocker

The GitHub remote currently advertises `stage-1a-pipeline` but does **not**
advertise `stage-1b-baseline`. The notebook intentionally stops instead of
cloning a different branch. Push the work branch before starting a fresh Colab
runtime. This workspace has no CUDA device, MUCS data, cached checkpoint,
KenLM, or `pyctcdecode`; consequently there is no Colab GPU output, dataset
validation result, model-download result, or smoke-test result to report here.

## Remaining work and blockers

- Full training, checkpoint selection, training curves, and GPU utilization
  have not been performed.
- KenLM binary construction or decoding has not been run. It requires the
  training corpus, `lmplz`, `build_binary`, the KenLM Python binding, and
  `pyctcdecode`.
- Evaluation on validation, test, or blind has not been run. No predictions,
  figures, measured WER/CER, official T-WER, or corpus error analysis have been
  generated.
- Official WER/T-WER scoring against raw MUCS references remains to be run.
  Local cleaned-text WER/CER diagnostics are not leaderboard-comparable.
- Transliteration and homophone error categories require manual review; the
  analysis script does not infer those causes.

The current checkout has no `data/` directory or cached checkpoint. Its Python
runtime reports PyTorch `2.14.1+cpu`, `torch.cuda.is_available() == False`, and
no installed `kenlm` or `pyctcdecode` modules. Full-scale training and KenLM
decoding therefore cannot run in the current environment.

## Commands when prerequisites are available

```powershell
python -m pytest tests -q
python -m src.training.smoke_test
python -m src.training.train --config configs/train_baseline.yaml --resume auto
python -m src.language_model --config configs/data.yaml --order 5
python -m src.evaluate --split val --checkpoint data/checkpoints/baseline_him4200/best
python -m src.evaluate --split test --checkpoint data/checkpoints/baseline_him4200/best
python -m src.evaluate --split blind --checkpoint data/checkpoints/baseline_him4200/best
```

Build and tune the language model using training data and validation data only.
Do not use test or blind references for LM training, tuning, or checkpoint
selection. Run the official scorer on raw test/blind references after settings
are frozen. Label all resulting figures with split, decoder, alignment condition
(`UnA`), and metric variant.

## Files changed for Stage 1B

- `src/inference.py`
- `src/language_model.py`
- `src/evaluate.py`
- `src/metrics/cer.py`
- `src/analysis/__init__.py`
- `src/analysis/error_analysis.py`
- `configs/decode.yaml`
- `env/requirements-colab.txt`
- `notebooks/colab_stage1b_baseline.ipynb`
- `tests/test_pipeline.py`
- `README.md`
- `docs/stage_1b_code_report.md`
