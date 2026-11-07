# Stage 1A Report: Data Pipeline and CTC Training Infrastructure

**Date:** 7 November 2026
**Status:** complete. The pipeline is tested and checkpointable. **Full-scale training has not been run.**

Every number below was measured on this machine. The sources are `docs/dataset_stats.json`, `docs/vocab_report.json`, `results/encoder_*.json`, `results/gpu_profile.json`, `results/smoke_test.json` and `results/pilot_300_log.csv`.

## Decision needed: which encoder

The proposed encoder could not be verified, so nothing here has run on it.

| | `ai4bharat/indicwav2vec-hindi` (proposed) | `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` (used) |
|---|---|---|
| Access | **Blocked: gated, HTTP 401** | Open, MIT licence |
| Architecture | Not inspectable until unlocked | wav2vec 2.0 base, 94.42 M parameters |
| Fit with Chadha et al. | Different group's model | Same author, same 4,200 h Hindi lineage |

**Recommendation:** use him-4200 for Stage 1B. It is the same architecture type, so this is a checkpoint choice and not an architecture change. It may not be the same weights as the paper's: the paper calls Hindi-4200 pretrained on unlabelled data, while this model card describes fine-tuning on 4,200 h of labelled data from CLSRIL-23. Report it as "same lineage, identity unverified".

To evaluate the proposed encoder instead, accept its terms on Hugging Face, run `hf auth login`, then run:

```
python -m src.models.verify_encoder --config configs/model_indicwav2vec.yaml
```

If it passes, change the `includes` line in `configs/train_baseline.yaml`. No code changes are needed.

## 1. Repository audit

- **Before this stage:** research documents only, with no code, notebooks or configs.
- **Reused:** dataset facts, the baseline choice and the hyperparameter suggestions from `implementation_brief.md`.
- **Corrections to the brief,** all confirmed against primary sources:

| Brief says | Actually |
|---|---|
| No T-WER scorer exists | The official `compute_wer.py` and `hin2engwords.txt` ship inside the blind archive |
| Hindi-4200 cannot be obtained | A Hugging Face release exists under the first author (see above) |
| `indicwav2vec-hindi` is verified present | It exists but is gated |
| Score the released baseline hypotheses (gate A0) | None are released, so 25.53 cannot be reproduced without running Kaldi |
| Plan for a 16 GB GPU or a 15 h subset | 1.6 M samples per batch fits on this 6 GB GPU, so the full data is feasible |
| Train has 44,249 utterances | That is close to the unique-sentence count; there are 52,825 utterances |

## 2. Dataset statistics

Acquired with `python -m src.data.download`. All three archives passed gzip CRC and size checks, and their SHA-256 digests are in `docs/SHA256SUMS`. No manual access step is needed.

| | train | val (from train) | test | blind |
|---|---|---|---|---|
| Recordings | 499 | 22 | 30 | 35 |
| Utterances | 50,509 | 2,316 | 3,136 | 4,034 |
| Unique sentences | 42,124 | 2,225 | 2,886 | 3,830 |
| Empty transcripts | 0 | 0 | 0 | 8 |
| Segment hours | 85.55 | 4.01 | 5.18 | 6.24 |
| Duration p50 / p99 / max (s) | 5 / 19 / 57 | 6 / 15 / 25 | 5 / 22 / 41 | 5 / 15 / 37 |
| Words | 604,999 | 28,469 | 37,595 | 47,419 |
| Latin-script words | 25.11 % | 23.76 % | 24.26 % | **32.27 %** |
| Devanagari words | 73.78 % | 75.88 % | 75.05 % | 66.25 % |
| Utterances whose sentence is in train | n/a | 13.21 % | 41.55 % | 4.55 % |

**Audio audit.** Every file was read in full. All 586 recordings are 16 kHz, mono, 16-bit PCM, with none missing, corrupt, truncated, silent or non-finite. Whole-recording train audio totals 89.86 h, matching the published figure. There are no text/segment mismatches, and train and val share 0 recordings and 0 utterances.

**What this shows.**
- **Script shift:** blind has 7 percentage points more Latin-script words than train, because its references were annotated differently. Expect WER to lag T-WER on blind.
- **Overlap:** the 41.55 % and 4.55 % figures are per utterance against train minus val, on cleaned text, so they will not equal the paper's 33.9 % and 2.1 %.
- **Timestamps:** segment boundaries look like whole seconds (duration p50 is exactly 5.0 s and p95 exactly 12.0 s), so boundaries are coarse.
- **Filtering:** training drops 341 utterances over 20 s (2.9 % of speech) and 7 whose labels exceed the output frames.

Blind labels are never used for training or tuning. `src/training/train.py` raises an error for any split other than train or val, and a unit test enforces this.

## 3. Files

| Area | Files |
|---|---|
| Data | `src/data/download.py`, `manifest.py`, `preprocessing.py`, `dataset.py` |
| Model | `src/models/tokenizer.py`, `ctc_model.py`, `verify_encoder.py` |
| Training | `src/training/train.py`, `profile_gpu.py`, `smoke_test.py` |
| Shared | `src/utils/config.py`, `src/metrics/wer.py` |
| Config | `configs/data.yaml`, `model_him4200.yaml`, `model_indicwav2vec.yaml`, `train_baseline.yaml`, `train_smoke.yaml` |
| Environment | `env/requirements.txt`, `env/requirements.lock`, `scripts/build_kenlm.sh` |
| Tests and Colab | `tests/test_pipeline.py` (35 tests), `notebooks/colab_stage1a.ipynb` |
| Modified | `.gitignore`, `README.md`, `implementation_brief.md` (correction notice) |

## 4. Model and tokenizer

**Model.** wav2vec 2.0 base: 12 layers, 12 heads, 768 hidden units, 320× downsampling, about 50 frames per second. The CNN feature encoder stays frozen. A new linear CTC head maps 768 to 120 outputs. Of those rows, 62 are copied from the checkpoint and 58 start fresh, including all Latin letters and digits.

**How CTC handles unaligned data.** The model outputs a distribution over the vocabulary plus blank for each frame. CTC needs no frame labels: it sums the probability of every frame path that reduces to the transcript after merging repeats and removing blanks. The only requirement is that the transcript is no longer than the frame sequence. A blank between two identical characters is what preserves a real double letter.

**Tokenizer: 120 tokens,** built from `train.jsonl` only.

| Group | Count | Decision |
|---|---|---|
| Specials | 5 | `<pad>` (id 0) is the CTC blank; `|` (id 4) is the word delimiter |
| Devanagari | 63 | Unchanged |
| Latin | 26 | Lower case, matching the official scorer |
| Digits | 10 | Kept as characters, not spelled out |
| Symbols | 16 | Spoken in these tutorials, for example `*` ("asterisk") and `+` |

- **Rare characters:** symbols under 10 occurrences are removed, following Chadha et al. (23 characters). `<unk>` is never a training target.
- **Round trip:** exact for 99.915 % of transcripts. The 43 others differ only by those removed characters.
- **Text cleaning:** NFC, invisible characters removed, sentence punctuation removed, whitespace collapsed. Scripts are preserved; there is no transliteration.
- **Visarga:** `अत:` and similar words use an ASCII colon as a letter, so 12 stems found in train are mapped to `ः` rather than stripped.
- **Audio:** zero-mean, unit-variance normalisation per utterance in the feature extractor. This makes Chadha et al.'s loudness normalisation redundant, so it is not applied.

**Bug found by testing: the checkpoint's real blank is `<s>`.** Its config says `pad_token_id=1`, but on real speech `<s>` (id 0) takes 77.2 % of frames, which is the fairseq convention. Trusting the config made the model output nothing at all. The first smoke run still reported success because it only required the loss to fall, so the check was tightened. `source_blank` is now explicit in the model config, and `verify_encoder` measures it on real audio and fails on a mismatch.

## 5. Smoke-test results

`python -m src.training.smoke_test`, on real training audio with the GPU. All 12 steps passed, and each one raises an error on failure.

| # | Step | Result |
|---|---|---|
| 1 | Load audio | 4 utterances of 9, 3, 2 and 3 s, float32 mono |
| 2 | Preprocess | Mixed-script text preserved |
| 3 | Encode | Round trip exact; label lengths 88, 44, 16, 27 |
| 4 | Batch | Input `[4, 144000]`; attention mask matches the true lengths |
| 5 | Load model | Head retargeted to 120 tokens |
| 6 | Forward | Logits `[4, 449, 120]`; output frames exceed label lengths |
| 7 | CTC loss | 4.17, finite |
| 8 | Training step | Head weights moved; feature encoder unchanged |
| 9 | Save checkpoint | 377.9 MB |
| 10 | Reload and infer | Largest logit difference 0.0 |
| 11 | Overfit 4 utterances, 500 steps | WER 63.33 → **0.00**; loss 3.47 → 0.17 |
| 12 | Resume after 3 of 6 updates | Weight difference 5.5e-5, below the 3.1e-4 between two identical runs |

Step 11's output matched the code-switched reference exactly, Latin words included. Unit tests: 35 passed.

**Pilot run, 300 updates on the full train and val data.** This measures speed and stability only.

| Update | Train loss | Val loss | Val WER |
|---|---|---|---|
| 150 | 2.40 | 1.87 | 44.97 |
| 300 | 2.11 | 1.66 | 43.06 |

No batches were skipped. These WERs come from greedy decoding on cleaned text after about 5 minutes of training, so they are not an accuracy result and must not be compared with the leaderboard.

## 6. GPU and memory requirements

Measured on the longest real utterances with the encoder unfrozen, mixed precision and gradient checkpointing. The GPU is an RTX 3050 6 GB, with 4.4 GB free at the start.

| Samples per batch | Peak allocated | Peak reserved | s/step |
|---|---|---|---|
| 640 k | 1.7 GB | 1.8 GB | 0.28 |
| 1.28 M | 2.3 GB | 2.9 GB | 0.57 |
| **1.6 M (configured)** | **2.7 GB** | **3.5 GB** | **0.68** |
| 2.24 M | 3.3 GB | 4.3 GB | 1.03 |
| 2.56 M | 3.6 GB | 4.7 GB | 1.15 |

- **Time for full training:** about **4.5 hours** for 20,000 updates, from the pilot's 0.75 s per update plus about 50 s per validation. That is about 6 passes over the data, an estimate.
- **Disk:** about 30 GB in use, plus about 1.1 GB per retained checkpoint. 121 GB is free on D:.
- **RAM:** 15 GB total with little free. `num_workers` is 0 because Windows workers are costly and audio reads are cheap.
- **Smaller GPUs:** lower `max_samples_per_batch` and raise `gradient_accumulation` to keep the same update size.

## 7. Known limitations

- **The proposed encoder is unverified,** and its `source_blank` value is an assumption until `verify_encoder` runs on it.
- **Almost all training hyperparameters are guesses.** Chadha et al. publish none, and each is marked `# GUESSED` in the configs. 20,000 updates is untested.
- **Colab is unexecuted.** The notebook is written and the code path is unit-tested, but it has not been run on Colab. It also needs the repository pushed.
- **Only Windows and Python 3.14 are tested.** A fresh install from `env/requirements.txt` has not been run.
- **Training segments are not realigned.** The ReA step is not implemented; the known misalignment affects train and test, and the organisers report blind was hand-corrected.
- **Attention mask:** this group-norm checkpoint normally runs without one, but it is passed so CTC gets true lengths. Padding can slightly shift normalisation. Length-grouped batches limit this, but the effect is not measured.
- **Validation is not speaker-disjoint,** since no speaker labels exist, and 13.21 % of its utterances repeat a train sentence.
- **Label noise remains.** A few transcripts misuse the visarga as punctuation (such as `हैः`, 28 times).
- **Laptop heat:** the GPU reached 78 °C during profiling. Multi-hour runs need ventilation.

## 8. Launching full-scale training

```powershell
# One-time setup, if starting from a clean clone
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r env\requirements.txt
.venv\Scripts\python -m src.data.download --only test blind train
.venv\Scripts\python -m src.data.manifest --config configs/data.yaml
.venv\Scripts\python -m src.models.tokenizer --config configs/data.yaml

# Checks
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python -m src.training.smoke_test

# Full training: about 4.5 hours. Use the same command to resume after a crash.
.venv\Scripts\python -m src.training.train --config configs/train_baseline.yaml --resume auto
```

Outputs go to `data/checkpoints/baseline_him4200/`: `best/`, `step-*/`, `train_log.csv` and `run_meta.json`. Override any value with `--set`, for example `--set training.seed=7 --set run_name=baseline_seed7`.

## 9. Handoff

See `docs/stage_1b_handoff.md`.
