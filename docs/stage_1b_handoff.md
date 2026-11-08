> **Completed.** This was the plan handed from Stage 1A to Stage 1B. The results are in [stage_1b_report.md](stage_1b_report.md), and that report's section F is the handoff to Stage 2.

# Stage 1B Handoff

**Goal:** a complete, experimentally validated baseline: full training, greedy and KenLM decoding, official scoring and error analysis.

**Starting point:** everything in `docs/stage_1a_report.md` works. Full training has not been run. Do not begin Stage 2 work.

## Settled before training

1. **Encoder: him-4200,** decided 7 November 2026. It is configured and verified. Using `ai4bharat/indicwav2vec-hindi` would need a Hugging Face login and a passing `verify_encoder` run.
2. **Seeds.** Run the final configuration with 3 seeds, since no paper here reports variance.

## Rules that protect the result

- **Never use `inference/hin2engwords.txt` for training, the LM or tuning.** It was built from the blind references, so it is scoring-only.
- **Build the LM from `train.jsonl` and any external text, never test or blind.** Exclude val if you tune LM weights on val.
- **Tune only on val.** That covers LM weight, word-insertion penalty and beam settings.
- **Blind is the headline split.** Report test separately, because 41.55 % of its utterances repeat a train sentence.
- **Label every number** with its split, its decoder, and the alignment condition (UnA, since realignment is not implemented).

## Training

```powershell
.venv\Scripts\python -m src.training.train --config configs/train_baseline.yaml --resume auto
```

- It takes about 4.5 hours and resumes after a crash.
- `data/checkpoints/baseline_rtx3050/train_log.csv` already logs train loss, val loss and val WER, and `best/` holds the best checkpoint by val WER.
- That val WER is for selecting checkpoints only: greedy, on cleaned text.
- The pilot leveled off near 43 % val WER by update 300 because its LR schedule ended there. It predicts nothing about the full run.

## Decoding

- **Greedy:** `greedy_decode` in `src/models/ctc_model.py` already exists. Load models with `load_model`.
- **Batching:** the checkpoint uses group normalisation, so compare batched inference with batch size 1 on a sample before trusting batched numbers.
- **KenLM:** PyPI has no Windows wheel. `scripts/build_kenlm.sh` builds it with MinGW, and the build was tested with a small decode.
- **pyctcdecode:** it is deliberately absent from `env/requirements.txt`, because its `numpy<2` pin conflicts with `numpy>=2.3`, and numpy 1.26 is broken on Python 3.14. Install it with `pip install --no-deps pyctcdecode` plus `pygtrie` and `hypothesis`.
- **Decoder labels:** the blank is id 0 and `|` must map to a space. Keep label order equal to `vocab.json`.
- **Paper settings:** beam width 3000 and a 5-gram KenLM LM on IndicCorp, top 500,000 words. IndicCorp availability is unverified. Train transcripts alone will leave many blind words out of the LM, since OOV is 19.6 %.

## Scoring

Use the official scorer. It is in `data/interim/extracted/blind/subtask2_blindtest_wReadme/inference/`.

```powershell
$B = "data/interim/extracted/blind/subtask2_blindtest_wReadme"
.venv\Scripts\python -I -X utf8 "$B/inference/compute_wer.py" --reffile "$B/hindi/files/text" --hypfile hyp.txt --punctmappingfile "$B/inference/punct.mapping.txt" --wordmappingfile "$B/inference/hin2engwords.txt"
```

Tested: it prints 0.00 on identical text and 100.00 on empty hypotheses.

- **Hypothesis format:** `<utterance id> <text>`, one per line.
- **Include every utterance.** The script raises `KeyError` on a missing id, including the 8 with empty references. Use `blind.jsonl`, which keeps all 4,034.
- **Use `-X utf8`.** Without it the script fails on Windows, and `-I` makes it ignore `PYTHONUTF8`.
- **It lower-cases and maps spoken punctuation in hypotheses,** such as `slash` to `/`. T-WER is therefore official, not approximate.
- **References are the raw `text` file, not our cleaned text.** They contain symbol tokens such as `@` and `.`, and our cleaned text differs, so do not score against `text` from the manifest.
- **The map only covers blind.** For a T-WER-style figure on val or test, state that it uses a map built for a different split, or skip it.

## Metrics to add

| Metric | Status |
|---|---|
| WER | `src/metrics/wer.py`, agrees with jiwer in tests. Use the official scorer for reported numbers |
| T-WER | Official scorer |
| CER | To implement; `score()` can be reused at character level |

## Targets

Set before running, on Hindi-English blind:

| System | WER | T-WER |
|---|---|---|
| Official baseline | 25.53 | 23.80 |
| Chadha et al. (Ekstep) | 21.77 | 20.75 |
| Best on the leaderboard | 16.43 | 15.51 |

**Success:** beat 25.53 / 23.80 and land in the 21–24 WER band. 21.77 is a reference point, not a reproduction target, because checkpoint identity is unverified.

## Things to watch

- **Script shift.** Blind is 32.27 % Latin words and train 25.11 %. Expect a large gap between WER and T-WER, which should appear in the error analysis.
- **Latin learning.** Latin rows start untrained. If English words stay transliterated late in training, consider a longer `freeze_encoder_updates`.
- **Sanity gate A0 cannot run as written.** No baseline hypotheses are released, so 25.53 cannot be reproduced without running Kaldi. Say this in the report.
- **Heat.** Monitor GPU temperature during the multi-hour run.

## Deliverables

1. Trained model with its best checkpoint and logs.
2. Greedy and beam-search results on val, test and blind.
3. WER, CER and official T-WER tables, with graphs.
4. Error analysis, split by English and Hindi tokens.
5. README and implementation documentation updates.
