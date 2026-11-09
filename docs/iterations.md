# Improvement Iterations

Iteration 1 is the Stage 1B baseline, tagged `iteration1` and described in [stage_1b_report.md](stage_1b_report.md). Each later iteration changes one thing, so its effect can be measured against the iteration it builds on.

## Rules

These carry over from Stage 1B:

- **Data:** test and blind text is never used for training, the language model or tuning, and `hin2engwords.txt` (built from blind references) is used only for scoring.
- **Tuning:** every iteration tunes its decoder on val, and its settings are committed before its test or blind decoding.
- **Choosing between iterations:** done on val only, with `val_wer_folded`: val WER after mapping both sides through the train-only script map, so an iteration that changes the script of its targets is judged fairly. Test and blind are reported for every iteration but never used to choose.
- **Isolation:** each iteration has its own run directory, logit cache, decoder config and results directory, so earlier results are never overwritten.
- **Seeds:** single runs, seed 42. Differences under about 1 WER may be noise.

## Plan

| Iteration | Change | Builds on | Config |
|---|---|---|---|
| 1 | Baseline (Stage 1B) | – | `configs/train_baseline.yaml` |
| 2 | More training: warm start from iteration 1 for 44,000 updates at half the peak learning rate, with batches grouped by duration (1.74× faster, measured) | 1 | `configs/train_iteration2.yaml` |
| 3 | Script normalisation: English words written in Devanagari mapped to Latin in train and val transcripts and in the LM, using a train-only map. Same recipe as iteration 2 | 1, compared with 2 | `configs/train_iteration3.yaml` |
| 4 | Remove training utterances whose transcript does not match the audio (7.03 % of them), found by scoring train with iteration 1's model. Same recipe as iteration 2 | 2, the better of 2 and 3 on val | `configs/train_iteration4.yaml` |

### Why these changes

The iteration 1 error analysis and the literature point at the data rather than the architecture:

- **Iteration 2:** val WER was still falling when iteration 1's schedule ended.
- **Iteration 3:** words in the wrong script cost about 1.3 WER points on blind.
- **Iteration 4:** misaligned segment edges cost about 0.9 points on blind. The top MUCS system reported its biggest gains from transcript cleanup.

## Script map (iteration 3)

`python -m src.data.script_normalisation` builds the map from `train.jsonl` and CMUdict only. A Devanagari word is mapped to a Latin training word when both of these hold:

- **Sound:** the Latin word's CMUdict pronunciation, spelled out in Devanagari by rule, matches the Devanagari word.
- **Context:** the two words appear among the same neighbours, scored again after mapping.

Short words need stronger context evidence, because many collide with Hindi words (की, the Hindi for "of", sounds like "key").

The result maps 587 words and changes 3.67 % of training tokens. Common Hindi words are left alone. The full list, with counts and similarity scores, is in `results/script_map_report.json`.

The rules were fixed before iteration 1's blind evaluation had been run, and were not changed after it. Some frequent loanwords are therefore deliberately left unmapped, such as सेव and यूज़र, whose context evidence was too weak.

## Misalignment filter (iteration 4)

MUCS segments are cut from long recordings, and some cuts leave a transcript that the audio does not hold. CTC has to emit every character of the transcript somewhere in the audio, so these utterances teach the model to invent words or skip speech. The filter runs in two steps, both on the training split only:

1. `python -m src.analysis.score_train --config configs/train_baseline.yaml --checkpoint data/checkpoints/baseline_rtx3050/best --device cpu` decodes every training utterance with iteration 1's model. This took 6.7 hours on three CPU threads while iteration 2 trained on the GPU.
2. `python -m src.data.filter_train --manifest-dir <manifests> --output-dir <manifests>_clean` drops an utterance when its character error rate is above **50** in both of two views:
   - **As written.**
   - **In one script:** English words are spelled in Devanagari through CMUdict, as for the script map, and both sides are reduced to consonant skeletons.

   The second view keeps utterances that differ only in script (कंसोल विंडो against "console window"). The first keeps English spellings CMUdict lacks ("acetate" against "acitate").

Iteration 1's model gets a median of 4.6 % of characters wrong on training utterances, but some it still gets mostly wrong after six epochs on them.

### What it drops

The filter drops 3,550 of 50,509 training utterances (7.03 %), or 5.65 of 85.55 hours (6.61 %).

| Duration | Share dropped |
|---|---|
| Under 5 s | 9.9 % |
| 5–10 s | 4.5 % |
| 10–20 s | 8.9 % |
| Over 20 s | 10.6 % (45 of 423) |

The commonest dropped case is a full sentence paired with two or three seconds of audio. Val, test and blind are copied unchanged. The list of dropped utterances, with examples on both sides of the threshold, is in `results/train_filter_report.json`.

### How the threshold was chosen

The threshold was set by reading random training utterances, before any model was trained on the result, and it is not tuned on val:

- **Above 60:** all 12 read were misaligned.
- **50 to 60:** 26 of 28 were misaligned, usually with the first or second half of the transcript missing from the audio, or words repeated in the transcript ("portrait portrait landscape landscape").
- **40 to 50:** most of the 16 read still had a word or two of extra or missing speech at an edge, mixed with aligned speech the model simply got wrong (a URL read out as "dot" and "slash", chemistry terms). This band is kept.

It is a data-quality filter, not a way to fit the GPU. No utterance is dropped for its length, and the language model still uses every training transcript, because the text is valid even where the audio is cut.

## Results

Every result uses KenLM beam search with that iteration's own val-tuned settings. The table is regenerated by `python -m src.analysis.compare_iterations`, and its source is `results/iterations.md`.

| Iteration | Val WER | Val WER, script-folded | Test WER | Test T-WER (approx.) | Blind WER | Blind T-WER | Blind CER |
|---|---|---|---|---|---|---|---|
| 1 | 20.08 | 17.75 | 31.29 | 28.25 | 21.54 | 20.22 | 13.65 |
| 2 | 18.65 | 16.54 | 30.19 | 27.22 | 21.01 | 19.74 | 13.43 |
| 3 | 16.75 | 16.75 | 30.73 | 27.55 | 19.36 | 18.82 | 11.71 |
| 4 | 18.43 | 16.36 | 29.98 | 27.06 | 20.44 | 19.20 | 13.17 |

Blind WER and T-WER are confirmed with the organisers' `compute_wer.py` for every iteration. Iteration 3's plain val WER is measured against its own script-normalised references, so only its script-folded val WER is comparable with the other rows.

### Iteration 2: more training

**Training.** Iteration 2 started from iteration 1's best weights, with a fresh optimiser and schedule. It ran 44,000 updates in 6 h 59 min. The schedule had a 5 % warm-up and a 35 % hold at half iteration 1's peak learning rate, then a cosine decay to zero.

During training, greedy val WER (the same check both runs use) improved as follows:

| Point in iteration 2 | Greedy val WER |
|---|---|
| Start (iteration 1's best, from update 37,872) | 23.77 |
| Warm-up peak | 24.93 |
| Best, update 42,000 | **21.34** |

It was still improving slightly when the schedule ended. Val loss rose from 0.76 to 0.86 over the same stretch: the model became more confident, so its remaining mistakes cost more.

**Decoder.** Tuned on val, as before. The settings were α 0.4, β 1.0, unknown-word offset −5 and beam width 500 (iteration 1 used α 0.35), and they were committed before test or blind was decoded. The settings are in `configs/decode_iteration2.yaml`.

**Results.** Every split improves with both decoders:

| Measure | Iteration 1 | Iteration 2 | Change |
|---|---|---|---|
| Val WER, script-folded (KenLM) | 17.75 | 16.54 | −1.21 |
| Blind WER (KenLM) | 21.54 | 21.01 | −0.53 |
| Blind T-WER (KenLM) | 20.22 | 19.74 | −0.48 |
| Val WER (greedy) | 24.07 | 21.73 | −2.34 |
| Blind WER (greedy) | 24.96 | 23.80 | −1.16 |

The blind gain is within single-seed noise. It is consistent across splits and decoders, though.

Greedy decoding gains more than KenLM, because the language model now has less to correct. On val, the LM's contribution falls from 3.99 to 3.08 WER points.

**Where the gain comes from.** The gain is spread across error types, not one fix. On blind, the error rate falls for both Hindi words (12.01 % → 11.55 %) and English words (31.04 % → 30.08 %). Substitutions drop by 251 and deletions by 82, while insertions rise by 80.

The full breakdowns are in `results/iteration2/error_analysis/`.

### Iteration 3: script normalisation

**Training.** Iteration 3 used iteration 2's recipe exactly, with the same warm start, schedule, batching and seed. Only the targets differ. It ran 44,000 updates in 6 h 59 min.

During training, greedy val WER against the script-normalised references improved as follows:

| Point in iteration 3 | Greedy val WER |
|---|---|
| Start (iteration 1's best, scored against the same references) | 23.62 |
| Update 2,000 | 21.33 |
| Best, the last update (44,000) | **19.04** |

Most of the first drop is the mapped words switching to Latin script. When script is ignored, iteration 1 already scores 21.78 on the same check.

**Decoder.** Tuned on val with the script-normalised language model, which is built from every training transcript after mapping. The settings were α 0.4, β 0.5, unknown-word offset −5 and beam width 1000, and they were committed before test or blind was decoded. They are in `configs/decode_iteration3.yaml`.

**Results.**

| Measure | Iteration 2 | Iteration 3 | Change |
|---|---|---|---|
| Val WER, script-folded (KenLM) | 16.54 | 16.75 | +0.21 |
| Val WER, script-folded (greedy) | 19.89 | 19.50 | −0.39 |
| Test WER (KenLM) | 30.19 | 30.73 | +0.54 |
| Test T-WER, approx. (KenLM) | 27.22 | 27.55 | +0.33 |
| Blind WER (KenLM) | 21.01 | 19.36 | −1.65 |
| Blind T-WER (KenLM) | 19.74 | 18.82 | −0.92 |
| Blind CER (KenLM) | 13.43 | 11.71 | −1.72 |

**Choice for iteration 4.** On the measure that decides, iteration 3 is 0.21 points worse than iteration 2. That is well inside single-seed noise, but the rule is to take the lower value, so iteration 4 builds on iteration 2. Blind played no part in the choice, and would have pointed the other way.

With script ignored, iteration 3's greedy output is slightly better than iteration 2's. KenLM gains less on it, though: 2.75 points against 3.35. With KenLM, iteration 3 has more deletions (1,234 against 1,167).

**Why val and blind disagree.** Script normalisation changes how English words are written, not how well they are heard, and the splits write English differently:

| Split | Share of words written in a mapped Devanagari form |
|---|---|
| Train | 3.7 % |
| Val | 4.4 % |
| Test | 5.2 % |
| Blind | 0.1 % |

Blind's references hardly ever use the Devanagari forms in the map. Iteration 3 almost never outputs them either (once in all of blind), while iteration 2 does for 1.5 % of its blind words. So plain WER and CER on blind reward iteration 3 for matching how the blind references are written:

- **English words:** on blind, the error rate falls from 30.08 % to 25.72 %, while Hindi words stay at 11.5 %.
- **Script errors:** errors in script only, and incorrect transliterations, account for 509 of the 788 fewer errors.
- **T-WER:** the organisers' T-WER forgives some transliteration, and it improves less (−0.92).

Test is written like the training data, and moves the other way. The script-folded val measure deliberately ignores script, and on it the two iterations are level.

The full breakdowns are in `results/iteration3/error_analysis/`.

### Iteration 4: misalignment filter

**Training.** Iteration 4 used iteration 2's recipe exactly, without the 3,550 misaligned training utterances. The number of updates stayed at 44,000, which took 7 h 2 min. With 6.6 % less audio, an epoch is 5,171 updates rather than 5,535, so the run covered about 8.5 epochs.

The val references are the same as iteration 2's, so the checks during training compare directly. Greedy val WER was lower than iteration 2's at 21 of the 22 checks, by 0.62 points on average. Iteration 4's best checkpoint, at update 40,000, scored **20.89**; iteration 2's best, at update 42,000, scored 21.34.

Val loss ended higher, at 0.91 against 0.86. Val was not filtered, so it still holds misaligned utterances, and a model that no longer trains on them probably pays more for them.

**Decoder.** Tuned on val with the same language model as iteration 2. The settings were α 0.5, β 0, unknown-word offset −10 and beam width 200, and they were committed before test or blind was decoded. They are in `configs/decode_iteration4.yaml`.

**Results.** Every split improves with both decoders:

| Measure | Iteration 2 | Iteration 4 | Change |
|---|---|---|---|
| Val WER, script-folded (KenLM) | 16.54 | 16.36 | −0.18 |
| Val WER, script-folded (greedy) | 19.89 | 19.40 | −0.49 |
| Test WER (KenLM) | 30.19 | 29.98 | −0.21 |
| Test T-WER, approx. (KenLM) | 27.22 | 27.06 | −0.16 |
| Blind WER (KenLM) | 21.01 | 20.44 | −0.57 |
| Blind T-WER (KenLM) | 19.74 | 19.20 | −0.54 |
| Blind WER (greedy) | 23.80 | 23.34 | −0.46 |

Each gain is within single-seed noise, but they all point the same way. Iteration 4 has the lowest script-folded val WER so far, so it is the base for the next iteration.

**Where the gain comes from.** Greedy decoding has no settings to tune, so it isolates the model. With it, the gain is mostly fewer substitutions, on both Hindi and English words:

| Greedy word error rate | Iteration 2 | Iteration 4 |
|---|---|---|
| Val, Hindi words | 15.40 % | 15.00 % |
| Val, English words | 28.32 % | 27.71 % |
| Blind, Hindi words | 13.10 % | 12.78 % |
| Blind, English words | 35.32 % | 34.24 % |

With KenLM, blind insertions fall from 1,099 to 955. That fall comes mostly from the lower word-insertion score (β 0 against 1.0), not from the filter.

**Checkpoint averaging, checked and not pursued.** Averaging iteration 4's last four checkpoints (updates 40,000 to 44,000) gives 20.92 greedy val WER, against 20.89 for the best single checkpoint. The learning rate had decayed almost to zero over those updates, so the checkpoints barely differ.

The full breakdowns are in `results/iteration4/error_analysis/`.
