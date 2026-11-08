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
| 4 | Remove training utterances whose transcript does not match the audio (7.03 % of them), found by scoring train with iteration 1's model | Best of 2 and 3 | Written once 2 and 3 are compared |

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
