# The inference and evaluation backend

How a recording becomes text and a number, which modules do it, and what was
checked. For installing on a machine without a GPU see [cpu_setup.md](cpu_setup.md).
For the model itself and how it was trained, see the
[Stage 1B report](stage_1b_report.md) and [iterations.md](iterations.md).

## Modules

| Module | Role |
|---|---|
| `src/inference.py` | `Transcriber` (load once, transcribe many files), audio validation, the `python -m src.inference` command |
| `src/beam_search.py` | pyctcdecode beam search over log-probabilities; picks the language-model backend |
| `src/arpa_lm.py` | Pure-Python reader for the ARPA n-gram file, used when the `kenlm` package is absent |
| `src/models/ctc_model.py` | Loads the wav2vec 2.0 CTC checkpoint; greedy decoding |
| `src/data/preprocessing.py` | Audio loading (mono, 16 kHz) and the training text cleaner |
| `src/metrics/wer.py`, `cer.py`, `official.py` | Alignment, WER, CER, the organisers' scoring rules |
| `src/score_files.py` | Scores reference and hypothesis files without a model |
| `src/analysis/error_analysis.py` | Categorises errors from a predictions file |
| `src/evaluate.py` | Full-split evaluation from cached log-probabilities (unchanged; needs the dataset) |
| `configs/inference.yaml` | Defaults: the final model, its decode settings, the validation limits |

## Transcribing

```
audio file ─► validate ─► load_audio ─► feature extractor ─► wav2vec 2.0 CTC ─► log-softmax
                                                                                   │
                       greedy: merge repeats, drop blanks  ◄───────────────────────┤
                       beam:   pyctcdecode + 5-gram LM     ◄───────────────────────┘
```

`Transcriber(checkpoint, decoder, decode_config, kenlm, lm_backend, device, limits)` loads the
checkpoint and, for beam decoding, the language model, once. `Transcriber.from_config()` reads
`configs/inference.yaml`; keyword arguments override it. `transcribe(path, start, end)` returns a
`Transcription` with `text`, `decoder`, `start`, `end`, `duration`, `sample_rate`, `lm_backend`,
`processing_seconds` and `warnings`; `to_dict()` is what `--json` prints.

Utterances are processed one at a time. The checkpoint uses group normalisation in its feature
encoder, so padding a batch would change the features.

### Validation

Checked before any audio reaches the model. Each failure is an `AudioValidationError` with the
file name and the reason; the command line prints it (or a JSON object with an `error` key) and
continues with the next file, exiting 1 at the end.

| Case | Result |
|---|---|
| File missing, 0 bytes, unreadable header, no frames | error |
| Range with `start < 0`, `end <= start`, `start` past the end, `end` past the end by more than 0.01 s | error |
| Shorter than `min_seconds` (0.1 s: fewer than 5 output frames) | error |
| Longer than `max_seconds` (60 s: the longest training utterance is 57 s) | error, with the advice to cut with `--start`/`--end` |
| NaN or infinite samples | error |
| Peak amplitude below `silence_peak` (0.0001) | empty text, warning, model not run |
| More than one channel, or not 16 kHz | converted, warning |
| Samples outside [-1, 1] | warning (the feature extractor normalises) |

The checkpoint directory is checked too: `config.json`, `preprocessor_config.json`, `vocab.json`,
weights, the five special tokens, contiguous ids with `<pad>` at 0, and the model's output size
against the vocabulary. A missing or broken artifact is an `ArtifactError`, exit code 2, with the
path that was looked at.

### Decoders

- **Greedy** (`--decoder greedy`, the default): best token per frame, repeats merged, blanks and
  the never-target tokens `<s>`, `</s>`, `<unk>` removed. Needs only torch and transformers.
- **Beam** (`--decoder beam`): pyctcdecode with the 5-gram KenLM trained on the training
  transcripts, with the settings tuned on val for iteration 4
  (`configs/decode_iteration4.yaml`: alpha 0.5, beta 0.0, beam 200, unk −10). Needs pyctcdecode
  and a language-model backend.

### Language-model backends

`--lm-backend auto` (default) uses the `kenlm` package when it imports, else the ARPA reader.

The `kenlm` package has no Windows wheel and needs a C++ toolchain plus the Windows SDK to build,
which is often not available on a laptop. `src/arpa_lm.py` reads `train_5gram.arpa`, the text file
`train_5gram.klm` was compiled from, into dictionaries and scores words with standard back-off.
It implements only the part of `kenlm.Model` that pyctcdecode calls (`BeginSentenceWrite`,
`NullContextWrite`, `BaseScore`, `__contains__`), with context tuples as states. The trained model
is not changed; the same file is read a different way.

**Agreement with KenLM was measured, not assumed.** On 2,003 sentences (500 lines sampled from the
LM's training text, the same lines with their words shuffled so that back-off to shorter n-grams
is exercised, and sentences with out-of-vocabulary words; 26,145 word scores in all), the ARPA
reader's per-word log10 probabilities differ from KenLM's by at most 6.3×10⁻⁷ (mean 5.5×10⁻⁸),
which is float32 rounding. `results/cpu_eval/lm_check/comparison.json` is the report; the two
score files beside it were produced by `python -m src.arpa_lm` and
`scripts/kenlm_reference.py` (KenLM ran under WSL, see cpu_setup.md), and
`scripts/compare_lm_scores.py` compared them. The unit tests also compare the two whenever `kenlm`
is importable, and `scripts/compare_beam_backends.py` decodes saved log-probabilities with both
backends to confirm that the beam search returns the same text.

Costs: the ARPA reader parses the 68 MB 5-gram file in Python each time a beam `Transcriber` is
created (3.5 s and about 250 MB on the machine used here), where KenLM memory-maps the binary
file instantly. Per
utterance the decoders cost about the same, because pyctcdecode caches LM scores by text.

### Devices

`--device auto` picks CUDA when available, otherwise CPU. `--device cuda` on a machine without
CUDA is an error rather than a silent fallback. No code path requires a GPU.

## Scoring files: `src.score_files`

Takes reference and hypothesis text and nothing else: Kaldi-style `id text` files, a CSV of
pairs, or the JSON lines that `src.evaluate` and `src.inference --manifest` write. It reuses
`src.metrics`, so its numbers are the ones in the committed results: rescoring the six
`results/iteration4/evaluations/*.jsonl` files reproduces every value in their `*_metrics.json`
(`results/cpu_eval/scoring/`).

Three normalisations are scored and named separately, because they answer different questions:

| Name | Rule | Metrics |
|---|---|---|
| official | lower-case both sides; `punct.mapping.txt` on the hypothesis; empty references skipped. The organisers' `compute_wer.py` rules. | `wer`, `cer` |
| T-WER | official, then `hin2engwords.txt` on the hypothesis (blind, where it is the official metric) or on both sides (test, `twer_approximate`). Not reported for val or custom data unless `--twer` asks. | `twer` / `twer_approximate` |
| cleaned | `clean_transcript` (the training cleaner) on both sides | `wer_cleaned`, `cer_cleaned` |
| script-folded | official, then both sides through the train-only script map `data/processed/script_map/hi2en_train.json`, so a word in either script counts as correct. Only when that file exists; it is built from the training data by `src.data.script_normalisation` and is not part of `checkpoints.zip`. | `wer_folded` |

Outputs: `summary.json` (the metrics, S/D/I counts under each normalisation, the sources and
rules used, the 40 most frequent error pairs), `utterances.csv` (per-utterance counts, WER, CER
and both scored strings), `word_errors.csv` (every substitution, deletion and insertion with the
script of each word) and `confusions.csv`. Existing files are not overwritten without `--force`.

## Error analysis: `src.analysis.error_analysis`

Unchanged in what it reported before (error types, substitutions split into listed
transliteration / near transliteration / homophone / other, errors by derived language, by
switch point and by duration, top confusions and examples), plus a reference-script by
hypothesis-script matrix for substitutions and per-script deletion and insertion counts, which
need no word map because scripts are visible in the text. `--csv-dir` writes every table as CSV.
Language labels remain derived from script and the organisers' word map, as the report's
`limitations` say; the word map is used for test and blind only.

## Tests

`python -m pytest tests -q`. `tests/test_inference_cpu.py` builds a two-layer random wav2vec 2.0
model in a temporary directory, so the checkpoint checks, every validation case above, greedy
decoding, beam decoding with the ARPA backend, config loading, the exit codes and the manifest
mode all run in seconds without the real weights. Tests that need `kenlm` skip when it is not
importable; one test loads the real iteration 4 checkpoint and skips when `checkpoints.zip` is
not unpacked. `tests/test_score_files.py` covers the readers, the normalisations, T-WER modes,
the outputs and the rescoring of the committed val results.

## What was and was not run here

- Greedy and beam (ARPA backend) transcription of the final model ran on CPU on a synthetic
  English text-to-speech clip, to check that the pipeline works end to end. The output is not
  meaningful: the model was trained on Hindi-English tutorial speech, and no MUCS audio was
  available on this machine. Accuracy is the committed evaluation, not anything measured here.
- Full-split evaluation (`src.evaluate`) was not re-run: it needs the dataset and cached
  log-probabilities. The committed results were rescored from their JSONL files instead.
- The ARPA backend was validated against real KenLM at the word-score level (above) and at the
  beam-output level with `scripts/compare_beam_backends.py`; see cpu_setup.md for the commands.
