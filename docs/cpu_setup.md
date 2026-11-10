# Running on a CPU-only machine

Transcribing and scoring with the final model need no GPU, no dataset and no training. This is
what was done on a Windows 11 laptop without CUDA; the Linux and macOS equivalents are noted.
Commands are for PowerShell from the repository root.

## 1. Python environment

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r env\requirements.txt
```

The CPU wheel of torch is a tenth of the size of the CUDA one. Any torch 2.x build works; the
CUDA index in the README is for training. (`python3 -m venv .venv` and `.venv/bin/python` on
Linux and macOS.)

Beam decoding also needs pyctcdecode. It pins an old numpy that does not install on Python 3.14,
so install it without dependencies and add the two it really needs:

```powershell
.venv\Scripts\python -m pip install --no-deps pyctcdecode
.venv\Scripts\python -m pip install pygtrie hypothesis
```

Greedy decoding works without this step.

## 2. The model, language model and scoring files

Weights are never committed. `checkpoints.zip` (not in git either) holds everything inference
and scoring need; unpack it in the repository root so that these exist:

```
data/checkpoints/iteration4/best/      config.json, model.safetensors, preprocessor_config.json, vocab.json
data/language_model/                   train_5gram.klm, train_5gram.arpa, unigrams.txt, train.txt
data/interim/extracted/blind/subtask2_blindtest_wReadme/inference/
                                       punct.mapping.txt, hin2engwords.txt, compute_wer.py
```

`src.inference` tells you which file is missing if the layout differs. `MUCS_DATA_ROOT` moves
the whole `data/` tree elsewhere, as for training.

## 3. The language model: KenLM or the ARPA reader

Beam search scores words with the 5-gram model. Two backends read it; `--lm-backend auto`
(the default) takes the first that works:

1. **`kenlm`**, the C++ package, reads `train_5gram.klm`. PyPI has no Windows wheel
   (`kenlm`, `pypi-kenlm` and `kenlm-windows` were tried), and `pip install kenlm` compiles
   it, which needs a C++ compiler *and* the Windows SDK headers. On the machine used here the
   Visual Studio Build Tools lacked the SDK and the build failed; `scripts/build_kenlm.sh` is
   an alternative for MinGW-w64 that is wired to Python 3.14 and `.venv`, and was not used.
   On Linux, macOS and WSL it installs in a minute:

   ```bash
   sudo apt install python3-dev zlib1g-dev   # Debian/Ubuntu; Homebrew users need nothing extra
   pip install kenlm
   ```

2. **The ARPA reader** (`src/arpa_lm.py`, `--lm-backend arpa`) reads `train_5gram.arpa` in pure
   Python. Same model, same scores: see the validation below. It costs 3.5 s and about 250 MB
   to parse the file when a beam `Transcriber` is created, and nothing per utterance.

Nothing needs to be chosen on Windows: with `kenlm` absent, `auto` uses the ARPA reader and the
output says `"lm_backend": "arpa"`.

### How the ARPA reader was validated

Real KenLM ran under WSL (Ubuntu) in its own virtual environment; everything else ran on
Windows. The results are committed under `results/cpu_eval/lm_check/`.

```powershell
# Windows: score 2,003 sentences (train sample, shuffled copies, OOV words) with the ARPA reader
.venv\Scripts\python -m src.arpa_lm --count 500 --seed 0 --output results\cpu_eval\lm_check\arpa_scores.jsonl
# WSL: score the same sentences with real KenLM from the .klm
wsl -d Ubuntu -- bash -lc "cd /mnt/c/<repo> && ~/kenlm_venv/bin/python scripts/kenlm_reference.py --klm data/language_model/train_5gram.klm --sentences results/cpu_eval/lm_check/arpa_scores.jsonl --output results/cpu_eval/lm_check/kenlm_scores.jsonl"
# Windows: compare per-word log10 scores
.venv\Scripts\python scripts\compare_lm_scores.py --arpa results\cpu_eval\lm_check\arpa_scores.jsonl --kenlm results\cpu_eval\lm_check\kenlm_scores.jsonl --output results\cpu_eval\lm_check\comparison.json
```

Result: 26,145 word scores, largest difference 6.3×10⁻⁷ (float32 rounding), none over the
10⁻⁴ tolerance.

The beam search itself was then compared. The real model's log-probabilities for a test clip
were saved from Windows (`results/cpu_eval/lm_check/beam_inputs/`), and in WSL, where both
backends are available, `scripts/compare_beam_backends.py` decoded them and 60 perturbed
variants (noise, dropped frames, repeated frames) with each backend under iteration 4's tuned
settings:

```bash
pip install --no-deps pyctcdecode && pip install numpy pyyaml pygtrie hypothesis   # in the WSL venv
python scripts/compare_beam_backends.py --logprobs results/cpu_eval/lm_check/beam_inputs/tts_en_logprobs.npy --vocab results/cpu_eval/lm_check/beam_inputs/vocab.json --decode-config configs/decode_iteration4.yaml --output results/cpu_eval/lm_check/beam_equivalence.json
```

Result: 61 cases, 0 disagreements (`results/cpu_eval/lm_check/beam_equivalence.json`). Per
utterance the two backends took the same time (0.25 s); loading took 0.5 s for KenLM and 8.6 s
for the ARPA reader on WSL's slower view of the Windows disk.

This shows that the ARPA backend gives the same text as KenLM for the same log-probabilities.
It does not re-run the full evaluation: the committed WER figures were produced with KenLM on a
GPU machine and have not been reproduced here, because the dataset is not on this machine.

## 4. Check the installation

```powershell
.venv\Scripts\python -m pytest tests -q
```

Expected on a CPU without `kenlm`: all tests pass, 5 skipped (the kenlm comparisons). With the
checkpoint unpacked, one test loads the real model, which takes a few seconds.

Then transcribe something. There is no sample recording in the repository (MUCS audio is
CC BY-SA and not committed); any WAV, MP3, FLAC or OGG file will do, and the command tells you
if the file is unusable.

```powershell
.venv\Scripts\python -m src.inference --audio path\to\audio.wav --json
.venv\Scripts\python -m src.inference --audio path\to\audio.wav --decoder beam --json
```

## 5. What it costs on a CPU

Measured on the laptop used here (Windows 11, no CUDA), with a 6.7 s clip:

| Step | Time |
|---|---|
| Import torch and transformers, load the 1.2 GB checkpoint | 5–40 s, depending on the disk cache |
| Parse the ARPA model and build the beam decoder (beam only) | 5 s |
| Greedy transcription of the clip | 3.5 s |
| Beam transcription of the clip (beam 200) | 5.6 s |

Loading dominates, so transcribe many files in one process: `--audio a.wav b.wav`,
`--manifest list.csv --output predictions.jsonl`, or `Transcriber` from Python. Memory: about
2 GB for greedy, 2.3 GB with the ARPA reader.

## 6. Troubleshooting

| Message | Meaning |
|---|---|
| `checkpoint directory not found` | `checkpoints.zip` is not unpacked, or `MUCS_DATA_ROOT` points elsewhere |
| `the kenlm package is not installed` | you asked for `--lm-backend kenlm`; use `auto` or `arpa` |
| `KenLM beam decoding requires pyctcdecode` | step 1, second block |
| `ARPA language model not found` | `train_5gram.arpa` must sit beside `train_5gram.klm` |
| `--device cuda was requested but CUDA is not available` | use `--device cpu` or `auto` |
| `audio is too short` / `too long` / `NaN or infinite samples` | the file is not transcribable as is; `--start`/`--end` cut long files |
| `Found entries of length > 1 in alphabet` | a pyctcdecode notice about the `<pad>`-style labels; harmless |
