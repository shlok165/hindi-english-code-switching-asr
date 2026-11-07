# Implementation Brief

**Project:** Replicate and improve a Hindi–English code-switched ASR system.
**Prepared:** 4 November 2026, from the research in `literature/`, `datasets/`, `replication_candidates/` and `improvement_ideas/`.
**Status:** research phase complete; **no experiment code has been written and nothing has been trained.**

> **Correction, 7 November 2026.** Stage 1A found that parts of this brief are out of date: an official T-WER scorer does exist, `ai4bharat/indicwav2vec-hindi` is gated, a Hindi-4200 release exists on Hugging Face, no baseline hypotheses are released, and the full data fits a 6 GB GPU. See the corrections table in `docs/stage_1a_report.md`. Where they disagree, that report wins.

---

## 0. Read this first — four verified facts that change the plan

1. **The Vakyansh "Hindi-4200" checkpoint is dead.** The paper being replicated depends on it. Its bucket returns `HTTP 403 UserProjectAccountProblem — the billing account for the owning project is disabled`, bucket-wide. **Substitute `ai4bharat/indicwav2vec-hindi`** (Apache-2.0, verified present with `pytorch_model.bin` and a CTC vocab). This makes the work a reproduction of the *approach*, not of the number.
2. **There is no T-WER scorer in the MUCS baseline repo.** Only standard Kaldi `score.sh`. T-WER must be implemented from its definition, and the organisers' version used *manual* annotation of the blind references, so yours is an approximation. Label it as such everywhere.
3. **Do not evaluate only on the MUCS test set.** Test–train sentence overlap is 33.9 %; blind–train overlap is 2.1 %. **The blind test set is the headline number.**
4. **No paper in this literature reports a random seed or run-to-run variance.** Most of the effects we are chasing are 1–3 WER. Fix seeds and run the final configuration 2–3 times, or you cannot tell an improvement from noise.

---

## 1. The chosen paper

**Primary system to match:** Harveen Singh Chadha, Priyanshi Shah, Ankur Dhuriya, Neeraj Chhimwal, Anirudh Gupta, Vivek Raghavan, *Code Switched and Code Mixed Speech Recognition for Indic languages*, arXiv:2203.16578 (2022) — "Approach C2". PDF: `literature/pdfs/chadha_2022_arxiv_cs_codemixed_asr_indic.pdf`.
Leaderboard identity: team **Ekstep**, MUCS 2021 Subtask-2.

**Benchmark / task definition (peer-reviewed):** Anuj Diwan et al., *MUCS 2021: Multilingual and code-switching ASR challenges for low resource Indian languages*, Interspeech 2021, https://doi.org/10.21437/Interspeech.2021-1339. PDF: `literature/pdfs/diwan_2021_mucs2021_challenge.pdf`.

**Repositories.**
- Baseline recipe (Kaldi + ESPnet, branch `master`): https://github.com/navana-tech/baseline_recipe_is21s_indic_asr_challenge — the Hindi code-switching baseline lives at `is21-subtask2-kaldi/hindi_baseline/`, and the only scorer is `is21-subtask2-kaldi/hindi_baseline/local/score.sh`.
- Chadha et al.'s fine-tuning platform: https://github.com/Open-Speech-EkStep/vakyansh-wav2vec2-experimentation (last pushed 22 Sep 2022). **Code is live; its checkpoints are not.**
- Model index (links broken, kept for provenance): https://github.com/Open-Speech-EkStep/vakyansh-models

---

## 2. Data and how to get it

**Corpus: MUCS 2021 Subtask-2, Hindi–English.** Licence **CC BY-SA 4.0**. No request form; direct download.

| Split | Hours | Utterances (unique sentences) | Download |
|---|---|---|---|
| train | 89.86 | 44,249 | `https://www.openslr.org/resources/104/Hindi-English_train.tar.gz` (7.3 GB) |
| test | 5.18 | 2,890 | `https://www.openslr.org/resources/104/Hindi-English_test.tar.gz` (443 MB) |
| blind test | 6.24 | 3,831 | `https://www.openslr.org/resources/104/subtask2_blindtest_wReadme.tar.gz` (1.0 GB, both pairs) |

- Index page with mirrors: https://www.openslr.org/104/
- Official data page: https://navana-tech.github.io/MUCS2021/data.html
- **Critical:** the Hindi–English data was **replaced on 16 February 2021**. Discard any earlier copy.
- Audio: 16 kHz, 16-bit, mono. Domain: Spoken Tutorial technical lectures. Vocabulary 17,830 words (the OpenSLR page says 17,877 — minor discrepancy, note it).
- Blind Hindi–English references are **34.2 % English words**; OOV is 12.5 % (test) and 19.6 % (blind).
- **No speaker labels exist.** The "520 speakers" figure assumes one speaker per tutorial, so a speaker-disjoint split is not constructible. Do not claim one.

**Order of work:** download **test + blind first** (1.4 GB) and get scoring working before pulling the 7.3 GB training archive.

**Known defect — alignment.** Part of the training data has misaligned transcript timestamps. The organisers report both unaligned (UnA) and force-realigned (ReA) results; realignment moves Hindi–English GMM-HMM test WER from 44.30 to 31.56. **Plan to realign**, and record which condition every number belongs to.

---

## 3. Preprocessing

1. **Audio:** keep 16 kHz mono; normalise to float32 in [-1, 1]. No resampling needed (unlike MUCS Subtask-1, which is 8 kHz).
2. **Segmentation:** use the provided Kaldi `segments` file for sentence timestamps. Force-realign at tutorial level and recompute segment boundaries (the ReA condition) before training.
3. **Text normalisation — document every rule:** Unicode NFC; strip punctuation; collapse whitespace; decide and record a single casing convention for Latin-script tokens. Keep an un-normalised copy of the references for scoring.
4. **Script handling (this is improvement #2, run it as an ablation):** build an English-word lexicon (CMUdict ∪ corpus vocabulary), detect English tokens written in Devanagari, and map them to one canonical script. Train both the all-Latin and the all-Devanagari variant — **we do not know which wins**.
5. **Tokenizer:** character/CTC vocabulary built from the normalised training transcripts. Record the final vocabulary size; it will differ between the two script variants.
6. **Splits:** train on the 89.86 h train set with a held-out validation slice (Biswas et al. reserve 4 h from train — follow that and state it). Report on **blind**; if you also report test, report it separately and note the 33.9 % overlap.

---

## 4. Model and hyperparameters

**Architecture:** wav2vec 2.0 encoder + linear CTC head.

**Encoder:** `ai4bharat/indicwav2vec-hindi` (Hugging Face, Apache-2.0). It is already a Hindi CTC ASR model, so this is a warm start, not a raw SSL checkpoint — adapt the output vocabulary accordingly.
Alternatives for the bake-off: `facebook/wav2vec2-xls-r-300m`, `openai/whisper-small`, `facebook/mms-1b-all` (**CC-BY-NC-4.0 — non-commercial, flag it if used**). Avoid `ai4bharat/indicwav2vec_v1_vakyansh` (HTTP 401) and verify `ai4bharat/indicwav2vec_v1_hindi` loads (0 downloads) before relying on it.

**Specified by the paper:**

| Item | Value | Source |
|---|---|---|
| Loss | CTC on a linear layer over the context network | Chadha et al. |
| Language model | KenLM 5-gram trained on IndicCorp | Chadha et al. |
| Decoding | Beam search, **beam width 3000**, LM vocabulary = top **500,000** words by frequency | Chadha et al. |
| Toolkit | fairseq (their Vakyansh wrapper); 🤗 Transformers is an acceptable substitute — **say so** | Chadha et al. |

**NOT specified — you must choose and document these.** This is the honest cost of the paper's 4-page methods section:

| Item | Status | Suggested starting value |
|---|---|---|
| Learning rate | **absent** | 1e-4 with tri-stage schedule (10 % warmup / 40 % hold / 50 % decay) |
| Batch size | **absent** | max ~1.6 M samples per batch via gradient accumulation |
| Training steps / epochs | **absent** | 20–40 k updates; early-stop on validation WER |
| Optimiser | **absent** | AdamW, β=(0.9, 0.98), weight decay 0.01 |
| Encoder freeze schedule | **absent** | freeze the feature extractor throughout; freeze the transformer for the first 2–4 k updates |
| SpecAugment / masking | **absent** | wav2vec 2.0 defaults (mask prob 0.065, mask length 10) |
| LM weight / word insertion penalty | **absent** | tune on the validation slice — do not tune on test or blind |
| Random seed | **absent (in every paper)** | fix it; run the final config with ≥3 seeds |
| Hardware / runtime | **absent** | — |

**These suggested values are my defaults, not the paper's.** Record them as guesses in the results table.

---

## 5. Evaluation

**Metrics to report, on the blind test set:**
1. **WER** — standard; use the Kaldi scorer or `jiwer` and verify the two agree on the baseline hypotheses.
2. **T-WER** — "counts an English word in the reference text as correctly predicted if it is in English transliterated form in the native script" (Diwan et al.). **You must implement this**; it is not in the repo. Needs an English-word lexicon plus a Devanagari↔Latin transliteration map. Mark it **approximate**.
3. **CS-WER or CBA** — only if you do improvement #3. CS-WER (Sharma et al. 2020) `= 1 − N/M` over words on both sides of switch points; CBA (Biswas et al. 2025) = share of correctly recognised switch-point bigrams. They run in opposite directions; pick one and define it in the report.
4. **CMI** — report the code-mixing index of the evaluation data when describing the corpus, not as a system metric.

**Do not report MER** unless you also report WER — MER is only used by Biswas et al. and is not leaderboard-comparable.

---

## 6. Target numbers

All on the **MUCS Subtask-2 Hindi–English blind test set**, from the official leaderboard (https://navana-tech.github.io/MUCS2021/leaderboard.html):

| System | WER | T-WER |
|---|---|---|
| Official baseline (= Diwan et al. GMM-HMM, unaligned) | **25.53** | **23.80** |
| Chadha et al. / Ekstep — the system being reproduced | **21.77** | **20.75** |
| Best on WER: JHU-CLSP/GoVivace (Wiesner et al.) | 16.43 | 15.51 |
| Best on T-WER average: CSTR (Klejch et al.) | 20.41 | 15.64 |

**Success criterion.** Because the encoder is substituted and the T-WER scorer is self-implemented, **do not treat 21.77 as a reproduction target.** Success = **beat 25.53 WER / 23.80 T-WER on the blind set and land in the 21–24 WER band** occupied by mid-table leaderboard systems. State this criterion in the report *before* running.

**Sanity gate.** Before training anything, score the released baseline hypotheses (or re-run the Kaldi baseline) and confirm your scorer reproduces 25.53 WER. If it does not, stop and fix the scorer.

---

## 7. First improvement to implement

**Improvement #2 from `improvement_ideas/ideas.md`: script normalisation of the training targets plus a transliteration-aware lexicon.**

Chosen first because (a) it is the cheapest real gain, (b) it is the most Hindi–English-specific intervention available, (c) it forces you to build the T-WER scorer, which everything else needs, and (d) Wiesner et al. measured roughly 2 WER from the lexicon/transliteration component **on this exact corpus** (NoDup Hindi–English: 26.6 → 26.4 with a transliteration map → 25.5 with a learned lexicon).

**Ablation ladder — one variable at a time, same seed:**

| Run | Configuration | Purpose |
|---|---|---|
| A0 | Baseline hypotheses, scored by our scorer | Scorer validation (must give 25.53 / 23.80) |
| A1 | IndicWav2Vec + CTC, raw transcripts, no LM | Our reference point |
| A2 | A1 + KenLM 5-gram | Isolates the LM contribution |
| A3 | A2 + script normalisation → all-Devanagari | Script variant 1 |
| A4 | A2 + script normalisation → all-Latin | Script variant 2 |
| A5 | Better of A3/A4 + homophone merging | Srivastava & Sitaram's step |
| A6 | A5 with 3 seeds | Variance — is the gain real? |

**Expected: 1–3 WER over A2.** Direction of normalisation is genuinely unknown; that is the experiment.

Then improvement #1 (encoder bake-off) on top of the winning pipeline, then #4 (borrowed-word vs true-switch stratified analysis, nearly free), then #3 if time allows. **Drop #5** (concatenated tokenizer) — it needed 10,000 h of synthetic data in the paper it comes from.

---

## 8. Suggested repository structure

```
hindi-english-cs-asr/
├── README.md                     # task, data licence, how to reproduce, results table
├── env/
│   ├── environment.yml           # pinned conda env
│   └── requirements.txt          # pinned pip (transformers, datasets, torchaudio, jiwer, kenlm, indic-nlp-library)
├── configs/
│   ├── data.yaml                 # paths, split definitions, sample rate
│   ├── model_indicw2v.yaml       # encoder, CTC head, freeze schedule
│   ├── train_a1.yaml … a6.yaml   # one config per ablation run; every guessed value marked `# GUESSED`
│   └── decode.yaml               # beam 3000, LM weight, word insertion penalty
├── data/                         # gitignored
│   ├── raw/                      # OpenSLR tarballs
│   ├── interim/                  # realigned segments, manifests
│   └── processed/                # normalised transcripts per script variant
├── src/
│   ├── download.py               # fetch + checksum OpenSLR archives
│   ├── prepare_manifest.py       # Kaldi segments → HF datasets manifest
│   ├── realign.py                # tutorial-level forced alignment (the ReA condition)
│   ├── normalize_text.py         # NFC, punctuation, casing; --script {latin,devanagari,none}
│   ├── lexicon.py                # English-word lexicon, Devanagari↔Latin map, homophone merging
│   ├── train_ctc.py              # fine-tuning loop; seeds fixed from config
│   ├── decode.py                 # KenLM beam search
│   ├── metrics/
│   │   ├── wer.py
│   │   ├── twer.py               # APPROXIMATE - document the deviation from the official scorer
│   │   └── cs_wer.py             # switch-point metric (improvement #3)
│   └── analysis/
│       ├── error_by_language.py  # English vs Hindi token error rates
│       └── loanword_split.py     # borrowed words vs true switches (improvement #4)
├── scripts/
│   ├── 00_sanity_score_baseline.sh
│   └── 01_run_ablation.sh        # A0 … A6
├── results/
│   ├── results.csv               # run_id, config, seed, split, WER, T-WER, CS-WER, notes
│   └── figures/
└── docs/
    └── guessed_hyperparameters.md  # every value not specified by the paper, with justification
```

**Reproducibility checklist (enforce in code, not prose):**
- Set and log `seed` for Python, NumPy, PyTorch and the dataloader; log it in `results.csv`.
- Log the exact git commit, config hash and package versions with every run.
- Checksum the downloaded archives and record them in the README.
- Never tune on test or blind — only on the held-out validation slice.
- Record for every number: **which split** (test / blind / val), **which alignment condition** (UnA / ReA), and **which metric variant** (WER / approximate T-WER).

---

## 9. Things that will waste your time if nobody warns you

- Installing **Kaldi** just to score. You do not need it; `jiwer` plus the released baseline hypotheses is enough for the sanity gate.
- Assuming **MUCS Subtask-1 Hindi (95.05 h, 8 kHz)** is the code-switched set. It is monolingual Hindi from storybooks. The code-switched set is Subtask-2, 89.86 h, 16 kHz.
- Looking for **frame-level or word-level language tags** in MUCS. There are none. Any language label you use is derived and noisy.
- Reaching for the **Microsoft MSCS / WSTCSMC 2020** corpus for LID or diarization. It has **no Hindi–English pair** — only Gujarati-, Tamil- and Telugu-English.
- Citing **"the MUCS baseline"** without saying which. There are six Hindi–English baseline numbers in Diwan et al.; the leaderboard one is 25.53 / 23.80.
- Comparing your WER against **Biswas et al.'s 28.8** — that is MER, a different metric.

---

## 10. Open questions to settle before coding

1. **All-Latin or all-Devanagari** normalisation as the default? (Run both; but pick a default for the headline table.)
2. Is an **approximated T-WER** acceptable for the report, with the deviation documented?
3. Is substituting **`ai4bharat/indicwav2vec-hindi`** for the dead Hindi-4200 checkpoint acceptable, given it makes this a reproduction of the approach rather than of the number?
4. **Full 89.86 h or a 15 h subset** for the ablations? The subset roughly halves GPU cost but confounds ablation and final runs.
5. Is using an **arXiv preprint** as the system being replicated acceptable, given the number is independently confirmed on the official leaderboard?

---

## 11. Source documents

- `literature/papers.md` — 29 annotated papers, 12 comparability warnings, the full MUCS Subtask-2 leaderboard, PDF inventory, open questions.
- `literature/comparison_table.csv` — the same 29 papers, one row each.
- `literature/pdfs/` — 30 open-access PDFs.
- `datasets/datasets.md` — 7 Hindi–English corpora with hours, licence, access, annotation granularity, script and feasibility, plus an exclusions table.
- `replication_candidates/candidates.md` — 12 candidates scored on 5 axes, top 3 with trade-offs, 8 ruled out with reasons.
- `improvement_ideas/ideas.md` — 5 improvements with mechanism, effort, expected effect and supporting paper.
