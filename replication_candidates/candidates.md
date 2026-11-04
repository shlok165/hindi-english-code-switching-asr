# Replication Candidates — Scoring and Recommendation

**Compiled:** 4 November 2026 · Companion to `literature/papers.md` and `datasets/datasets.md`.

## Summary

- **Recommendation: replicate Chadha et al. (2022) "Approach C2" — wav2vec 2.0 + CTC + n-gram LM on MUCS 2021 Hindi–English.** Target: **21.77 % WER / 20.75 % T-WER** on the official blind test set, against the challenge baseline of **25.53 / 23.80**. It is the only strong Hindi–English system where the data, the pretrained encoder, the training code and the comparison number are *all* public and all reachable on one GPU.
- **Runner-up: Biswas et al. (2025) Whisper adaptation** — the most interesting methods and the most complete ablation ladder, but it reports MER instead of WER/T-WER, so you cannot place your result on the leaderboard, and its headline system is Whisper large-v2 (1.54 B params).
- **Third: Kumar et al. (2021) dual-script E2E** — peer-reviewed, clean ablation, leaderboard-comparable, but no code and it needs IIT-Madras's Common Label Set parser, which is a substantial unpublished dependency.
- **Explicitly ruled out:** everything built on proprietary Microsoft data (Taneja, Sharma, Srivastava ×2, Verma), everything requiring IITG-HingCoS (Ganji, Dhawan), and the whole language-diarization line (no real Hindi–English data exists).
- **The uncomfortable trade-off:** the single most replicable paper (Chadha et al.) is an **arXiv preprint with no verified peer-reviewed venue** and the thinnest methods section in the set. The project brief prioritises peer-reviewed work. My resolution is to treat the **peer-reviewed** Diwan et al. (Interspeech 2021) as the benchmark and task definition, and Chadha et al. as the *system* being matched — which is exactly how the leaderboard frames it anyway (team "Ekstep", 21.77 % WER, rank 7 of 16).

> ### ⚠ Verified blocker — the exact encoder is gone
>
> I checked, rather than assuming. **The Vakyansh "Hindi-4200" checkpoint can no longer be downloaded.** The model index at `Open-Speech-EkStep/vakyansh-models` still lists `hindi_pretrained_4kh.pt` (wav2vec 2.0 Base, 4,200 h Hindi), but the hosting bucket returns:
>
> ```
> HTTP 403  <Code>UserProjectAccountProblem</Code>
> The project to be billed is associated with an absent billing account.
> ```
>
> That is bucket-wide, not object-specific, so **every Vakyansh open model (ASR, TTS, punctuation, LID) is currently unreachable.** The GitHub code repositories are unaffected (`vakyansh-wav2vec2-experimentation`, 90 stars, last pushed 22 Sep 2022).
>
> **Consequence:** you cannot replicate Chadha et al. *exactly*. You can replicate the *approach* with a substituted Hindi wav2vec 2.0 encoder. The verified substitute is **`ai4bharat/indicwav2vec-hindi`** (Apache-2.0, 8,124 downloads, ships `pytorch_model.bin` + CTC vocab, already fine-tuned for Hindi ASR — so it is a warm start rather than a raw SSL checkpoint). This is a *different* encoder from Hindi-4200, so **21.77 WER stops being a reproduction target and becomes a reference point.** Say so explicitly in the report; do not present a substituted-encoder result as a reproduction of their number.
>
> **Also verified:** the MUCS baseline repository contains only standard Kaldi WER scoring (`is21-subtask2-kaldi/hindi_baseline/local/score.sh`). **There is no T-WER scorer in the repo**, so transliterated WER has to be implemented from the definition in Diwan et al. and will not be bit-identical to the organisers' manually annotated version.

---

## 1. Scoring

Five axes, 1–5, higher is better. **Code** = is working code public? **Data** = can I get the data? **Compute** = does it fit one GPU / Colab? **Detail** = are hyperparameters, preprocessing and splits specified? **Clarity** = are the reported numbers unambiguous enough to compare against?

| ID | Paper | Code | Data | Compute | Detail | Clarity | Total | Verdict |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| P05 | **Chadha et al. 2022** — wav2vec 2.0 CS ASR | 4 * | **5** | **4** | 1 * | **5** | **19** | **Recommended** |
| P09 | **Biswas et al. 2025** — Whisper + augmentation | 1 | **5** | 3 | 3 | 2 | 14 | Runner-up |
| P04 | **Kumar et al. 2021** — dual script E2E | 1 | **5** | 3 | 3 | **4** | 16 | Third |
| P01 | Diwan et al. 2021 — MUCS baseline | **5** | **5** | 2 | 3 | 3 | 18 | Reproduce as *scaffolding*, not as the project |
| P08 | Antony et al. 2022 — Transformer-Transducer | 1 | **5** | 3 | 2 | 1 | 12 | Not recommended |
| P02 | Wiesner et al. 2021 — JHU hybrid (best score) | 1 | 4 | 1 | 2 | 2 | 10 | Not recommended |
| P03 | Klejch et al. 2021 — CSTR hybrid | 1 | 2 | 1 | 1 | 2 | 7 | Not recommended |
| S07 | Dhawan et al. 2023 — concatenated tokenizer | **5** | 3 | 1 | 2 | 2 | 13 | Not recommended (10 k h synthetic) |
| S06 | Verma et al. 2023 — common Indic representation | 1 | 1 | 3 | 1 | 2 | 8 | Blocked (private data) |
| P13/O03 | Mishra & Prasanna — language diarization | 4 | 1 | 4 | 2 | 3 | 14 | Blocked (no real Hi-En data) |
| P12 | Ganji et al. 2020 — joint LID | 1 | 1 | 4 | 2 | 3 | 11 | Blocked (HingCoS unobtainable) |
| P06/P07/P10/S03 | Taneja / Sharma / Srivastava ×2 | 1 | 1 | 3 | 2 | 3 | 10 | Blocked (proprietary Microsoft data) |

\* Downgraded after verification: the fine-tuning **code** is live on GitHub but the **pretrained checkpoint it depends on is no longer hosted** (Code 5→4), and with the encoder substituted the already-thin hyperparameter reporting leaves nothing pinned down (Detail 2→1). It still ranks first because no other candidate has open data, open code, an open leaderboard target and single-GPU cost together.

### Axis justifications

**Code.** Only three candidates ship anything runnable. Chadha et al. point to `Open-Speech-EkStep/vakyansh-wav2vec2-experimentation` **and** published pretrained checkpoints — that is the whole pipeline. Diwan et al. publish the full Kaldi/ESPnet baseline recipe. Dhawan et al. (2023) ship their work inside NVIDIA NeMo. Mishra et al. release the diarization model code. Every other candidate states "none".

**Data.** 5 = MUCS 2021 Hindi–English, direct download from OpenSLR under CC BY-SA 4.0, no request form (see `datasets/datasets.md` §2.1). 4 = Wiesner et al., because the MUCS data is open but their NoDup/Dup split definition and their external LM crawl are not released. 3 = Dhawan et al. 2023, because the 10 k h synthetic corpus must be regenerated and their MUCS evaluation is on a privately cleaned subset. 2 = Klejch et al., who depend on ~200 h of crawled YouTube per language that cannot be redistributed. 1 = proprietary or unobtainable.

**Compute.** This is my estimate, with the basis stated.
- 4 (Chadha): wav2vec 2.0 **base**, 95 M parameters, CTC fine-tuning on 89.86 h at 16 kHz. Comparable published fine-tunes of this size run in the low tens of GPU-hours. On a single 16 GB GPU I would budget **~30–50 h wall-clock for the full 89.86 h**, or **~6–10 h on a 15 h subset**, plus KenLM training on CPU. Fits Colab if checkpointing across sessions is handled. The paper states no training time, so this is my estimate, not theirs.
- 3 (Biswas): Whisper **large-v2** at 1.54 B parameters will not fine-tune on a free Colab T4; `whisper-small` (244 M) or `medium` with LoRA will. Their stated budget — max 5 k update steps — is modest, but the model size is the binding constraint, and swapping to a smaller Whisper changes the result.
- 3 (Kumar, Antony): ESPnet Transformer / Transducer trained for 20–40 epochs on ~90 h. Feasible on one GPU but slower than CTC fine-tuning and much more sensitive to the Noam schedule.
- 2 (Diwan baseline): Kaldi GMM-HMM + TDNN LF-MMI. Not GPU-bound but CPU- and install-heavy; Kaldi is the single biggest practical obstacle in this whole list.
- 1 (Wiesner, Klejch, Dhawan 2023): Wiesner needs PyChain LF-MMI plus Librispeech-960 pre-training plus RNNLM rescoring plus MBR; Klejch needs 200 h of YouTube per language; Dhawan needs 10 k h of synthetic audio and 200 training epochs of a 120 M Conformer-RNNT.

**Detail.** Nobody scores above 3. **Not one of the 28 papers reports a random seed or run-to-run variance.** Chadha et al. score 2 because batch size, learning rate, epochs, optimiser and hardware are all absent — you get the architecture and the checkpoint but must guess the schedule. Diwan, Kumar and Biswas score 3 because they give architecture and the main schedule parameters but omit batch size, seeds and the train/validation split inside the released training hours.

**Clarity.** 5 (Chadha): a single table giving Hindi–English blind-set WER and T-WER side by side with the official baseline, and the number matches their leaderboard entry exactly (21.77 / 20.75). 4 (Kumar): per-pair dev and blind WER, matching their leaderboard entry (22.06 vs 22.0 in the paper) — the small difference is unexplained. 3 (Diwan): six different Hindi–English baseline numbers, all legitimate but easily conflated. 2 (Biswas): MER only, with MER's tokenisation deferred to a citation. 1 (Antony): the reference number they improve over cannot be traced to any single system in the challenge paper, and their own split is never named.

---

## 2. Top 3, with the trade-offs

### 1st — Chadha et al. (2022), *Code Switched and Code Mixed Speech Recognition for Indic languages*

**What you would replicate.** "Approach C2": take a wav2vec 2.0 encoder pretrained on unlabelled Hindi, fine-tune it with CTC on the 89.86 h MUCS Hindi–English training set, and decode with a KenLM 5-gram built from IndicCorp text (beam width 3000, top 500 k words). Report WER and T-WER on the official blind test set.

**Target numbers.** Hindi–English blind: **WER 21.77, T-WER 20.75** (baseline 25.53 / 23.80). Independently confirmed by the official leaderboard entry for team *Ekstep*.

**Why it wins.**
- Every ingredient is downloadable: the corpus (OpenSLR SLR104, CC BY-SA 4.0), the fine-tuning code (`vakyansh-wav2vec2-experimentation`), the pretrained encoders (`vakyansh-models`), and KenLM.
- The comparison is against an *official blind test set with a public leaderboard*, so your result is placeable among 16 real systems rather than against a number you have to trust.
- It isolates one clean scientific claim — a monolingual Hindi SSL encoder beats multilingual XLSR-53 on this pair — which is a replicable finding, not just a leaderboard position.
- 95 M parameters and CTC loss is the cheapest route to a competitive number on this task.

**What you would have to guess (and this is the honest cost).**
- **Learning rate, schedule, batch size, number of updates, freeze schedule, optimiser.** All absent. Standard wav2vec 2.0 CTC practice gives sensible defaults, but these choices are worth several WER points, so your reproduction may land anywhere in roughly **20–25 WER** rather than exactly 21.77.
- **The "Hindi-4200" checkpoint is gone.** See the verified blocker in the summary: the Vakyansh hosting bucket returns HTTP 403 `UserProjectAccountProblem` (billing account absent), bucket-wide. You must substitute **`ai4bharat/indicwav2vec-hindi`** (Apache-2.0, verified present with `pytorch_model.bin` and a CTC vocab). `ai4bharat/indicwav2vec_v1_vakyansh` returns HTTP 401 and `ai4bharat/indicwav2vec_v1_hindi` has 0 downloads — verify whichever you pick actually loads before building on it.
- **T-WER has no released scorer — confirmed.** The baseline repo ships only Kaldi `score.sh` (`is21-subtask2-kaldi/hindi_baseline/local/score.sh`). T-WER must be reimplemented from the definition, which needs an English-word lexicon plus a Devanagari↔Latin transliteration map. The organisers produced their reference by **manual annotation** of the blind set, so your T-WER will be an approximation and should be labelled as such.
- **Text normalisation and the LM text.** IndicCorp is large and the exact subset, tokenisation and casing rules are unstated.

**Honest expectation.** With the original encoder available I would have predicted **within 1–3 WER of 21.77**. With a substituted encoder and a self-implemented T-WER scorer, I will not predict a specific gap — that would be a number I cannot justify. The defensible success criterion is: **beat the official baseline of 25.53 WER / 23.80 T-WER on the blind set, and land in the 21–24 WER region occupied by the mid-table leaderboard systems.** Treat 21.77 as a reference point, not as a reproduction target, and say so in the report.

**Why it is only an arXiv preprint.** No peer-reviewed venue could be verified; the paper itself says "submitted to Interspeech 2022". Its leaderboard entry is independently verifiable on the official challenge site, which is the stronger evidence here. Frame the project as *replicating the Ekstep MUCS 2021 Subtask-2 system described in Chadha et al. (2022), benchmarked against Diwan et al. (Interspeech 2021)*.

### 2nd — Biswas et al. (2025), *Adapting Whisper for low-resource Hindi–English Code-Mix speech*

**Trade-off.** The best *ideas* in the whole set and the only switch-point-level metric at scale (Code-Switch Bigram Accuracy), on the same open MUCS data. Two of its findings are cheap to re-test and genuinely surprising: prompting Whisper with `<|hi|>` instead of `<|en|>` is worth 7.3 MER points for free (44.7 → 37.4), and 31 % of the gain is reachable with *no real code-mixed audio at all*.

**Why not first.**
- **No code.** Everything — the on-the-fly concatenation algorithm, the Llama prompting, the Parler-TTS rendering — must be rebuilt.
- **MER only.** You cannot place the result on the leaderboard or compare it to any of the other 12 primary papers. For a project whose deliverable is a comparison, that is a real cost.
- **Whisper large-v2 (1.54 B)** is outside free-Colab reach; dropping to `whisper-small` makes the numbers incomparable to the paper.
- Three generative dependencies (Llama-3.3-70B, Parler-TTS, WhisperX) each add their own unreported settings.

**When to prefer it.** If you have a 24 GB+ GPU and care more about an interesting method than about leaderboard placement.

### 3rd — Kumar et al. (2021), *Dual Script E2E Framework*

**Trade-off.** Peer-reviewed at Interspeech, leaderboard-comparable (22.06 WER / 20.97 T-WER on the blind set), a clean and interesting idea — decode Common Label Set phonemes and native script from two heads simultaneously so Hindi–English and Bengali–English can share acoustic data — and a well-structured ablation (Systems M–Q).

**Why not higher.**
- **No code**, and the **unified parser / Common Label Set** grapheme-to-phoneme mapping is an IIT-Madras in-house resource that the paper does not release. Rebuilding the CLS mapping for Devanagari plus an English g2p path is a project in itself.
- Its dev-set baselines correspond to the organisers' *unaligned* row, so its "5 % improvement" is against a different baseline condition than the one most other papers quote.
- Its own paper reports 22.0 on the blind set while the leaderboard records 22.06 — a small unexplained discrepancy.
- The dual-script model is *worse* than baseline on the hi-en dev set (28.9 vs 27.7) and only better on blind, so the mechanism is not cleanly established.

---

## 3. Ruled out, with reasons

| Paper | Blocking reason |
|---|---|
| Taneja et al. 2019; Sharma et al. 2020; Srivastava & Sitaram 2018; Srivastava et al. 2019 | Proprietary Microsoft Hindi–English corpora (46–47 h CS, 353 h Hindi, 133 h English). **Unobtainable** — the numbers can never be reproduced, only the methods re-implemented on other data. |
| Verma et al. 2023 | The Hindi–English result is on a private 136 h call-centre corpus; fine-tuning hyperparameters are in an appendix absent from the proceedings PDF. |
| Ganji et al. 2020 (LID); Dhawan et al. 2020 (target-set reduction); Ganji et al. 2019 (corpus) | All require **IITG-HingCoS**, for which no public download link exists. Also, the test-set size is internally inconsistent across these papers (100 vs 1,152 sentences). |
| Mishra & Prasanna 2024 / Mishra et al. 2023 (language diarization) | The code is public and the method is the best available, but **there is no real Hindi–English language-diarization data**. Their Hindi–English evaluation is single-speaker synthetic TTS; their real-data evaluation has no Hindi. Replicating means first *building* a dataset, which converts a replication project into a data-collection project. |
| Wiesner et al. 2021 | Best published number (16.43 WER), but needs PyChain LF-MMI hybrid training, Librispeech-960 pre-training, a 790 K–34 M-word external text crawl, a learned lexicon, RNNLM rescoring and MBR decoding — and their reported 19.5 % is on a private NoDup split. Highest-effort, lowest-reproducibility combination in the set. |
| Klejch et al. 2021 | Depends on ~200 h of crawled YouTube audio per language that cannot be redistributed, plus a custom OpenFST/OpenGRM phoneme-mapping tool that is not released. |
| Dhawan et al. 2023 | Excellent code (NeMo), but requires generating 10,000 h of synthetic code-switched audio per language pair and 200 epochs of a 120 M Conformer-RNNT. Far outside single-GPU reach. Its MUCS evaluation is also on a privately cleaned 5 h subset, so the 28.78 WER is not leaderboard-comparable. |
| Diwan et al. 2021 | Not ruled out — but reproducing a published *baseline* is scaffolding, not a contribution. Do reproduce the scoring pipeline from this repo; don't make it the project. |
| Antony et al. 2022 | No code, incomplete hyperparameters, internally inconsistent data volumes, and a reference number that cannot be traced to a single system in the source paper. You would not know what you had replicated. |

---

## 4. What "success" should mean for the recommended project

1. **Reproduce the scoring pipeline first.** Score the published baseline hypotheses (or re-run the Kaldi GMM-HMM baseline) and confirm you get 25.53 WER / 23.80 T-WER on the Hindi–English blind set. If your scorer disagrees with the official baseline, nothing downstream is comparable.
2. **Then reproduce the approach** with `ai4bharat/indicwav2vec-hindi` substituted for the unavailable Hindi-4200 encoder. **Success = beating 25.53 WER / 23.80 T-WER on the blind set and landing in the 21–24 WER band**; 21.77 is a reference point, not a target, because the encoder differs. Document every hyperparameter you had to guess.
3. **Report on the blind set, and report Dup/NoDup separately** if you also report test-set numbers — test–train sentence overlap is 33.9 %.
4. **Fix a seed and run at least 2–3 seeds for the final configuration.** No paper in this literature does this, so it is a genuine contribution on its own: it tells you whether your improvement in §`improvement_ideas/ideas.md` is real.

---

## 5. Open questions for this decision

1. ~~Is the Vakyansh Hindi-4200 checkpoint still downloadable?~~ **Answered: no.** The bucket is dead (billing disabled). Open question becomes: is substituting `ai4bharat/indicwav2vec-hindi` acceptable, given it makes this a reproduction of the *approach* rather than of the number?
2. ~~Does the MUCS baseline repository include a T-WER scorer?~~ **Answered: no, WER only.** Open question becomes: is an approximated T-WER (own transliteration map) acceptable for the report, with the limitation stated?
3. Is the project allowed to use an **arXiv preprint** as the system being replicated, given the brief's preference for peer-reviewed work, when the number is independently confirmed on the official leaderboard?
4. Would you rather optimise for **leaderboard comparability** (Chadha) or for **methodological interest** (Biswas)? The two pull in opposite directions here.
