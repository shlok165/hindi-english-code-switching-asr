# Hindi–English Code-Switched Speech Corpora

**Compiled:** 4 November 2026 · **Scope:** Hindi–English code-switched *speech* only. Text-only code-mixing benchmarks and other language pairs are listed in §4 as exclusions, because several of them are routinely mistaken for Hindi–English speech resources.

**No audio was downloaded.** Everything below is links, sizes and access instructions.

## Summary

- **Exactly one Hindi–English code-switched speech corpus is openly downloadable with a clear licence and a published baseline: MUCS 2021 Subtask-2 (OpenSLR SLR104, CC BY-SA 4.0, 89.86 h train).** Use it. Everything else is either synthetic, tiny, or has no documented public download.
- **CS-FLEURS (2025) is the one genuinely new option** — Hindi–English is covered in all three of its subsets, it is on Hugging Face, but the speech is read or TTS-synthesised, not spontaneous, and the licence is **non-commercial**.
- **No Hindi–English corpus has frame-level or 200 ms language tags.** MUCS gives sentence-level transcripts in mixed script; language identity must be inferred per *word* from the script and a lexicon. The only Hindi–English resource with explicit *word*-level language tags is the ETS dialog corpus (§2.5), which has 200 transcribed calls and no public download.
- **The big trap:** the Microsoft code-switched corpus (MSCS / WSTCSMC 2020) that almost the whole "Indian code-switched LID and language diarization" literature uses **has no Hindi–English pair** — it is Gujarati-, Tamil- and Telugu-English. See §4.

---

## 1. At a glance

| # | Corpus | Hours (Hindi–English) | Licence | Access | Annotation granularity | Script | Course-project feasible? |
|---|---|---|---|---|---|---|---|
| 2.1 | **MUCS 2021 Subtask-2 Hin-Eng** | 89.86 train / 5.18 test / 6.24 blind | CC BY-SA 4.0 | Direct download (OpenSLR SLR104) | Utterance-level transcript + segment timestamps; **no language tags** | Mixed Devanagari + Latin | **Yes — the default choice** |
| 2.2 | **CS-FLEURS** (Hindi–English subset) | Part of 294 h over 113 pairs; Hin-Eng in READ, XTTS and MMS subsets | CC BY-NC 4.0 | Direct download (HF `byan/cs-fleurs`) | Sentence transcript; matrix/embedded language known by construction | Per release (synthetic from text) | Yes, as a **supplementary / zero-shot eval** set |
| 2.3 | **IITG-HingCoS** | ~25 h (9,251 utts, 101 spk) | Not stated | **No public link found** — contact IIT Guwahati | Utterance-level transcript | Mixed | No (access unresolved) |
| 2.4 | **IIITH-HE-CM** | 11 h (8,804 utts, 142 spk) | Not stated | **No public link found** — contact IIIT Hyderabad | Utterance-level transcript | Mixed | No (access unresolved) |
| 2.5 | **ETS code-switched dialog corpus** | 555 calls collected, 200 transcribed (hours not stated) | Not stated | **No public link found** — contact ETS R&D | **Word-level `<En>`/`<Hi>` language tags** + utterance-level language label | Latin (romanised Hindi in examples) | No (too small, access unresolved) |
| 2.6 | **TTSF-LD / TTSF-LC** (synthetic) | 10,000 train + 4,000 test CS utterances (hours not stated) | Not stated | **No public link found** | Change-point and segment language labels by construction | N/A (built from TTS) | No (access unresolved) |
| 2.7 | Proprietary Microsoft Hindi–English CS sets | 46–47 h (Srivastava/Taneja/Sharma line of work) | Proprietary | **Not available** | Utterance-level | Mixed | No |

---

## 2. Corpus detail

### 2.1 MUCS 2021 Subtask-2, Hindi–English — **recommended**

- **Hours:** train **89.86 h**, test **5.18 h**, blind test **6.24 h**. 44,249 / 2,890 / 3,831 unique sentences; 520 / 30 / 35 "speakers"; vocabulary 17,830 / 3,212 / 3,527 words (the OpenSLR page states 17,877 for the training vocabulary — a minor discrepancy with the paper's Table 2).
- **Audio:** 16 kHz, 16-bit, single channel.
- **Domain:** Spoken Tutorial lectures on technical/programming topics. The code-switching is driven by technical vocabulary, so it is **largely intra-sentential English noun/term insertion into Hindi matrix sentences** — exactly the short-insertion case of interest, but in one narrow register.
- **Licence:** **CC BY-SA 4.0** (stated on the OpenSLR SLR104 page).
- **Access — direct download, no request form:**
  - Index page: https://www.openslr.org/104/
  - `Hindi-English_train.tar.gz` — **7.3 GB**
  - `Hindi-English_test.tar.gz` — **443 MB**
  - `subtask2_blindtest_wReadme.tar.gz` — **1.0 GB** (contains both pairs)
  - Mirrors are offered on the OpenSLR page (EU ×2, CN). The challenge site additionally lists `https://www.openslr.org/resources/104/Hindi-English_train.zip` and an IISc mirror at `http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi-English_train.tar.gz`.
  - Official challenge data page: https://navana-tech.github.io/MUCS2021/data.html — **note its warning that the Hindi–English and Bengali–English data were replaced on 16 February 2021; discard any earlier copy.**
- **Annotation granularity:** **utterance level only.** You get a Kaldi-style `text` file and a `segments` file with sentence timestamps derived from the tutorial audio. There are **no language tags of any kind** — not per word, not per frame.
- **Transcription script:** **mixed.** Hindi in Devanagari, English words sometimes in Latin and sometimes transliterated into Devanagari. This inconsistency is the reason the challenge defines Transliterated WER. The blind-test Hindi–English references are **34.2 % English words**.
- **Known defects, straight from the challenge paper:**
  - Transcript/timestamp **misalignment** in part of the training data. The organisers provide both unaligned (UnA) and force-realigned (ReA) results; realignment moves Hindi–English GMM-HMM test WER from 44.30 to 31.56.
  - **Test–train sentence overlap is 33.9 %**, while blind–train overlap is only 2.1 %. Reporting on the test set alone substantially overstates generalisation.
  - OOV rate 12.5 % (test) and 19.6 % (blind).
  - **Speaker labels do not exist.** The "520 speakers" figure assumes one speaker per tutorial. A genuinely speaker-disjoint split is therefore not constructible.
- **Baseline recipe:** https://github.com/navana-tech/baseline_recipe_is21s_indic_asr_challenge
- **Official leaderboard** (so you know what "good" is): https://navana-tech.github.io/MUCS2021/leaderboard.html — best Hindi–English blind score 16.43 % WER / 15.51 % T-WER; baseline 25.53 / 23.80.
- **Course-project verdict: yes.** ~8 GB download, a documented baseline, an official blind set and a public leaderboard to compare against. A 15 h subset (as used by Biswas et al. 2025) fits comfortably on a single Colab GPU.
- **Third-party mirror, use with caution:** `sajalmadan0909/mucs-hindi-english-codemix-asr` exists on Hugging Face (29 downloads, licence listed as *unknown*). Provenance is unverified — prefer OpenSLR.

### 2.2 CS-FLEURS (Hindi–English subset)

- **Source paper:** Brian Yan, Injy Hamed, Shuichiro Shimizu, Vasista Sai Lodagala, William Chen et al., *CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset*, Interspeech 2025. https://doi.org/10.21437/Interspeech.2025-2247 · PDF saved as `_work/csfleurs.pdf` (not in `literature/pdfs/` — it is a corpus reference, not a reviewed paper).
- **Scale:** the whole dataset is **294 h across 52 languages and 113 code-switched pairs**; Hindi (`hin`) is an explicitly listed matrix language, and per the paper's Table 2 the `hin`–`eng` pair appears in **all three** subsets (READ ✓, XTTS ✓, MMS ✓). It is additionally one of the four matrix languages in the 60-pair non-English test set. Per-subset totals: READ test 17 h, XTTS train 128 h / dev 15 h / test1 36 h / test2 42 h, MMS test 56 h. **The per-pair Hindi–English hours are not broken out in the paper.**
- **Licence:** **CC BY-NC 4.0** (non-commercial) per the Hugging Face dataset card.
- **Access:** `https://huggingface.co/datasets/byan/cs-fleurs` (released by the first author; 9,422 downloads, last modified 17 September 2025). A Hindi–English READ test slice has also been re-uploaded by a third party as `Trelis/cs-fleurs-hineng-read-test`.
- **Annotation granularity:** sentence-level transcript. Because the code-switched sentences are *generated* from parallel monolingual FLEURS sentences (GPT-4o prompted), the matrix and embedded language of every token is known by construction, which makes word-level language labels derivable.
- **Transcription script:** derived from FLEURS text; the paper classifies pairs as same-script or distinct-script, and Hindi–English is a **distinct-script** pair.
- **Speech type:** **read or synthetic, never spontaneous.** READ = human speakers reading synthetically generated code-switched sentences; XTTS and MMS = TTS output. Domain is Wikipedia.
- **Course-project verdict: yes, as a secondary set.** Good for a zero-shot / cross-domain generalisation check next to MUCS, and the only easy way to get clean derived word-level language labels. Do **not** use it as the headline benchmark — the acoustics are not real conversational code-switching, and the non-commercial licence restricts reuse.

### 2.3 IITG-HingCoS (Hinglish code-switching database)

- **Source:** Sreeram Ganji, Kunal Dhawan, Rohit Sinha, *IITG-HingCoS corpus: A Hinglish code-switching database for automatic speech recognition*, Speech Communication, 2019. https://doi.org/10.1016/j.specom.2019.04.007 (**paywalled**). Earlier open preprint: https://arxiv.org/abs/1810.00662
- **Hours:** the papers that use it report **9,251 Hindi–English code-switching utterances, ~25 h, 101 speakers (64 M / 37 F)**, 8 kHz, utterances 2–30 s, with a train/dev/test split of 7,015 / 1,152 / 2,136 sentences (Dhawan et al. 2020). The 2018 preprint describes a smaller snapshot: 7,005 utterances from 71 speakers, split 5,500 / 1,505. **Treat ~25 h / 101 speakers as the figure for the released corpus, and note the discrepancy.**
- **Licence:** not stated in any document I could access.
- **Access:** **no public download link was found from this environment.** The journal paper is the only authoritative description and it is paywalled. Route: contact the authors at IIT Guwahati.
- **Annotation granularity:** utterance-level transcripts. The Ganji et al. (2020) LID work derives character- and word-level language tags from these transcripts, so word-level tags are obtainable but are not distributed as a separate layer.
- **Transcription script:** mixed Hindi (Devanagari) and English; the sentences were crawled from Hindi web blogs, so the script choice reflects real online Hinglish usage.
- **Speech type:** **read.** Each of the 101 speakers read 100 predefined code-switching sentences over a toll-free phone line in 5 sessions, from varied acoustic environments — so channel and environment variation is real, but the code-switching patterns are scripted.
- **Course-project verdict: no, unless access is secured.** It is the second-most-important Hindi–English corpus and the only one with a published spoken-LID benchmark (word-level LID error 16.60 %), but there is no documented way to obtain it.

### 2.4 IIITH-HE-CM (IIIT Hyderabad Hindi–English code-mixed database)

- **Source:** Banothu Rambabu, Suryakanth V Gangashetty, *Development of IIITH Hindi English Code Mixed Speech Database*, SLTU 2018. https://doi.org/10.21437/SLTU.2018-23 (open access; PDF saved).
- **Hours:** **11 h total** — 8,804 utterances from **142 bilingual speakers (71 M / 71 F)**. Speaker-disjoint split: train 110 speakers / 6,820 utterances / ~8 h; test 32 speakers / 1,984 utterances / ~3 h.
- **Text:** 8,804 phonetically balanced sentences selected from a 9-million-sentence crawl, across lifestyle, sports, gadgets and health; 12,850 unique Hindi words and 4,390 unique English words.
- **Licence:** not stated.
- **Access:** **no public download link found.** Route: contact the Speech Lab, IIIT Hyderabad.
- **Annotation granularity:** utterance-level transcripts.
- **Transcription script:** mixed.
- **Speech type:** **read, phonetically balanced.** This is why its WERs (7–17 %) are far below any MUCS number — it is not a comparable difficulty.
- **Course-project verdict: no** (no access). Noted here because it is the only Hindi–English corpus with a clean, gender-balanced, **speaker-disjoint** split by design — which is exactly what MUCS lacks.
- **Caveat:** the paper reports two experiment sets (EXP1 and EXP2) with very different WERs and never says what distinguishes them, so its published numbers are not usable as targets.

### 2.5 ETS code-switched human–machine dialog corpus (English–Hindi)

- **Source:** Vikram Ramanarayanan, David Suendermann-Oeft, *"Jee haan, I'd like both, por favor": Elicitation of a Code-Switched Corpus of Hindi–English and Spanish–English Human–Machine Dialog*, Interspeech 2017. https://doi.org/10.21437/Interspeech.2017-1198 (open access; PDF saved as `ramanarayanan_2017_cs_dialog_corpus.pdf`).
- **Size:** **555 English–Hindi calls collected, 200 transcribed**, >4,400 transcribed tokens (2,195 unique English + 2,274 unique Hindi). **Hours are not stated.**
- **Licence / access:** neither is stated in the paper. **No public download link found.** Route: contact ETS R&D.
- **Annotation granularity:** **the richest of any Hindi–English speech corpus.** Utterance-level language labels (English 32 %, Hindi 35 %, both 33 %) *and* inline word-level language tags of the form `<En>Uh what</En> <Hi>Aapke paas kya hai</Hi> <En>available?</En>`.
- **Transcription script:** the published examples romanise Hindi in Latin script.
- **Speech type:** **spontaneous human–machine dialog** — crowd workers ordering from an automated barista through the HALEF cloud dialog system, with items designed to elicit intra-sentential switching. Audio *and* video were captured.
- **Course-project verdict: no** — 200 transcribed calls is too small to train anything and access is unresolved. Listed because it is the only Hindi–English speech resource with native word-level language tags, which makes it the natural evaluation set for a code-switch *detector* if access can be arranged.

### 2.6 TTSF-LD / TTSF-LC (synthetic Hindi–English, for language diarization)

- **Source:** Jagabandhu Mishra, S. R. Mahadeva Prasanna — *Implicit Self-supervised Language Representation for Spoken Language Diarization* (TASLP 2024, preprint arXiv:2308.10470) and *Spoken language change detection inspired by speaker change detection* (preprint arXiv:2302.05265).
- **Size:** TTSF-LD — 10,000 training and 4,000 test code-switched utterances synthesised from the IITM-TTS corpus (a single Hindi female speaker plus her English utterances), first 5 h per language for training; mean monolingual segment 6.5 s Hindi / 5.2 s English. TTSF-LC — 4,000 utterances with one to five language change points. **Hours of the generated audio are not stated.**
- **Licence / access:** not stated; **no public download link found.** The authors release code for the human-study component (https://github.com/jagabandhumishra/HUMAN-SUBJECTIVE-STUDY-FOR-LCD-and-SCD) and for the W2V diarization model (https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization), but not the TTSF corpora.
- **Annotation granularity:** segment language labels and change-point times, exact by construction.
- **Course-project verdict: no — and more importantly, do not treat its numbers as a Hindi–English benchmark.** This is **single-speaker synthetic speech with ~5–6 s monolingual segments**, i.e. the easiest possible setting. The same system that scores JER 6.38 here scores JER 60.4 on real (non-Hindi) code-switched data with ~0.5 s secondary segments. If you need a Hindi–English diarization set, building one by segmenting MUCS is more defensible than citing these.

### 2.7 Proprietary Microsoft Hindi–English CS corpora

Used by Srivastava & Sitaram (2018), Srivastava et al. (2019), Taneja et al. (2019) and Sharma et al. (2020). Figures as reported: ~46 h / 41,276 utts / 429 speakers train with 5.6 h test and 5.7 h dev (conversational, 16.6 % English tokens but 40.23 % of the unique vocabulary English); Taneja et al. add 353 h monolingual Hindi, 133 h Indian-accented English and 12.6 k code-mixed text utterances; Sharma et al. use ~12 h of real CS speech. Verma et al. (2023) use a separate 136 h Hinglish call-centre corpus.

**None of these is obtainable.** They are listed so that their published WERs are not mistaken for reproducible targets.

---

## 3. Monolingual and pretrained resources you will actually need

The best-performing Hindi–English methods all lean on out-of-domain monolingual data or a pretrained encoder. These were verified to exist:

| Resource | What for | Link | Licence (as listed) |
|---|---|---|---|
| `ai4bharat/indicwav2vec-hindi` | Hindi-specialised wav2vec 2.0 encoder | https://huggingface.co/ai4bharat/indicwav2vec-hindi | apache-2.0 |
| `facebook/wav2vec2-xls-r-300m` | Multilingual SSL encoder | https://huggingface.co/facebook/wav2vec2-xls-r-300m | apache-2.0 |
| `facebook/mms-1b-all` | Massively multilingual encoder/ASR | https://huggingface.co/facebook/mms-1b-all | **cc-by-nc-4.0 (non-commercial)** |
| `openai/whisper-small` / `openai/whisper-large-v2` | Foundation ASR for fine-tuning or prompting | https://huggingface.co/openai/whisper-small | apache-2.0 |
| Vakyansh wav2vec 2.0 models (incl. Hindi-4200) | The encoder behind the best open MUCS Hindi–English system | https://github.com/Open-Speech-EkStep/vakyansh-models | see repo |
| Mozilla Common Voice (Hindi, English) | Monolingual audio for the on-the-fly code-mix simulation recipes | https://commonvoice.mozilla.org | CC0 |

**Caution:** `ai4bharat/indicwav2vec_v1_vakyansh` returned HTTP 401 — it is gated or no longer public. Check availability before planning around it. Common Voice Hindi is small: Antony et al. (2022) report only 21 validated hours (and 16 h elsewhere in the same paper) against 2,224 h of English — check the live Common Voice stats page for current figures rather than relying on either number.

---

## 4. Explicitly excluded — commonly mistaken for Hindi–English speech corpora

| Resource | What it actually is | Why excluded |
|---|---|---|
| **Microsoft MSCS / WSTCSMC 2020 shared task** | Code-switched speech with 200 ms language tags for **Gujarati–English, Tamil–English, Telugu–English** | **Contains no Hindi–English pair.** This is the single most common mistake in this area — Liu et al. (2021), Mishra et al. (2023), Rangan et al. (2020) and the MSCS half of Mishra & Prasanna (2024) all use it, and none of them has Hindi data. |
| **MUCS 2021 Subtask-1 Hindi (95.05 h)** | **Monolingual** Hindi read speech from storybooks, 8 kHz | Not code-switched. Easy to confuse with the 89.86 h Subtask-2 Hindi–English set. |
| **GramVaani Hindi (100 h)** | Spontaneous telephone Hindi with regional variation | Monolingual Hindi, not code-switched. |
| **SEAME (192 h)** | Conversational **Mandarin–English** | Different pair. Listed because it is the corpus most Hindi–English papers benchmark *against* rhetorically. |
| **Bangor Miami (35 h), ArzEn (12 h), Soapies (14 h), ESCWA (3 h)** | Spanish–English, Egyptian Arabic–English, South African pairs, Arabic dialects | Different pairs (sizes per CS-FLEURS Table 1). |
| **LinCE, GLUECoS, Hindi–English UD treebank, Dey & Fung (2014) CS corpus** | **Text-only** code-mixing benchmarks | No audio. The Dey & Fung LREC 2014 "A Hindi-English Code-Switching Corpus" is a *text* corpus and is frequently miscited as speech. |
| **LAHAJA, Vistaar, IndicSUPERB, Shrutilipi** | Multi-accent / benchmark / large-scale **monolingual** Indic ASR resources | Useful as pretraining or accent-robustness data, not as code-switched corpora. |
| **NIT Goa bilingual code-switch corpus (70 utterances)** | **Kannada–English** | Different pair and far too small. |

---

## 5. Practical recommendation

1. **Build on MUCS 2021 Subtask-2 Hindi–English.** It is the only open, licensed, benchmarked Hindi–English code-switched speech corpus, and the official blind-test leaderboard gives you an unambiguous comparison.
2. **Download the test and blind sets first** (443 MB + 1.0 GB) and reproduce the published baseline scoring before committing to the 7.3 GB training archive.
3. **Decide the split question explicitly.** Because test–train sentence overlap is 33.9 % and blind–train overlap is 2.1 %, report on the **blind test set**, or report Dup/NoDup separately as Wiesner et al. do. A number on the test set alone is not comparable to the leaderboard.
4. **Add CS-FLEURS Hindi–English as an out-of-domain check**, and note the non-commercial licence in your report.
5. **If the project needs switch-point or language-diarization labels, you will have to derive them** — MUCS has none. The honest options are: derive word-level language tags from script plus an English lexicon (noisy, and the script is inconsistent), or force-align and emit segment labels (noisy, and the corpus has known alignment problems). Either way, state the label noise as a limitation rather than treating derived labels as ground truth.
