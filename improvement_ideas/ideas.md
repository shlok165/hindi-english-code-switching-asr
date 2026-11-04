# Improvement Ideas for the Recommended Replication

**Baseline being improved:** wav2vec 2.0 + CTC + n-gram LM on MUCS 2021 Subtask-2 Hindi–English, evaluated on the official blind test set (Chadha et al. 2022, "Approach C2"; encoder substituted with `ai4bharat/indicwav2vec-hindi` because Hindi-4200 is no longer hosted — see `replication_candidates/candidates.md`).

**Reference points:** official baseline **25.53 WER / 23.80 T-WER**; Chadha et al. **21.77 / 20.75**; best leaderboard entry **16.43 / 15.51** (JHU-CLSP/GoVivace).

## Summary

| # | Change | Effort | Expected effect | Confidence | Main supporting paper |
|---|---|---|---|---|---|
| 1 | Encoder bake-off on one fixed split (IndicWav2Vec vs XLS-R vs Whisper-small) | **Low–med** (~15 GPU-h) | 1–4 WER spread between encoders | **High** that a spread exists; low on which wins | Chadha et al. 2022; Biswas et al. 2025 |
| 2 | Script normalisation of targets + transliteration-aware lexicon | **Low–med** (~6 GPU-h + 1 day coding) | 1–3 WER | **Medium–high** | Wiesner et al. 2021; Verma et al. 2023 |
| 3 | Switch-point-weighted synthetic augmentation for 1–2 word English insertions | **Medium** (~20 GPU-h + 2 days) | 0–2 WER overall; larger on CS-WER | **Medium** | Taneja et al. 2019; Biswas et al. 2025 |
| 4 | Borrowed-word vs true-switch stratified error analysis (+ loanword lexicon) | **Low** for analysis, med for fix | ~0 WER; high report value | **High** that the analysis is novel | Srivastava & Sitaram 2018 |
| 5 | Token-level LID auxiliary head via concatenated tokenizer | **Med–high** (~25 GPU-h) | −1 to +1 WER | **Low** — this is the weakest idea | Dhawan et al. 2023 |

**If you only do two: do #2 then #1.** #2 is the cheapest real gain and is the one most specific to Hindi–English; #1 is the controlled comparison that nobody in this literature has published.

**Weakest idea: #5** — see §5 for why I would drop it before the others.

---

## 1. Encoder bake-off on a single fixed split

**The change.** Fine-tune the *same* CTC head, on the *same* MUCS Hindi–English training data, with the *same* hyperparameters and seed, on top of four encoders: `ai4bharat/indicwav2vec-hindi` (Hindi-specialised), `facebook/wav2vec2-xls-r-300m` (multilingual SSL), `openai/whisper-small` (foundation encoder-decoder, fine-tuned rather than CTC-headed), and optionally `facebook/mms-1b-all`. Report WER and T-WER on the blind set for all four.

**Why it should help, specifically for Hindi–English.** Chadha et al. found that a *monolingual Hindi* encoder beat multilingual XLSR-53 on this exact task — which is counter-intuitive, because the task contains English. If that holds, it says the Hindi matrix language dominates the acoustic difficulty and English insertions are short enough to ride on Hindi acoustics. If it does not hold with a different Hindi encoder, Chadha's claim was checkpoint-specific. Either outcome is a result. The Indian-English accent in this corpus is also closer to Hindi phonetics than to the English in XLS-R's pretraining mix, which is a mechanism for why in-language pretraining could win.

**Effort.** Low–medium. ~4 fine-tuning runs; at ~15 h training each on a 15 h subset or longer on the full 89.86 h. Call it **10–20 GPU-hours** plus a day of plumbing. No new data, no new code beyond a config sweep.

**Expected effect and uncertainty.** I expect a **1–4 WER spread across encoders**, based on Chadha et al.'s own reported preference for Hindi-4200 over XLSR-53 and on the general size gap between a 95 M base model and a 300 M XLS-R. **I will not predict which wins** — the published evidence is one data point from an unavailable checkpoint, and Whisper-small has never been evaluated on this split at all.

**Supporting papers.** Chadha et al. 2022 (Hindi-4200 > XLSR-53 on this corpus); Biswas et al. 2025 (Whisper on the same corpus, but reported with MER). As a **cautionary** counterpoint: Jansen van Vüren & Niesler, *Improving Under-Resourced Code-Switched Speech Recognition: Large Pre-trained Models or Architectural Interventions*, Interspeech 2023, https://doi.org/10.21437/Interspeech.2023-1841 — a **five-language South African** study (isiZulu, isiXhosa, Sesotho, Setswana, Sepedi + English; bilingual sets from ~10 minutes to ~22 h). Note it concerns **language models, not acoustic encoders**, and it concludes *against* the large-pretrained-model reflex: a small adapted LSTM beat fine-tuned M-BERT by 0.17 % absolute WER in n-best rescoring with a seven-fold parameter reduction. Cite it as evidence that "bigger pretrained model" is not automatic in under-resourced code-switching — not as evidence that it wins.

**Why it is first-choice.** Open question #6 in `literature/papers.md` is precisely that no controlled encoder comparison exists on one Hindi–English split. This fills a real, citable gap for almost no methodological risk.

---

## 2. Script normalisation of the training targets + transliteration-aware lexicon

**The change.** Three parts, each independently ablatable:
1. **Normalise the training transcripts** so that every English word appears in one script. Build an English-word lexicon (CMUdict plus the corpus vocabulary), detect English tokens written in Devanagari, and map them to a single canonical form — either all-Latin or all-Devanagari. Train two variants and compare.
2. **Merge homophone pairs** created by the inconsistency, following Srivastava & Sitaram's procedure: identify vocabulary items that share a pronunciation after grapheme-to-phoneme conversion and collapse them onto the more frequent anchor.
3. **Score consistently**: implement T-WER from the definition in Diwan et al. and report WER, T-WER and your own normalisation-consistent WER side by side.

**Why it should help, specifically for Hindi–English.** This is the single most Hindi–English-specific problem in the literature and it is quantified: the MUCS blind references are **34.2 % English words**, and **over 50 % of all errors fall on them** for every baseline architecture. Srivastava & Sitaram found 442 mergeable vocabulary pairs in a 19 k-word Hindi–English vocabulary, of which 154 were cross-script. Every one of those pairs is a token the CTC head must learn to spell two ways from identical acoustics — pure label noise. Wiesner et al. showed the effect concretely: adding a transliteration map moved their Hindi–English NoDup WER from 26.6 to 26.4, and a *learned* lexicon took it to 25.5.

**Effort.** Low–medium. One day to build the lexicon and normaliser, then 2–3 fine-tuning runs (~6 GPU-hours if you reuse a warm checkpoint). The T-WER scorer is needed anyway, since the MUCS repo does not ship one.

**Expected effect and uncertainty.** **1–3 WER**, with higher confidence than any other idea here, because Wiesner et al. measured ~2 WER from the lexicon/transliteration component alone on this exact corpus. The uncertainty is in direction of normalisation: all-Devanagari keeps one script and one tokenizer but destroys the Latin forms the T-WER scorer rewards; all-Latin matches T-WER but romanising Hindi is lossy. **I genuinely do not know which wins — that is the experiment.**

**Supporting papers.** Wiesner et al. 2021 (transliteration map + learned lexicon, measured on MUCS Hindi–English); Srivastava & Sitaram 2018 (homophone merging and poWER, with the 442/154 breakdown); Verma et al. 2023 (Common Indic Representation, the most aggressive version of this idea); Klejch et al. 2021 (phoneme-mapping English into the Hindi inventory, top T-WER on the blind set).

---

## 3. Switch-point-weighted synthetic augmentation for short English insertions

**The change.** Generate extra code-mixed training audio that *over-represents one- and two-word English insertions*, rather than uniformly simulating code-switching. Concretely: (a) measure the empirical distribution of English-insertion span lengths in the MUCS training transcripts; (b) force-align the training audio; (c) splice monolingual Hindi (Common Voice / IndicSUPERB) with single English words from Common Voice English at the phone-transition points, with the span-length distribution deliberately skewed toward length 1–2; (d) mix this synthetic audio with real data rather than simply appending it.

**Why it should help, specifically for Hindi–English.** The project brief names very short English insertions as a target, and the MUCS training set contains 176,563 switch points across 606,456 words — frequent, but dominated by whatever span lengths naturally occur. Taneja et al. showed that *how* you choose splice points matters: phone-transition-aware splicing (SynPT) beat naive concatenation consistently (e.g. 60.89 → 59.05 dev WER at 100 h of synthetic data). Sharma et al. showed that *mixing* synthetic with real beats appending it. Biswas et al. showed on-the-fly concatenation is worth real CBA gains at switch points. Nobody has combined these with a deliberate bias toward short insertions.

**Effort.** Medium. Forced alignment plus a splicing pipeline is ~2 days of work; then ~20 GPU-hours for the augmented runs. Everything needed is open (MUCS, Common Voice, torchaudio/MFA).

**Expected effect and uncertainty.** **0–2 WER overall, but a larger effect on switch-point metrics.** Overall WER is dominated by long monolingual Hindi stretches, so a switch-point intervention is structurally limited in how much WER it can move — this is exactly why you must report **CS-WER (Sharma et al.'s definition) or CBA (Biswas et al.'s) alongside WER**, or you will measure a real improvement as noise. Confidence medium: the three component findings are each published, but their combination on this corpus is untested.

**Supporting papers.** Taneja et al. 2019 (SynSL/SynPT splicing and CM-WER); Sharma et al. 2020 (mixup of synthetic and real, CS-bias loss, CS-WER); Biswas et al. 2025 (on-the-fly CS simulation, CBA at switch points).

---

## 4. Borrowed words versus true switches: stratified error analysis

**The change.** Partition the English tokens in the MUCS Hindi–English blind references into (a) **established loanwords** now part of everyday Hindi (*train*, *phone*, *computer*, *file*) and (b) **genuine insertions** (technical terms, longer English phrases). Build the loanword list from frequency in monolingual Hindi text plus the Devanagari-script rate in the corpus — a word written in Devanagari most of the time is behaving as a borrowed Hindi word, not as a switch. Then report WER, substitution rate and deletion rate separately for the two classes, and test whether adding Devanagari pronunciations for loanwords to the lexicon helps.

**Why it should help, specifically for Hindi–English.** This is the distinction the project brief singles out and that **no paper in the 28 reviewed actually evaluates against**. Srivastava & Sitaram got closest: of 442 mergeable pairs, 288 were within-language spelling variants and 154 were cross-script borrowings — but they report an aggregate 1.2 % relative WER gain, not a per-class error rate. The mechanism for a real fix is narrow but concrete: loanwords should probably be modelled with Hindi phonology and Devanagari orthography, and true switches with English phonology, and current systems treat both the same way.

**Effort.** Low for the analysis (a day of scripting on top of an existing decode). Medium if you also build the loanword-aware lexicon and retrain.

**Expected effect and uncertainty.** **Essentially zero WER improvement expected from the analysis itself**, and I would be surprised if the lexicon intervention moved overall WER by more than ~0.5. I am recommending it anyway because it is the highest-value *reporting* contribution available: it answers a question the literature has left open, it directly serves the error-analysis requirement of the final report, and it costs almost nothing. **Do not plan it as your headline improvement.**

**Supporting paper.** Srivastava & Sitaram 2018 (the only quantification of the borrowing/spelling-variant split in a Hindi–English corpus).

---

## 5. Token-level LID auxiliary head via a concatenated tokenizer — *the weakest idea*

**The change.** Replace the single CTC vocabulary with two disjoint ID ranges, one per language, so every emitted token carries its language identity. Optionally add an auxiliary per-frame language-classification loss.

**Why it might help.** Forcing the model to commit to a language per token could sharpen decisions at switch boundaries, and it yields token-level LID for free — which would let you report a Hindi–English LID number where almost none exist.

**Effort.** Medium–high: a custom tokenizer, a modified loss, and roughly 25 GPU-hours of retraining.

**Expected effect and uncertainty.** **−1 to +1 WER. I expect no reliable gain, and I am flagging this as the idea to drop first.** Three reasons:
1. Dhawan et al.'s concatenated tokenizer produced its gains while training on **10,000 hours of synthetic code-switched audio**. At 90 h you are asking a doubled output vocabulary to be learned from a fraction of the data per token — the most likely outcome is worse, not better.
2. Their own real-data MUCS Hindi result (28.78 WER on a privately cleaned subset) is *worse* than the straightforward wav2vec 2.0 baseline you would already have.
3. It conflicts with improvement #2: if you have normalised the script, the language of a token is largely recoverable from its orthography anyway, so the auxiliary signal is partly redundant.

**If you want token-level LID**, the cheaper route is to derive it post hoc from the normalised output script plus the English lexicon from #2, and validate it against hand-labelled switch points on a small slice.

**Supporting paper.** Dhawan et al. 2023 (CALCS) — cited here as the source of the idea *and* as the evidence that it needs far more data than this project has.

---

## 6. Suggested experimental order

1. Reproduce the baseline scoring (confirm 25.53 / 23.80 on the blind set).
2. Reproduce the approach with `ai4bharat/indicwav2vec-hindi` → establishes your own reference number.
3. **Improvement #2** (script normalisation) — cheapest real gain, and it produces the T-WER scorer everything else needs.
4. **Improvement #1** (encoder bake-off) — run on top of the #2 pipeline so the comparison is on normalised targets.
5. **Improvement #4** (stratified analysis) — free, runs on decodes you already have.
6. **Improvement #3** (short-insertion augmentation) — only if time remains; report CS-WER alongside WER.
7. Drop **#5** unless 3–6 all finish early.

**Throughout: fix seeds and run the final configuration 2–3 times.** No paper in this literature reports seeds or variance, so without this you cannot distinguish a 1 WER improvement from run-to-run noise — and 1 WER is the size of most of the effects above.

---

## 7. Open questions

1. Should normalisation target **all-Latin** or **all-Devanagari**? This determines the tokenizer and interacts with how T-WER scores you.
2. Is there an existing, licence-compatible **Hindi loanword frequency list**? If not, deriving one from IndicCorp frequency plus corpus script rate is a judgement call that should be documented.
3. Do you want to report **CS-WER (Sharma et al.) or CBA (Biswas et al.)** as the switch-point metric? They point in opposite directions (lower-is-better vs higher-is-better) and neither is standard.
4. Is a 15 h training subset acceptable for the ablations, with only the final configuration trained on the full 89.86 h? That roughly halves total GPU cost but adds a confound between ablation and final runs.
