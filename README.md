# Hindi–English Code-Switching Speech — Research Dossier

Research phase for a project to replicate and then improve a Hindi–English code-switched ASR system.
**No experiment code here and nothing has been trained** — implementation is handed off separately.

## Contents

| Path | What it is |
|---|---|
| `literature/papers.md` | 29 annotated papers (13 primary Hindi–English, 7 supporting, 9 other-pair with justifications), 12 comparability warnings, the official MUCS 2021 Subtask-2 leaderboard, PDF inventory, open questions |
| `literature/comparison_table.csv` | The same 29 papers, 19 columns each |
| `literature/pdfs/` | 30 open-access PDFs — **git-ignored by default**, see below |
| `datasets/datasets.md` | 7 Hindi–English speech corpora with hours, licence, access, annotation granularity, script and course-project feasibility, plus an exclusions table |
| `replication_candidates/candidates.md` | 12 candidates scored on 5 axes, top 3 with trade-offs, 8 ruled out with reasons |
| `improvement_ideas/ideas.md` | 5 improvements with mechanism, effort, expected effect and uncertainty |
| `implementation_brief.md` | Full implementation brief: data, preprocessing, specified *and* unspecified hyperparameters, metrics, targets, ablation ladder, repo structure |
| `_work/` | Fetch logs, extracted text and numeric-verification output — **git-ignored**, see `_work/README.md` |

## Headline findings

1. **MUCS 2021 Subtask-2** is the only open, licensed, benchmarked Hindi–English code-switched speech corpus: 89.86 h train / 5.18 h test / 6.24 h blind, CC BY-SA 4.0, OpenSLR SLR104.
2. **The Microsoft MSCS / WSTCSMC 2020 corpus has no Hindi–English pair** (Gujarati-, Tamil- and Telugu-English only), so there is no real Hindi–English language-diarization or frame-level LID benchmark. All published Hindi–English diarization numbers are on synthetic single-speaker speech.
3. **Recommendation:** replicate Chadha et al. (2022) "Approach C2" — wav2vec 2.0 + CTC + KenLM — with `ai4bharat/indicwav2vec-hindi` substituted, because the Vakyansh "Hindi-4200" checkpoint is no longer hosted (HTTP 403, billing account absent). Official blind-set baseline 25.53 WER / 23.80 T-WER; Chadha et al. 21.77 / 20.75; best leaderboard entry 16.43 / 15.51.

## A note on the PDFs

`literature/pdfs/` holds 30 open-access PDFs (21.3 MB) from the ISCA Archive, ACL Anthology and arXiv. They are **excluded from version control by default**: those papers are free to read, but redistributing the files in a repository is a separate permission that varies by publisher, and four of the cited papers are paywalled and represented only by their preprints. Every paper is cited with title, authors, venue, year and DOI in `literature/papers.md`, so the dossier is fully usable without them. Remove the `literature/pdfs/` line from `.gitignore` only if you have checked the terms and the repository is private.

## Provenance

Every number in these documents was read from a fetched PDF or from the official challenge site and cross-checked against the source text; see `_work/README.md` for the retrieval log, the tools used, and the known limitations.
