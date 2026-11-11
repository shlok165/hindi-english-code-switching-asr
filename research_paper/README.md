# Research paper

**Fine-Tuning Self-Supervised Speech Representations for Hindi–English Code-Switched Speech Recognition: A Controlled Study of Data Quality, Script Normalisation and Language-Model Decoding**

Nehal Abbott (2024AIB1010) and Shlok (2024AIB1016), IIT Ropar.

The paper documents what this repository implemented and measured: wav2vec 2.0 + CTC + 5-gram KenLM on MUCS 2021 Subtask-2, the five training iterations and how the final model (iteration 4) was chosen, the error analysis, and the CPU inference application added afterwards. It is a two-column, A4, IEEE-style paper of 11 pages including references.

## Contents

| Path | What it is |
|---|---|
| `hindi_english_code_switching_asr.docx` | The paper (Word), with native Word equations |
| `hindi_english_code_switching_asr.pdf` | The same paper exported by Microsoft Word |
| `references.bib` | The 32 references in BibTeX; the paper cites them in IEEE style, numbered by first use |
| `figures/` | The 8 figures as PDF and SVG (vector) and 400 dpi PNG (embedded in the Word file) |
| `scripts/make_figures.py` | Draws every figure from the committed result files |
| `scripts/build_paper.js` | Builds the Word file: text, tables, equations, figure and citation numbering |
| `scripts/export_pdf.ps1` | Exports the Word file to PDF with Microsoft Word |

| Figure | File | Data |
|---|---|---|
| 1 | `fig1_architecture` | Diagram of the system |
| 2 | `fig2_preprocessing` | Diagram of `src/data/preprocessing.py` and the tokenizer |
| 3 | `fig3_timeline` | Dates of the research documents, iteration tags and commits |
| 4 | `fig4_wer_iterations` | `results/iterations.json` |
| 5 | `fig5_training_curves` | `results/train_log.csv`, `results/iteration{2,4,5}/train_log.csv` |
| 6 | `fig6_blind_metrics` | `results/iterations.json` |
| 7 | `fig7_error_distribution` | `results/iteration4/error_analysis/blind_beam.json` |
| 8 | `fig8_inference_workflow` | Diagram of `src/inference.py`, `src/arpa_lm.py`, `app.py` |

## Where the numbers come from

Every number in the paper is read from files committed in this repository. Nothing was re-measured for the paper.

| Topic | Source |
|---|---|
| Results of the five iterations (WER, CER, T-WER, script-folded WER) | `results/iterations.json`, `results/*/evaluations/*_metrics.json` |
| Corpus statistics, tokenizer, encoder facts | `docs/stage_1a_report.md` |
| Baseline training, LM, decoder tuning, baseline error analysis | `docs/stage_1b_report.md`, `results/lm_report.json` |
| Iterations 2–5, filter, realignment, edge-position analysis | `docs/iterations.md`, `results/train_filter_report.json`, `results/train_realign_report.json`, `results/realign_checks.json`, `results/iteration5/cut_effects.json` |
| Error analysis of the final model | `results/iteration4/error_analysis/` |
| Training curves and durations | `results/train_log.csv`, `results/iteration*/train_log.csv` |
| CPU backend, ARPA reader validation, dashboard | `docs/backend.md`, `docs/cpu_setup.md`, `results/cpu_eval/lm_check/` |
| Published and leaderboard numbers | `literature/papers.md` (each value read from the source paper or the official leaderboard) |

Reporting conventions the paper follows:

- The metrics are the committed results of GPU training runs. They were not re-measured on CPU: the CPU path was checked end to end on synthetic audio only, because the dataset is not part of the repository.
- Choices between iterations were made on the script-folded validation WER only. Test and blind are reported for every iteration and were never used to choose.
- Iteration 3's plain validation WER is scored against its own script-normalised references, so only its script-folded value is comparable with the other iterations.
- T-WER is official only on blind; on test it is an approximation and labelled as one.
- Published values are labelled as such. This project's blind results were obtained after the challenge with the released references and the official `compute_wer.py`; they are not leaderboard submissions.
- Every result is a single training run (seed 42). No significance testing was done.

## Rebuilding the paper

From the repository root. The figures need only the project environment; the Word file needs Node.js; the PDF needs Microsoft Word on Windows.

```powershell
.venv\Scripts\python research_paper\scripts\make_figures.py
npm install --prefix research_paper\scripts docx@9.15.3
node research_paper\scripts\build_paper.js
powershell -ExecutionPolicy Bypass -File research_paper\scripts\export_pdf.ps1
```

`build_paper.js` stops with an error if a figure, table, equation or section is out of order, if a reference is never cited, or if a table is wider than a column. `npm install --prefix` puts `node_modules` under `research_paper/scripts/`, which is git-ignored.
