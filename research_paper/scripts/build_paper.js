/*
 * Builds research_paper/hindi_english_code_switching_asr.docx.
 *
 *   npm install docx@9.15.3            (once, anywhere on NODE_PATH)
 *   node research_paper/scripts/build_paper.js
 *
 * Every number in the text comes from the committed result files and reports
 * named in research_paper/README.md. Figures are drawn by make_figures.py.
 * Citations are numbered in order of first use (IEEE style).
 */
"use strict";
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, ImageRun, Math: OMath, MathFraction, MathFunction, MathRun,
  MathSubScript, MathSubSuperScript, MathSum, MathSuperScript, Packer, PageNumber, Paragraph, SectionType,
  Table, TableCell, TableRow, TabStopType, TextRun, VerticalAlign, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "research_paper", "figures");
const OUT = path.join(ROOT, "research_paper", "hindi_english_code_switching_asr.docx");

// ---------------------------------------------------------------- page geometry (twips)
const PAGE_W = 11906, PAGE_H = 16838;            // A4
const MARGIN_LR = 1008, MARGIN_TOP = 1080, MARGIN_BOTTOM = 1080;
const GUTTER = 340;
const COL_W = Math.floor((PAGE_W - 2 * MARGIN_LR - GUTTER) / 2);   // 4775 twips, about 3.32 in
const FONT = { ascii: "Times New Roman", hAnsi: "Times New Roman", cs: "Nirmala UI", eastAsia: "Times New Roman" };
const MONO = { ascii: "Courier New", hAnsi: "Courier New", cs: "Nirmala UI" };

// ---------------------------------------------------------------- numbering registries
const FIGS = ["arch", "prep", "timeline", "wer", "curves", "blind", "errors", "workflow"];
const TABS = ["corpus", "config", "iterations", "results", "lm", "published", "errortypes", "examples", "edges"];
const EQS = ["ctc", "ctcloss", "greedy", "kn", "beam", "wer", "twer", "fold", "select", "filter"];
const SECS = ["intro", "related", "data", "method", "design", "results", "errors", "impl", "discussion", "limits", "conclusion"];
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];
const num = (list, key, roman) => {
  const i = list.indexOf(key);
  if (i < 0) throw new Error(`unknown label ${key}`);
  return roman ? ROMAN[i] : String(i + 1);
};

// ---------------------------------------------------------------- references (IEEE)
const REFS = {
  diwan2021mucs: "A. Diwan *et al.*, “MUCS 2021: Multilingual and code-switching ASR challenges for low resource Indian languages,” in *Proc. Interspeech*, 2021, doi: 10.21437/Interspeech.2021-1339.",
  openslr104: "OpenSLR, “SLR104: MUCS 2021 code-switching speech data (Hindi–English, Bengali–English),” [Online]. Available: https://www.openslr.org/104/",
  baevski2020wav2vec: "A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 12449–12460.",
  graves2006ctc: "A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks,” in *Proc. 23rd Int. Conf. Machine Learning (ICML)*, 2006, pp. 369–376, doi: 10.1145/1143844.1143891.",
  heafield2011kenlm: "K. Heafield, “KenLM: Faster and smaller language model queries,” in *Proc. Sixth Workshop on Statistical Machine Translation*, Edinburgh, Scotland, 2011, pp. 187–197.",
  chadha2022codeswitched: "H. S. Chadha, P. Shah, A. Dhuriya, N. Chhimwal, A. Gupta, and V. Raghavan, “Code switched and code mixed speech recognition for Indic languages,” arXiv:2203.16578, 2022.",
  srivastava2018homophone: "B. M. L. Srivastava and S. Sitaram, “Homophone identification and merging for code-switched speech recognition,” in *Proc. Interspeech*, 2018, doi: 10.21437/Interspeech.2018-1171.",
  wiesner2021noisy: "M. Wiesner *et al.*, “Training hybrid models on noisy transliterated transcripts for code-switched speech recognition,” in *Proc. Interspeech*, 2021, doi: 10.21437/Interspeech.2021-2127.",
  conneau2021xlsr: "A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and M. Auli, “Unsupervised cross-lingual representation learning for speech recognition,” in *Proc. Interspeech*, 2021, doi: 10.21437/Interspeech.2021-329.",
  gupta2021clsril: "A. Gupta, H. S. Chadha, P. Shah, N. Chhimwal, A. Dhuriya, R. Gaur, and V. Raghavan, “CLSRIL-23: Cross lingual speech representations for Indic languages,” arXiv:2107.07402, 2021.",
  chadha2022vakyansh: "H. S. Chadha, A. Gupta, P. Shah, N. Chhimwal, A. Dhuriya, R. Gaur, and V. Raghavan, “Vakyansh: ASR toolkit for low resource Indic languages,” arXiv:2203.16512, 2022.",
  radford2023whisper: "A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in *Proc. 40th Int. Conf. Machine Learning (ICML)*, PMLR vol. 202, 2023, pp. 28492–28518.",
  taneja2019exploiting: "K. Taneja, S. Guha, P. Jyothi, and B. Abraham, “Exploiting monolingual speech corpora for code-mixed speech recognition,” in *Proc. Interspeech*, 2019, doi: 10.21437/Interspeech.2019-1959.",
  sharma2020augmented: "Y. Sharma, B. Abraham, K. Taneja, and P. Jyothi, “Improving low resource code-switched ASR using augmented code-switched TTS,” in *Proc. Interspeech*, 2020, doi: 10.21437/Interspeech.2020-2402.",
  srivastava2019e2e: "B. M. L. Srivastava, B. Abraham, S. Sitaram, R. Mehta, and P. Jyothi, “End-to-end ASR for code-switched Hindi–English speech,” arXiv:1906.09426, 2019.",
  antony2022transducer: "A. Antony, S. R. Kota, A. Lade, Spoorthy V, and S. G. Koolagudi, “An improved transformer transducer architecture for Hindi–English code switched speech recognition,” in *Proc. Interspeech*, 2022, doi: 10.21437/Interspeech.2022-10763.",
  dhawan2023unified: "K. Dhawan, D. Rekesh, and B. Ginsburg, “Unified model for code-switching speech recognition and language identification based on concatenated tokenizer,” in *Proc. 6th Workshop on Computational Approaches to Linguistic Code-Switching*, 2023, doi: 10.18653/v1/2023.calcs-1.7.",
  biswas2025whisper: "A. Biswas, O. Shevelev, A. Abdaoui, V. Tyagi, and A. Boumadane, “Adapting Whisper for low-resource Hindi–English code-mix speech with on-the-fly augmentation & LLM-synthesised data,” in *Proc. Interspeech*, 2025, doi: 10.21437/Interspeech.2025-447.",
  klejch2021cstr: "O. Klejch, E. Wallington, and P. Bell, “The CSTR system for multilingual and code-switching ASR challenges for low resource Indian languages,” in *Proc. Interspeech*, 2021, doi: 10.21437/Interspeech.2021-1035.",
  kumar2021dualscript: "M. G. Kumar *et al.*, “Dual script E2E framework for multilingual and code-switching ASR,” in *Proc. Interspeech*, 2021, doi: 10.21437/Interspeech.2021-978.",
  verma2023asr: "T. Verma, A. Shree, and A. Modi, “ASR for low resource and multilingual noisy code-mixed speech,” in *Proc. Interspeech*, 2023, doi: 10.21437/Interspeech.2023-757.",
  hannun2014firstpass: "A. Y. Hannun, A. L. Maas, D. Jurafsky, and A. Y. Ng, “First-pass large vocabulary continuous speech recognition using bi-directional recurrent DNNs,” arXiv:1408.2873, 2014.",
  garg2018dual: "S. Garg, T. Parekh, and P. Jyothi, “Dual language models for code switched speech recognition,” in *Proc. Interspeech*, 2018, doi: 10.21437/Interspeech.2018-1343.",
  kneser1995improved: "R. Kneser and H. Ney, “Improved backing-off for M-gram language modeling,” in *Proc. IEEE Int. Conf. Acoustics, Speech, and Signal Processing (ICASSP)*, vol. 1, 1995, pp. 181–184, doi: 10.1109/ICASSP.1995.479394.",
  chen1999empirical: "S. F. Chen and J. Goodman, “An empirical study of smoothing techniques for language modeling,” *Computer Speech & Language*, vol. 13, no. 4, pp. 359–394, 1999, doi: 10.1006/csla.1999.0128.",
  mucs2021leaderboard: "MUCS 2021 organisers, “MUCS 2021 challenge leaderboard,” [Online]. Available: https://navana-tech.github.io/MUCS2021/leaderboard.html",
  cmudict: "Carnegie Mellon University, “The CMU Pronouncing Dictionary,” [Online]. Available: http://www.speech.cs.cmu.edu/cgi-bin/cmudict",
  park2019specaugment: "D. S. Park *et al.*, “SpecAugment: A simple data augmentation method for automatic speech recognition,” in *Proc. Interspeech*, 2019, pp. 2613–2617, doi: 10.21437/Interspeech.2019-2680.",
  loshchilov2019adamw: "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.",
  pyctcdecode: "Kensho Technologies, “pyctcdecode: A fast and lightweight Python-based CTC beam search decoder for speech recognition,” version 0.5.0. [Online]. Available: https://github.com/kensho-technologies/pyctcdecode",
  wolf2020transformers: "T. Wolf *et al.*, “Transformers: State-of-the-art natural language processing,” in *Proc. Conf. Empirical Methods in Natural Language Processing: System Demonstrations*, 2020, pp. 38–45, doi: 10.18653/v1/2020.emnlp-demos.6.",
  paszke2019pytorch: "A. Paszke *et al.*, “PyTorch: An imperative style, high-performance deep learning library,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019, pp. 8024–8035.",
};
const citeOrder = [];
function citeNumbers(keys) {
  const nums = keys.map((k) => {
    if (!REFS[k]) throw new Error(`unknown reference ${k}`);
    if (!citeOrder.includes(k)) citeOrder.push(k);
    return citeOrder.indexOf(k) + 1;
  }).sort((a, b) => a - b);
  const parts = [];
  for (let i = 0; i < nums.length; i++) {
    let j = i;
    while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    parts.push(j - i >= 2 ? `[${nums[i]}]–[${nums[j]}]` : nums.slice(i, j + 1).map((n) => `[${n}]`).join(", "));
    i = j;
  }
  return parts.join(", ");
}

// ---------------------------------------------------------------- inline markup
// **bold**  *italic*  `code`  _{sub}  ^{sup}  [@key;@key]  {F:fig} {T:table} {E:eq} {S:section}
const TOKEN = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|_\{[^}]+\}|\^\{[^}]+\}|\[@[^\]]+\]|\{[FTES]:[a-z0-9]+\})/;
function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(TOKEN)) {
    if (!part) continue;
    // Nirmala UI, the only Devanagari font assumed present, draws large; scale it to match Times New Roman.
    const r = (t, extra = {}) => out.push(new TextRun({ text: t, font: FONT, ...base, ...extra,
      sizeComplexScript: Math.round((base.size || 20) * 0.85) }));
    if (part.startsWith("**")) r(part.slice(2, -2), { bold: true });
    else if (part.startsWith("*")) r(part.slice(1, -1), { italics: !base.italics });
    else if (part.startsWith("`")) out.push(new TextRun({ text: part.slice(1, -1), font: MONO, size: Math.round((base.size || 20) * 0.88) }));
    else if (part.startsWith("_{")) r(part.slice(2, -1), { subScript: true });
    else if (part.startsWith("^{")) r(part.slice(2, -1), { superScript: true });
    else if (part.startsWith("[@")) r(citeNumbers(part.slice(1, -1).split(";").map((k) => k.trim().replace(/^@/, ""))));
    else if (/^\{[FTES]:/.test(part)) {
      const kind = part[1], key = part.slice(3, -1);
      r(kind === "F" ? num(FIGS, key) : kind === "T" ? num(TABS, key, true) : kind === "E" ? `(${num(EQS, key)})` : num(SECS, key, true));
    } else r(part);
  }
  return out;
}

// ---------------------------------------------------------------- block builders
const body = [];
const P = (text, opts = {}) => body.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, indent: { firstLine: opts.noIndent ? 0 : 200 },
  spacing: { after: 0, line: 238 }, children: runs(text, opts.run || {}), keepNext: opts.keepNext,
}));
let secIndex = 0, subIndex = 0;
const H1 = (key, title) => {
  if (SECS[secIndex] !== key) throw new Error(`section order: expected ${SECS[secIndex]}, got ${key}`);
  secIndex++; subIndex = 0;
  body.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 }, keepNext: true,
    children: [new TextRun({ text: `${ROMAN[secIndex - 1]}. ${title}`, font: FONT, size: 20, smallCaps: true })],
  }));
};
const H2 = (title) => {
  subIndex++;
  body.push(new Paragraph({
    spacing: { before: 120, after: 40 }, keepNext: true,
    children: [new TextRun({ text: `${String.fromCharCode(64 + subIndex)}. ${title}`, font: FONT, size: 20, italics: true })],
  }));
};
const BULLETS = (items) => items.forEach((t) => body.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, indent: { left: 260, hanging: 160 }, spacing: { after: 20, line: 238 },
  children: [new TextRun({ text: "• ", font: FONT, size: 20 }), ...runs(t)],
})));

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
let figCount = 0;
function FIGURE(key, file, caption, widthIn = 3.3) {
  figCount++;
  if (FIGS[figCount - 1] !== key) throw new Error(`figure order: expected ${FIGS[figCount - 1]}, got ${key}`);
  const f = path.join(FIG, file);
  const { w, h } = pngSize(f);
  const width = Math.round(widthIn * 96), height = Math.round(width * h / w);
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width, height } })] }));
  body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 140, line: 220 },
    children: [new TextRun({ text: `Fig. ${figCount}. `, font: FONT, size: 16 }), ...runs(caption, { size: 16 })] }));
}

let tabCount = 0;
const cellBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const rule = (size) => ({ style: BorderStyle.SINGLE, size, color: "000000" });
function TABLE(key, caption, header, rows, widths, opts = {}) {
  tabCount++;
  if (TABS[tabCount - 1] !== key) throw new Error(`table order: expected ${TABS[tabCount - 1]}, got ${key}`);
  const total = widths.reduce((a, b) => a + b, 0);
  if (total > COL_W) throw new Error(`table ${key} is ${total} twips wide, column is ${COL_W}`);
  const size = opts.size || 15;
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 140, after: 0 }, keepNext: true,
    children: [new TextRun({ text: `TABLE ${ROMAN[tabCount - 1]}`, font: FONT, size: 16, smallCaps: true })] }));
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
    children: runs(caption, { size: 16, smallCaps: true }) }));
  const align = opts.align || widths.map((_, i) => (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER));
  const mkRow = (cells, kind, isLast) => new TableRow({
    cantSplit: true,
    children: cells.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      margins: { top: 15, bottom: 15, left: 50, right: 50 },
      verticalAlign: VerticalAlign.CENTER,
      borders: {
        top: kind === "head" ? rule(8) : cellBorder,
        bottom: kind === "head" ? rule(4) : isLast ? rule(8) : cellBorder,
        left: cellBorder, right: cellBorder,
      },
      children: [new Paragraph({ alignment: align[i], spacing: { after: 0, line: 220 }, keepNext: true,
        children: runs(String(c), { size, bold: kind === "head" ? false : undefined }) })],
    })),
  });
  const tableRows = [mkRow(header, "head", false), ...rows.map((r, i) => mkRow(r, "body", i === rows.length - 1))];
  body.push(new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER, rows: tableRows }));
  body.push(new Paragraph({ spacing: { after: opts.note ? 0 : 100 }, children: [] }));
  if (opts.note) body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 210 }, children: runs(opts.note, { size: 14 }) }));
}

// equations
const mr = (t, style) => new MathRun(style ? { text: t, style } : t);
const up = (t) => mr(t, "plain");
const sub = (base, s) => new MathSubScript({ children: base, subScript: s });
const sup = (base, s) => new MathSuperScript({ children: base, superScript: s });
const subsup = (base, s, p) => new MathSubSuperScript({ children: base, subScript: s, superScript: p });
const frac = (n, d) => new MathFraction({ numerator: n, denominator: d });
const fn = (name, args) => new MathFunction({ name: [up(name)], children: args });
let eqCount = 0;
function EQ(key, children) {
  eqCount++;
  if (EQS[eqCount - 1] !== key) throw new Error(`equation order: expected ${EQS[eqCount - 1]}, got ${key}`);
  body.push(new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(COL_W / 2) }, { type: TabStopType.RIGHT, position: COL_W }],
    spacing: { before: 80, after: 80 },
    children: [new TextRun({ text: "\t", font: FONT }), new OMath({ children }), new TextRun({ text: `\t(${eqCount})`, font: FONT, size: 20 })],
  }));
}

function ALGORITHM(title, lines) {
  const cells = [
    new TableRow({ children: [new TableCell({ width: { size: COL_W - 100, type: WidthType.DXA },
      margins: { top: 30, bottom: 30, left: 80, right: 80 },
      borders: { top: rule(8), bottom: rule(4), left: cellBorder, right: cellBorder },
      children: [new Paragraph({ spacing: { after: 0 }, children: runs(title, { size: 16 }) })] })] }),
    new TableRow({ children: [new TableCell({ width: { size: COL_W - 100, type: WidthType.DXA },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      borders: { top: cellBorder, bottom: rule(8), left: cellBorder, right: cellBorder },
      children: lines.map((l) => new Paragraph({ spacing: { after: 10, line: 220 }, indent: { left: 200, hanging: 200 }, children: runs(l, { size: 16 }) })) })] }),
  ];
  body.push(new Table({ width: { size: COL_W - 100, type: WidthType.DXA }, columnWidths: [COL_W - 100], alignment: AlignmentType.CENTER, rows: cells }));
  body.push(new Paragraph({ spacing: { after: 100 }, children: [] }));
}

// ---------------------------------------------------------------- data
const RES = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "iterations.json"), "utf8"));
const it = (n) => RES[`iteration${n}`];
const f2 = (x) => Number(x).toFixed(2);
const gains = (split, key) => [1, 2, 3, 4, 5].map((n) => it(n)[`${split}_greedy`][key] - it(n)[`${split}_beam`][key]);
const range = (xs) => `${f2(Math.min(...xs))}–${f2(Math.max(...xs))}`;

// ================================================================ CONTENT
const TITLE = "Fine-Tuning Self-Supervised Speech Representations for Hindi–English Code-Switched Speech Recognition: A Controlled Study of Data Quality, Script Normalisation and Language-Model Decoding";

const ABSTRACT = "Hindi–English code-switched speech combines two phonologies within an utterance, writes English words in either Devanagari or Latin script, and offers little labelled data, which together make automatic speech recognition (ASR) difficult. This paper documents a compute-constrained recognition system for the MUCS 2021 Subtask-2 corpus, about 90 hours of code-switched spoken tutorials. A wav2vec 2.0 base encoder of the Vakyansh Hindi lineage is fine-tuned with connectionist temporal classification (CTC) over a 120-symbol mixed-script character inventory, and decoded either greedily or by beam search guided by a 5-gram Kneser–Ney language model trained only on training transcripts. Five iterations, each changing one factor, examine continued training, script normalisation of the training targets, removal of misaligned training utterances, and CTC-based realignment of segment boundaries. Every model and decoder choice was made on a held-out validation set with a script-folded word error rate (WER); test and blind-test sets were decoded once per model and never used for selection. The selected system (iteration 4) reaches a blind-test WER of 20.44% and a transliterated WER (T-WER) of 19.20%, both confirmed with the organisers’ scoring script, against 25.53% and 23.80% for the official challenge baseline. Error analysis attributes most remaining errors to English technical vocabulary, segment-boundary effects and inconsistent script use, and shows that realignment improves recognition of correctly segmented speech while lowering scores under the released segmentation. All results come from single training runs.";

const KEYWORDS = "code-switching, automatic speech recognition, wav2vec 2.0, connectionist temporal classification, n-gram language model, Hindi–English";

// ---------------------------------------------------------------- I. Introduction
H1("intro", "Introduction");
P("Automatic speech recognition (ASR) has moved from hybrid hidden Markov model systems towards end-to-end neural models, and more recently towards encoders pretrained by self-supervision on untranscribed audio and fine-tuned on modest amounts of labelled speech [@baevski2020wav2vec;@conneau2021xlsr]. These advances have been measured mostly on monolingual benchmarks. Multilingual speakers, however, frequently alternate between languages within a single utterance. In India, Hindi–English code-switching is common in education, technology and media, where English technical terms are embedded in Hindi sentences [@diwan2021mucs].");
P("Code-switched speech is harder to recognise than either language alone for three reasons. First, the acoustic model must handle two phoneme inventories and the transitions between them inside one utterance. Second, labelled code-switched speech is scarce: the largest openly licensed Hindi–English corpus, MUCS 2021 Subtask-2, contains about 90 hours of training audio [@diwan2021mucs;@openslr104]. Third, the written form is not standardised. English words appear in Latin script in some transcripts and in Devanagari transliteration in others, so a hypothesis can be phonetically correct and still count as a word error [@srivastava2018homophone]. The MUCS organisers report that more than half of all errors fall on English words, which make up about a third of the reference words, and introduce a transliterated WER (T-WER) to reduce the penalty for script choice [@diwan2021mucs].");
P("Chadha et al. [@chadha2022codeswitched] showed that a wav2vec 2.0 encoder pretrained on Hindi audio, fine-tuned with connectionist temporal classification (CTC) [@graves2006ctc] and decoded with an n-gram language model (LM) [@heafield2011kenlm], reaches 21.77% WER on the MUCS blind test without the lexicon engineering of hybrid systems. Their report omits most training hyperparameters, however, and published MUCS systems rarely isolate the effect of individual data-quality interventions or report how decisions were made [@wiesner2021noisy;@chadha2022codeswitched].");
P("**Problem statement.** This work asks how far a single-GPU, self-supervised CTC system can be improved on Hindi–English code-switched speech by interventions on training duration, target script and training-segment quality, when every decision is made on validation data only and the effect of each intervention is measured in isolation.");
P("**Contributions.** The paper makes four contributions:");
BULLETS([
  "A documented implementation of wav2vec 2.0 fine-tuning with CTC and 5-gram KenLM beam search on MUCS 2021 Subtask-2, with strict separation of training, validation, test and blind-test data. Blind-test WER and T-WER are confirmed with the organisers’ scoring script.",
  "A controlled study of five iterations, each changing one factor: continued training, script-normalised targets, a misalignment filter and CTC realignment of training segments. Iterations are selected with a script-folded validation WER that is insensitive to the script in which English words are written.",
  "Evidence that data-quality interventions interact with the conventions of the evaluation data: script normalisation and realignment lower blind-test error but not validation error, and an analysis attributes the disagreement to how each split writes English words and transcribes speech at segment boundaries.",
  "An error analysis by error type, script, language-switch position and utterance duration, and a CPU-compatible inference path whose pure-Python ARPA language-model reader was validated against KenLM.",
]);
P("Section {S:related} reviews related work, Section {S:data} describes the data, Section {S:method} the method and Section {S:design} the experimental design. Sections {S:results} and {S:errors} report results and errors, Section {S:impl} the implementation, and Sections {S:discussion}–{S:conclusion} discuss findings, limitations and conclusions. Fig. {F:arch} summarises the system studied: a self-supervised wav2vec 2.0 encoder with a CTC output layer over a mixed-script character inventory, decoded greedily or with an n-gram LM.");
FIGURE("arch", "fig1_architecture.png", "System architecture. The encoder and CTC head are fine-tuned jointly; the language model is used only at decoding time. Script-folded WER is used for model selection on validation data, and T-WER is the official blind-test metric.");

// ---------------------------------------------------------------- II. Related work
H1("related", "Related Work");
H2("Self-supervised and multilingual speech representations");
P("wav2vec 2.0 learns contextual speech representations by masking latent features of raw audio and solving a contrastive task over quantised targets; fine-tuning with a CTC head on a few hours of labelled speech yields competitive recognition [@baevski2020wav2vec]. XLSR extends the approach to cross-lingual pretraining on 53 languages [@conneau2021xlsr]. For Indian languages, CLSRIL-23 pretrains a wav2vec 2.0 model on about 10,000 hours of audio in 23 Indic languages [@gupta2021clsril], and the Vakyansh toolkit releases pretrained and fine-tuned Indic checkpoints together with data-preparation tools [@chadha2022vakyansh]. Weakly supervised encoder–decoder models such as Whisper offer an alternative foundation [@radford2023whisper].");
H2("Code-switched speech recognition");
P("Early Hindi–English work relied on proprietary corpora and hybrid systems. Taneja et al. synthesised code-mixed speech by splicing monolingual fragments [@taneja2019exploiting], and Sharma et al. augmented training with code-switched text-to-speech [@sharma2020augmented]. Srivastava et al. found end-to-end CTC and attention models 15–30 WER points worse than hybrid TDNN systems on 47 hours of conversational data [@srivastava2019e2e], suggesting that from-scratch end-to-end models need more data than such corpora provide. Later work studied transformer transducers with monolingual pretraining [@antony2022transducer], concatenated tokenisers that emit language identity [@dhawan2023unified], and Whisper adaptation with synthetic code-mixed data [@biswas2025whisper]. These studies use different data and metrics, which limits direct comparison.");
H2("The MUCS 2021 challenge");
P("MUCS 2021 released Hindi–English and Bengali–English code-switched corpora with GMM-HMM, TDNN and transformer baselines, and defined T-WER [@diwan2021mucs]. The organisers noted that the released segment timestamps are imprecise and reported baselines trained on unaligned and realigned segments. The strongest Hindi–English blind-test WER, 16.43%, came from a hybrid LF-MMI system with a learned lexicon, transliteration maps and removal of badly aligned segments [@wiesner2021noisy]. The best T-WER, 15.64%, came from a CNN-TDNN system with phoneme mapping for English words and external data [@klejch2021cstr]. A dual-script end-to-end model reached 22.06% [@kumar2021dualscript;@mucs2021leaderboard], and the wav2vec 2.0 system of Chadha et al. 21.77% [@chadha2022codeswitched]. Verma et al. addressed script inconsistency by transliterating all text into a common representation before fine-tuning an Indic wav2vec 2.0 model [@verma2023asr].");
H2("CTC and language-model decoding");
P("CTC trains a frame-level classifier against an unsegmented label sequence by summing over all alignments [@graves2006ctc]. Its frame-independent outputs are commonly combined with an n-gram LM in a prefix beam search [@hannun2014firstpass]. Kneser–Ney smoothing remains the standard for such LMs [@kneser1995improved;@chen1999empirical], and KenLM provides efficient querying [@heafield2011kenlm]. For code-switching specifically, dual language models that combine monolingual LMs through a switching model have been proposed [@garg2018dual].");
H2("Research gap");
P("Among MUCS Hindi–English systems, the self-supervised CTC approach is the most reproducible on a single GPU, but its published description lacks training details and ablations [@chadha2022codeswitched]. The literature reviewed for this project reports neither random seeds nor run-to-run variance, and does not separate the effect of training-data quality from that of evaluation conventions such as script choice and segment boundaries. The present work addresses this gap with a controlled, validation-only study of one system, and reports every split, including those on which an intervention does not help.");

// ---------------------------------------------------------------- III. Dataset and preprocessing
H1("data", "Dataset and Preprocessing");
H2("MUCS 2021 Subtask-2 Hindi–English corpus");
P("The corpus consists of recordings of technical spoken tutorials in which Hindi sentences embed English terms, released under CC BY-SA 4.0 [@diwan2021mucs;@openslr104]. It is lecture-style speech, not conversational speech. Training audio totals 89.86 hours of whole recordings. A full audit found all 586 recordings to be 16 kHz, mono, 16-bit PCM, with no missing, corrupt, silent or non-finite files. Table {T:corpus} summarises the segments used. A validation set of 4.01 hours was held out from the training split as 22 whole recordings, selected with a fixed seed, so that no recording contributes to both training and validation. The released test and blind-test sets were kept unchanged.");
TABLE("corpus", "Segment statistics of the MUCS 2021 Hindi–English data as used",
  ["Split", "Rec.", "Utt.", "Hours", "Latin words", "Sent. in train"],
  [["Train", "499", "50,509", "85.55", "25.11%", "–"],
   ["Val (from train)", "22", "2,316", "4.01", "23.76%", "13.21%"],
   ["Test", "30", "3,136", "5.18", "24.26%", "41.55%"],
   ["Blind", "35", "4,034", "6.24", "32.27%", "4.55%"]],
  [1150, 560, 760, 640, 820, 830],
  { note: "Rec.: recordings; Utt.: segments. Latin words: share of words in Latin script. Sent. in train: share of utterances whose cleaned sentence also occurs in the training split (train minus val). Blind contains 8 empty references, which are skipped in scoring." });
P("Three properties of the corpus shape the experiments. First, blind references write English in Latin script more often than the other splits (32.27% against 25.11% for training), reflecting a different annotation convention. Second, 41.55% of test utterances repeat a training sentence against 4.55% for blind, so test results may partly reflect memorisation. Third, segment boundaries are coarse: 99% of segments start and end on a whole second in every split, and some recordings carry transcripts offset by several seconds from the audio [@diwan2021mucs].");
H2("Audio preprocessing");
P("Fig. {F:prep} shows the preprocessing pipeline. Segments are read from the recordings by their time stamps, down-mixed to mono by channel averaging and resampled to 16 kHz with a polyphase filter. For MUCS audio the last two steps are identities, since every recording is already 16 kHz mono, but they let the inference path accept arbitrary audio. Each segment is normalised to zero mean and unit variance by the encoder’s feature extractor, as in pretraining, so no separate loudness normalisation is applied. No audio was truncated. Training excludes segments shorter than 0.5 s or longer than 60 s, which removes none, and 7 segments whose transcripts contain more characters than the encoder produces frames, which CTC cannot align. Training therefore used 50,502 segments.");
FIGURE("prep", "fig2_preprocessing.png", "Audio and transcript preprocessing. Resampling and down-mixing are identities for the 16 kHz mono MUCS recordings but are applied to arbitrary input at inference time.");
H2("Text normalisation and tokenizer");
P("Transcripts are normalised to Unicode NFC, stripped of invisible formatting characters, lower-cased (affecting Latin script only), cleared of sentence punctuation and collapsed to single spaces. Twelve word stems in which an ASCII colon stands for the Devanagari visarga are restored to the proper character. Scripts are preserved: a word stays in the script chosen by the transcriber, and the baseline applies no transliteration.");
P("The tokenizer is a character inventory built from training transcripts only. It contains 120 symbols: 63 Devanagari characters, 26 lower-case Latin letters, 10 digits, 16 symbols that are spoken in programming tutorials (for example `*` and `+`), and 5 special tokens. The CTC blank is `<pad>` (index 0) and the word delimiter is `|`. Characters occurring fewer than 10 times are removed, following Chadha et al. [@chadha2022codeswitched]. Encoding followed by decoding reproduces 99.915% of training transcripts exactly; the remainder differ only in removed rare characters.");

// ---------------------------------------------------------------- IV. Methodology
H1("method", "Proposed Methodology");
P("The system of Fig. {F:arch} has three parts: a wav2vec 2.0 encoder, a linear CTC output layer, and two decoders, one greedy and one guided by an n-gram LM.");
H2("Acoustic encoder");
P("The encoder is `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`, a wav2vec 2.0 base model with 94.42 M parameters [@baevski2020wav2vec;@chadha2022vakyansh]. A seven-layer convolutional feature encoder (kernels 10, 3, 3, 3, 3, 2, 2; strides 5, 2, 2, 2, 2, 2, 2) maps 16 kHz audio to 512-dimensional frames at a 20 ms hop (stride 320 samples). A twelve-layer transformer with 768 hidden units and 12 attention heads then produces contextual representations **h**_{t}. The checkpoint belongs to the same Vakyansh Hindi lineage as the encoder of Chadha et al. [@chadha2022codeswitched], but whether its weights are identical to theirs could not be verified: that paper describes its encoder as pretrained on unlabelled Hindi audio, whereas this checkpoint’s model card describes fine-tuning on 4,200 hours of labelled Hindi from a CLSRIL-23 initialisation [@gupta2021clsril]. The study therefore reproduces the method, not the published number.");
H2("CTC output layer");
P("A linear layer maps each 768-dimensional frame representation to 120 logits, followed by a log-softmax, giving frame posteriors *p*_{t}(*k* | **x**) over the inventory. Of the 120 rows, 62 are copied from the checkpoint’s output layer for characters it already modelled, and 58 are newly initialised, including all Latin letters and digits. Testing showed that the checkpoint’s actual blank symbol is `<s>`, which occupies 77.2% of frames on real speech, rather than the padding token named in its configuration. This blank is mapped to the new blank index; trusting the configuration produced empty output.");
P("For a target label sequence **y** and input **x** with *T* frames, CTC [@graves2006ctc] marginalises over all frame-level paths π that collapse to **y** under the mapping ℬ, which merges repeated labels and removes blanks:");
EQ("ctc", [mr("P"), mr("("), mr("y"), mr("|"), mr("x"), mr(")"), mr("="),
  new MathSum({ subScript: [mr("π∈"), sup([mr("ℬ")], [mr("−1")]), mr("(y)")], children: [
    subsup([up("∏")], [mr("t=1")], [mr("T")]), sub([mr("p")], [mr("t")]), mr("("), sub([mr("π")], [mr("t")]), mr("|"), mr("x"), mr(")")] })]);
P("and the training loss is the negative log-likelihood", { noIndent: true });
EQ("ctcloss", [sub([mr("ℒ")], [up("CTC")]), mr("="), mr("−"), fn("ln", [mr("P"), mr("("), mr("y"), mr("|"), mr("x"), mr(")")])]);
P("averaged over the utterances of a batch. CTC needs no frame-level alignment, only that the label sequence is no longer than the frame sequence, which motivates the length filter of Section {S:data}.", { noIndent: true });
H2("Greedy decoding");
P("Greedy decoding selects the most probable symbol in each frame and applies ℬ:");
EQ("greedy", [mr("ŷ"), mr("="), mr("ℬ"), mr("("), sub([mr("k")], [mr("1")]), mr(", …, "), sub([mr("k")], [mr("T")]), mr(")"), mr(",  "),
  sub([mr("k")], [mr("t")]), mr("="), sub([up("argmax")], [mr("k")]), sub([mr("p")], [mr("t")]), mr("("), mr("k"), mr("|"), mr("x"), mr(")")]);
P("followed by mapping the delimiter to a space and removing the special tokens that are never training targets. Greedy decoding needs no language model and isolates the acoustic model in the comparisons below.", { noIndent: true });
H2("Language model and beam search");
P("A 5-gram word LM was trained on the 50,509 training transcripts (604,999 tokens, vocabulary 17,005 words) after the same normalisation as the acoustic targets. Validation text was excluded because it is used for tuning, and test and blind-test text was never read. The model uses interpolated Kneser–Ney smoothing [@kneser1995improved;@chen1999empirical]:");
EQ("kn", [sub([up("P")], [up("KN")]), mr("("), mr("w|h"), mr(")"), mr("="),
  frac([up("max"), mr("("), mr("c(hw)"), mr("−"), sub([mr("D")], [mr("n")]), mr(", 0)")], [mr("c(h)")]),
  mr("+"), mr("λ(h)"), sub([up("P")], [up("KN")]), mr("("), mr("w|h′"), mr(")")]);
P("where *h* is the history, *h*′ its shortened form, *D*_{n} an order-specific discount (0.61, 0.74, 0.85, 0.90 and 0.82 for orders 1–5), λ(*h*) the normalising back-off weight, and lower orders use continuation counts. All observed n-grams were kept (17,008 / 157,702 / 304,587 / 368,621 / 378,722 by order). The model is stored in the ARPA text format (68.5 MB) and compiled to the KenLM binary format (27.45 MB) [@heafield2011kenlm]. Because `lmplz` could not be built on the development machine, the ARPA file was produced by a Kneser–Ney builder implemented in the project and compiled with KenLM’s `build_binary`.", { noIndent: true });
P("Beam search uses pyctcdecode [@pyctcdecode], which extends CTC prefix beam search [@hannun2014firstpass] with word-level LM scores. A word sequence **w** is scored as");
EQ("beam", [mr("Q(w)"), mr("="), fn("ln", [sub([up("P")], [up("CTC")])]), mr("+"), mr("α"), fn("ln", [sub([up("P")], [up("LM")])]),
  mr("+"), mr("β|w|"), mr("+"), mr("γ"), sub([mr("N")], [up("OOV")])]);
P("where P_{CTC} = *P*(**w** | **x**) is the acoustic probability of the word sequence, P_{LM} = *P*(**w**) its LM probability, |**w**| its number of words, N_{OOV} the number of its words outside the LM vocabulary, α the LM weight, β a word-insertion score, and γ ≤ 0 a penalty for each word outside the LM vocabulary. These and the beam width are tuned on validation data (Section {S:design}). Out-of-vocabulary rates of the LM are 2.69%, 2.95% and 3.65% of word tokens on validation, test and blind, rising to 8.47% of blind’s Latin-script tokens.", { noIndent: true });
H2("Evaluation metrics");
P("WER is the word-level edit distance between reference and hypothesis, summed over utterances:");
EQ("wer", [up("WER"), mr("="), frac([mr("S+D+I")], [mr("N")]), mr("×100%")]);
P("where *S*, *D* and *I* are substitutions, deletions and insertions, and *N* is the number of reference words. Scoring follows the organisers’ rules: both sides are lower-cased, spoken punctuation in the hypothesis is mapped to symbols, and empty references are skipped. Character error rate (CER) applies the same formula to Unicode code points, including spaces; it is not an official challenge metric. T-WER additionally maps each hypothesis word through the organisers’ word list μ, which converts Devanagari spellings of English words to Latin:", { noIndent: true });
EQ("twer", [new MathRun({ text: "T-WER", normalText: true }), mr("="), frac([new MathSum({ subScript: [mr("u")], children: [up("ED"), mr("("), sub([mr("r")], [mr("u")]), mr(", μ("), sub([mr("h")], [mr("u")]), mr("))")] })],
  [new MathSum({ subScript: [mr("u")], children: [mr("|"), sub([mr("r")], [mr("u")]), mr("|")] })]), mr("×100%")]);
P("where ED is the word edit distance and *r*_{u}, *h*_{u} are the reference and hypothesis of utterance *u*. T-WER is official only on the blind test. On the test set its references are not annotated for it, so a two-sided approximation is reported and labelled as such. Because the word list was built from blind references, it is used for scoring only and never for training or tuning. For model selection the study defines a script-folded WER,", { noIndent: true });
EQ("fold", [sub([up("WER")], [up("fold")]), mr("="), frac([new MathSum({ subScript: [mr("u")], children: [up("ED"), mr("(φ("), sub([mr("r")], [mr("u")]), mr("), φ("), sub([mr("h")], [mr("u")]), mr("))")] })],
  [new MathSum({ subScript: [mr("u")], children: [mr("|"), sub([mr("r")], [mr("u")]), mr("|")] })]), mr("×100%")]);
P("in which φ maps both reference and hypothesis through a training-only map of 587 English words from Devanagari to Latin script. The map is built from training transcripts and the CMU Pronouncing Dictionary [@cmudict]: a Devanagari word is mapped when its pronunciation, rendered by rule from the dictionary, matches a Latin training word and the two occur in similar contexts. Script-folded WER counts a word as correct in either script, so iterations that change the script of their targets are compared fairly.", { noIndent: true });

// ---------------------------------------------------------------- V. Experimental design
H1("design", "Experimental Design");
H2("Protocol and selection");
P("All experiments follow the protocol of Algorithm 1. Each iteration changes exactly one factor relative to the iteration it builds on. The decoder is tuned on validation data in five stages: a 7 × 5 grid of α and β at beam width 50, 14 finer points around the best, three OOV penalties, beam widths from 50 to 3,000 (a wider beam had to improve WER by at least 0.05 points to be chosen), and a check of milder OOV penalties. Each iteration evaluated 58–61 configurations. The chosen settings were committed to version control before test or blind-test audio was decoded. Iterations are compared by");
EQ("select", [sup([mr("i")], [mr("*")]), mr("="), sub([up("argmin")], [mr("i")]), subsup([up("WER")], [up("fold")], [up("val")]), mr("(i)"), mr(",")]);
P("the beam-search, script-folded validation WER of Eq. {E:fold}. Test and blind-test results are reported for every iteration but played no part in any choice.", { noIndent: true });
ALGORITHM("**Algorithm 1.** Protocol of one iteration *i*", [
  "**Input:** training segments *D*_{tr}^{(i)}, validation set *D*_{val}, initial weights θ_{0}, seed 42.",
  "1: Fine-tune θ on *D*_{tr}^{(i)} with the CTC loss of Eq. {E:ctcloss}; every 2,000 updates, keep the checkpoint with the lowest greedy validation WER.",
  "2: Cache frame log-posteriors for validation, test and blind (one forward pass per utterance, float16).",
  "3: Tune (α, β, γ, beam width) of Eq. {E:beam} on *D*_{val} only; freeze and commit the decoder configuration.",
  "4: Decode test and blind once with the frozen configuration; score WER, CER and T-WER; confirm blind scores with the organisers’ script.",
  "5: Record WER_{fold}^{val}(*i*); select the final model with Eq. {E:select}.",
  "**Output:** checkpoint, frozen decoder configuration, and metrics for all splits.",
]);
H2("Baseline configuration (iteration 1)");
P("Table {T:config} lists the training configuration of the baseline. The whole network, including the convolutional feature encoder, was fine-tuned. Three parameter groups were used because a pilot comparison on 150 validation utterances showed that a uniform learning rate of 10^{−5} never produced Latin-script output (0% of output words after 4,000 updates), whereas raising the transformer and head to 10^{−4} produced 29.6% Latin words after 2,000 updates, with validation loss 1.17 against 1.58. A faster head alone did not help, which indicates that the transformer, not the output layer, must learn when English is being spoken. The convolutional encoder kept 10^{−5} because a single update at 10^{−4} sharply increased error in a smoke test.");
TABLE("config", "Baseline training configuration (iteration 1)",
  ["Setting", "Value"],
  [["Encoder", "him-4200, wav2vec 2.0 base, 94.42 M parameters, fully fine-tuned"],
   ["Updates / batch", "38,000 updates (about 6 epochs); 1 utterance per step, 8-step accumulation"],
   ["Learning rates", "CNN 1×10^{−5}; transformer and CTC head 1×10^{−4}"],
   ["Schedule", "Tri-stage: 10% linear warm-up, 40% hold, 50% cosine decay to zero"],
   ["Optimiser", "AdamW [@loshchilov2019adamw], β = (0.9, 0.98), weight decay 0.01, gradient clipping 1.0"],
   ["Regularisation", "Time masking p = 0.065, span 10 frames [@park2019specaugment;@baevski2020wav2vec]"],
   ["Precision / memory", "FP16 autocast with gradient scaling; gradient checkpointing"],
   ["Selection", "Greedy validation WER every 2,000 updates; best at update 37,872"],
   ["Seed / hardware", "42 (single run); RTX 3050 6 GB laptop GPU; 8 h 52 min"]],
  [1250, 3500], { size: 15, align: [AlignmentType.LEFT, AlignmentType.LEFT] });
H2("Iterations");
P("Table {T:iterations} summarises the five iterations and Fig. {F:timeline} their chronology. Iterations 2–5 start from the best checkpoint of iteration 1 with a fresh optimiser and schedule.");
FIGURE("timeline", "fig3_timeline.png", "Research timeline (2026). Dates are those of the research documents, tagged iterations and implementation commits. Durations are training times on the RTX 3050. The inference backend and dashboard were developed after all experiments.");
TABLE("iterations", "Experimental iterations",
  ["It.", "Change (builds on)", "Train data", "Decoder (α, β, γ, beam)"],
  [["1", "Baseline fine-tuning from him-4200", "50,502 utt.", "0.35, 1.0, −5, 500"],
   ["2", "Continued training, 44,000 updates, halved learning rates, duration batching (1)", "50,502 utt.", "0.40, 1.0, −5, 500"],
   ["3", "As 2, English words in Devanagari mapped to Latin in targets and LM (1)", "50,502 utt.", "0.40, 0.5, −5, 1000"],
   ["4", "As 2, misaligned training utterances removed (2)", "46,959 utt., 79.90 h", "0.50, 0.0, −10, 200"],
   ["5", "As 4, training segments re-cut by CTC alignment instead of removed (4)", "48,805 seg., 81.09 h", "0.35, 1.0, −5, 50"]],
  [300, 2150, 1000, 1300], { size: 14, align: [AlignmentType.CENTER, AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER] });
P("**Iteration 2 (continued training).** Iteration 1’s validation WER was still falling when its schedule ended. Iteration 2 continued for 44,000 updates at half the peak learning rates (5×10^{−6} for the CNN, 5×10^{−5} elsewhere) with a 5% warm-up and 35% hold. Batches were grouped by duration up to 60 s of audio per step without accumulation, which was measured to be 1.74× faster than one utterance per step. Training took 6 h 59 min.");
P("**Iteration 3 (script normalisation).** The baseline error analysis found that words in the wrong script cost about 1.3 WER points on blind. Iteration 3 used iteration 2’s recipe with targets in which the 587 mapped English words (3.67% of training tokens) are written in Latin script; the LM was rebuilt on the mapped text. It was compared with iteration 2 to measure the effect of the targets alone.");
P("**Iteration 4 (misalignment filter).** Because segment boundaries are coarse, some training transcripts describe speech that is not in their audio, which teaches CTC to invent or skip words. Each training utterance was decoded greedily with iteration 1’s model and removed when");
EQ("filter", [up("min"), mr("("), sub([up("CER")], [up("raw")]), mr(", "), sub([up("CER")], [up("skel")]), mr(")"), mr(">50"), mr(",")]);
P("where CER_{raw} compares the strings as written and CER_{skel} compares consonant skeletons after spelling English words in Devanagari via the pronouncing dictionary, so that utterances differing only in script are kept. The threshold was set by inspecting random training utterances before any model was trained on the result: all 12 inspected above 60 were misaligned, as were 26 of 28 between 50 and 60, while the 40–50 band mixed misalignment with ordinary recognition errors. The filter removed 3,550 utterances (7.03%, 5.65 h); the commonest case was a full sentence paired with 2–3 s of audio. Otherwise iteration 4 repeats iteration 2, which it builds on because iteration 2 had the lower script-folded validation WER (16.54 against 16.75).", { noIndent: true });
P("**Iteration 5 (CTC realignment).** Rather than discarding misaligned segments, iteration 5 re-cut them. Each training recording was scored with iteration 4’s model in 12 s windows every 8 s, chosen after measuring that 30 s windows caused the model to stop emitting partway through (0.928 output characters per transcript character, against 0.971). The recording’s transcripts were then aligned in order by a banded CTC Viterbi search within 15 s of their released times. Cuts were placed in the middle of the pauses between transcripts, or within 0.3 s of the transcribed speech where a pause contained speech no transcript accounts for (1.5 h in total, left out). The misalignment filter was then reapplied to the new segments. The median segment start moved by 0.57 s, the filter flagged 1,704 segments instead of 6,645, and 3,032 of the utterances removed in iteration 4 were recovered, giving 48,805 segments (81.09 h). Validation, test and blind keep their released boundaries.");

// ---------------------------------------------------------------- VI. Results
H1("results", "Results and Quantitative Evaluation");
P("All metrics below are the committed results of the training runs described above, measured on the original GPU workstation. They are reported experimental results, not accuracy measurements newly reproduced in the CPU environment of Section {S:impl}, which has no copy of the dataset. Unless stated otherwise, results use KenLM beam search with each iteration’s own validation-tuned settings. Each iteration is a single training run with seed 42; no statistical significance testing was performed, and differences below about one WER point may be within run-to-run variation.");
H2("Performance across iterations");
TABLE("results", "WER, T-WER and CER (%) of the five iterations with KenLM beam search",
  ["It.", "Val WER", "Val fold.", "Test WER", "Test T-WER†", "Blind WER", "Blind T-WER", "Blind CER"],
  [1, 2, 3, 4, 5].map((n) => {
    const r = it(n);
    const cell = (v) => (n === 4 ? `**${f2(v)}**` : f2(v));
    return [n === 4 ? "**4**" : String(n), n === 3 ? `${f2(r.val_beam.wer)}*` : f2(r.val_beam.wer), cell(r.val_beam.wer_folded), f2(r.test_beam.wer),
      f2(r.test_beam.twer_approximate), f2(r.blind_beam.wer), f2(r.blind_beam.twer), f2(r.blind_beam.cer)];
  }),
  [340, 600, 600, 600, 640, 640, 680, 640],
  { size: 14, note: "Val fold.: script-folded validation WER, Eq. {E:fold}, the selection criterion. †Approximate T-WER; test references are not annotated for T-WER. *Measured against iteration 3’s own script-normalised references and not comparable with the other rows; its script-folded value is comparable. Bold: selected model. Blind WER and T-WER were confirmed with the organisers’ compute_wer.py for every iteration." });
P(`Table {T:results} and Fig. {F:wer} report all iterations. Continued training (iteration 2) gave the largest single improvement in script-folded validation WER, from ${f2(it(1).val_beam.wer_folded)} to ${f2(it(2).val_beam.wer_folded)}, and improved every split under both decoders. Script normalisation (iteration 3) reached ${f2(it(3).val_beam.wer_folded)}, level with iteration 2 within run-to-run variation. The misalignment filter (iteration 4) reached the lowest value, ${f2(it(4).val_beam.wer_folded)}, and improved on iteration 2 on every split: test WER ${f2(it(2).test_beam.wer)} to ${f2(it(4).test_beam.wer)}, blind WER ${f2(it(2).blind_beam.wer)} to ${f2(it(4).blind_beam.wer)} and blind T-WER ${f2(it(2).blind_beam.twer)} to ${f2(it(4).blind_beam.twer)}. Each of these gains is small, but all point in the same direction. Realignment (iteration 5) was worse on validation (${f2(it(5).val_beam.wer_folded)}) and test (${f2(it(5).test_beam.wer)}), and better on blind (${f2(it(5).blind_beam.wer)} WER, ${f2(it(5).blind_beam.twer)} T-WER).`);
FIGURE("wer", "fig4_wer_iterations.png", "WER of the five iterations with KenLM beam search. The script-folded validation WER is the selection criterion; test and blind were not used for any decision.");
P("Fig. {F:curves} shows greedy validation WER during training. Iteration 1 converged from 43.9% at update 2,000 to 23.77% at update 37,872. Iteration 4 was below iteration 2 at 21 of 22 checkpoints, by 0.62 points on average, and iteration 5 above iteration 4 at 19 of 22, by 0.66 points on average. Iteration 5’s training loss ended far lower than iteration 4’s (0.22 against 0.37) while its validation loss ended higher (1.04 against 0.91), consistent with targets that fit the training audio better but differ from the conventions of the validation data.");
FIGURE("curves", "fig5_training_curves.png", "Greedy validation WER (cleaned text) at each checkpoint. Iterations 2, 4 and 5 start from iteration 1’s best checkpoint; dots mark each run’s selected checkpoint. Iteration 3 is omitted because it is scored against different (script-normalised) references.");
H2("Selection of the final model");
P(`By Eq. {E:select}, iteration 4 is selected: its script-folded validation WER of ${f2(it(4).val_beam.wer_folded)} is the lowest of the five. It is also the best of the five on test WER (${f2(it(4).test_beam.wer)}). On blind it scores ${f2(it(4).blind_beam.wer)} WER and ${f2(it(4).blind_beam.twer)} T-WER, which is not the lowest: iteration 3 has the lowest blind WER (${f2(it(3).blind_beam.wer)}) and iteration 5 the lowest blind T-WER (${f2(it(5).blind_beam.twer)}). Choosing either on that basis would select the model on the test data whose score is then reported, which biases the reported result. Section {S:errors} relates both blind gains to properties of the blind references rather than to better recognition on validation data.`);
H2("Generalisation across validation, test and blind sets");
P(`Test WER is ${range([1, 2, 3, 4, 5].map((n) => it(n).test_beam.wer - it(n).blind_beam.wer))} points higher than blind WER across the iterations, although both are held-out recordings of the same domain. The difference is consistent with poorer segment alignment on test: in the baseline analysis, errors on the first and last word account for 3.95% of test words against 3.35% on blind and 2.22% on validation, and the MUCS organisers observed a similar gap between unaligned and realigned conditions [@diwan2021mucs]. Test also shares more sentences with training (Table {T:corpus}). Test and blind numbers are therefore not interchangeable, and blind, whose references the organisers report as manually corrected, is the headline split.`);
P("Raw and script-folded validation WER differ by about two points for every iteration whose targets keep the original scripts (for example 18.43 against 16.36 for iteration 4). This difference measures how many validation errors are pure script mismatches between Devanagari and Latin spellings of the same English word, which the folded metric forgives. For iteration 3, whose targets are already normalised, the raw and folded values coincide by construction.");
FIGURE("blind", "fig6_blind_metrics.png", "Blind-test WER, T-WER and CER with KenLM beam search. T-WER forgives Devanagari spellings of English words listed by the organisers; CER is not an official metric.");
P("Fig. {F:blind} compares blind WER, T-WER and CER. T-WER is 0.5–1.4 points below WER for every iteration, and CER varies far less than WER across iterations except where the script of English words changes (iteration 3), because a script change alters every character of a word.");
H2("Effect of language-model decoding");
TABLE("lm", "Greedy and KenLM beam-search WER (%) and the LM gain",
  ["It.", "Val fold. greedy", "Val fold. beam", "Gain", "Blind greedy", "Blind beam", "Gain"],
  [1, 2, 3, 4, 5].map((n) => {
    const r = it(n);
    return [String(n), f2(r.val_greedy.wer_folded), f2(r.val_beam.wer_folded), f2(r.val_greedy.wer_folded - r.val_beam.wer_folded),
      f2(r.blind_greedy.wer), f2(r.blind_beam.wer), f2(r.blind_greedy.wer - r.blind_beam.wer)];
  }),
  [340, 760, 700, 560, 760, 720, 560], { size: 14 });
P(`Table {T:lm} isolates the LM. Beam search lowers blind WER by ${range(gains("blind", "wer"))} points and script-folded validation WER by ${range(gains("val", "wer_folded"))} points. The gain shrinks as the acoustic model improves: on blind it falls from ${f2(it(1).blind_greedy.wer - it(1).blind_beam.wer)} points for iteration 1 to ${f2(it(4).blind_greedy.wer - it(4).blind_beam.wer)} for iteration 4. For the baseline, the LM lowered blind CER by only 0.39 points while lowering WER by 3.42, so most of its benefit is in snapping nearly correct spellings onto known words (for example “syzan interpreter” to “python interpreter”). Section {S:errors} shows that this comes at the cost of more script errors.`);
H2("Comparison with published systems");
TABLE("published", "Hindi–English blind-test results (%)",
  ["System", "WER", "T-WER", "Source"],
  [["Official baseline (GMM-HMM)", "25.53", "23.80", "Published [@diwan2021mucs]"],
   ["Chadha et al., wav2vec 2.0 + CTC + LM", "21.77", "20.75", "Published [@chadha2022codeswitched]"],
   ["Dual-script E2E (IITM-SMT-Lab)", "22.06", "20.97", "Leaderboard [@kumar2021dualscript;@mucs2021leaderboard]"],
   ["CSTR (CNN-TDNN, LF-MMI)", "20.41", "15.64", "Leaderboard [@klejch2021cstr;@mucs2021leaderboard]"],
   ["JHU-CLSP/GoVivace (hybrid)", "16.43", "15.51", "Leaderboard [@wiesner2021noisy;@mucs2021leaderboard]"],
   ["This work, iteration 1 (baseline)", "21.54", "20.22", "Measured"],
   ["**This work, iteration 4 (selected)**", "**20.44**", "**19.20**", "Measured"]],
  [1950, 520, 560, 1720], { size: 14, align: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.LEFT],
    note: "Published values are reported by their authors or the challenge leaderboard and were not reproduced here. This work’s results were obtained after the challenge with the released blind references and the official scoring script; they are not leaderboard submissions, and systems differ in external data and language-model text." });
P("Table {T:published} places the results beside published blind-test numbers. The selected model is 5.09 WER and 4.60 T-WER points below the official baseline and 1.33 WER and 1.55 T-WER points below the published result of the closest method [@chadha2022codeswitched], whose 5-gram LM was trained on the external IndicCorp text and decoded with a beam of 3,000. The comparison is indicative only: the encoder checkpoint may differ from theirs, and the result is a single run. Among published Hindi–English leaderboard entries, only the two hybrid systems with extensive lexicon and data engineering report lower WER [@wiesner2021noisy;@klejch2021cstr].");

// ---------------------------------------------------------------- VII. Error analysis
H1("errors", "Error Analysis");
P("This section analyses the selected model on blind with beam search unless stated otherwise (47,764 reference words in 4,026 scored utterances). Words are labelled by script, and a Latin word or a Devanagari word in the organisers’ word list counts as English; transliteration and homophone labels come from phonetic rules and are approximate.");
H2("Error types");
TABLE("errortypes", "Blind-test word errors, baseline and selected model (beam)",
  ["Category", "It. 1", "It. 4", "Change"],
  [["Other substitutions", "4,129", "3,758", "−371"],
   ["Right word, wrong script", "608", "572", "−36"],
   ["Homophone spellings", "342", "333", "−9"],
   ["Incorrect transliteration", "279", "326", "+47"],
   ["Deletions", "3,911", "3,817", "−94"],
   ["Insertions", "1,019", "955", "−64"],
   ["Total (WER %)", "10,288 (21.54)", "9,761 (20.44)", "−527"],
   ["Error rate, Hindi words", "12.01%", "11.37%", "−0.64"],
   ["Error rate, English words", "31.04%", "29.58%", "−1.46"]],
  [1900, 1000, 1000, 850], { size: 14, note: "Word-group error rates count substitutions and deletions per reference word (31,413 Hindi and 15,301 English words)." });
P("Table {T:errortypes} and Fig. {F:errors}(a) break down the errors. Deletions (3,817) and substitutions without a phonetic relation to the reference (3,758) dominate. English words, 32% of blind reference words, carry 51.4% of word errors, with an error rate of 29.58% against 11.37% for Hindi words, in line with the MUCS organisers’ observation [@diwan2021mucs]. Words recognised correctly but written in the wrong script cost 572 errors (1.20 points); the LM adds such errors, because its training text writes English words in both scripts. For the selected model, the right-word-wrong-script count rises from 487 with greedy decoding to 572 with beam search, while other substitutions fall from 5,219 to 3,758.");
FIGURE("errors", "fig7_error_distribution.png", "Errors of the selected model on blind (beam search). (a) Error counts by type, with their contribution in WER points. (b) Error rates for Hindi and English words and for the first word after a language switch (substitutions and deletions per reference word), and WER by utterance duration.");
H2("Script variation, transliteration and homophones");
P("Table {T:examples} lists verbatim examples from the selected model’s blind output. The most frequent confusions are script pairs: “page” written as पेज (108 times), “save” as सेव (62) and “click” as क्लिक (58). Incorrect transliterations include “key” rendered as की, which is also the Hindi postposition “of”, and English terms such as “nucleotide” spelled in Devanagari forms absent from the organisers’ list. Homophone errors are spelling variants with no acoustic difference, such as गयी/गई and लिये/लिए, which transcribers use inconsistently in every split [@srivastava2018homophone].");
TABLE("examples", "Examples from the selected model’s blind output",
  ["Category", "Reference excerpt", "Hypothesis excerpt"],
  [["Wrong script", "base pairs", "base पेयर्स"],
   ["Wrong script", "format में save", "format में सेव"],
   ["Transliteration", "enter key दो बार", "enter की दो बार"],
   ["Homophone", "दिखाई गयी", "दिखाई गई"],
   ["Rare term", "fasta और genbank files", "master और genvinc files"],
   ["Switch point", "data files parse करना", "data files pass करता है"],
   ["Edge insertion", "tutorial में हम सीखेंगे", "इस tutorial में हम सीखेंगे"],
   ["Edge deletion", "स्वागत है इस", "स्वागत है"],
   ["Spoken digits", "version 14 . 10", "version 1410"]],
  [1150, 1800, 1800], { size: 14, align: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.LEFT] });
H2("Language switches, duration and segment boundaries");
P("Errors concentrate at language switches into English: the first word after a Hindi-to-English switch is wrong 21.41% of the time, against 14.47% after an English-to-Hindi switch (Fig. {F:errors}(b)). At a switch into English, the model must change phonology and script at once, often on a rare technical term such as “genbank”, whereas the word after a switch back into Hindi is typically a frequent function word. By duration, WER is lowest for 5–10 s utterances (18.80%), higher for those under 5 s (22.70%), where boundary effects weigh more, and highest above 20 s (39.48%), where long English command sequences and spelled-out input such as file names lose their word boundaries.");
P("Segment boundaries are a major error source. The most deleted token is the spoken symbol “.” (204 times), and the three most inserted words (है, अब, यह) typically begin or end a sentence. In the examples of Table {T:examples}, the edge insertion of इस is a word spoken in this segment’s audio but assigned by the transcript to the previous segment. In the baseline analysis, 51% of insertions fell before the first or after the last reference word, and excess edge deletions alone cost an estimated 0.9 blind WER points that no acoustic model can recover with the released segmentation.");
H2("Why realignment helps blind but not validation");
TABLE("edges", "Change in greedy WER (points) from iteration 4 to 5, by error position",
  ["Error position", "Val", "Test", "Blind"],
  [["Before the first correct word", "+0.78", "+0.96", "−0.20"],
   ["Between first and last correct word", "−0.75", "−1.10", "+0.36"],
   ["After the last correct word", "+0.82", "+1.48", "−1.04"],
   ["No word correct", "+0.18", "+0.83", "−0.13"],
   ["Total", "+1.02", "+2.19", "−1.00"]],
  [2350, 800, 800, 800], { size: 14 });
P("Table {T:edges} counts each greedy-decoding error by its position in the utterance; a segment boundary can only cause errors at the edges. Inside utterances, iteration 5 makes fewer errors than iteration 4 on validation and test. At the edges of validation and test it makes more, chiefly deletions of words that the references include although the cut has partly removed their audio (deletions after the last correct word rise from 402 to 677 on validation). Iteration 4, trained on the same kind of cuts, learned to complete such words; iteration 5, trained on segments that contain exactly their transcript, did not. On blind the same habit becomes an error for iteration 4: it inserts 908 words before the first or after the last correct word, against 274 for iteration 5, because blind references appear to stop where the audio does (one ends on देख, a word cut short). This interpretation is supported by inspected examples but remains a hypothesis about annotation practice.");
P("A further check re-cut the validation set with the same realignment procedure, for analysis only, since the re-cut uses validation references. On the 1,578 validation segments whose released cut contains only its own speech, the two models tie (16.19 against 16.17 greedy WER). On the 738 segments whose released cut passes through speech, iteration 5 is worse (34.76 against 31.63). With all segments re-cut, iteration 5 is better (15.39 against 16.41), even though the cuts were placed with iteration 4’s model. Re-cutting lowers both models’ greedy WER by about 5 points, an optimistic estimate of what the released boundaries cost.");

// ---------------------------------------------------------------- VIII. Implementation and reproducibility
H1("impl", "Implementation and Reproducibility");
H2("Software architecture");
P("The system is implemented in Python with PyTorch [@paszke2019pytorch] and Hugging Face Transformers [@wolf2020transformers]. Modules cover data download with checksum verification and manifest construction, the tokenizer, model loading with blank remapping, training with exact resumption, a logit cache that stores float16 log-posteriors so that both decoders see identical acoustics, LM construction, decoder tuning, evaluation, error analysis and plotting. Training code refuses any split other than training and validation, and the LM builder refuses non-training text. The repository contains 118 unit tests covering the metrics (checked against hand-worked cases and an independent WER implementation), the official scoring rules, the data guards, decoding and the CPU inference path. Every training run records its configuration, the version-control commit and whether the working tree was clean.");
H2("Research experiments versus the inference application");
P("The experiments of Sections {S:design}–{S:errors} were run on an RTX 3050 6 GB laptop GPU with an AMD Ryzen 7 7840HS CPU, Windows 11, Python 3.14, PyTorch 2.14 and pyctcdecode 0.5.0. After the experiments, an inference application was added for CPU-only machines (Fig. {F:workflow}). It is separate from the research results: it reuses the trained iteration 4 checkpoint and decoder settings but changes no model, setting or reported metric.");
FIGURE("workflow", "fig8_inference_workflow.png", "Inference workflow of the application layer developed after the experiments. It runs on CPU or GPU and reuses the selected model and its validation-tuned decoder settings.");
P("A `Transcriber` class loads the checkpoint and, for beam decoding, the LM once, validates each input (missing, empty, corrupt, non-finite, silent, shorter than 0.1 s or longer than 60 s) before the model runs, and returns text with timing and warnings. Utterances are processed one at a time, because the encoder’s group normalisation makes padded batches change the features. Without a GPU, decoding takes a few seconds per clip after a one-time load of 5–40 s. Because the `kenlm` package has no Windows wheel, a pure-Python reader of the same ARPA file was implemented for beam search. Over 2,003 sentences (26,145 words, including shuffled and out-of-vocabulary sentences), its per-word log-probabilities differ from KenLM’s by at most 6.3×10^{−7}, which is float32 rounding, and beam decoding with the two backends produced identical text in all 61 test cases. Parsing the ARPA file costs about 3.5 s and 250 MB per load. A scoring tool reproduces the committed iteration 4 metrics exactly from the saved hypotheses, and a Streamlit dashboard provides an overview, CPU transcription of uploaded audio, and the results table. CPU inference was verified end to end on synthetic audio only; no MUCS audio was transcribed on CPU, and no accuracy figure in this paper comes from the CPU path.");
H2("Reproducibility");
P("The dataset is downloaded from OpenSLR with checksums [@openslr104]; model weights and data are not stored in the repository. Each iteration is reproduced by its configuration file, and the evaluation script tunes the decoder on validation data and commits the settings before decoding test and blind. Exact package versions are recorded in a lock file. Every number in this paper is read from committed result files, and the figures are regenerated from them by a script distributed with the paper.");

// ---------------------------------------------------------------- IX. Discussion
H1("discussion", "Discussion");
P("**Language-model decoding.** A 5-gram LM trained on only 50,509 transcripts lowered blind WER by about three points for every iteration, more than any training intervention. Its benefit is lexical: it corrects misspellings onto known words and barely changes CER. It also reproduces the inconsistencies of its training text, writing more English words in Devanagari, and it cannot help with words absent from its 17,005-word vocabulary, which include 8.47% of blind’s Latin-script tokens. The shrinking LM gain as the acoustic model improves (Table {T:lm}) shows the two components correcting overlapping errors; the tuned LM weight and insertion score differ between iterations, so decoder settings must be re-tuned for each acoustic model.");
P("**Acoustic versus data interventions.** The largest validation gain came from more training, not from a change of data or targets, indicating that the baseline was undertrained under the single-GPU budget. Removing 7% of the training utterances improved every split, consistent with reports that tossing badly aligned segments helps hybrid systems on this corpus [@wiesner2021noisy]. Realignment produced a model that recognises correctly segmented speech better, yet scored worse under the released validation and test segmentation. A data-quality intervention can therefore be beneficial for recognition and harmful for a benchmark score at the same time, depending on how the evaluation data were segmented and transcribed.");
P("**Mixed-script evaluation.** The disagreement between validation and blind for iterations 3 and 5 shows that evaluation conventions are part of the task definition for code-switched ASR. Blind writes English in Latin script far more often than the training data and appears to transcribe only fully audible words; a model can gain on blind by matching these conventions without recognising better. Script-folded scoring removes the first effect from model selection, and T-WER removes part of it from the official score. No standard metric removes the effect of segment boundaries. Reporting raw WER, T-WER and a script-insensitive WER together, as done here, makes such effects visible.");
P("**Generalisation and practical implications.** Test WER above 29% for every model shows that results on this corpus depend strongly on segmentation quality, and that a single split can mislead. For deployed Indian multilingual ASR, where audio would typically be segmented at pauses by a voice-activity detector rather than at whole seconds, the re-cut analysis suggests that realigned training could be preferable, but no benchmark result in this study measures that setting. A lightweight self-supervised CTC model with an n-gram LM is competitive with much larger systems on a modest GPU, and it can run on CPU for inference.");

// ---------------------------------------------------------------- X. Limitations and future work
H1("limits", "Limitations and Future Work");
BULLETS([
  "**Single seed.** Every result comes from one training run with seed 42. No variance was measured, so differences below about one WER point, including iteration 4’s 0.18-point validation lead over iteration 2, may not be reproducible.",
  "**Dataset diversity.** All data are spoken programming and science tutorials. Results may not transfer to conversational, noisy or telephone speech, to other domains, or to other speaker populations; no speaker labels exist, so the validation split is not speaker-disjoint.",
  "**Accent and pronunciation.** No accent or pronunciation analysis was possible, because the corpus provides no speaker metadata; English technical terms pronounced with Hindi phonology remain a dominant error source.",
  "**Acoustic errors.** Rare English terms, spelled-out commands and long English passages are recognised poorly, and spoken punctuation and digits are frequently wrong.",
  "**Language-model vocabulary.** The LM is trained on 605 k words of in-domain transcripts only, without the external text used by other systems, and has a closed 17,005-word vocabulary.",
  "**Evaluation constraints.** T-WER is official only on blind and approximate on test; the script map used for selection depends on training data; segment boundaries of the evaluation sets are imprecise. The encoder’s identity with that of Chadha et al. is unverified, and the published comparisons are not controlled.",
]);
P("Future work should repeat the key comparisons with several seeds; evaluate alternative encoders such as XLS-R and Whisper on the same split [@conneau2021xlsr;@radford2023whisper]; adopt transliteration-aware decoding that treats Latin and Devanagari spellings of a word as equivalent; train the LM on larger normalised text; normalise spoken symbols and digits in training targets as the scorer does; and evaluate on audio segmented by voice-activity detection to measure the deployment benefit of realigned training.");

// ---------------------------------------------------------------- XI. Conclusion
H1("conclusion", "Conclusion");
P("This paper presented a controlled study of a wav2vec 2.0 CTC recogniser with 5-gram LM decoding for Hindi–English code-switched speech on MUCS 2021. With every choice made on validation data, the selected model, trained with continued fine-tuning and a misalignment filter, reaches 20.44% WER and 19.20% T-WER on the blind test, 5.09 and 4.60 points below the official baseline, on a single 6 GB GPU. Continued training and removal of misaligned training utterances improved every split, whereas script normalisation and CTC realignment improved the blind test but not the validation set used for selection. The analysis traces these disagreements to differences in how the splits write English words and transcribe speech at segment boundaries, and shows that English technical vocabulary, script inconsistency and segment boundaries account for most remaining errors. These findings indicate that, for code-switched ASR, data quality and evaluation conventions deserve as much attention as model architecture, and that results should be reported across splits and metrics together with the decision protocol behind them.");

// ---------------------------------------------------------------- References
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 }, keepNext: true,
  children: [new TextRun({ text: "References", font: FONT, size: 20, smallCaps: true })] }));
const unused = Object.keys(REFS).filter((k) => !citeOrder.includes(k));
if (unused.length) throw new Error(`references never cited: ${unused.join(", ")}`);
citeOrder.forEach((key, i) => body.push(new Paragraph({
  alignment: AlignmentType.LEFT, indent: { left: 360, hanging: 360 }, spacing: { after: 30, line: 200 },
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 15 }), ...runs(REFS[key], { size: 15 })],
  tabStops: [{ type: TabStopType.LEFT, position: 360 }],
})));

// ---------------------------------------------------------------- front matter (single column)
const front = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160, line: 300 },
    children: [new TextRun({ text: TITLE, font: FONT, size: 40 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
    children: [new TextRun({ text: "Nehal Abbott (2024AIB1010)", font: FONT, size: 22 }), new TextRun({ text: "  ", font: FONT, size: 22 }),
      new TextRun({ text: "Shlok (2024AIB1016)", font: FONT, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Indian Institute of Technology Ropar, Rupnagar, Punjab, India", font: FONT, size: 20, italics: true })] }),
];
const abstractBlock = [
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 220 },
    children: [new TextRun({ text: "Abstract—", font: FONT, size: 18, bold: true, italics: true }), ...runs(ABSTRACT, { size: 18, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 220 },
    children: [new TextRun({ text: "Index Terms—", font: FONT, size: 18, bold: true, italics: true }), ...runs(KEYWORDS, { size: 18, bold: true })] }),
];

const footer = () => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] });
const page = { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN_TOP, bottom: MARGIN_BOTTOM, left: MARGIN_LR, right: MARGIN_LR } };

const doc = new Document({
  creator: "Nehal Abbott; Shlok",
  lastModifiedBy: "Nehal Abbott; Shlok",
  title: "Hindi–English Code-Switched Speech Recognition",
  description: "Research paper on wav2vec 2.0 + CTC + KenLM for MUCS 2021 Subtask-2",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [
    { properties: { page }, footers: { default: footer() }, children: front },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: GUTTER } }, footers: { default: footer() },
      children: [...abstractBlock, ...body] },
  ],
});

if (figCount !== FIGS.length) throw new Error(`placed ${figCount} of ${FIGS.length} figures`);
if (tabCount !== TABS.length) throw new Error(`placed ${tabCount} of ${TABS.length} tables`);
if (eqCount !== EQS.length) throw new Error(`placed ${eqCount} of ${EQS.length} equations`);
if (secIndex !== SECS.length) throw new Error(`placed ${secIndex} of ${SECS.length} sections`);
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${citeOrder.length} references, ${figCount} figures, ${tabCount} tables, ${eqCount} equations)`);
});
