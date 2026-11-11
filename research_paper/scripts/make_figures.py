"""Draw every figure in the paper from the committed results.

    .venv\\Scripts\\python research_paper/scripts/make_figures.py

Each figure is written to research_paper/figures/ as PDF and SVG (vector) and as
a 400 dpi PNG (embedded in the Word file). Numbers are read from results/ at run
time; nothing is typed in by hand except the labels of the diagrams.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "research_paper" / "figures"

COLUMN = 3.35  # inches: one column of the two-column layout

# Reference categorical palette (slots 1-3 validate all-pairs for colour-vision deficiency).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
TINT = {"blue": "#e3eefc", "orange": "#fde6dc", "aqua": "#dcf4ea", "grey": "#f1f0ed", "yellow": "#fdf0d2"}
EDGE = {"blue": BLUE, "orange": ORANGE, "aqua": AQUA, "grey": MUTED, "yellow": YELLOW}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 400})):
        fig.savefig(OUT / f"{name}{suffix}", bbox_inches="tight", pad_inches=0.02, facecolor="white", **kwargs)
    plt.close(fig)


def style_axes(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- diagrams

def box(ax, x, y, w, h, text, tone="grey", size=7, weight="normal", dashed=False):
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                           linewidth=0.8, edgecolor=EDGE[tone], facecolor=TINT[tone],
                           linestyle=(0, (3, 2)) if dashed else "solid")
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, weight=weight, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.8, color=INK2, shrinkA=0, shrinkB=0))


def canvas(height: float):
    fig = plt.figure(figsize=(COLUMN, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def figure_architecture() -> None:
    fig, ax = canvas(4.7)
    w, h = 0.78, 0.072
    box(ax, 0.5, 0.955, w, h, "Speech segment $x$\n16 kHz mono waveform")
    box(ax, 0.5, 0.855, w, h, "Per-utterance normalisation\n(zero mean, unit variance)")
    # wav2vec 2.0 encoder group: boxes inside, caption along the bottom edge
    ax.add_patch(FancyBboxPatch((0.07, 0.535), 0.86, 0.25, boxstyle="round,pad=0.004,rounding_size=0.015",
                                linewidth=0.7, edgecolor=BLUE, facecolor="none", linestyle=(0, (3, 2))))
    box(ax, 0.5, 0.735, w - 0.06, h, "CNN feature encoder: 7 temporal convolutions\ntotal stride 320 samples (20 ms frames)", "blue")
    box(ax, 0.5, 0.62, w - 0.06, h, "Transformer context network\n12 layers, 768 dimensions, 12 heads", "blue")
    ax.text(0.5, 0.553, "wav2vec 2.0 encoder (Vakyansh him-4200, 94.4 M parameters)", fontsize=6.3,
            color=BLUE, va="center", ha="center", style="italic")
    yh = 0.46
    box(ax, 0.5, yh, w, 0.085, "Linear CTC head (768 $\\rightarrow$ 120) + log-softmax\n"
        "frame posteriors $p_t(k \\mid x)$, trained with the CTC loss", "orange", size=6.8)
    arrow(ax, 0.5, 0.955 - h / 2, 0.5, 0.855 + h / 2)
    arrow(ax, 0.5, 0.855 - h / 2, 0.5, 0.735 + h / 2)
    arrow(ax, 0.5, 0.735 - h / 2, 0.5, 0.62 + h / 2)
    arrow(ax, 0.5, 0.535, 0.5, yh + 0.0425)
    yd = 0.315
    box(ax, 0.27, yd, 0.42, 0.085, "Greedy decoding\narg-max, merge repeats,\ndrop blanks", "aqua", size=6.6)
    box(ax, 0.73, yd, 0.42, 0.085, "Beam search (pyctcdecode)\nscore = CTC + $\\alpha$ LM + $\\beta$ words\n+ OOV penalty", "aqua", size=6.6)
    arrow(ax, 0.43, yh - 0.0425, 0.27, yd + 0.0425)
    arrow(ax, 0.57, yh - 0.0425, 0.73, yd + 0.0425)
    # the language model feeds the beam search from below
    box(ax, 0.81, 0.19, 0.3, 0.07, "5-gram Kneser-Ney LM\n(training text only)", "yellow", size=6.3)
    arrow(ax, 0.81, 0.19 + 0.035, 0.81, yd - 0.0425)
    yo = 0.085
    box(ax, 0.4, yo, 0.62, 0.06, "Hypothesis text (Devanagari and Latin)", "grey", size=6.8)
    arrow(ax, 0.27, yd - 0.0425, 0.27, yo + 0.03)
    arrow(ax, 0.6, yd - 0.0425, 0.6, yo + 0.03)
    ax.text(0.4, 0.018, "Scoring: WER, CER, T-WER, script-folded WER", fontsize=6.3, color=INK2, ha="center", va="center")
    arrow(ax, 0.4, yo - 0.03, 0.4, 0.034)
    save(fig, "fig1_architecture")


def figure_preprocessing() -> None:
    fig, ax = canvas(3.75)
    w, h = 0.44, 0.07
    ax.text(0.25, 0.975, "Audio", ha="center", va="center", fontsize=7.5, weight="bold", color=BLUE)
    ax.text(0.75, 0.975, "Transcript", ha="center", va="center", fontsize=7.5, weight="bold", color=ORANGE)
    audio = ["Recording (WAV, 16-bit PCM)", "Slice segment [start, end)", "Down-mix to mono\n(channel mean)",
             "Resample to 16 kHz (polyphase;\nidentity for MUCS audio)", "Length filters: 0.5-60 s;\nlabel length $\\leq$ frames",
             "Zero-mean, unit-variance\nnormalisation"]
    text = ["Raw transcript (mixed script)", "Unicode NFC; remove\ninvisible characters", "Lower-case Latin;\nrestore visarga (:)",
            "Remove sentence punctuation;\ncollapse whitespace", "Characters to 120 tokens\n('|' = word boundary)",
            "Integer label sequence $y$"]
    ys = [0.9 - i * 0.128 for i in range(6)]
    for i, y in enumerate(ys):
        box(ax, 0.25, y, w, h + (0.012 if "\n" in audio[i] else 0), audio[i], "blue", size=6.5)
        box(ax, 0.75, y, w, h + (0.012 if "\n" in text[i] else 0), text[i], "orange", size=6.5)
        if i:
            for x in (0.25, 0.75):
                arrow(ax, x, ys[i - 1] - 0.041, x, y + 0.041)
    yb = 0.06
    box(ax, 0.5, yb, 0.7, 0.06, "Training pair $(x, y)$ $\\rightarrow$ CTC loss", "grey", size=6.8)
    arrow(ax, 0.25, ys[-1] - 0.041, 0.4, yb + 0.03)
    arrow(ax, 0.75, ys[-1] - 0.041, 0.6, yb + 0.03)
    save(fig, "fig2_preprocessing")


def figure_timeline() -> None:
    events = [
        ("4 Nov", "Literature review: 29 papers, MUCS leaderboard, implementation brief", "grey", ""),
        ("7 Nov", "Stage 1A: data pipeline, tokenizer, smoke test, GPU profiling", "blue", ""),
        ("8 Nov", "Iteration 1: baseline fine-tuning (38,000 updates, 8 h 52 min)", "orange", ""),
        ("8 Nov", "Iteration 2: continued training from iteration 1 (6 h 59 min)", "orange", ""),
        ("9 Nov", "Iteration 3: script-normalised targets (6 h 59 min)", "orange", ""),
        ("9 Nov", "Iteration 4: misalignment filter (7 h 02 min)", "orange", "selected model"),
        ("9 Nov", "Iteration 5: CTC realignment of training segments (6 h 52 min)", "orange", ""),
        ("10 Nov", "CPU inference backend, ARPA LM reader, Streamlit dashboard", "aqua", "after the experiments"),
    ]
    fig, ax = canvas(2.85)
    n = len(events)
    ys = [0.94 - i * (0.88 / (n - 1)) for i in range(n)]
    ax.plot([0.165, 0.165], [ys[-1], ys[0]], color=GRID, linewidth=1.6, zorder=1)
    for (date, label, tone, note), y in zip(events, ys):
        ax.text(0.13, y, date, ha="right", va="center", fontsize=7, color=INK2)
        ax.scatter([0.165], [y], s=26, color=EDGE[tone], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(0.2, y, label, ha="left", va="center", fontsize=6.7, color=INK)
        if note:
            ax.text(0.2, y - 0.052, note, ha="left", va="center", fontsize=6.2, color=EDGE[tone], style="italic")
    ax.text(0.02, 0.995, "2026", fontsize=6.5, color=MUTED, va="top")
    save(fig, "fig3_timeline")


def figure_inference_workflow() -> None:
    fig, ax = canvas(3.55)
    w = 0.84
    steps = [
        ("Input: audio file (any common format) or manifest (CSV/JSONL)", "grey"),
        ("Artifact checks: checkpoint, vocabulary, decoder config, LM", "grey"),
        ("Audio validation: missing, empty, corrupt, non-finite,\nsilent, shorter than 0.1 s or longer than 60 s", "grey"),
        ("Transcriber: load iteration 4 once (CPU or CUDA)", "blue"),
    ]
    ys = [0.94, 0.835, 0.715, 0.595]
    for (text, tone), y in zip(steps, ys):
        box(ax, 0.5, y, w, 0.075 if "\n" in text else 0.06, text, tone, size=6.6)
    for a, b in zip(ys, ys[1:]):
        arrow(ax, 0.5, a - 0.035, 0.5, b + 0.04)
    yd = 0.445
    box(ax, 0.27, yd, 0.4, 0.085, "Greedy CTC decoding\n(no language model)", "aqua", size=6.5)
    box(ax, 0.73, yd, 0.4, 0.085, "Beam search, 5-gram LM:\nKenLM binary, or pure-Python\nARPA reader without kenlm", "aqua", size=6.3)
    arrow(ax, 0.42, ys[-1] - 0.03, 0.27, yd + 0.0425)
    arrow(ax, 0.58, ys[-1] - 0.03, 0.73, yd + 0.0425)
    yo = 0.285
    box(ax, 0.5, yo, w, 0.075, "Output: text, JSON lines (text, timing, warnings),\nor the Streamlit dashboard", "orange", size=6.6)
    arrow(ax, 0.27, yd - 0.0425, 0.42, yo + 0.0375)
    arrow(ax, 0.73, yd - 0.0425, 0.58, yo + 0.0375)
    box(ax, 0.5, 0.135, w, 0.075, "Optional scoring (src.score_files): WER, CER,\nT-WER, script-folded WER, error tables", "grey", size=6.6)
    arrow(ax, 0.5, yo - 0.0375, 0.5, 0.135 + 0.0375)
    ax.text(0.5, 0.035, "Application layer added after the experiments; it does not change the reported metrics.",
            ha="center", va="center", fontsize=6.1, color=INK2, style="italic")
    save(fig, "fig8_inference_workflow")


# ---------------------------------------------------------------- charts

def iterations() -> dict:
    return json.loads((RESULTS / "iterations.json").read_text(encoding="utf-8"))


def grouped_bars(ax, groups, series, colours, labels, value_fmt="{:.2f}", label_size=5.4):
    n = len(series)
    width = 0.8 / n
    for i, (values, colour, label) in enumerate(zip(series, colours, labels)):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, values, width=width, color=colour, edgecolor="white", linewidth=0.7, label=label, zorder=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.35, value_fmt.format(v), ha="center", va="bottom",
                    fontsize=label_size, color=INK2, rotation=90)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


def figure_wer_iterations() -> None:
    data = iterations()
    names = list(data)
    fig, ax = plt.subplots(figsize=(COLUMN, 2.35))
    series = [[data[n]["val_beam"]["wer_folded"] for n in names], [data[n]["test_beam"]["wer"] for n in names],
              [data[n]["blind_beam"]["wer"] for n in names]]
    grouped_bars(ax, [f"It. {n[-1]}" for n in names], series, [BLUE, ORANGE, AQUA],
                 ["Val (script-folded, selection)", "Test", "Blind"])
    style_axes(ax)
    ax.set_ylabel("WER (%), KenLM beam search")
    ax.set_ylim(0, 40)
    ax.get_xticklabels()[3].set_weight("bold")
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13), handlelength=1.2, columnspacing=1)
    ax.text(3, 36.3, "selected", ha="center", fontsize=6.3, color=INK2, style="italic")
    save(fig, "fig4_wer_iterations")


def figure_blind_metrics() -> None:
    data = iterations()
    names = list(data)
    fig, ax = plt.subplots(figsize=(COLUMN, 2.25))
    series = [[data[n]["blind_beam"]["wer"] for n in names], [data[n]["blind_beam"]["twer"] for n in names],
              [data[n]["blind_beam"]["cer"] for n in names]]
    grouped_bars(ax, [f"It. {n[-1]}" for n in names], series, [BLUE, ORANGE, AQUA], ["WER", "T-WER", "CER"])
    style_axes(ax)
    ax.set_ylabel("Blind test error rate (%)")
    ax.set_ylim(0, 27)
    ax.get_xticklabels()[3].set_weight("bold")
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13), handlelength=1.2)
    save(fig, "fig6_blind_metrics")


def val_checks(path: Path) -> list[tuple[int, float]]:
    with path.open(encoding="utf-8") as handle:
        return [(int(r["update"]), float(r["val_wer"])) for r in csv.DictReader(handle) if r.get("val_wer")]


def figure_training_curves() -> None:
    start = 37872  # iteration 1's best checkpoint, where iterations 2-5 begin
    runs = [("Iteration 1", RESULTS / "train_log.csv", 0, BLUE),
            ("Iteration 2", RESULTS / "iteration2" / "train_log.csv", start, ORANGE),
            ("Iteration 4", RESULTS / "iteration4" / "train_log.csv", start, AQUA),
            ("Iteration 5", RESULTS / "iteration5" / "train_log.csv", start, MAGENTA)]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.3))
    for label, path, offset, colour in runs:
        points = val_checks(path)
        xs = [(u + offset) / 1000 for u, _ in points]
        ys = [v for _, v in points]
        ax.plot(xs, ys, color=colour, linewidth=1.3, label=label, zorder=3)
        best = min(range(len(ys)), key=ys.__getitem__)
        ax.scatter([xs[best]], [ys[best]], s=14, color=colour, edgecolor="white", linewidth=0.7, zorder=4)
    ax.axvline(start / 1000, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.text(start / 1000 + 0.8, 43, "warm start of\niterations 2-5", fontsize=6.2, color=INK2, va="top")
    style_axes(ax, "both")
    ax.set_xlabel("Cumulative updates (thousands)")
    ax.set_ylabel("Greedy val WER (%), cleaned text")
    ax.set_ylim(19.5, 45)
    ax.set_xlim(0, 84)
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.4)
    save(fig, "fig5_training_curves")


def figure_errors() -> None:
    analysis = json.loads((RESULTS / "iteration4" / "error_analysis" / "blind_beam.json").read_text(encoding="utf-8"))
    types = analysis["error_types"]
    words = analysis["reference_words"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COLUMN, 3.9), gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.55})
    kinds = [("Deletions", types["deletion"]), ("Other substitutions", types["other_substitution"]),
             ("Insertions", types["insertion"]), ("Right word, wrong script", types["script_only"]),
             ("Homophone spellings", types["homophone_confusion"]),
             ("Incorrect transliteration", types["incorrect_transliteration"])]
    kinds.sort(key=lambda item: item[1])
    labels = [k for k, _ in kinds]
    counts = [v for _, v in kinds]
    bars = ax1.barh(labels, counts, color=BLUE, height=0.62, edgecolor="white", linewidth=0.7, zorder=2)
    for bar, v in zip(bars, counts):
        ax1.text(v + 60, bar.get_y() + bar.get_height() / 2, f"{v:,} ({100 * v / words:.2f} pts)", va="center",
                 fontsize=6.3, color=INK2)
    style_axes(ax1, "x")
    ax1.set_xlim(0, 5600)
    ax1.set_xlabel("Word errors on blind (final model, KenLM)")
    ax1.set_title("(a) Error types", loc="left", fontsize=7.5, color=INK)

    lang, switch, duration = analysis["by_language"], analysis["by_switch"], analysis["by_duration"]
    groups = [("Hindi words", lang["hi"]["error_rate"]), ("English words", lang["en"]["error_rate"]),
              ("Hindi$\\rightarrow$English switch", switch["hi_to_en"]["error_rate"]),
              ("English$\\rightarrow$Hindi switch", switch["en_to_hi"]["error_rate"]),
              ("Utterances < 5 s", duration["under 5 s"]["wer"]), ("5-10 s", duration["5-10 s"]["wer"]),
              ("10-20 s", duration["10-20 s"]["wer"]), ("> 20 s", duration["over 20 s"]["wer"])]
    labels2 = [g for g, _ in groups][::-1]
    rates = [r for _, r in groups][::-1]
    colours = [AQUA] * 4 + [ORANGE] * 4
    bars = ax2.barh(labels2, rates, color=colours, height=0.62, edgecolor="white", linewidth=0.7, zorder=2)
    for bar, v in zip(bars, rates):
        ax2.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", fontsize=6.3, color=INK2)
    style_axes(ax2, "x")
    ax2.set_xlim(0, 45)
    ax2.set_xlabel("Error rate on blind (%)")
    ax2.set_title("(b) By word group and utterance duration", loc="left", fontsize=7.5, color=INK)
    save(fig, "fig7_error_distribution")


def main() -> None:
    figure_architecture()
    figure_preprocessing()
    figure_timeline()
    figure_wer_iterations()
    figure_blind_metrics()
    figure_training_curves()
    figure_errors()
    figure_inference_workflow()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
