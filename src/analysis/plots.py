"""Draw the Stage 1B figures from logged results. Nothing here is typed in by hand.

    python -m src.analysis.plots --run data/checkpoints/baseline_rtx3050

Colours come from a palette checked for colour-blind separation. Greedy and
KenLM keep the same colour in every figure. Figures that do not have their
source files yet are skipped with a message, so this can run mid-training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import ROOT
from src.utils.config import data_path

GREEDY, KENLM = "#2a78d6", "#eb6834"
SERIES = {"greedy": ("Greedy CTC", GREEDY), "beam": ("KenLM beam search", KENLM)}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SPLIT_NAMES = {"val": "Validation\n(tuning set)", "test": "Test", "blind": "Blind test"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "font.size": 10, "legend.frameon": False,
})


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a log. After a resume, the last row for each update wins."""
    with path.open(encoding="utf-8") as handle:
        rows = {row["update"]: row for row in csv.DictReader(handle)}
    return sorted(rows.values(), key=lambda row: int(row["update"]))


def save(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output / name, dpi=160)
    plt.close(figure)
    print("wrote", output / name)


def points(rows: list[dict[str, str]], column: str) -> tuple[list[int], list[float]]:
    kept = [row for row in rows if row.get(column)]
    return [int(row["update"]) for row in kept], [float(row[column]) for row in kept]


def plot_losses(rows: list[dict[str, str]], output: Path, epochs: list[dict[str, str]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.2))
    for column, label, colour in (("train_loss", "Training loss", GREEDY), ("val_loss", "Validation loss", KENLM)):
        x, y = points(rows, column)
        # Validation is sparse, and a line through few points would hide them.
        axis.plot(x, y, color=colour, linewidth=2, label=label,
                  marker="o" if column == "val_loss" else None, markersize=5)
    for row in epochs:
        if row["partial"] == "False":
            axis.axvline(int(row["update"]), color=GRID, linewidth=1, linestyle="--")
            axis.annotate(f"epoch {row['epoch']}", (int(row["update"]), 1), xycoords=("data", "axes fraction"),
                          xytext=(-3, -3), textcoords="offset points", rotation=90, va="top", ha="right",
                          color=MUTED, fontsize=8)
    axis.set(title="Training and validation CTC loss", xlabel="Update", ylabel="CTC loss (per character)")
    # Below the axis, so it cannot cover the epoch labels along the top.
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save(figure, output, "loss_curves.png")


def plot_single(rows: list[dict[str, str]], output: Path, column: str, title: str, label: str, name: str) -> None:
    x, y = points(rows, column)
    if not x:
        return
    figure, axis = plt.subplots(figsize=(8, 3.6))
    axis.plot(x, y, color=GREEDY, linewidth=2, marker="o" if len(x) < 40 else None, markersize=4)
    axis.set(title=title, xlabel="Update", ylabel=label)
    save(figure, output, name)


def load_metrics(results: Path) -> dict[tuple[str, str], dict]:
    found = {}
    for split in SPLIT_NAMES:
        for decoder in SERIES:
            path = results / "evaluations" / f"{split}_{decoder}_metrics.json"
            if path.exists():
                found[(split, decoder)] = json.loads(path.read_text(encoding="utf-8"))
    return found


def plot_metric(metrics: dict, output: Path, keys: tuple[str, ...], title: str, label: str, name: str) -> None:
    splits = [s for s in SPLIT_NAMES
              if any(key in metrics.get((s, d), {}) for d in SERIES for key in keys)]
    if not splits:
        return
    figure, axis = plt.subplots(figsize=(6.4, 4))
    width = 0.34
    for offset, (decoder, (series, colour)) in zip((-width / 2 - 0.01, width / 2 + 0.01), SERIES.items()):
        for position, split in enumerate(splits):
            row = metrics.get((split, decoder), {})
            value = next((row[key] for key in keys if key in row), None)
            if value is None:
                continue
            bars = axis.bar(position + offset, value, width, color=colour,
                            label=series if position == 0 else None)
            axis.bar_label(bars, labels=[f"{value:.2f}"], padding=3, fontsize=9, color=INK)
    axis.set_xticks(range(len(splits)), [SPLIT_NAMES[s] for s in splits])
    axis.set(title=title, ylabel=label)
    axis.grid(axis="x", visible=False)
    axis.margins(y=0.16)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save(figure, output, name)


def plot_error_types(path: Path, output: Path) -> None:
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    types = report["error_types"]
    named = {
        "Deletions": types.get("deletion", 0),
        "Insertions": types.get("insertion", 0),
        "Right word, wrong script": types.get("script_only", 0),
        "Incorrect transliteration": types.get("incorrect_transliteration", 0),
        "Homophone confusion": types.get("homophone_confusion", 0),
        "Other substitutions": types.get("other_substitution", 0),
    }
    ordered = sorted(named.items(), key=lambda item: item[1])
    total = sum(named.values())
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    bars = axis.barh([k for k, _ in ordered], [v for _, v in ordered], color=GREEDY, height=0.6)
    axis.bar_label(bars, labels=[f"{v:,} ({100 * v / total:.1f}%)" for _, v in ordered],
                   padding=4, fontsize=9, color=INK)
    axis.set(title="Where the errors fall (KenLM beam search, blind test)", xlabel="Word errors")
    axis.grid(axis="y", visible=False)
    axis.margins(x=0.24)
    save(figure, output, "error_categories.png")


def write_samples(results: Path, output: Path, count: int = 12) -> None:
    """Sample comparison as Markdown, because matplotlib cannot shape Devanagari."""
    paths = {d: results / "evaluations" / f"blind_{d}.jsonl" for d in SERIES}
    if not all(path.exists() for path in paths.values()):
        return
    rows = {d: [json.loads(line) for line in paths[d].open(encoding="utf-8")] for d in SERIES}
    pairs = [(g, b) for g, b in zip(rows["greedy"], rows["beam"]) if g["reference"]]
    chosen = pairs[:: max(len(pairs) // count, 1)][:count]
    lines = ["# Sample predictions (blind test)", "",
             "Evenly spaced utterances, so this sample is not picked to flatter either decoder.", ""]
    for number, (greedy, beam) in enumerate(chosen, 1):
        lines += [f"## {number}. {greedy['id']} ({greedy['duration']:.1f} s)", "",
                  f"- **Reference:** {greedy['reference']}",
                  f"- **Greedy CTC:** {greedy['hypothesis']}",
                  f"- **KenLM beam:** {beam['hypothesis']}", ""]
    (output / "sample_predictions.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("wrote", output / "sample_predictions.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="data/checkpoints/baseline_rtx3050")
    args = parser.parse_args()
    run = data_path(args.run)
    results = ROOT / "results"
    output = results / "figures"

    rows = read_rows(run / "train_log.csv")
    epochs = []
    if (run / "epoch_log.csv").exists():
        with (run / "epoch_log.csv").open(encoding="utf-8") as handle:
            epochs = list(csv.DictReader(handle))
    plot_losses(rows, output, epochs)
    plot_single(rows, output, "lr", "Learning rate: transformer and CTC head (CNN at one tenth)",
                "Learning rate", "learning_rate.png")
    plot_single(rows, output, "val_wer", "Validation WER during training (greedy, cleaned text)",
                "WER (%)", "val_wer.png")

    metrics = load_metrics(results)
    plot_metric(metrics, output, ("wer",), "Word error rate", "WER (%)", "wer_comparison.png")
    plot_metric(metrics, output, ("cer",), "Character error rate", "CER (%)", "cer_comparison.png")
    plot_metric(metrics, output, ("twer", "twer_approximate"),
                "T-WER: official on blind test, approximate on test", "T-WER (%)", "twer_comparison.png")
    plot_error_types(results / "error_analysis" / "blind_beam.json", output)
    write_samples(results, output)


if __name__ == "__main__":
    main()
