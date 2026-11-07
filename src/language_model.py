"""Build a KenLM n-gram model from the training manifest only.

Requires the KenLM ``lmplz`` and ``build_binary`` executables on PATH. The
manifest is deliberately fixed to ``train.jsonl``; validation, test, and blind
transcripts are never read by this command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from src.utils.config import data_path, load_config


def training_text(manifest: Path) -> list[str]:
    if manifest.name != "train.jsonl":
        raise ValueError(f"language-model input must be train.jsonl, got {manifest.name}")
    lines = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            entry = json.loads(line)
            if entry.get("split") != "train":
                raise ValueError(f"{manifest}:{line_number}: non-training row in training manifest")
            text = " ".join(entry.get("text", "").split())
            if text:
                lines.append(text)
    if not lines:
        raise ValueError(f"no non-empty training transcripts in {manifest}")
    return lines


def build_model(
    manifest: Path,
    output_dir: Path,
    order: int = 5,
    lmplz: str = "lmplz",
    build_binary: str = "build_binary",
) -> tuple[Path, Path]:
    if order < 1:
        raise ValueError("n-gram order must be positive")
    missing = [command for command in (lmplz, build_binary) if shutil.which(command) is None]
    if missing:
        raise FileNotFoundError(f"KenLM executable(s) not found on PATH: {', '.join(missing)}")

    lines = training_text(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = output_dir / "train.txt"
    arpa = output_dir / f"train_{order}gram.arpa"
    binary = output_dir / f"train_{order}gram.klm"
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        [lmplz, "-o", str(order), "--discount_fallback", "--text", str(corpus), "--arpa", str(arpa)],
        check=True,
    )
    subprocess.run([build_binary, str(arpa), str(binary)], check=True)
    return arpa, binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/language_model"))
    parser.add_argument("--lmplz", default="lmplz")
    parser.add_argument("--build-binary", default="build_binary")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = data_path(config["data"]["manifest_dir"]) / "train.jsonl"
    output_dir = data_path(args.output_dir)
    arpa, binary = build_model(manifest, output_dir, args.order, args.lmplz, args.build_binary)
    print(json.dumps({"arpa": str(arpa), "binary": str(binary), "order": args.order}, indent=2))


if __name__ == "__main__":
    main()
