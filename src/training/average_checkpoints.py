"""Average the weights of several checkpoints from one run into a new model.

    python -m src.training.average_checkpoints --output data/checkpoints/RUN/average \
        data/checkpoints/RUN/step-036000 data/checkpoints/RUN/step-038000

Every input must have the same vocabulary and weights. The output directory
works anywhere a checkpoint does: load_model, logit_cache, evaluate.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import src  # noqa: F401  (sets the model cache before transformers is imported)
import torch
from safetensors.torch import load_file, save_file

from src.utils.config import data_path

COPIED_FILES = ("config.json", "preprocessor_config.json", "vocab.json")


def average_state_dicts(states: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Mean of each float tensor, summed in float64. Other tensors come from the first.

    States are consumed one at a time, so only one checkpoint is held in memory.
    """
    totals: dict[str, torch.Tensor] = {}
    kept: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    count = 0
    for state in states:
        if count and set(state) != set(dtypes):
            raise ValueError("checkpoints hold different weights")
        for key, tensor in state.items():
            if not count:
                dtypes[key] = tensor.dtype
                if tensor.is_floating_point():
                    totals[key] = tensor.double()
                else:
                    kept[key] = tensor.clone()
            elif key in totals:
                if tensor.shape != totals[key].shape:
                    raise ValueError(f"{key} has different shapes")
                totals[key] += tensor.double()
        count += 1
    if not count:
        raise ValueError("nothing to average")
    averaged = {key: (total / count).to(dtypes[key]) for key, total in totals.items()}
    averaged.update(kept)
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = [data_path(path) for path in args.checkpoints]
    output = data_path(args.output)

    vocab = (sources[0] / "vocab.json").read_text(encoding="utf-8")
    for source in sources[1:]:
        if (source / "vocab.json").read_text(encoding="utf-8") != vocab:
            raise ValueError(f"{source} has a different vocabulary")
    averaged = average_state_dicts(load_file(source / "model.safetensors") for source in sources)

    output.mkdir(parents=True, exist_ok=True)
    save_file(averaged, output / "model.safetensors", metadata={"format": "pt"})
    for name in COPIED_FILES:
        shutil.copy2(sources[0] / name, output / name)
    (output / "averaged_from.json").write_text(
        json.dumps([str(source) for source in sources], indent=2), encoding="utf-8")
    print(f"averaged {len(sources)} checkpoints into {output}")


if __name__ == "__main__":
    main()
