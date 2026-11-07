"""Prepare a NEW English-language lake study before its first freeze or recording."""

import argparse
import json
from pathlib import Path

from experiments.lake_media_english import english_missions
from experiments.lake_missions.assets import save
from experiments.lake_missions.prepare import prepare


def prepare_english(output, assets):
    prepare(output, assets)
    path = output / "missions.json"
    data = json.loads(path.read_text())
    data["missions"] = english_missions(data["missions"])
    save(path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("results/lake_assets"))
    args = parser.parse_args()
    prepare_english(args.output, args.assets)
