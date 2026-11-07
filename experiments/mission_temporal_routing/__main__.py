"""Train, freeze, execute, independently validate, and report temporal routing."""

import argparse
from pathlib import Path

from experiments.mission_temporal_routing import protocol, report, study, train, validate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("mode", choices=("train", "prepare", "run", "validate", "report"))
p.add_argument("--output", type=Path, required=True)
p.add_argument("--parent", type=Path)
p.add_argument("--training", type=Path)
p.add_argument("--cohort", type=Path)
p.add_argument("--original-training", type=Path)
p.add_argument("--flight")
a = p.parse_args()
if a.mode == "train":
    train.run(a.cohort, a.original_training, a.output)
elif a.mode == "prepare":
    protocol.prepare(a.parent, a.training, a.output)
elif a.mode == "run":
    study.run(a.output, a.flight, 15)
elif a.mode == "validate":
    validate.run(a.output)
else:
    report.run(a.output)
