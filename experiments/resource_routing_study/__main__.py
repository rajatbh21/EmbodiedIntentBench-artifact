"""Freeze, run and validate an exploratory battery intervention."""

import argparse
from pathlib import Path

from experiments.resource_routing_study import protocol, report, study, validate, witness

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("mode", choices=("prepare", "run", "witness", "batch", "validate", "report"))
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cohort", type=Path)
p.add_argument("--training", type=Path)
p.add_argument("--risk-training", type=Path)
p.add_argument("--selection", type=Path)
p.add_argument("--flight")
a = p.parse_args()
if a.mode == "prepare":
    protocol.prepare(a.cohort, a.training, a.risk_training, a.selection, a.output)
elif a.mode in ("run", "witness", "batch"):
    if a.mode in ("run", "batch"):
        study.run(a.output, a.flight, 15)
    if a.mode in ("witness", "batch"):
        witness.run(a.output, a.flight)
elif a.mode == "validate":
    validate.run(a.output)
else:
    report.run(a.output)
