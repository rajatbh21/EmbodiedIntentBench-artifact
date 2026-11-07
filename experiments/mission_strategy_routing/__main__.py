"""Run the frozen distinct-strategy routing experiment."""

import argparse
from pathlib import Path

from experiments.mission_strategy_routing import protocol, report, study, validate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("mode", choices=("prepare", "run", "validate", "report"))
p.add_argument("--output", type=Path, required=True)
p.add_argument("--parent", type=Path)
p.add_argument("--flight")
a = p.parse_args()
if a.mode == "prepare":
    protocol.prepare(a.parent, a.output)
elif a.mode == "run":
    study.run(a.output, a.flight, 15)
elif a.mode == "validate":
    validate.run(a.output)
else:
    report.run(a.output)
