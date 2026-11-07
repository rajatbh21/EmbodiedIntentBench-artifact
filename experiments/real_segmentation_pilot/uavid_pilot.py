"""The UAVid real-data pilot: real imagery, real models, one standard bundle (V3 P4).

    python -m experiments.real_segmentation_pilot.uavid_pilot \\
      --output results/uavid_pilot

Runs the two pretrained torchvision configurations (tiled) over the curated UAVid
subset, writes the empirical-bundle source files, builds and validates the bundle with
the existing tooling, and prints where the ordinary benchmark CLI can run it.

Honesty ledger for everything this produces:

- predictions: ``model_generated`` — real pretrained models over real UAV imagery
  (COCO/VOC checkpoints, not aerial-trained; low recall is a domain observation).
- measurements: labelled ``estimated`` as a whole because the *energy* column is an
  estimate (assumed average package power x measured time). Latency itself is
  measured wall-clock around the full tiled forward pass with device sync.
- ground truth: real UAVid annotations; person instances derived as connected
  components of the semantic Humans class (see the adapter's provenance).
- outputs live under ``results/`` (gitignored): UAVid derivatives are CC BY-NC-SA
  and are never committed to this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.real_segmentation_pilot.build_pilot import ConfigSpec, run_pilot
from experiments.real_segmentation_pilot.torchvision_models import (
    build_deeplab_strong,
    build_lraspp_light,
)
from experiments.real_segmentation_pilot.uavid import build_adapter

__all__ = ["main"]

#: Assumed average package power while inferring on Apple silicon. An assumption for
#: the energy *estimate*, recorded as such — never a hardware measurement.
_ASSUMED_POWER_W = 20.0
_ENERGY_SOURCE = (
    "assumed 20 W Apple-silicon package power during inference; assumption, not telemetry"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.real_segmentation_pilot.uavid_pilot")
    parser.add_argument("--output", type=Path, default=Path("results/uavid_pilot"))
    parser.add_argument("--created-at", default="2026-07-30T00:00:00Z")
    args = parser.parse_args(argv)

    adapter = build_adapter()
    print(json.dumps(adapter.provenance(), indent=2))

    models = [
        (
            build_lraspp_light(),
            ConfigSpec(
                config_id="CFG_LOCAL_LIGHT",
                model_id="lraspp_mobilenet_v3_large",
                precision="fp32",
                quality_tier="low",
                prediction_provenance="model_generated",
                measurement_provenance="estimated",
            ),
        ),
        (
            build_deeplab_strong(),
            ConfigSpec(
                config_id="CFG_LOCAL_STRONG",
                model_id="deeplabv3_resnet50",
                precision="fp32",
                quality_tier="high",
                prediction_provenance="model_generated",
                measurement_provenance="estimated",
            ),
        ),
    ]

    result = run_pilot(
        adapter,
        models,
        sources_dir=args.output / "sources",
        bundle_dir=args.output / "bundle",
        bundle_id="UAVID_PILOT",
        average_power_w=_ASSUMED_POWER_W,
        energy_source=_ENERGY_SOURCE,
        # The schema's vocabulary: real captured imagery from an external source. The
        # full description (dataset, license, derivation) lives in the adapter's
        # provenance printed above and in the pilot documentation.
        data_origin="external_capture",
        contract={
            "quality_metric": "target_recall",
            "quality_operator": ">=",
            "quality_threshold": 0.5,
            # Generous: the pilot evaluates perception grounding, not time pressure.
            "deadline_s": float(len(adapter.frames())) * 2.0 + 10.0,
        },
        # The builder's default scaffold path outlasts the deadline by design, which
        # makes every mission end deadline_exceeded. This pilot's mission nominally
        # ends when the frames are flown: complete the path shortly after the last
        # frame so the deadline constraint measures lateness, not the scaffold.
        mission={"path_length_m": 5.0 * (len(adapter.frames()) + 1.0)},
        created_at=args.created_at,
    )
    print(json.dumps(result.to_dict(), indent=2))
    bundle = args.output / "bundle"
    print(
        "\nrun it with:\n"
        f"  .venv/bin/python -m missionbench.run_benchmark \\\n"
        f"    --episode {bundle}/episodes/episode.json \\\n"
        f"    --contract {bundle}/contracts/contract.json \\\n"
        f"    --policy rule_based --executor replay"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
