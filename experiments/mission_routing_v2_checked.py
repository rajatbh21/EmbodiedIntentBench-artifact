"""Execute the frozen routing study only after the physical-route gate passes."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.lake_missions.assets import sha
from experiments.mission_routing_v2.flights import verify
from experiments.mission_routing_v2_route_validation import audit


def require_route(root):
    p = verify(root)
    revision = p.get("route_revision")
    if (
        revision
        and sha(Path(__file__).with_name("mission_routing_v2_route_validation.py"))
        != revision["reviser_sha256"]
    ):
        raise ValueError("Route revision implementation changed")
    result = audit(root)
    if not result["all_passed"]:
        raise ValueError(
            "Physical route gate failed; do not infer or compare policies on this cohort"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("infer", "study", "validate", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight")
    parser.add_argument("--power", type=int, default=15)
    args = parser.parse_args()
    require_route(args.output)
    if args.mode == "infer":
        from experiments.mission_routing_v2.infer import run

        run(args.output, args.flight)
    elif args.mode == "study":
        from experiments.mission_routing_v2.study import run

        run(args.output, args.flight, args.power)
    elif args.mode == "validate":
        from experiments.mission_routing_v2.validate import run

        run(args.output)
    else:
        from experiments.mission_routing_v2.report import run

        run(args.output)


if __name__ == "__main__":
    main()
