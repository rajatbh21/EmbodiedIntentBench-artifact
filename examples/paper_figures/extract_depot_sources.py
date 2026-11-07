"""Extract compact numeric Fig. 4 sources from the original Depot output directory.

This optional provenance utility requires the historical analysis.json and
recall_transfer.json; those full files and their recording/prediction inputs are
not bundled. Their exact digests must match the preserved paper figure data.
No neural model or simulator is run. No machine-specific source paths are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def extract(source, package=HERE):
    source, package = Path(source), Path(package)
    figure = json.loads((package / "figure_data.json").read_text())["fig4"]
    originals = {}
    for filename, key in (("analysis.json", "depot.analysis"),
                          ("recall_transfer.json", "depot.recall_transfer")):
        content = (source / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != figure["raw_sha256"][key]:
            raise ValueError(filename + " does not match the original paper source SHA-256")
        originals[key] = json.loads(content)
    analysis, recall = originals["depot.analysis"], originals["depot.recall_transfer"]
    # Preserve original row order and numeric fields. Dispatch schedules and
    # redundant selection counts do not enter any Fig. 4 reduction.
    cases = [{k: v for k, v in row.items() if k not in {"schedule", "selections"}}
             for row in analysis["policy_cases"] if row["policy"].startswith("static:")]
    return {
        "schema_version": 1,
        "original_source_sha256": figure["raw_sha256"],
        "selection": {
            "policy_cases": "analysis.json /policy_cases; static policies only; schedule and selections omitted",
            "recall_rows": "recall_transfer.json /rows; all rows, unchanged",
        },
        "evaluation_labels_fitted": recall["evaluation_labels_fitted"],
        "predictor_changed": recall["predictor_changed"],
        "policy_cases": cases,
        "recall_rows": recall["rows"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Historical directory containing analysis.json and recall_transfer.json")
    parser.add_argument("--output", type=Path, default=HERE / "sources/depot_rows.json")
    args = parser.parse_args(argv)
    try:
        result = extract(args.source)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "Depot extraction failed: " + str(error) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"policy_cases": len(result["policy_cases"]),
                      "recall_rows": len(result["recall_rows"])}))


if __name__ == "__main__":
    main()
