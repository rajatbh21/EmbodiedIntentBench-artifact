"""Compare a complete measured-profile replay to the included reference rows."""
import argparse
import json
from pathlib import Path


def load_rows(path):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    by_id = {row["job_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate job IDs in {path}")
    return by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Replay output directory")
    args = parser.parse_args()
    references = Path(__file__).resolve().parents[1] / "references" / "replay"
    expected = {}
    for name in ("declared", "orin_15wv2_cpu4", "orin_15wv2_cuda"):
        rows = load_rows(references / name / "rows.jsonl")
        if expected.keys() & rows.keys():
            raise ValueError("Duplicate job IDs across reference files")
        expected.update(rows)
    actual = load_rows(args.output / "rows.jsonl")
    missing = expected.keys() - actual.keys()
    extra = actual.keys() - expected.keys()
    different = [key for key in expected.keys() & actual.keys() if expected[key] != actual[key]]
    print(f"Expected {len(expected)} rows; got {len(actual)}. "
          f"Missing: {len(missing)}; extra: {len(extra)}; different: {len(different)}.")
    if missing or extra or different:
        for key in sorted(different)[:5]:
            print(f"Different: {key}")
        raise SystemExit(1)
    print("All rows match the recorded reference.")


if __name__ == "__main__":
    main()
