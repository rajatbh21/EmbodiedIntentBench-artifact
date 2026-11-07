"""Summarize the released intent-adaptation results and compare replay rows."""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index(rows):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in ("profile", "contract", "gate", "policy", "run_id"))
        if key in result:
            raise ValueError(f"Duplicate row: {key}")
        result[key] = row
    return result


def same(a, b):
    if isinstance(a, float) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, help="New rows.jsonl; subsets are permitted")
    args = parser.parse_args()
    rows = read_rows(HERE / "reference_rows.jsonl")
    reference = index(rows)
    if len(rows) != 10368:
        raise ValueError("The reference is expected to contain 10,368 unique rows")
    off = {k[:2] + k[3:]: v for k, v in reference.items() if not k[2]}
    on = {k[:2] + k[3:]: v for k, v in reference.items() if k[2]}
    if off.keys() != on.keys() or len(off) != 5184:
        raise ValueError("Reference rows do not form 5,184 matched pairs")
    lost = sum(off[k]["success"] and not on[k]["success"] for k in off)
    gained = sum(not off[k]["success"] and on[k]["success"] for k in off)
    if (lost, gained) != (0, 3):
        raise ValueError("Reference mission-outcome counts differ")
    print(f"Matched pairs: {len(off)}; successes lost: {lost}; failures becoming successes: {gained}")
    for contract in ("confirm", "time_critical", "confirm_strict", "revisit"):
        subsets = [[r for r in rows if r["profile"] == "orin_15wv2_cuda"
                    and r["policy"] == "fixed_local_mask" and r["contract"] == contract
                    and r["gate"] is gate] for gate in (False, True)]
        if any(len(s) != 72 for s in subsets):
            raise ValueError(f"Incomplete GPU fixed-local conditions: {contract}")
        energy = [sum(r["task_energy_j"] for r in s) / len(s) for s in subsets]
        successes = [sum(r["success"] for r in s) for s in subsets]
        print(f"{contract}: success {successes[0]}/72 -> {successes[1]}/72; "
              f"mean modeled energy {energy[0]:.2f} -> {energy[1]:.2f} J "
              f"({100 * (1-energy[1]/energy[0]):.1f}% reduction)")
    device = json.loads((HERE / "device_measurements.json").read_text())
    runs = device["runs"]
    for contract in ("confirm", "confirm_strict"):
        groups = [[r for r in runs if r["key"] == contract + "|" + gate]
                  for gate in ("ungated", "gated")]
        if any(len(g) != 6 for g in groups):
            raise ValueError("Expected six device runs per schedule")
        means = [sum(r["inference_j"] for r in g) / len(g) for g in groups]
        print(f"Device {contract}: mean inference energy {means[0]:.1f} -> {means[1]:.1f} J "
              f"({100 * (1-means[1]/means[0]):.1f}% reduction)")
    if args.compare:
        actual = index(read_rows(args.compare))
        if not actual:
            raise ValueError("No generated rows to compare")
        errors = []
        for key, row in actual.items():
            expected = reference.get(key)
            if expected is None or row.keys() != expected.keys():
                errors.append((key, "missing reference or different fields"))
                continue
            bad = [field for field in row if not same(row[field], expected[field])]
            if bad:
                errors.append((key, bad))
        if errors:
            raise ValueError(f"{len(errors)} differing rows; first: {errors[0]}")
        print(f"Compared {len(actual)} generated rows with the supplied reference: all match")


if __name__ == "__main__":
    main()
