"""Freeze explicitly assigned development/validation/evaluation inputs and detect leakage.

Protocol checks only: disjoint identities do not prove statistical independence.
Never label historical development results as previously unseen evaluation data.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from missionbench.schemas.loading import DocumentReader, read_json_object
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.scenario import digest, load_scenario


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partitions(version: str) -> set[str]:
    if version == "0.1":
        return {"development", "evaluation"}
    if version == "0.2":
        return {"development", "validation", "evaluation"}
    raise ValueError("unsupported split version")


def freeze(source: Path, output: Path) -> dict:
    data = read_json_object(source)
    reader = DocumentReader(
        data, context=str(source), allowed_fields=("split_schema_version", "entries")
    )
    version = reader.get_str("split_schema_version")
    required = partitions(version)
    entries = reader.get_object_list(
        "entries", allowed_fields=("case_id", "partition", "family_id", "scenario", "recording")
    )
    rows = []
    seen = {key: {} for key in ("family_id", "world_sha256", "recording_identity")}
    ids = set()
    for item in entries:
        case_id = item.get_str("case_id")
        partition = item.get_str("partition")
        if partition not in required or case_id in ids:
            raise ValueError("partition must match split version and case IDs must be unique")
        ids.add(case_id)
        scenario_path = (source.parent / item.get_str("scenario")).resolve()
        recording = (source.parent / item.get_str("recording")).resolve()
        scenario = load_scenario(scenario_path)
        replay = ReplayBackend(scenario, recording)  # Validates every recorded frame hash.
        replay.close()
        contract_path = (scenario_path.parent / scenario.source["contract_path"]).resolve()
        row = {
            "case_id": case_id,
            "partition": partition,
            "family_id": item.get_str("family_id"),
            "world_sha256": digest(scenario.world),
            "recording_identity": scenario.recording_identity,
            "scenario_path": str(scenario_path),
            "scenario_sha256": checksum(scenario_path),
            "contract_path": str(contract_path),
            "contract_sha256": checksum(contract_path),
            "recording_path": str(recording),
            "recording_sha256": checksum(recording / "recording.json"),
        }
        for key, mapping in seen.items():
            if row[key] in mapping and mapping[row[key]] != partition:
                raise ValueError(f"cross-partition leakage: {key}")
            mapping[row[key]] = partition
        rows.append(row)
    if {r["partition"] for r in rows} != required:
        raise ValueError(f"all partitions are required: {sorted(required)}")
    manifest = {
        "split_schema_version": version,
        "protocol": "frozen_input_integrity_only",
        "independence_verified": False,
        "entries": rows,
    }
    # Never replace an existing frozen manifest.
    with output.open("x") as handle:
        import json

        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(path: Path) -> dict:
    manifest = read_json_object(path)
    version = manifest.get("split_schema_version")
    required = partitions(version)
    rows = manifest["entries"]
    if not rows or {r["partition"] for r in rows} != required:
        raise ValueError(f"all partitions are required: {sorted(required)}")
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate case IDs")
    seen = {key: {} for key in ("family_id", "world_sha256", "recording_identity")}
    for row in rows:
        for name in ("scenario", "contract"):
            if checksum(Path(row[f"{name}_path"])) != row[f"{name}_sha256"]:
                raise ValueError(f"frozen {name} changed: {row['case_id']}")
        recording = Path(row["recording_path"])
        if checksum(recording / "recording.json") != row["recording_sha256"]:
            raise ValueError(f"frozen recording changed: {row['case_id']}")
        scenario = load_scenario(Path(row["scenario_path"]))
        if (
            digest(scenario.world) != row["world_sha256"]
            or scenario.recording_identity != row["recording_identity"]
        ):
            raise ValueError("frozen identity metadata differs from scenario")
        for key, mapping in seen.items():
            if row[key] in mapping and mapping[row[key]] != row["partition"]:
                raise ValueError(f"cross-partition leakage: {key}")
            mapping[row[key]] = row["partition"]
        replay = ReplayBackend(scenario, recording)
        replay.close()
    return {
        "split_schema_version": version,
        "status": "verified",
        "manifest_sha256": checksum(path),
        "entry_count": len(manifest["entries"]),
    }


def verify_membership(manifest: Path, scenario: Path, recording: Path, partition: str) -> dict:
    verification = verify(manifest)
    matches = [
        row
        for row in read_json_object(manifest)["entries"]
        if row["scenario_path"] == str(scenario.resolve())
        and row["recording_path"] == str(recording.resolve())
        and row["partition"] == partition
    ]
    if len(matches) != 1:
        raise ValueError(
            "run must match exactly one frozen scenario/recording in the requested partition"
        )
    return {
        **verification,
        "partition": partition,
        "case_id": matches[0]["case_id"],
        "family_id": matches[0]["family_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.output is None:
            parser.error("freeze requires --output")
        freeze(args.input, args.output)
        print(f"Frozen split: {args.output}")
    else:
        result = verify(args.input)
        if args.output:
            save_json(args.output, result)
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
