"""Audit downloaded field records without inventing missing telemetry or GT.

Archives are read without extraction. Output is environment evidence, not a
perception or mission-score dataset. Time-weighted power includes recorded idle.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from itertools import pairwise
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def number(value):
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite measurement")
    return result


def span(values):
    finite = sorted(v for v in values if v is not None)
    if not finite:
        return dict(count=0, minimum=None, median=None, maximum=None)
    mid = len(finite) // 2
    median = finite[mid] if len(finite) % 2 else (finite[mid - 1] + finite[mid]) / 2
    return dict(count=len(finite), minimum=finite[0], median=median, maximum=finite[-1])


def integrate_power(samples):
    if len(samples) < 2:
        raise ValueError("need at least two telemetry samples")
    points = [
        (number(row["time"]), number(row["battery_voltage"]), number(row["battery_current"]))
        for row in samples
    ]
    if any(any(v is None for v in p) for p in points):
        raise ValueError("missing telemetry: no interpolation is implicit")
    energy = 0.0
    gaps = []
    for (ta, va, ia), (tb, vb, ib) in pairwise(points):
        if tb <= ta:
            raise ValueError("timestamps must be strictly increasing")
        gaps.append(tb - ta)
        energy += (va * ia + vb * ib) / 2 * (tb - ta)
    duration = points[-1][0] - points[0][0]
    return dict(
        duration_s=duration,
        energy_j=energy,
        energy_wh=energy / 3600,
        recorded_mean_power_w=energy / duration,
        max_sample_gap_s=max(gaps),
        negative_power_samples=sum(v * i < 0 for _, v, i in points),
        positive_net_energy=energy > 0,
        calibration_status=(
            "requires platform and phase review; not automatically a flight-power calibration"
        ),
        power_scope="entire recorded interval, including idle/takeoff/landing",
    )


def verify_downloads(root):
    manifest = json.loads((root / "download_manifest.json").read_text())
    for item in manifest:
        path = root / item["dataset"] / item["name"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("download changed: " + str(path))
    return manifest


def analyze(root, output):
    manifest = verify_downloads(root)
    output.mkdir(parents=True, exist_ok=False)
    prefix = "C_search_experiments/data/"
    with zipfile.ZipFile(root / "swarm_steward/C_search_experiments.zip") as archive:
        field = rows(archive.read(prefix + "search_experiment_manifest.csv").decode())
        telemetry = rows(archive.read(prefix + "search_experiment_drone_telemetry.csv").decode())
        exclusions = rows(archive.read(prefix + "search_experiment_excluded_runs.csv").decode())
    field_cases = []
    for r in field:
        field_cases.append(
            dict(
                run_id=r["run_id"],
                session=r["session"],
                source_outcome=r["victim_detection_status"],
                drone_count=int(r["drone_count"]),
                median_altitude_m=number(r["median_altitude_m"]),
                camera_depression_deg=-number(r["median_gimbal_pitch_deg"]),
                observed_speed_mps=number(r["coverage_speed_mps"]),
                duration_s=60 * number(r["run_duration_min"]),
                source_quality_flags=r["telemetry_quality_flags"].split(";"),
                imported_as="environment_reference_only",
                limitations=[
                    "curated summaries, not raw replay",
                    "candidate counts are not false-positive labels",
                    "no RGB or mask GT in this component",
                ],
            )
        )
    parameters = {r["flight"]: r for r in rows((root / "cmu_m100/parameters.csv").read_text())}
    flights = []
    with zipfile.ZipFile(root / "cmu_m100/flights.zip") as archive:
        for info in sorted(archive.infolist(), key=lambda x: x.filename):
            if not info.filename.endswith(".csv"):
                continue
            if info.file_size > 10_000_000:
                raise ValueError("unexpected telemetry file size")
            fid = Path(info.filename).stem
            data = rows(archive.read(info).decode())
            record = dict(
                flight_id=fid,
                source_member=info.filename,
                parameters=parameters.get(fid),
                sample_count=len(data),
            )
            try:
                record.update(integrate_power(data), status="valid")
            except ValueError as exc:
                record.update(status="invalid", error=str(exc))
            flights.append(record)
    valid = [r for r in flights if r["status"] == "valid"]
    report = dict(
        schema_version="0.1",
        provenance=manifest,
        field=dict(
            cases=field_cases,
            run_count=len(field),
            telemetry_rows=len(telemetry),
            excluded_source_runs=len(exclusions),
            altitude_m=span(r["median_altitude_m"] for r in field_cases),
            speed_mps=span(r["observed_speed_mps"] for r in field_cases),
            duration_s=span(r["duration_s"] for r in field_cases),
        ),
        energy=dict(
            flights=flights,
            flight_count=len(flights),
            valid_count=len(valid),
            statuses=dict(Counter(r["status"] for r in flights)),
            recorded_mean_power_w=span(r["recorded_mean_power_w"] for r in valid),
            source_platform=(
                "DJI Matrice 100; do not transplant watts to Mini 4 Pro or simulated UAV"
            ),
            wind_note="onboard relative airflow, not automatically ambient wind",
            units_note="position_z is not assumed AGL; payload values retained without clipping",
        ),
    )
    (output / "evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source, args.output)
    print(
        json.dumps(
            dict(
                field_runs=result["field"]["run_count"],
                energy_flights=result["energy"]["flight_count"],
                valid_energy_flights=result["energy"]["valid_count"],
            )
        )
    )


if __name__ == "__main__":
    main()
