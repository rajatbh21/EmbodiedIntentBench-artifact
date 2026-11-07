"""Acquire the versioned physical cohort and enforce admission before neural evaluation."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from experiments.lake_missions.assets import save, sha


def utc():
    return datetime.now(UTC).isoformat()


def run(root):
    from experiments.mission_environment_transfer_v3.cohort import record, verify
    from experiments.mission_environment_transfer_v3.sensor import run as screen
    from experiments.mission_environment_transfer_v3.sensor import verify_gate
    from experiments.mission_transfer_comparison_v3.protocol import verify as verify_study

    p = verify(root)
    verify_study(root / "study")
    if (root / "acquisition_order.json").exists() or any(
        (root / flight / name).exists()
        for flight in p["cases"]
        for name in ("recording", "live", "predictions")
    ):
        raise FileExistsError("Preserve every acquisition attempt; use a versioned cohort")
    save(
        root / "acquisition_order.json",
        dict(
            created_utc=utc(),
            source_sha256=sha(Path(__file__)),
            cohort_protocol_sha256=sha(root / "protocol.json"),
            study_protocol_sha256=sha(root / "study/protocol.json"),
            methods_frozen_before_this_capture_run=True,
            flights=list(p["cases"]),
            neural_inference_started=False,
            execution="Sequential physical recordings with unchanged capture schedule",
        ),
    )
    completed = []
    try:
        for flight in p["cases"]:
            start = time.monotonic()
            record(root, flight)
            completed.append(
                dict(
                    flight=flight,
                    elapsed_s=time.monotonic() - start,
                    completed_utc=utc(),
                    recording_sha256=sha(root / flight / "recording/recording.json"),
                )
            )
            save(
                root / "acquisition_progress.json",
                dict(expected=len(p["cases"]), finished=len(completed), cases=completed),
            )
            print(completed[-1], flush=True)
        result = screen(root)
        save(
            root / "acquisition_completion.json",
            dict(
                acquisition_complete=True,
                admitted_to_model_evaluation=result["passed"],
                complete=result["passed"],
                cases=completed,
                completed_utc=utc(),
                sensor_gate_sha256=sha(root / "sensor_gate.json"),
                physical_route_validation_sha256=sha(root / "physical_route_validation.json"),
                source_sha256=sha(Path(__file__)),
                neural_inference_started=False,
            ),
        )
        # screen() returns diagnostic data even when admission fails. Verification
        # is deliberately separate and mandatory: capture completion is not admission.
        verify_gate(root)
        print(dict(acquisition_complete=True, admitted_to_model_evaluation=True), flush=True)
        return result
    except Exception as error:
        save(
            root / "acquisition_failure.json",
            dict(
                complete=False,
                admitted_to_model_evaluation=False,
                completed_recordings=len(completed),
                error_type=type(error).__name__,
                error=str(error),
                created_utc=utc(),
                source_sha256=sha(Path(__file__)),
                neural_inference_started=False,
                original_attempt_preserved=True,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    run(parser.parse_args().cohort)
