"""Bind descriptive analyses to the admitted study and exact physical recording."""

from pathlib import Path

from experiments.lake_missions.assets import sha
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_transfer_comparison_v4.protocol import check_prediction_admission


def check_diagnostics(diagnostics, protocol, study, cohort):
    expected = {
        "protocol_sha256": sha(cohort / "protocol.json"),
        "comparison_protocol_sha256": sha(study / "protocol.json"),
        "routing_admission_sha256": protocol["routing_admission_sha256"],
    }
    if any(diagnostics.get(k) != v for k, v in expected.items()):
        raise ValueError("Diagnostic evidence belongs to a different admitted comparison")
    if set(diagnostics["flights"]) != {c["source_flight"] for c in protocol["cases"].values()}:
        raise ValueError("Incomplete diagnostic source-flight grid")


def checked_prediction(cohort, flight, protocol, scenario):
    folder = cohort / flight / "predictions"
    manifest = checked_manifest(folder, cohort / "protocol.json")
    check_prediction_admission(manifest, protocol)
    if manifest.get("recording_identity") != scenario.recording_identity or manifest.get(
        "recording_sha256"
    ) != sha(cohort / flight / "recording/recording.json"):
        raise ValueError("Prediction bank belongs to a different physical recording")
    return manifest


def implementation_sha256():
    return {str(Path(__file__).resolve()): sha(Path(__file__))}
