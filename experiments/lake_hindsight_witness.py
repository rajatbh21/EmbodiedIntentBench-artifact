"""Find evaluator-only schedules and verify them with the unchanged physical replay runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.lake_fast_evaluation import load_fast_scenario
from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_search import HindsightSchedule, search
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.objectives import score
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import CachedExecutor
from experiments.v4_benchmark.runner import run_mission


class VerifiedMasks:
    """Load a hash-verified mask on demand; inference latency still comes from specs."""

    def __init__(self, directory, model):
        self.directory, self.model = directory, model

    def __contains__(self, key):
        return key in self.model["predictions"]

    def __getitem__(self, key):
        import hashlib

        import numpy as np

        item = self.model["predictions"][key]
        path = (self.directory / item["file"]).resolve()
        assert path.parent == self.directory.resolve() and sha(path) == item["sha256"]
        with np.load(path, allow_pickle=False) as data:
            mask = data["prediction_mask"].copy()
        assert mask.dtype == np.bool_ and mask.ndim == 2
        assert hashlib.sha256(mask.tobytes()).hexdigest() == item["mask_sha256"]
        return mask


def run(output, environment, mission_id, beam_width):
    from experiments.v4_benchmark.calibrated_router import model_signature

    p = read(output / "protocol.json")
    source, prior = Path(p["source"]), Path(p["prior"])
    freeze(source)
    suite = read(source / "suite.json")
    assets = Path(suite["assets"])
    base = load(source / "scenario.json", suite["environments"][environment], assets)
    control = next(s for s in base.executors if s.config_id == "CFG_MASKRCNN_1280")
    case = output / environment / "missions" / mission_id
    scenario = load_fast_scenario(
        case / "scenario.json", suite["environments"][environment], assets, control
    )
    mission = next(
        m for m in read(source / "missions.json")["missions"] if m["mission_id"] == mission_id
    )
    outcomes_path = output / environment / "outcomes.json"
    destination = output / "hindsight" / environment / mission_id
    destination.mkdir(parents=True, exist_ok=False)
    save(
        destination / "protocol.json",
        dict(
            schema_version="1.0",
            gt_aware=True,
            deployable_policy=False,
            beam_width=beam_width,
            search_sha256=sha(Path(__file__).with_name("lake_hindsight_search.py")),
            verifier_sha256=sha(Path(__file__)),
            outcomes_sha256=sha(outcomes_path),
            scenario_sha256=sha(case / "scenario.json"),
            missions_sha256=sha(source / "missions.json"),
            source_protocol_sha256=sha(output / "protocol.json"),
            failure_rule="Beam failure is inconclusive; exact relaxations may certify impossibility",
        ),
    )
    result = search(mission, scenario, read(outcomes_path), beam_width)
    save(destination / "search.json", result)
    print(environment, mission_id, result["conclusion"], flush=True)
    if result["schedule"] is None:
        return result
    original = Path(read(prior / "protocol.json")["prior"])
    directories = [
        original / environment / "predictions",
        prior / environment / "predictions",
        output / environment / "predictions",
    ]
    available = {}
    for directory in directories:
        manifest = read(directory / "manifest.json")
        complete = read(directory / "completion.json")
        assert complete["complete"] and complete["manifest_sha256"] == sha(
            directory / "manifest.json"
        )
        assert manifest["recording_identity"] == scenario.recording_identity
        for config, model in manifest["models"].items():
            available[config] = directory, model
    executors = {}
    for spec in scenario.executors:
        directory, model = available[spec.config_id]
        assert model["signature"] == model_signature(spec)
        executors[spec.config_id] = CachedExecutor(
            spec, VerifiedMasks(directory, model), sha(directory / "manifest.json")
        )
    backend = ReplayBackend(scenario, source / environment / "recording")
    poses = {t: f["body_pose"] for t, f in backend.frames.items()}
    execution = destination / "execution"
    router = HindsightSchedule(result["schedule"])
    try:
        metrics = run_mission(
            scenario,
            backend,
            router,
            executors,
            router_name="GT_AWARE_OFFLINE_WITNESS",
            output=execution,
        )
    finally:
        backend.close()
    events = read(execution / "events.json")
    actual = [
        dict(capture_id=e["observation_id"], config_id=e["execution"]["config_id"]) for e in events
    ]
    assert actual == result["schedule"]
    assert all(e["capture_body_pose"] == poses[e["capture_timestamp_ns"]] for e in events)
    objective = score(mission, events, metrics)
    assert objective["mission_success"], "Planner candidate failed canonical runner verification"
    predicted = (
        scenario.duration_s * scenario.flight_power_w
        + result["planned_nonflight_and_tail_energy_j"]
    )
    assert abs(predicted - metrics["cumulative_energy_j"]) < 1e-6
    save(execution / "objectives.json", objective)
    save(
        destination / "verified.json",
        dict(
            gt_aware=True,
            deployable_policy=False,
            mission_success=True,
            optimality_claim=False,
            exact_capture_schedule=True,
            exact_capture_poses=True,
            exact_energy_accounting=True,
            final_battery_frac=metrics["final_battery_frac"],
            protocol_sha256=sha(destination / "protocol.json"),
            search_sha256=sha(destination / "search.json"),
            **{
                f"{n}_sha256": sha(execution / f"{n}.json")
                for n in ("results", "events", "objectives")
            },
        ),
    )
    print(environment, mission_id, "VERIFIED FEASIBLE WITNESS", flush=True)
    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--environment", required=True, choices=("day", "dusk_haze"))
    parser.add_argument("--mission", required=True)
    parser.add_argument("--beam-width", type=int, default=500)
    args = parser.parse_args()
    run(args.output, args.environment, args.mission, args.beam_width)
