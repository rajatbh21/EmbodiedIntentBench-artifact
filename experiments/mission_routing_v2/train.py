"""Evaluator-only dense development calibration; discard identities before routing."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_causal_router import split_fingerprint
from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import validate_table
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking_study import scenario_and_cache
from experiments.mission_routing_v2.features import NAMES, encode, fit
from experiments.mission_routing_v2.models import NEW
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame


def checked_manifest(folder, protocol):
    manifest = read(folder / "manifest.json")
    completion = read(folder / "completion.json")
    if (
        not completion["complete"]
        or completion["manifest_sha256"] != sha(folder / "manifest.json")
        or manifest["protocol_sha256"] != sha(protocol)
        or manifest["inference_sha256"] != sha(folder / "inference.json")
    ):
        raise ValueError("Unverified inference cache")
    return manifest


def run(root):
    p = read(root / "protocol.json")
    specs = tuple(ExecutorConfigSpec(**s) for s in p["specs"])
    source, prior = Path(p["source"]), Path(p["prior"])
    all_features, tables, lineages, latency = [], {}, [], {}
    for environment in p["environments"]:
        folder = root / environment
        manifest = checked_manifest(folder, root / "protocol.json")
        scenario, _, _ = scenario_and_cache(prior, source, environment, "sector_rescue")
        if manifest["recording_identity"] != scenario.recording_identity or manifest[
            "recording_sha256"
        ] != sha(source / environment / "recording/recording.json"):
            raise ValueError("Cache does not belong to development flight")
        lineages.append(split_fingerprint(scenario, scenario.scenery_sha256))
        scorers = {
            s.config_id: MissionEvaluator(
                scenario.matching_iou_threshold,
                len(scenario.targets),
                scenario.observation_interval_s,
            )
            for s in specs
        }
        masks = {}
        for s in specs:
            model = manifest["models"][s.config_id]
            if model["signature"] != model_signature(s):
                raise ValueError("Cache model signature mismatch")
            masks[s.config_id] = VerifiedMasks(folder, model)
        rows, features, times = [], [], []
        backend = ReplayBackend(scenario, source / environment / "recording")
        try:
            for index, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                t = (timestamp - scenario.schedule.epoch_ns) / 1e9
                times.append(t)
                features.append(encode(frame.rgb).tolist())
                truth = evaluation_frame(frame, index, t, scenario)
                for s in specs:
                    rows.append(
                        dict(
                            capture_id=index,
                            config_id=s.config_id,
                            **scorers[s.config_id]
                            .update(truth, masks[s.config_id][rgb_key(frame.rgb)])
                            .to_dict(),
                        )
                    )
        finally:
            backend.close()
        validate_table(rows, times, [s.config_id for s in specs])
        save(folder / "outcomes.json", rows)
        save(folder / "features.json", dict(names=NAMES, values=features))
        tables[environment] = rows
        all_features.extend(features)
        measurements = read(folder / "inference.json")
        for s in specs:
            values = [
                r["measured_wall_clock_s"]
                for r in measurements
                if r["config_id"] == s.config_id and r["measured_wall_clock_s"] is not None
            ]
            value = float(np.percentile(values, 95)) if s.config_id in NEW else s.mission_latency_s
            latency[s.config_id] = max(latency.get(s.config_id, 0), math.ceil(value * 100) / 100)
    models = {}
    for s in specs:
        quality = []
        for rows in tables.values():
            for row in rows:
                if row["config_id"] != s.config_id:
                    continue
                recall = len(row["matched_target_ids"]) / max(1, len(row["visible_target_ids"]))
                precision = row["matched_components"] / max(1, row["predicted_components"])
                quality.append(recall * precision)
        cadence = max(2, 2 * math.ceil((latency[s.config_id] + 0.11) / 2))
        models[s.config_id] = dict(
            signature=model_signature(s),
            quality=fit(all_features, quality),
            reward=fit(all_features, np.array(quality) * 2 / cadence),
        )
    profile = dict(
        schema_version="mission_routing_calibration_v2",
        feature_names=list(NAMES),
        training_lineage=lineages,
        models=models,
        training_scope="Both complete development flights; no evaluation labels",
        target="Per-frame visible recall times component precision; empty frames score zero",
        method="Full-information offline ridge regression, alpha=1 fixed before evaluation",
    )
    save(root / "calibration.json", profile)
    save(
        root / "costs.json",
        dict(
            latency_s=latency,
            assumed_active_cpu_power_w=15,
            energy_j={c: 15 * t for c, t in latency.items()},
            energy_measured=False,
            scope="Development laptop max-environment p95; copied prior profiles for old models",
        ),
    )
    diagnostic = {}
    for env, rows in tables.items():
        by_config = {
            s.config_id: sum(
                len(r["matched_target_ids"]) for r in rows if r["config_id"] == s.config_id
            )
            for s in specs
        }
        groups = [
            [r for r in rows if r["capture_id"] == i]
            for i in sorted({r["capture_id"] for r in rows})
        ]
        diagnostic[env] = dict(
            fixed_matches=by_config,
            best_fixed=max(by_config.values()),
            free_best_per_frame=sum(max(len(r["matched_target_ids"]) for r in g) for g in groups),
            free_union=sum(
                len(set().union(*(set(r["matched_target_ids"]) for r in g))) for g in groups
            ),
            visible_target_observations=sum(len(g[0]["visible_target_ids"]) for g in groups),
            interpretation="GT diagnostic; ignores latency, energy and repeat evidence",
        )
    save(root / "complementarity.json", diagnostic)
    save(
        root / "training_complete.json",
        dict(
            calibration_sha256=sha(root / "calibration.json"),
            costs_sha256=sha(root / "costs.json"),
            diagnostic_sha256=sha(root / "complementarity.json"),
        ),
    )
    print(diagnostic, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    run(parser.parse_args().screen)
