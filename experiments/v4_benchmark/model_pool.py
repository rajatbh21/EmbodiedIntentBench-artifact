"""Build an editable, provenance-labelled local segmentation candidate pool.

All weights are official COCO/VOC pretrained baselines, NOT aerial fine-tunes.
Costs and ordinal tiers are initial assumptions until separately profiled.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.v4_benchmark.backends import save_json
from experiments.v4_benchmark.scenario import digest, exported_source, finite_tree, load_scenario


def candidates():
    rows = []
    for key, model, kind, w, h, latency, energy, tier, weights in [
        (
            "LRASPP_320",
            "lraspp_mobilenet_v3_large",
            "semantic",
            320,
            240,
            0.05,
            3,
            "low",
            "COCO_WITH_VOC_LABELS_V1",
        ),
        (
            "DEEPLAB_MOBILE_320",
            "deeplabv3_mobilenet_v3_large",
            "semantic",
            320,
            240,
            0.1,
            4,
            "medium",
            "COCO_WITH_VOC_LABELS_V1",
        ),
        (
            "DEEPLAB_R50_320",
            "deeplabv3_resnet50",
            "semantic",
            320,
            240,
            0.5,
            8,
            "high",
            "COCO_WITH_VOC_LABELS_V1",
        ),
        (
            "MASKRCNN_320",
            "maskrcnn_resnet50_fpn_v2",
            "instance",
            320,
            240,
            0.5,
            8,
            "high",
            "COCO_V1",
        ),
        (
            "MASKRCNN_640",
            "maskrcnn_resnet50_fpn_v2",
            "instance",
            640,
            480,
            1.5,
            16,
            "high",
            "COCO_V1",
        ),
    ]:
        params = dict(
            model_id=model,
            weights_id=weights,
            task_class="person",
            input_width_px=w,
            input_height_px=h,
            probability_threshold=0.5,
            device="cpu",
            dtype="float32",
            latency_mode="configured",
            warmup_runs=1,
            min_component_px=0,
        )
        if kind == "instance":
            params["score_threshold"] = 0.5
        rows.append(
            dict(
                config_id=f"CFG_{key}",
                model_strategy_id=f"{model}_{w}x{h}",
                kind=f"torch_{kind}_segmentation",
                mission_latency_s=latency,
                energy_j_per_call=energy,
                communication_mb_per_call=0,
                quality_tier=tier,
                parameters=params,
            )
        )
    return rows


def prepare(source: Path, output: Path, *, cost_profile: Path | None = None):
    scenario = load_scenario(source)
    data = exported_source(scenario)
    data["executor_configs"] = candidates()
    calibration = None
    if cost_profile is not None:
        from missionbench.v2.scenario import ExecutorConfigSpec
        from experiments.v4_benchmark.calibrated_router import model_signature

        calibration = json.loads(cost_profile.read_text())
        finite_tree(calibration)
        if calibration.get("calibration_schema_version") not in ("0.1", "0.2"):
            raise ValueError("unsupported cost profile")
        for spec in data["executor_configs"]:
            profile = calibration["models"][spec["config_id"]]
            if model_signature(ExecutorConfigSpec(**spec)) != profile["signature"]:
                raise ValueError("cost profile model mismatch")
            latency = profile["measured_latency_p95_s"]
            if not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency <= 0:
                raise ValueError("invalid measured latency")
            spec["mission_latency_s"] = math.ceil(latency / 0.01) * 0.01
    data["public_profiles"] = [
        dict(
            config_id=s["config_id"],
            expected_latency_ms=s["mission_latency_s"] * 1000,
            expected_upload_mb=0,
            quality_tier=s["quality_tier"],
        )
        for s in data["executor_configs"]
    ]
    data["mission"]["fallback_config_id"] = candidates()[0]["config_id"]
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "contract.json",
        json.loads((source.parent / scenario.source["contract_path"]).read_text()),
    )
    save_json(output / "scenario.json", data)
    # Load before publishing the manifest; verifies that the pool preserves scene identity.
    prepared = load_scenario(output / "scenario.json")
    assert prepared.recording_identity == scenario.recording_identity
    save_json(
        output / "pool_manifest.json",
        {
            "pool_schema_version": "0.1",
            "purpose": "development_candidate_screening",
            "models": data["executor_configs"],
            "source": "https://docs.pytorch.org/vision/main/models.html",
            "training_domain": "COCO/VOC; NOT VisDrone or aerial-finetuned",
            "quality_tiers": "Uncalibrated architectural priors; not empirical quality rankings",
            "costs": (
                "Local diagnostic p95 rounded up to 10ms; configured latency, assumed energy"
                if calibration
                else "Initial configured assumptions; energy is not measured"
            ),
            "cost_profile_sha256": digest(calibration) if calibration else None,
            "output_semantics": (
                "Binary person masks; instance masks unioned before connected-component scoring"
            ),
            "preloading": "All configurations resident; load and switching costs excluded",
        },
    )
    return output / "scenario.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-profile", type=Path)
    args = parser.parse_args()
    print(prepare(args.scenario, args.output, cost_profile=args.cost_profile))


if __name__ == "__main__":
    main()
