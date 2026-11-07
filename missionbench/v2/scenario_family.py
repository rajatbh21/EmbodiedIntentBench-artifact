"""Deterministic hard-scenario family generation (V2.4).

One hand-designed hard scenario proves possibility; a scenario *family* is what lets
the benchmark ask whether "adaptive beats static" is a pattern or a coincidence. This
module derives controlled variants of a base hard-trade-off scenario
(``demo_img1_hard_tradeoff.json``) that keep the qualitative structure fixed while
varying the quantities a field mission would not control:

- **Small early targets** (light-model-defeating, ~32 px): which capture slot they
  occupy on the early lanes, lateral offset, exact size within the validated 30-34 px
  band, pose, rotation.
- **Large late targets** (strong-cadence-defeating): which odd capture slot on the
  final lane, lateral offset, size, pose, rotation. Their ~1.5 s visibility windows
  stay centred near odd capture times — that misalignment with the strong executor's
  2-capture cadence *is* the family's defining stress, not a tunable.
- **Battery capacity** within a narrow band around the base scenario's value, which
  moves the adaptive policy's battery-pressure switch time and probes the fragility of
  its final margin.
- **Network conditions** (only for bases that declare a ``network_trace``, V3 P2):
  regime boundary times within a small window and within-regime link quality within a
  band. The regime *structure* — names, order, and reachability classes (good /
  degraded / disconnected / weak recovery) — is part of the base design and is never
  resampled; the family varies when the degradation arrives and how bad it is, which is
  exactly what a field mission would not control.

What is deliberately NOT varied: the executor configs (configured latencies/energies),
the contract (including its privacy level), the trajectory, the camera, the world
image, and the network regime structure. Outcome variation must come from where
targets fall, when the link turns, and how the real models respond — never from
forcing a winner. Seeds that break the hypothesis are kept and reported.

Determinism: every variant is a pure function of
``(FAMILY_VERSION, base scenario id, seed)``. All sampled values are rounded, recorded
under ``provenance.scenario_family``, and the emitted JSON is strict
(``allow_nan=False``, sorted keys), so identical inputs yield identical bytes. No
wall-clock, no global RNG.

Version history (a sampling change is a version bump — variant identity depends on it):

- **1.0** (never released): sampled late slots from the whole final lane (>= ~37 s)
  with +-0.2 s jitter. A 5-seed smoke run exposed two mismatches with the documented
  hard-scenario structure: late targets could appear *before* the adaptive policy's
  battery-pressure switch (measuring switch-timing luck, not adaptation value), and
  the jitter let wide targets clip the edge of a strong-cadence frame. Finding kept
  in the V2.4 report as sensitivity evidence.
- **1.1**: late slots start at >= 41 s (the post-switch window the hard-scenario
  design defines), late jitter is +-0.1 s, and the late window runs to just before
  path completion. On the base trajectory that fixes the late slot *set* at
  {41, 43, 45, 47}; late-target variation comes from jitter, lateral offset, size,
  pose, rotation, background patch, and battery capacity — not slot choice.
- **2.0** (V3 P2): the family generalises across base scenarios and gains network
  dimensions. Battery capacity is sampled *relative* to the base value (x0.97-1.03,
  reproducing the 1.1 band on the V2.4 base) instead of a hardcoded absolute band;
  late-target heights are sampled relative to the base late target of the same pose
  (x0.93-1.07) because "comfortably detectable by both models" is a property of each
  world's background, while the small-target band stays the absolute Stage-B-validated
  0.70-0.80 m. For bases declaring a ``network_trace``, regime boundary times jitter
  +-2.0 s and within-regime link quality scales x0.75-1.3 (uplink and downlink
  together; RTT x0.85-1.25); regime structure, order, reachability classes, and packet
  loss are base design and never resampled. All 1.1 variant identities change (the
  version participates in the RNG stream); the V2.4 30-seed result under 1.1 stands as
  recorded history.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Final

from missionbench.schemas.loading import SchemaValidationError, read_json_object
from missionbench.v2.scenario import V2Scenario, load_scenario
from missionbench.v2.trajectory import PolylineTrajectory, build_trajectory

__all__ = ["FAMILY_VERSION", "variant_id", "write_variant"]

#: Version of the family sampling scheme; bump on any change to what or how we sample.
FAMILY_VERSION: Final = "2.0"

#: Asset aspect ratios (width/height) of the poses the base scenario uses, from the
#: asset manifest's nominal physical dimensions.
_POSES: Final = {
    "walking": {"asset_id": "person_aerial_walking_001", "aspect": 0.755 / 1.5},
    "standing": {"asset_id": "person_aerial_standing_001", "aspect": 1.42 / 2.25},
}
#: Small-target ground extent (m): the Stage-B-validated "light model detects nothing"
#: band (~30-34 px at the base scenario's 42.7 px/m). Absolute — a perceptual constant
#: of the rendered pixel size, not a property of any one world.
_SMALL_HEIGHT_RANGE: Final = (0.70, 0.80)
#: Late-target height factor relative to the base scenario's late target of the same
#: pose. "Comfortably detectable by both models" depends on each world's background,
#: so the base scenario owns the operating point and the family probes around it.
_LATE_HEIGHT_REL_RANGE: Final = (0.93, 1.07)
_LATERAL_OFFSET_MAX_M: Final = 1.2
_ROTATION_MAX_DEG: Final = 45.0
#: Battery capacity factor relative to the base scenario's capacity (reproduces the
#: 1.1 absolute band on the V2.4 base: 6.806 Wh x 0.97-1.03 = 6.60-7.01 Wh).
_BATTERY_CAPACITY_REL_RANGE: Final = (0.97, 1.03)
#: Network sampling bands (bases with a ``network_trace`` only): regime boundary
#: jitter, a single link-quality factor applied to uplink and downlink together, and
#: an independent RTT factor. Packet loss and reachability are regime identity.
_REGIME_START_JITTER_S: Final = 2.0
_LINK_SCALE_RANGE: Final = (0.75, 1.3)
_RTT_SCALE_RANGE: Final = (0.85, 1.25)
#: Minimum surviving gap between consecutive regime starts after jitter.
_REGIME_MIN_GAP_S: Final = 1.0
#: Time jitter around the chosen capture slot. Small targets stay within +-0.3 s of an
#: even slot (their window always covers a strong-processed capture); late targets stay
#: within +-0.1 s of an odd slot (their window never covers one, up to a possible
#: few-pixel sliver of the widest targets at a frame edge — kept as genuine noise).
_SMALL_JITTER_S: Final = 0.3
_LATE_JITTER_S: Final = 0.1
#: Late targets belong to the post-battery-pressure window the hard-scenario design
#: defines; slots earlier than this measured switch-timing luck (family 1.0 finding).
_LATE_WINDOW_START_S: Final = 41.0
#: The final lane may end mid-slot; a capture still happens right up to path completion
#: (the reference scenario's last target sits 0.25 s before the end), so the late
#: window only needs a small end margin.
_LATE_END_MARGIN_S: Final = 0.25
#: Keep sampled times at least this far from a lane's ends (turns distort geometry).
_LANE_MARGIN_S: Final = 1.5
_SMALL_TARGET_COUNT: Final = 4
_LATE_TARGET_COUNT: Final = 4


def variant_id(base_scenario_id: str, seed: int) -> str:
    return f"{base_scenario_id}_SEED_{seed:04d}"


def write_variant(base_scenario_path: Path, seed: int, output_path: Path) -> Path:
    """Generate one deterministic variant, write it, and strictly re-validate it."""
    payload = generate_variant(base_scenario_path, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_scenario(output_path)  # the generator never ships an invalid scenario
    return output_path


def generate_variant(base_scenario_path: Path, seed: int) -> dict[str, Any]:
    """A variant payload as a pure function of (family version, base id, seed)."""
    if seed < 0:
        raise SchemaValidationError(f"seed must be non-negative, got {seed}")
    base_raw = read_json_object(base_scenario_path)
    base = load_scenario(base_scenario_path)
    rng = random.Random(
        f"missionbench.v2.scenario_family/{FAMILY_VERSION}/{base.scenario_id}/{seed}"
    )
    trajectory = build_trajectory(base.trajectory, base.drone.speed_mps)

    early_slots, late_slots = _capture_slots(trajectory)
    small_times = sorted(rng.sample(early_slots, _SMALL_TARGET_COUNT))
    late_times = sorted(rng.sample(late_slots, _LATE_TARGET_COUNT))

    late_poses = ["standing", "standing", "walking", "walking"]
    rng.shuffle(late_poses)
    base_late_heights = _base_late_heights(base)

    targets: list[dict[str, Any]] = []
    sampled: list[dict[str, Any]] = []
    for index, slot in enumerate(small_times):
        pose = rng.choice(["walking", "standing"])
        height = round(rng.uniform(*_SMALL_HEIGHT_RANGE), 3)
        targets.append(
            _target(
                object_id=f"TGT_S{index + 1}_{pose.upper()}_SMALL",
                pose=pose,
                height_m=height,
                slot=slot,
                jitter=round(rng.uniform(-_SMALL_JITTER_S, _SMALL_JITTER_S), 2),
                lateral=round(rng.uniform(-_LATERAL_OFFSET_MAX_M, _LATERAL_OFFSET_MAX_M), 3),
                rotation=round(rng.uniform(-_ROTATION_MAX_DEG, _ROTATION_MAX_DEG), 1),
                z_order=10 + index,
                trajectory=trajectory,
                base=base,
                sampled=sampled,
            )
        )
    for index, slot in enumerate(late_times):
        pose = late_poses[index]
        height = round(base_late_heights[pose] * rng.uniform(*_LATE_HEIGHT_REL_RANGE), 3)
        targets.append(
            _target(
                object_id=f"TGT_E{index + 1}_{pose.upper()}_LATE",
                pose=pose,
                height_m=height,
                slot=slot,
                jitter=round(rng.uniform(-_LATE_JITTER_S, _LATE_JITTER_S), 2),
                lateral=round(rng.uniform(-_LATERAL_OFFSET_MAX_M, _LATERAL_OFFSET_MAX_M), 3),
                rotation=round(rng.uniform(-_ROTATION_MAX_DEG, _ROTATION_MAX_DEG), 1),
                z_order=20 + index,
                trajectory=trajectory,
                base=base,
                sampled=sampled,
            )
        )

    distractor = _distractor(rng, trajectory, base)
    battery_wh = round(
        base.drone.battery_capacity_wh * rng.uniform(*_BATTERY_CAPACITY_REL_RANGE), 3
    )

    payload = dict(base_raw)
    payload["scenario_id"] = variant_id(base.scenario_id, seed)
    payload["random_seed"] = seed
    payload["drone"] = {**base_raw["drone"], "battery_capacity_wh": battery_wh}
    payload["objects"] = [*targets, distractor]

    family: dict[str, Any] = {
        "family_version": FAMILY_VERSION,
        "base_scenario_id": base.scenario_id,
        "base_scenario_sha256": hashlib.sha256(base_scenario_path.read_bytes()).hexdigest(),
        "seed": seed,
        "battery_capacity_wh": battery_wh,
        "targets": sampled,
    }
    base_trace = (base_raw.get("simulation") or {}).get("network_trace")
    if base_trace:
        sampled_trace = _sample_network_trace(rng, base_trace)
        payload["simulation"] = {**base_raw["simulation"], "network_trace": sampled_trace}
        family["network_regimes"] = sampled_trace

    payload["provenance"] = {
        **dict(base_raw.get("provenance") or {}),
        "scenario_family": family,
    }
    return payload


# --- sampling helpers ------------------------------------------------------------------------


def _base_late_heights(base: V2Scenario) -> dict[str, float]:
    """The base scenario's late-target height per pose — the family's anchor points."""
    asset_to_pose = {info["asset_id"]: pose for pose, info in _POSES.items()}
    heights: dict[str, float] = {}
    for obj in base.targets:
        if not obj.object_id.startswith("TGT_E"):
            continue
        pose = asset_to_pose.get(getattr(obj, "asset_id", None))
        if pose is not None and pose not in heights:
            heights[pose] = obj.height_m
    missing = [pose for pose in _POSES if pose not in heights]
    if missing:
        raise SchemaValidationError(
            f"the scenario family needs a base late target for every pose; "
            f"base {base.scenario_id!r} declares none for {missing}"
        )
    return heights


def _sample_network_trace(
    rng: random.Random, base_trace: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Jitter regime boundaries and scale within-regime link quality, structure fixed.

    The first regime keeps ``start_s`` 0.0 (the trace contract). Unreachable regimes
    (zero uplink or downlink) are copied verbatim — disconnection is regime identity,
    not a sampled quantity — as is ``packet_loss_frac`` everywhere, because crossing a
    transport's loss ceiling would silently change a regime's failure class.
    """
    sampled: list[dict[str, Any]] = []
    previous_start = 0.0
    for index, regime in enumerate(base_trace):
        entry = dict(regime)
        if index > 0:
            jitter = rng.uniform(-_REGIME_START_JITTER_S, _REGIME_START_JITTER_S)
            entry["start_s"] = round(float(regime["start_s"]) + jitter, 1)
            if entry["start_s"] < previous_start + _REGIME_MIN_GAP_S:
                raise SchemaValidationError(
                    f"network regime {entry.get('regime_id')!r} start "
                    f"{entry['start_s']} collides with the previous regime at "
                    f"{previous_start}; the base trace's gaps are too small for "
                    f"+-{_REGIME_START_JITTER_S} s jitter"
                )
        reachable = float(regime["uplink_mbps"]) > 0.0 and float(regime["downlink_mbps"]) > 0.0
        if reachable:
            link_scale = rng.uniform(*_LINK_SCALE_RANGE)
            rtt_scale = rng.uniform(*_RTT_SCALE_RANGE)
            entry["uplink_mbps"] = round(float(regime["uplink_mbps"]) * link_scale, 1)
            entry["downlink_mbps"] = round(float(regime["downlink_mbps"]) * link_scale, 1)
            entry["rtt_ms"] = round(float(regime["rtt_ms"]) * rtt_scale, 1)
        sampled.append(entry)
        previous_start = float(entry["start_s"])
    return sampled


# --- geometry helpers ------------------------------------------------------------------------


def _capture_slots(trajectory: PolylineTrajectory) -> tuple[list[int], list[int]]:
    """Usable integer capture times: even slots on the early lanes, odd on the last.

    Lanes are the trajectory's long straight segments; short connecting turns are
    excluded (with a margin) so every sampled position sits on a clean lane run.
    """
    spans = _lane_time_spans(trajectory)
    if len(spans) < 2:
        raise SchemaValidationError(
            "the scenario family needs at least two lanes (early + late); "
            f"found {len(spans)} straight segments"
        )
    *early_spans, late_span = spans
    early = [
        t
        for start, end in early_spans
        for t in range(int(start) + 1, int(end) + 1)
        if t % 2 == 0 and start + _LANE_MARGIN_S <= t <= end - _LANE_MARGIN_S
    ]
    late = [
        t
        for t in range(int(late_span[0]) + 1, int(late_span[1]) + 1)
        if t % 2 == 1
        and t >= _LATE_WINDOW_START_S
        and late_span[0] + _LANE_MARGIN_S <= t <= late_span[1] - _LATE_END_MARGIN_S
    ]
    if len(early) < _SMALL_TARGET_COUNT or len(late) < _LATE_TARGET_COUNT:
        raise SchemaValidationError(
            f"not enough capture slots for the family (early={len(early)}, "
            f"late={len(late)}); the base trajectory is too short"
        )
    return early, late


def _lane_time_spans(trajectory: PolylineTrajectory) -> list[tuple[float, float]]:
    """(start_s, end_s) of each long straight segment, in flight order."""
    import math

    lengths = [
        math.dist(a, b)
        for a, b in zip(trajectory.waypoints_m, trajectory.waypoints_m[1:], strict=False)
    ]
    longest = max(lengths)
    spans: list[tuple[float, float]] = []
    start = 0.0
    for length in lengths:
        end = start + length / trajectory.speed_mps
        if length >= 0.5 * longest:
            spans.append((start, end))
        start = end
    return spans


def _target(
    *,
    object_id: str,
    pose: str,
    height_m: float,
    slot: int,
    jitter: float,
    lateral: float,
    rotation: float,
    z_order: int,
    trajectory: PolylineTrajectory,
    base: V2Scenario,
    sampled: list[dict[str, Any]],
) -> dict[str, Any]:
    time_s = round(slot + jitter, 2)
    on_path = trajectory.position_at(time_s)
    position = (round(on_path[0], 3), round(on_path[1] + lateral, 3))
    _check_in_valid_region(base, object_id, position)
    width_m = round(height_m * _POSES[pose]["aspect"], 3)
    sampled.append(
        {
            "object_id": object_id,
            "pose": pose,
            "capture_slot_s": slot,
            "time_jitter_s": jitter,
            "centre_time_s": time_s,
            "lateral_offset_m": lateral,
            "height_m": height_m,
            "rotation_deg": rotation,
        }
    )
    return {
        "object_id": object_id,
        "class_id": "person",
        "is_target": True,
        "render_mode": "image_asset",
        "asset_id": _POSES[pose]["asset_id"],
        "position_m": list(position),
        "width_m": width_m,
        "height_m": height_m,
        "rotation_deg": rotation,
        "z_order": z_order,
    }


def _distractor(
    rng: random.Random, trajectory: PolylineTrajectory, base: V2Scenario
) -> dict[str, Any]:
    time_s = round(rng.uniform(18.0, 28.0), 2)
    on_path = trajectory.position_at(time_s)
    lateral = round(rng.uniform(-1.0, 1.0), 3)
    position = (round(on_path[0], 3), round(on_path[1] + lateral, 3))
    _check_in_valid_region(base, "DIS_DEBRIS", position)
    return {
        "object_id": "DIS_DEBRIS",
        "class_id": "debris",
        "is_target": False,
        "render_mode": "procedural_marker",
        "position_m": list(position),
        "width_m": 1.4,
        "height_m": 0.9,
        "rotation_deg": round(rng.uniform(-45.0, 45.0), 1),
        "z_order": 5,
        "appearance": {
            "shape": "rect",
            "body_rgb": [146, 95, 100],
            "stripe": False,
            "alpha": 1.0,
        },
    }


def _check_in_valid_region(base: V2Scenario, object_id: str, position: tuple[float, float]) -> None:
    region = base.world.valid_region_m
    if region is None:
        return
    x0, y0, x1, y1 = region
    x, y = position
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        raise SchemaValidationError(
            f"family generator placed {object_id!r} at {position} outside "
            f"valid_region_m {region}; adjust the family's slot/offset bounds"
        )
