"""Content and geometry identities for prospective transfer cohorts.

These hashes identify declared inputs, not statistical independence or equivalent
rendered images. Callers must verify every supplied asset hash against its bytes.
Source families are explicit provenance declarations: different versions/copies
of the same source map belong to the same family.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction

SCHEMA = "mission_transfer_identity_v1"
ROUTE_KINDS = {"step_setpoints", "piecewise_linear_reference"}
ANNOTATION_FIELDS = {"object_id", "label", "is_target"}


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite JSON numbers required")
    return Fraction(str(value))


def _canonical(value):
    """Preserve types while making 1, 1.0 and negative zero identical."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if type(value) in (int, float):
        return ["number", str(_number(value))]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("String JSON object keys required")
        return ["object", [[key, _canonical(value[key])] for key in sorted(value)]]
    if isinstance(value, (list, tuple)):
        return ["array", [_canonical(item) for item in value]]
    raise ValueError("Only finite JSON-compatible values are supported")


def _serialized(value):
    return json.dumps(_canonical(value), separators=(",", ":"), ensure_ascii=True)


def _digest(value):
    return hashlib.sha256(_serialized(value).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[a-fA-F0-9]{64}", value) is None:
        raise ValueError("A SHA-256 content digest is required")
    return value.lower()


def asset_content_digest(file_sha256: Mapping[str, str]) -> str:
    """Hash a multiset of verified render-dependency bytes, independent of paths.

    Supply the actual SDF/mesh/material/texture dependency inventory. Exclude ZIP
    wrappers, attribution text and retrieval metadata; they do not create a new
    map. Multiplicity is retained. This helper does not read or verify the files.
    """
    if not isinstance(file_sha256, Mapping) or not file_sha256:
        raise ValueError("A nonempty verified render-dependency inventory is required")
    if any(not isinstance(name, str) or not name for name in file_sha256):
        raise ValueError("Nonempty dependency paths required")
    return _digest(sorted(_sha(value) for value in file_sha256.values()))


def _position(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Three metric coordinates required")
    return tuple(_number(item) for item in value)


def canonical_route(waypoints, *, duration_s, route_kind="step_setpoints"):
    """Canonical timed command representation with conservative equivalences.

    Step setpoints may only merge consecutive identical positions. Collinear
    setpoints are NOT equivalent because their activation changes the controller.
    Linear references additionally merge exact equal-velocity segments. There is
    no coordinate rounding, geometric tolerance, rotation or time rescaling.
    """
    duration = _number(duration_s)
    if duration <= 0 or route_kind not in ROUTE_KINDS:
        raise ValueError("Positive duration and an explicit supported route kind required")
    if not isinstance(waypoints, (list, tuple)) or not waypoints:
        raise ValueError("Nonempty timed waypoints required")
    points = []
    for waypoint in waypoints:
        if not isinstance(waypoint, Mapping) or set(waypoint) != {"start_s", "position_m"}:
            raise ValueError("Only start_s and position_m may define a waypoint")
        time = _number(waypoint["start_s"])
        position = _position(waypoint["position_m"])
        if time < 0 or time > duration or (points and time <= points[-1][0]):
            raise ValueError("Waypoints must increase strictly within the mission duration")
        points.append((time, position))
    if points[0][0] != 0:
        raise ValueError("The route must start at time zero")
    if route_kind == "step_setpoints":
        reduced = [points[0]]
        for point in points[1:]:
            if point[1] != reduced[-1][1]:
                reduced.append(point)
    else:
        # The reference holds its last position through the declared mission end.
        # Adding the endpoint also canonicalizes redundant stationary suffixes.
        if points[-1][0] < duration:
            points.append((duration, points[-1][1]))
        reduced = []
        for point in points:
            reduced.append(point)
            while len(reduced) >= 3:
                (ta, a), (tb, b), (tc, c) = reduced[-3:]
                equal_velocity = all(
                    (vb - va) * (tc - tb) == (vc - vb) * (tb - ta)
                    for va, vb, vc in zip(a, b, c, strict=True)
                )
                if not equal_velocity:
                    break
                reduced.pop(-2)
    return [
        {"time_fraction": str(time), "position_fractions": [str(v) for v in position]}
        for time, position in reduced
    ]


def scene_identity(
    world,
    *,
    asset_sha256,
    map_family,
    duration_s,
    mesh_content_sha256=None,
    route_kind="step_setpoints",
    controller_parameters=None,
):
    """Identify a raw scenario world independently of evaluator labels/file paths.

    mesh_content_sha256 maps each raw mesh.manifest string to its verified content
    digest. A mesh without an explicit content binding is rejected. Caller-supplied
    controller_parameters must include gains/acceleration limits that differ from
    the common controller; max_speed_mps and initial position are always bound.
    """
    asset = _sha(asset_sha256)
    if not isinstance(map_family, str) or not map_family.strip():
        raise ValueError("Explicit source map family required")
    family = map_family.strip().casefold().rstrip("/")
    if not family:
        raise ValueError("Nonempty source map family required")
    meshes = {} if mesh_content_sha256 is None else mesh_content_sha256
    objects = world.get("objects")
    if not isinstance(objects, list):
        raise ValueError("A declared world object list is required")
    normalized = []
    for item in objects:
        if not isinstance(item, Mapping):
            raise ValueError("World objects must be JSON objects")
        obj = {key: value for key, value in item.items() if key not in ANNOTATION_FIELDS}
        _position(obj["position_m"])
        if "mesh" in obj:
            mesh = dict(obj["mesh"])
            manifest = mesh.pop("manifest", None)
            if manifest not in meshes or "asset" in mesh:
                raise ValueError("Bind each raw mesh manifest to verified content")
            mesh["content_sha256"] = _sha(meshes[manifest])
            obj["mesh"] = mesh
        normalized.append(obj)
    normalized.sort(key=_serialized)
    layout = _digest(normalized)
    route = _digest(
        dict(
            route_kind=route_kind,
            duration_s=duration_s,
            initial_position_fractions=[str(v) for v in _position(world["initial_position_m"])],
            max_speed_mps=world.get("max_speed_mps"),
            controller_parameters=controller_parameters or {},
            waypoints=canonical_route(
                world["waypoints"], duration_s=duration_s, route_kind=route_kind
            ),
        )
    )
    return dict(
        schema_version=SCHEMA,
        map_family=family,
        asset_sha256=asset,
        layout_geometry_sha256=layout,
        route_program_sha256=route,
        layout_group_sha256=_digest([asset, layout]),
        route_group_sha256=_digest([asset, route]),
        route_kind=route_kind,
    )


def assert_disjoint(training: Sequence[Mapping], evaluation: Mapping, *, unseen_map=False):
    """Reject exact geometry/program overlap and, optionally, map-family overlap.

    Equal coordinates in genuinely different maps are not automatically one
    spatial layout. This is an input-lineage gate, not a population-independence
    certificate. Acquisition still requires paired measured-pose checks.
    """
    if not training or type(unseen_map) is not bool:
        raise ValueError("Training identities and an explicit boolean map scope required")
    required = {
        "schema_version", "map_family", "asset_sha256", "layout_geometry_sha256",
        "route_program_sha256", "layout_group_sha256", "route_group_sha256", "route_kind",
    }
    for identity in [*training, evaluation]:
        if set(identity) != required or identity["schema_version"] != SCHEMA:
            raise ValueError("Complete supported transfer identity required")
        for key in required - {"schema_version", "map_family", "route_kind"}:
            _sha(identity[key])
        family = identity["map_family"]
        if (
            not isinstance(family, str)
            or not family
            or family != family.strip().casefold().rstrip("/")
            or identity["route_kind"] not in ROUTE_KINDS
        ):
            raise ValueError("Canonical map family and supported route kind required")
        for kind, digest_key in (
            ("layout", "layout_geometry_sha256"), ("route", "route_program_sha256")
        ):
            expected = _digest([identity["asset_sha256"], identity[digest_key]])
            if identity[f"{kind}_group_sha256"] != expected:
                raise ValueError("Inconsistent map-bound geometry identity")
    fields = ["layout_group_sha256", "route_group_sha256"]
    if unseen_map:
        fields.extend(("map_family", "asset_sha256"))
    overlaps = [key for key in fields if any(t[key] == evaluation[key] for t in training)]
    if overlaps:
        raise ValueError("Training/evaluation overlap: " + ", ".join(overlaps))
    return dict(complete=True, unseen_map=unseen_map, checked_fields=fields, overlaps=[])
