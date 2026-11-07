"""Extract textured props and place deterministic, terrain-grounded route scenery.

Only derived files in the requested output directory are written. Original Fuel
meshes, textures and prior cohorts remain unchanged. Tree and bush components are
grouped geometrically before placement; an entire world-positioned scenery mesh
is never copied as a prop. Physical colliders are conservative proxy volumes,
not claims of detailed branch/leaf contact geometry.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from experiments.lake_missions.placement import collision_bounds, terrain_height


NS_URI = "http://www.collada.org/2005/11/COLLADASchema"
NS = {"c": NS_URI}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mesh_parts(path):
    root = ET.parse(path).getroot()
    if root.findtext("c:asset/c:up_axis", namespaces=NS) != "Z_UP":
        raise ValueError("Expected the pinned Z_UP scenery source")
    unit = float(root.find("c:asset/c:unit", NS).get("meter"))
    transforms = {}

    def visit(node, parent):
        matrix = parent.copy()
        for child in node:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "matrix":
                matrix = matrix @ np.fromstring(child.text, sep=" ").reshape(4, 4)
            elif tag in {"translate", "rotate", "scale"}:
                raise ValueError("Unsupported source transform; do not guess")
        for instance in node.findall("c:instance_geometry", NS):
            transforms[instance.get("url")[1:]] = matrix
        for child in node.findall("c:node", NS):
            visit(child, matrix)

    for node in root.findall("c:library_visual_scenes/c:visual_scene/c:node", NS):
        visit(node, np.eye(4))
    parts = {}
    for geometry in root.findall("c:library_geometries/c:geometry", NS):
        mesh = geometry.find("c:mesh", NS)
        sources = {}
        for source in mesh.findall("c:source", NS):
            stride = int(source.find("c:technique_common/c:accessor", NS).get("stride"))
            sources[source.get("id")] = np.fromstring(
                source.findtext("c:float_array", namespaces=NS), sep=" "
            ).reshape(-1, stride)
        position_ref = mesh.find("c:vertices/c:input[@semantic='POSITION']", NS).get("source")[1:]
        triangles = mesh.findall("c:triangles", NS)
        if len(triangles) != 1:
            raise ValueError("Expected one material per source geometry")
        triangles = triangles[0]
        inputs = triangles.findall("c:input", NS)
        stride = max(int(item.get("offset")) for item in inputs) + 1
        indices = np.fromstring(triangles.findtext("c:p", namespaces=NS), sep=" ", dtype=int)
        indices = indices.reshape(-1, 3, stride)
        lookup = {(item.get("semantic"), item.get("set", "0")): item for item in inputs}
        vertices = indices[:, :, int(lookup["VERTEX", "0"].get("offset"))]
        normal = lookup["NORMAL", "0"]
        uv = lookup["TEXCOORD", "0"]
        matrix = transforms[geometry.get("id")]
        position = (sources[position_ref] @ matrix[:3, :3].T + matrix[:3, 3]) * unit
        normals = sources[normal.get("source")[1:]] @ np.linalg.inv(matrix[:3, :3])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        parts[geometry.get("id").removesuffix("-lib")] = dict(
            positions=position,
            vertices=vertices,
            normals=normals[indices[:, :, int(normal.get("offset"))]],
            uv=sources[uv.get("source")[1:]][indices[:, :, int(uv.get("offset"))]][:, :, :2],
        )
    return parts


def _components(part):
    """Triangle connectivity, welding duplicated UV-seam positions first."""
    positions, triangles = part["positions"], part["vertices"]
    parent = np.arange(len(positions))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(first, second):
        parent[find(first)] = find(second)

    welded = {}
    for index, point in enumerate(np.round(positions, 4)):
        key = tuple(point)
        if key in welded:
            union(index, welded[key])
        else:
            welded[key] = index
    for first, second, third in triangles:
        union(first, second)
        union(second, third)
    groups = defaultdict(list)
    for index, triangle in enumerate(triangles):
        groups[find(triangle[0])].append(index)
    return [np.asarray(group, dtype=int) for group in groups.values()]


def _points(part, selected):
    return part["positions"][part["vertices"][selected]].reshape(-1, 3)


def _bush_groups(part):
    """Group complete leaf triangles through neighboring occupied 1 m voxels."""
    centroids = part["positions"][part["vertices"]].mean(axis=1)
    cells = defaultdict(list)
    for index, cell in enumerate(np.floor(centroids).astype(int)):
        cells[tuple(cell)].append(index)
    pending = set(cells)
    groups = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        stack, selected = [seed], []
        while stack:
            current = stack.pop()
            selected.extend(cells[current])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbour = current[0] + dx, current[1] + dy, current[2] + dz
                        if neighbour in pending:
                            pending.remove(neighbour)
                            stack.append(neighbour)
        groups.append(np.asarray(sorted(selected), dtype=int))
    return groups


def _write_dae(path, part, selected, origin):
    """Write selected textured triangles in meters, preserving UV0 and normals."""
    positions = _points(part, selected) - origin
    normals = part["normals"][selected].reshape(-1, 3)
    uv = part["uv"][selected].reshape(-1, 2)
    root = ET.Element("COLLADA", xmlns=NS_URI, version="1.4.1")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "unit", name="meter", meter="1")
    ET.SubElement(asset, "up_axis").text = "Z_UP"
    library = ET.SubElement(root, "library_geometries")
    mesh = ET.SubElement(ET.SubElement(library, "geometry", id="prop", name="prop"), "mesh")
    for name, values, params in (("position", positions, "XYZ"), ("normal", normals, "XYZ"), ("uv", uv, "ST")):
        source = ET.SubElement(mesh, "source", id=name)
        ET.SubElement(source, "float_array", id=name + "-array", count=str(values.size)).text = " ".join(
            format(float(value), ".8g") for value in values.ravel()
        )
        accessor = ET.SubElement(ET.SubElement(source, "technique_common"), "accessor",
                                 source="#" + name + "-array", count=str(len(values)), stride=str(len(params)))
        for param in params:
            ET.SubElement(accessor, "param", name=param, type="float")
    vertices = ET.SubElement(mesh, "vertices", id="vertices")
    ET.SubElement(vertices, "input", semantic="POSITION", source="#position")
    triangles = ET.SubElement(mesh, "triangles", count=str(len(selected)))
    ET.SubElement(triangles, "input", semantic="VERTEX", source="#vertices", offset="0")
    ET.SubElement(triangles, "input", semantic="NORMAL", source="#normal", offset="1")
    ET.SubElement(triangles, "input", semantic="TEXCOORD", source="#uv", offset="2", set="0")
    ET.SubElement(triangles, "p").text = " ".join(str(index) for index in np.repeat(np.arange(len(positions)), 3))
    scene = ET.SubElement(ET.SubElement(root, "library_visual_scenes"), "visual_scene", id="scene")
    ET.SubElement(ET.SubElement(scene, "node", id="prop"), "instance_geometry", url="#prop")
    ET.SubElement(ET.SubElement(root, "scene"), "instance_visual_scene", url="#scene")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _extract_assets(assets, output):
    source = assets / "harmonic_terrain_objects/meshes/terrainobjects.dae"
    parts = _mesh_parts(source)
    selections = []
    for name in ("Rock01", "Rock02"):
        candidates = []
        for group in _components(parts[name]):
            points = _points(parts[name], group)
            size = np.ptp(points, axis=0)
            if 1.5 <= max(size[:2]) <= 5 and 1 <= size[2] <= 4:
                candidates.append((tuple(points.mean(axis=0)), group))
        if not candidates:
            raise ValueError("No isolated rock component in source")
        selections.append((name.lower(), [(name, sorted(candidates, key=lambda item: item[0])[0][1])], "connected triangles"))

    # A trunk component is vertical, slender, and at least 6 m tall. Assign all
    # bark/leaf triangles to their nearest trunk; select an isolated tree so no
    # neighboring crown overlaps the selected tree's horizontal envelope.
    trunks = []
    for group in _components(parts["JeffreyTreeBark"]):
        points = _points(parts["JeffreyTreeBark"], group)
        size = np.ptp(points, axis=0)
        if size[2] >= 6 and max(size[:2]) < 2.5 and size[2] > 4 * max(size[:2]):
            trunks.append((points.min(axis=0) + points.max(axis=0)) / 2)
    trunks = np.asarray(trunks)
    if len(trunks) < 2:
        raise ValueError("Cannot isolate source trees")
    spacing = np.linalg.norm(trunks[:, None, :2] - trunks[None, :, :2], axis=2)
    np.fill_diagonal(spacing, np.inf)
    assignments = {}
    for name in ("JeffreyTreeBark", "JeffreyTreeLeaves"):
        centroids = parts[name]["positions"][parts[name]["vertices"]].mean(axis=1)
        assignments[name] = np.argmin(np.linalg.norm(centroids[:, None, :2] - trunks[None, :, :2], axis=2), axis=1)
    trees = []
    for index in np.argsort(-spacing.min(axis=1)):
        selected = [(name, np.flatnonzero(assignments[name] == index)) for name in assignments]
        points = np.concatenate([_points(parts[name], group) for name, group in selected])
        size = np.ptp(points, axis=0)
        radius = np.linalg.norm(points[:, :2] - trunks[index, :2], axis=1).max()
        if 7 <= size[2] <= 19 and radius < spacing[index].min() / 2 and max(size[:2]) <= 13:
            trees.append(selected)
            if len(trees) == 2:
                break
    if not trees:
        raise ValueError("No separated complete tree crown can be extracted")
    for index, selected in enumerate(trees):
        selections.append(("pine_" + str(index + 1), selected, "nearest isolated trunk; complete bark and crown triangles"))

    bushes = []
    for group in _bush_groups(parts["Bush"]):
        points = _points(parts["Bush"], group)
        size = np.ptp(points, axis=0)
        if len(group) >= 40 and 1 <= max(size[:2]) <= 5 and 0.5 <= size[2] <= 3:
            bushes.append((tuple(points.mean(axis=0)), group))
    if not bushes:
        raise ValueError("No isolated source bush cluster")
    selections.append(("bush", [("Bush", sorted(bushes, key=lambda item: item[0])[0][1])], "connected occupied voxels of whole leaf triangles"))

    texture_names = {
        "Rock01": ("Rock01_Albedo.png", "Rock01_Normal.png"),
        "Rock02": ("Rock02_Albedo.png", "Rock02_Normal.png"),
        "JeffreyTreeBark": ("Bark_Albedo.png", "Bark_Normal.png"),
        "JeffreyTreeLeaves": ("TreeJeffreyPine_Albedo.png", None),
        "Bush": ("Bush01_Albedo.png", None),
    }
    derived = []
    for name, selected, method in selections:
        points = np.concatenate([_points(parts[part], group) for part, group in selected])
        low, high = points.min(axis=0), points.max(axis=0)
        origin = np.r_[(low[:2] + high[:2]) / 2, low[2]]
        visuals = []
        for part, group in selected:
            target = output / (name + "_" + part + ".dae")
            _write_dae(target, parts[part], group, origin)
            albedo, normal = texture_names[part]
            textures = assets / "harmonic_terrain_objects/materials/textures"
            visuals.append(dict(mesh=str(target.resolve()), mesh_sha256=_sha(target),
                                albedo=str(textures / albedo), normal=str(textures / normal) if normal else None,
                                source_geometry=part, triangle_count=len(group),
                                source_triangle_selection_sha256=hashlib.sha256(group.astype("<i8").tobytes()).hexdigest()))
        derived.append(dict(name=name, kind="tree" if name.startswith("pine") else ("bush" if name == "bush" else "rock"),
                            size_m=(high - low).tolist(), source_origin_m=origin.tolist(), extraction=method, visuals=visuals))
    return derived, source


def _route_array(route_points):
    if isinstance(route_points, dict):
        for key in ("positions_m", "points", "samples", "waypoints"):
            if key in route_points:
                route_points = route_points[key]
                break
    points = [point.get("position_m", point.get("position")) if isinstance(point, dict) else point for point in route_points]
    route = np.asarray(points, dtype=float)
    if route.ndim != 2 or route.shape[1] < 2 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("Need at least two finite route positions")
    return route


def _distance_to_route(point, route):
    starts, delta = route[:-1, :2], np.diff(route[:, :2], axis=0)
    along = np.sum((point - starts) * delta, axis=1) / np.maximum(np.sum(delta * delta, axis=1), 1e-12)
    nearest = starts + np.clip(along, 0, 1)[:, None] * delta
    return float(np.linalg.norm(nearest - point, axis=1).min())


def as_placement_bounds(layout_spec):
    """Return the tuples accepted by the existing target placement guard."""
    return [(entry["name"], np.asarray(entry["low"]), np.asarray(entry["high"])) for entry in layout_spec["collision_bounds"]]


def build_layout(assets, route_points, output_dir, seed=21028, *, count=72, target_positions=(), clearance_m=10.0):
    """Make 40--200 reproducible props beside a route, never from model outputs."""
    assets, output = Path(assets).resolve(), Path(output_dir).resolve()
    if not 40 <= count <= 200:
        raise ValueError("Scenery count must stay within the bounded 40--200 prop budget")
    output.mkdir(parents=True, exist_ok=True)
    derived, source = _extract_assets(assets, output)
    route = _route_array(route_points)
    rng = np.random.default_rng(seed)
    old_bounds = collision_bounds(assets)
    props, bounds, rejected = [], [], defaultdict(int)
    segment_lengths = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(segment_lengths)]
    if cumulative[-1] < 100:
        raise ValueError("Wide-area scenery requires a route at least 100 m long")
    region_count = int(np.clip(round(cumulative[-1] / 180), 9, 28))
    cluster_positions = np.linspace(0.025, 0.975, region_count) * cumulative[-1]
    choices = [item for item in derived if item["kind"] == "tree"] * 2 + derived
    for attempt in range(count * 100):
        if len(props) == count:
            break
        # A steep region must not prevent placement in every later region.
        # Cycle attempts rather than accepted objects, retaining all rejections.
        cluster = attempt % len(cluster_positions)
        along = float(np.clip(cluster_positions[cluster] + rng.normal(0, 25), 0, cumulative[-1] - 1e-6))
        segment = min(int(np.searchsorted(cumulative, along, side="right") - 1), len(route) - 2)
        delta = route[segment + 1, :2] - route[segment, :2]
        length = np.linalg.norm(delta)
        if length < 1e-8:
            rejected["zero_length_segment"] += 1
            continue
        tangent = delta / length
        centre = route[segment, :2] + tangent * (along - cumulative[segment])
        normal = np.array([-tangent[1], tangent[0]])
        point = centre + normal * rng.choice([-1, 1]) * rng.uniform(18, 42)
        if not -570 < point[0] < 570 or not -870 < point[1] < 270:
            rejected["terrain_edge"] += 1
            continue
        asset = choices[int(rng.integers(len(choices)))]
        scale = float(rng.uniform(0.75, 1.15))
        size = np.asarray(asset["size_m"]) * scale
        radius = float(np.linalg.norm(size[:2]) / 2)
        if _distance_to_route(point, route) < clearance_m + radius:
            rejected["route_corridor"] += 1
            continue
        if any(np.linalg.norm(point - np.asarray(target)[:2]) < 8 + radius for target in target_positions):
            rejected["target_corridor"] += 1
            continue
        ground = terrain_height(assets, *point)
        slope = math.hypot((terrain_height(assets, point[0] + 1, point[1]) - terrain_height(assets, point[0] - 1, point[1])) / 2,
                           (terrain_height(assets, point[0], point[1] + 1) - terrain_height(assets, point[0], point[1] - 1)) / 2)
        if ground < 1.2 or slope > 0.3:
            rejected["water_or_steep_ground"] += 1
            continue
        footprint = [terrain_height(assets, point[0] + dx * radius, point[1] + dy * radius)
                     for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        if max(footprint) - min(footprint) > min(2.0, max(0.7, size[2] * 0.4)):
            rejected["uneven_footprint"] += 1
            continue
        # Slightly bury the source bottom so the central stem/rock contacts the
        # terrain. The proxy encloses the entire visual, including tree crowns.
        z = ground - min(0.25, size[2] * 0.08)
        low, high = np.r_[point - radius, z], np.r_[point + radius, z + size[2]]
        if any(np.all(a < high + 1) and np.all(b > low - 1) for _, a, b in old_bounds):
            rejected["original_scenery_overlap"] += 1
            continue
        if any(np.all(np.asarray(entry["low"]) < high + 1) and np.all(np.asarray(entry["high"]) > low - 1) for entry in bounds):
            rejected["new_prop_overlap"] += 1
            continue
        name = f"wide_prop_{len(props):03d}"
        props.append(dict(name=name, asset=asset["name"], kind=asset["kind"], position_m=[*point.tolist(), z],
                          yaw_rad=float(rng.uniform(-math.pi, math.pi)), scale=scale, radius_m=radius,
                          height_m=float(size[2]), terrain_height_m=ground, local_slope=slope,
                          cluster=int(cluster), route_clearance_m=_distance_to_route(point, route) - radius))
        bounds.append(dict(name=name, low=low.tolist(), high=high.tolist(), type="conservative_proxy"))
    if len(props) < 40:
        raise ValueError(f"Only {len(props)} safely placed props; refusing insufficient broad scenery")
    specification = dict(
        schema_version="wide_area_scenery_v1", seed=seed, requested_count=count, actual_count=len(props),
        kind_counts={kind: sum(prop["kind"] == kind for prop in props) for kind in ("tree", "rock", "bush")},
        region_counts={str(index): sum(prop["cluster"] == index for prop in props) for index in range(len(cluster_positions))},
        rejected_candidates=dict(rejected),
        derived_assets=derived, props=props, collision_bounds=bounds,
        route_clearance_m=clearance_m, source_mesh=str(source), source_mesh_sha256=_sha(source),
        source_manifest_sha256=_sha(assets / "manifest.json"), implementation_sha256=_sha(Path(__file__)),
        model_outputs_used=False, placement=f"{region_count} arc-length regions; deterministic side-of-route clusters; ground and collision rejection",
        disabled_source_visuals=[dict(model="scenery_harmonic_terrain_objects", visual="Background Trees",
                                     source_mesh="background_trees.dae",
                                     reason="Original flat background tree billboards are not suitable for close nadir observations")],
        collision_semantics="Conservative full visual cylinders, including crowns; not detailed leaf or branch contact",
        limitations=["Known terrain derivative, not a new independent map", "Original flat background tree billboard visual layer is omitted in the derived world",
                     "Authored placements and proxy collisions; not surveyed real-world scenery"],
        attribution="Derived from OpenRobotics Harmonic Terrain Objects v1, CC BY 4.0; UVs and source textures retained",
        attribution_url="https://fuel.gazebosim.org/1.0/OpenRobotics/models/Harmonic%20Terrain%20Objects",
    )
    (output / "layout.json").write_text(json.dumps(specification, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "ATTRIBUTION.txt").write_text(specification["attribution"] + "\n" + specification["attribution_url"]
                                          + "\nhttps://creativecommons.org/licenses/by/4.0/\nDerived isolated mesh components and terrain-grounded placements. Source files unchanged.\n")
    return specification


def decorate_world(world_path, layout_spec):
    """Append the derived visuals and exactly matching proxy collision volumes."""
    tree = ET.parse(world_path)
    world = tree.getroot().find("world")
    for omitted in layout_spec.get("disabled_source_visuals", []):
        model = world.find(f"model[@name='{omitted['model']}']")
        if model is None:
            raise ValueError("Expected source scenery model is missing")
        matches = [(link, visual) for link in model.findall("link") for visual in link.findall("visual")
                   if visual.get("name") == omitted["visual"]]
        if len(matches) != 1 or Path(matches[0][1].findtext("geometry/mesh/uri", "")).name != omitted["source_mesh"]:
            raise ValueError("Source background visual differs; refusing broad deletion")
        matches[0][0].remove(matches[0][1])
    assets = {asset["name"]: asset for asset in layout_spec["derived_assets"]}
    for prop in layout_spec["props"]:
        if world.find(f"model[@name='{prop['name']}']") is not None:
            raise ValueError("Refusing duplicate scenery decoration")
        model = ET.SubElement(world, "model", name=prop["name"])
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = " ".join(str(value) for value in [*prop["position_m"], 0, 0, prop["yaw_rad"]])
        link = ET.SubElement(model, "link", name="scenery")
        for index, source in enumerate(assets[prop["asset"]]["visuals"]):
            if _sha(source["mesh"]) != source["mesh_sha256"]:
                raise ValueError("Derived mesh changed since layout freeze")
            visual = ET.SubElement(link, "visual", name=f"textured_component_{index}")
            mesh = ET.SubElement(ET.SubElement(visual, "geometry"), "mesh")
            ET.SubElement(mesh, "uri").text = source["mesh"]
            ET.SubElement(mesh, "scale").text = " ".join([str(prop["scale"])] * 3)
            material = ET.SubElement(visual, "material")
            ET.SubElement(material, "diffuse").text = "1 1 1 1"
            metal = ET.SubElement(ET.SubElement(material, "pbr"), "metal")
            ET.SubElement(metal, "albedo_map").text = source["albedo"]
            if source["normal"]:
                ET.SubElement(metal, "normal_map").text = source["normal"]
            ET.SubElement(metal, "roughness").text = "0.9"
            ET.SubElement(metal, "metalness").text = "0"
            ET.SubElement(visual, "cast_shadows").text = "true"
        collision = ET.SubElement(link, "collision", name="conservative_full_visual_proxy")
        ET.SubElement(collision, "pose").text = f"0 0 {prop['height_m'] / 2} 0 0 0"
        cylinder = ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder")
        ET.SubElement(cylinder, "radius").text = str(prop["radius_m"])
        ET.SubElement(cylinder, "length").text = str(prop["height_m"])
    ET.indent(tree)
    tree.write(world_path, encoding="utf-8", xml_declaration=True)
