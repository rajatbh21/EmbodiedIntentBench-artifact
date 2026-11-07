"""Conservative collision-aware placement, independent of model predictions.

Reject water, steep slopes, and human volumes overlapping scenery collision AABBs.
This is a conservative placement guard, not a human contact dynamics model.
"""

import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(None)
def height_image(assets):
    return np.asarray(
        Image.open(Path(assets) / "harmonic_terrain/materials/textures/Heightmap.png")
    )


def terrain_height(assets, x, y):
    image = height_image(str(Path(assets).resolve()))
    u, v = (x + 600) / 1200 * (image.shape[1] - 1), (300 - y) / 1200 * (image.shape[0] - 1)
    if not 0 <= u < image.shape[1] - 1 or not 0 <= v < image.shape[0] - 1:
        raise ValueError("Target outside terrain")
    i, j = int(v), int(u)
    dy, dx = v - i, u - j
    value = (
        image[i, j] * (1 - dx) * (1 - dy)
        + image[i, j + 1] * dx * (1 - dy)
        + image[i + 1, j] * (1 - dx) * dy
        + image[i + 1, j + 1] * dx * dy
    )
    return float(value / image.max() * 201.66 - 5)


def transform(pose):
    x, y, z, r, p, yaw = pose
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(yaw),
        math.sin(yaw),
    )
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return rotation, np.array([x, y, z])


@lru_cache(None)
def mesh_vertices(path):
    root = ET.parse(path).getroot()
    ns = {"c": root.tag.split("}")[0].strip("{")}
    if root.findtext("c:asset/c:up_axis", namespaces=ns) != "Z_UP":
        raise ValueError("Placement guard supports verified Z_UP collision meshes only")
    unit = float(root.find("c:asset/c:unit", ns).get("meter", "1"))
    geometries = {}
    for geom in root.findall("c:library_geometries/c:geometry", ns):
        mesh = geom.find("c:mesh", ns)
        ref = mesh.find("c:vertices/c:input[@semantic='POSITION']", ns).get("source")[1:]
        source = mesh.find(f"c:source[@id='{ref}']", ns)
        stride = int(source.find("c:technique_common/c:accessor", ns).get("stride"))
        if stride != 3:
            raise ValueError("Unexpected collision coordinate stride")
        geometries[geom.get("id")] = np.fromstring(
            source.findtext("c:float_array", namespaces=ns), sep=" "
        ).reshape(-1, 3)
    points = []

    def visit(node, parent):
        matrix = parent.copy()
        for child in node:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "matrix":
                matrix = matrix @ np.fromstring(child.text, sep=" ").reshape(4, 4)
            elif tag in ("translate", "rotate", "scale"):
                raise ValueError("Unexpected Collada transform; do not guess")
        for instance in node.findall("c:instance_geometry", ns):
            vertices = geometries[instance.get("url")[1:]]
            points.append((vertices @ matrix[:3, :3].T + matrix[:3, 3]) * unit)
        for child in node.findall("c:node", ns):
            visit(child, matrix)

    for node in root.findall("c:library_visual_scenes/c:visual_scene/c:node", ns):
        visit(node, np.eye(4))
    if not points:
        raise ValueError("Missing collision mesh scene")
    return np.concatenate(points)


def collision_bounds(assets):
    bounds = []
    for directory in ("harmonic_terrain_objects", "lake_house"):
        folder = Path(assets) / directory
        root = ET.parse(folder / "model.sdf").getroot()
        # These pinned assets have identity model/link transforms.
        for parent in [root.find("model"), *root.findall("model/link")]:
            if any(float(v) for v in parent.findtext("pose", "0 0 0 0 0 0").split()):
                raise ValueError("Nonidentity scenery parent transform")
        for collision in root.findall(".//collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh")
            if mesh is not None:
                uri = mesh.findtext("uri").removeprefix("model://Lake House/")
                points = mesh_vertices(str(folder / uri)) * np.fromstring(
                    mesh.findtext("scale", "1 1 1"), sep=" "
                )
            elif geometry.find("box") is not None:
                size = np.fromstring(geometry.findtext("box/size"), sep=" ")
                points = np.array(list(product((-0.5, 0.5), repeat=3))) * size
            elif geometry.find("cylinder") is not None:
                radius = float(geometry.findtext("cylinder/radius"))
                length = float(geometry.findtext("cylinder/length"))
                points = np.array(list(product((-0.5, 0.5), repeat=3))) * [
                    2 * radius,
                    2 * radius,
                    length,
                ]
            else:
                raise ValueError("Unsupported scenery collision; fail closed")
            rotation, offset = transform(
                list(map(float, collision.findtext("pose", "0 0 0 0 0 0").split()))
            )
            points = points @ rotation.T + offset
            bounds.append(
                (directory + "/" + collision.get("name"), points.min(axis=0), points.max(axis=0))
            )
    return bounds


def obstacles_at(x, y, z, bounds):
    # A 0.9m-wide, 1.9m-tall standing-person clearance volume. Tiny low surface
    # features under 15cm are allowed; higher scenery must not intersect it.
    low, high = np.array([x - 0.45, y - 0.45, z + 0.15]), np.array([x + 0.45, y + 0.45, z + 1.9])
    return [name for name, a, b in bounds if np.all(a < high) and np.all(b > low)]


def place(assets, desired, bounds, occupied):
    offsets = sorted(product(range(-8, 9), repeat=2), key=lambda p: (p[0] ** 2 + p[1] ** 2, p))
    for dx, dy in offsets:
        x, y = desired[0] + dx, desired[1] + dy
        z = terrain_height(assets, x, y)
        gradient = math.hypot(
            (terrain_height(assets, x + 1, y) - terrain_height(assets, x - 1, y)) / 2,
            (terrain_height(assets, x, y + 1) - terrain_height(assets, x, y - 1)) / 2,
        )
        if z < 1.2 or gradient > 0.3 or any(math.dist((x, y), p[:2]) < 3 for p in occupied):
            continue
        if not obstacles_at(x, y, z, bounds):
            return [x, y, z + 0.04]
    raise ValueError(f"No safe standing position near {desired}")
