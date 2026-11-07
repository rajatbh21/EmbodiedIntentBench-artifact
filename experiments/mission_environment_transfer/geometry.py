"""Conservative geometry-only placement using the unmodified source triangles."""

import xml.etree.ElementTree as ET
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np

from experiments.lake_missions.placement import transform


def pose_matrix(text):
    rotation, offset = transform(list(map(float, text.split())))
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, offset
    return result


@lru_cache(None)
def collada_triangles(filename, submesh=None):
    root = ET.parse(filename).getroot()
    ns = {"c": root.tag.split("}")[0].strip("{")}
    if root.findtext("c:asset/c:up_axis", namespaces=ns) != "Z_UP":
        raise ValueError("Only explicitly verified Z_UP meshes are supported")
    unit = float(root.find("c:asset/c:unit", ns).get("meter", "1"))
    geometries = {}
    for geom in root.findall("c:library_geometries/c:geometry", ns):
        mesh = geom.find("c:mesh", ns)
        source_id = mesh.find("c:vertices/c:input[@semantic='POSITION']", ns).get("source")[1:]
        source = mesh.find(f"c:source[@id='{source_id}']", ns)
        if int(source.find("c:technique_common/c:accessor", ns).get("stride")) != 3:
            raise ValueError("Unexpected position stride")
        vertices = np.fromstring(source.findtext("c:float_array", namespaces=ns), sep=" ").reshape(
            -1, 3
        )
        chunks = []
        if mesh.find("c:polylist", ns) is not None or mesh.find("c:polygons", ns) is not None:
            raise ValueError("Only source triangulated geometry is supported")
        for primitive in mesh.findall("c:triangles", ns):
            inputs = primitive.findall("c:input", ns)
            stride = 1 + max(int(i.get("offset")) for i in inputs)
            offset = int(next(i for i in inputs if i.get("semantic") == "VERTEX").get("offset"))
            indices = np.fromstring(
                primitive.findtext("c:p", namespaces=ns), sep=" ", dtype=int
            ).reshape(-1, stride)[:, offset]
            if len(indices) != 3 * int(primitive.get("count")):
                raise ValueError("Invalid triangle count")
            chunks.append(vertices[indices].reshape(-1, 3, 3))
        geometries[geom.get("id")] = np.concatenate(chunks)
    result = []

    def visit(node, parent, selected):
        matrix = parent.copy()
        selected = selected or submesh in (node.get("name"), node.get("id"))
        for child in node:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "matrix":
                matrix = matrix @ np.fromstring(child.text, sep=" ").reshape(4, 4)
            elif tag in ("translate", "rotate", "scale"):
                raise ValueError("Unsupported Collada node transform")
        if selected:
            for instance in node.findall("c:instance_geometry", ns):
                points = geometries[instance.get("url")[1:]]
                result.append((points @ matrix[:3, :3].T + matrix[:3, 3]) * unit)
        for child in node.findall("c:node", ns):
            visit(child, matrix, selected)

    for node in root.findall("c:library_visual_scenes/c:visual_scene/c:node", ns):
        visit(node, np.eye(4), submesh is None)
    if not result:
        raise ValueError("Missing requested mesh node")
    return np.concatenate(result)


@lru_cache(None)
def triangles(assets):
    folder = Path(assets) / "depot"
    model = ET.parse(folder / "model.sdf").getroot().find("model")
    parent = pose_matrix(model.findtext("pose", "0 0 0 0 0 0"))
    all_triangles = []
    for link in model.findall("link"):
        link_transform = parent @ pose_matrix(link.findtext("pose", "0 0 0 0 0 0"))
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Unexpected nonmesh source visual")
            uri = (folder / mesh.findtext("uri")).resolve()
            if not uri.is_relative_to(folder.resolve()):
                raise ValueError("Mesh path escaped source directory")
            if mesh.findtext("submesh/center", "false") != "false":
                raise ValueError("Unexpected centered submesh")
            points = collada_triangles(str(uri), mesh.findtext("submesh/name"))
            points = points * np.fromstring(mesh.findtext("scale", "1 1 1"), sep=" ")
            matrix = link_transform @ pose_matrix(visual.findtext("pose", "0 0 0 0 0 0"))
            all_triangles.append(points @ matrix[:3, :3].T + matrix[:3, 3])
    return np.concatenate(all_triangles)


@lru_cache(None)
def triangle_bounds(assets):
    points = triangles(str(Path(assets).resolve()))
    return points.min(axis=1), points.max(axis=1)


def clear_box(assets, low, high):
    """Reject any triangle AABB overlap; conservative, not exact contact analysis."""
    a, b = triangle_bounds(str(Path(assets).resolve()))
    return not bool(np.any(np.all(a < high, axis=1) & np.all(b > low, axis=1)))


def standing_clear(assets, x, y):
    return clear_box(assets, [x - 0.45, y - 0.45, 0.15], [x + 0.45, y + 0.45, 2.0])


def route_clear(assets, route, clearance=0.4):
    for a, b in pairwise(route):
        # Include segment interiors; 0.1m sampling plus 0.45m inflated boxes.
        for t in np.linspace(0, 1, 1 + int(np.ceil(np.linalg.norm(np.array(b) - a) / 0.1))):
            point = np.asarray(a) * (1 - t) + np.asarray(b) * t
            if not clear_box(assets, point - clearance - 0.05, point + clearance + 0.05):
                return False
    return True
