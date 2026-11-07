"""Export selected source triangles into a standalone collision mesh.

Gazebo 8's installed physics loadMesh path ignores SDF submesh selection for
unoptimized meshes. A complete world-coordinate mesh avoids that ambiguity.
Original material-bearing visuals are unchanged.
"""

from pathlib import Path

import numpy as np

from experiments.lake_missions.assets import sha
from experiments.mission_environment_transfer_v3.geometry import triangles


def export(assets, destination):
    points = triangles(str(Path(assets).resolve()))
    with destination.open("x") as stream:
        stream.write("# Derived from unchanged OpenRobotics Depot v6 source visuals\n")
        stream.write("o depot_collision\n")
        for triangle in points:
            for vertex in triangle:
                stream.write("v " + " ".join(format(float(v), ".17g") for v in vertex) + "\n")
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals[lengths > 0] /= lengths[lengths > 0, None]
        normals[lengths == 0] = [0.0, 0.0, 1.0]
        for normal in normals:
            stream.write("vn " + " ".join(format(float(v), ".17g") for v in normal) + "\n")
        for i in range(len(points)):
            stream.write("f " + " ".join(f"{3 * i + j + 1}//{i + 1}" for j in range(3)) + "\n")
    return dict(
        filename=destination.name,
        sha256=sha(destination),
        triangles=len(points),
        vertices=3 * len(points),
        normals=len(points),
        source_visual_instances=16,
        coordinate_frame="Source model world frame; units meters",
        collision_pose=[0, 0, 0, 0, 0, 0],
        collision_scale=[1, 1, 1],
        submesh_selection_in_physics=False,
        original_visuals_modified=False,
        degenerate_triangles_retained=int((lengths == 0).sum()),
        generator_sha256=sha(Path(__file__)),
        geometry_parser_sha256=sha(Path(__file__).with_name("geometry.py")),
    )


def read_triangles(path):
    vertices, faces = [], []
    with path.open() as stream:
        for line in stream:
            words = line.split()
            if not words:
                continue
            if words[0] == "v":
                vertices.append(list(map(float, words[1:])))
            elif words[0] == "f":
                faces.append([int(word.split("/")[0]) - 1 for word in words[1:]])
    points = np.asarray(vertices, dtype=float)
    indices = np.asarray(faces, dtype=int)
    if points.shape[1:] != (3,) or indices.shape[1:] != (3,):
        raise ValueError("Only standalone triangle meshes are supported")
    return points[indices]


def validate(assets, path):
    expected = triangles(str(Path(assets).resolve()))
    actual = read_triangles(path)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError("Exported collision coordinates/winding differ from source visuals")
    return dict(
        complete=True,
        triangle_count=len(actual),
        exact_float64_roundtrip=True,
        triangle_order_and_winding_equal=True,
    )
