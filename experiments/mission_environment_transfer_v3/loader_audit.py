"""Verify derivative triangles and normals using the installed Gazebo mesh loader."""

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

from experiments.lake_missions.assets import save, sha
from experiments.mission_environment_transfer_v3.assets import verify as verify_assets
from experiments.mission_environment_transfer_v3.geometry import triangles


def run(assets):
    manifest = verify_assets(assets)
    output = assets / "loader_audit"
    output.mkdir(exist_ok=False)
    source = Path(__file__).with_name("mesh_loader_audit.cc")
    executable = output / "installed_loader_audit"
    flags = shlex.split(
        subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "gz-common5-graphics"], text=True
        )
    )
    subprocess.run(["g++", "-O2", str(source), "-o", str(executable), *flags], check=True)
    coordinates, metadata = output / "triangles.csv", output / "metadata.csv"
    mesh = assets / manifest["collision_derivative_relative_path"]
    with (output / "loader.log").open("w") as log:
        subprocess.run(
            [
                str(executable.resolve()),
                str(mesh.resolve()),
                str(coordinates.resolve()),
                str(metadata.resolve()),
            ],
            check=True,
            stdout=log,
            stderr=log,
        )
    expected = triangles(str(assets.resolve()))
    actual = np.loadtxt(coordinates, delimiter=",").reshape(-1, 3, 3)
    rows = []
    for name, vertices, normals, indices, primitive in csv.reader(metadata.open()):
        rows.append(
            dict(
                name=name,
                vertices=int(vertices),
                normals=int(normals),
                indices=int(indices),
                triangles=primitive == "1",
            )
        )
    same = actual.shape == expected.shape and np.allclose(actual, expected, atol=2e-6, rtol=0)
    normals_valid = bool(rows) and all(
        r["normals"] == r["vertices"] and r["triangles"] for r in rows
    )
    result = dict(
        complete=bool(same and normals_valid),
        coordinate_absolute_tolerance_m=2e-6,
        relative_tolerance=0,
        max_coordinate_error_m=float(np.max(np.abs(actual - expected))) if same else None,
        source_and_loaded_triangle_counts=[len(expected), len(actual)],
        triangle_order_winding_equal_within_tolerance=bool(same),
        normals_valid=normals_valid,
        loaded_submeshes=rows,
        mesh_sha256=sha(mesh),
        cpp_source_sha256=sha(source),
        audit_source_sha256=sha(Path(__file__)),
        executable_sha256=sha(executable),
        coordinates_sha256=sha(coordinates),
        metadata_sha256=sha(metadata),
        installed_gz_common_graphics_version=subprocess.check_output(
            ["pkg-config", "--modversion", "gz-common5-graphics"], text=True
        ).strip(),
    )
    save(output / "validation.json", result)
    if not result["complete"]:
        raise ValueError("Installed Gazebo collision loader differs from source visual triangles")
    return result


def verify_audit(assets):
    manifest = verify_assets(assets)
    output = assets / "loader_audit"
    result = json.loads((output / "validation.json").read_text())
    if (
        not result["complete"]
        or not result["normals_valid"]
        or not result["triangle_order_winding_equal_within_tolerance"]
        or result["mesh_sha256"] != manifest["collision"]["sha256"]
        or result["source_and_loaded_triangle_counts"] != [manifest["collision"]["triangles"]] * 2
        or result["coordinate_absolute_tolerance_m"] != 2e-6
        or result["relative_tolerance"] != 0
        or result["max_coordinate_error_m"] > 2e-6
    ):
        raise ValueError("Collision loader evidence does not match current derivative")
    for path, key in (
        (Path(__file__), "audit_source_sha256"),
        (Path(__file__).with_name("mesh_loader_audit.cc"), "cpp_source_sha256"),
        (output / "installed_loader_audit", "executable_sha256"),
        (output / "triangles.csv", "coordinates_sha256"),
        (output / "metadata.csv", "metadata_sha256"),
    ):
        if sha(path) != result[key]:
            raise ValueError("Changed installed-loader evidence: " + str(path))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    run(parser.parse_args().assets)
