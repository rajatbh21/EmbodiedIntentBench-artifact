"""Bind map transfer to rendered bytes and geometry, independently of GT names."""

from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import sha, verify
from experiments.mission_transfer_identity import asset_content_digest, scene_identity

RENDER_SUFFIXES = {
    ".sdf",
    ".dae",
    ".obj",
    ".stl",
    ".mesh",
    ".material",
    ".png",
    ".jpg",
    ".jpeg",
    ".dds",
    ".tga",
    ".bmp",
    ".vert",
    ".frag",
    ".glsl",
}


def render_files(files):
    return {
        name: digest
        for name, digest in files.items()
        if Path(name).suffix.lower() in RENDER_SUFFIXES and "thumbnails" not in Path(name).parts
    }


def scene(doc, assets, family):
    manifest = verify(assets)
    files = {
        name: digest
        for model in manifest["models"]
        for name, digest in render_files(model["files"]).items()
    }
    meshes = {}
    for obj in doc["world"]["objects"]:
        if "mesh" not in obj:
            continue
        path = Path(obj["mesh"]["manifest"])
        if str(path) in meshes:
            continue
        inventory = render_files(read(path)["files"])
        for name, expected in inventory.items():
            target = (path.parent / name).resolve()
            if not target.is_relative_to(path.parent.resolve()) or sha(target) != expected:
                raise ValueError("Changed target render dependency")
        meshes[str(path)] = asset_content_digest(inventory)
    return scene_identity(
        doc["world"],
        asset_sha256=asset_content_digest(files),
        map_family=family,
        duration_s=doc["mission"]["duration_s"],
        mesh_content_sha256=meshes,
        route_kind="step_setpoints",
        controller_parameters={
            "implementation_sha256": sha(Path(__file__).parents[1] / "v4_gazebo_pilot/transport.py")
        },
    )


def training_inventory(training, previous):
    plan = read(training / "plan.json")
    development = Path(plan["cohort"])
    protocol = read(development / "protocol.json")
    original = Path(read(Path(protocol["screen"]) / "protocol.json")["source"])
    rows = []
    cache = {}
    for item in plan["inventory"]:
        if item["group"] == "original_development":
            path = original / "scenario.json"
            assets = Path(read(original / "suite.json")["assets"])
        else:
            path = development / item["name"] / "rapid_survey/scenario.json"
            assets = Path(protocol["assets"])
        key = str(path), str(assets)
        if key not in cache:
            cache[key] = scene(read(path), assets, "fuel/openrobotics/harmonic-lake")
        rows.append(
            dict(
                name=item["name"],
                source=str(path.resolve()),
                source_sha256=sha(path),
                asset_manifest=str((assets / "manifest.json").resolve()),
                asset_manifest_sha256=sha(assets / "manifest.json"),
                identity=cache[key],
            )
        )
    # Exclude the preceding completed evaluation map/layouts as well as actual training.
    prior = read(previous / "protocol.json")
    for flight in prior["cases"]:
        path = previous / flight / "rapid_survey/scenario.json"
        assets = Path(prior["assets"])
        rows.append(
            dict(
                name="prior_evaluation_" + flight,
                source=str(path.resolve()),
                source_sha256=sha(path),
                asset_manifest=str((assets / "manifest.json").resolve()),
                asset_manifest_sha256=sha(assets / "manifest.json"),
                identity=scene(read(path), assets, "fuel/openrobotics/harmonic-lake"),
            )
        )
    return rows
