"""Combine the original textured Harmonic lake scenery with our physical drone.

Original Fuel assets are never edited. Relative SDF resource paths are resolved
locally; demo furniture, mascot, boats and their control plugins are not included.
Original terrain, tree, rock, house, deck and water geometry/materials are retained.
"""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.lake_missions.assets import verify
from experiments.routing_mission.world_worker import build_world as pilot_world


def build_world(path, *, asset_directory, environment, **kwargs):
    asset_directory = Path(asset_directory).resolve()
    manifest = verify(asset_directory)
    pilot_world(path, environment=environment, **kwargs)
    tree = ET.parse(path)
    world = tree.getroot().find("world")
    world.remove(world.find("model[@name='ground']"))
    for entry in manifest["models"]:
        folder = asset_directory / entry["directory"]
        model = copy.deepcopy(ET.parse(folder / "model.sdf").getroot().find("model"))
        model.set("name", "scenery_" + entry["directory"])
        static = model.find("static")
        if static is None:
            static = ET.SubElement(model, "static")
        static.text = "true"
        for parent in model.iter():
            for node in list(parent):
                if node.tag == "plugin" and node.get("filename") != "gz-sim-shader-param-system":
                    parent.remove(node)
                elif node.tag == "script" and parent.find("pbr") is not None:
                    # Old Ogre1 fallback directories are absent in some Fuel bundles.
                    # Ogre2 uses the supplied PBR material, not this fallback script.
                    parent.remove(node)
        for node in model.iter():
            if (
                node.tag
                in {
                    "uri",
                    "albedo_map",
                    "normal_map",
                    "roughness_map",
                    "metalness_map",
                    "light_map",
                    "diffuse_map",
                    "environment_map",
                    "diffuse",
                    "normal",
                    "vertex",
                    "fragment",
                    "value",
                }
                and node.text
            ):
                ref = node.text.strip()
                ref = ref.removeprefix("model://" + entry["name"] + "/")
                if ref.startswith(("materials/", "meshes/")):
                    resolved = (folder / ref).resolve()
                    if not resolved.exists() or not resolved.is_relative_to(folder):
                        raise ValueError(f"Unresolved scenery reference: {ref}")
                    node.text = str(resolved)
        world.append(model)
    for camera in world.findall(".//sensor/camera"):
        camera.find("clip/far").text = "1500"
    for sensor in world.findall(".//sensor"):
        if sensor.get("type") in ("camera", "segmentation"):
            sensor.find("update_rate").text = "2"
    world.find("light/direction").text = "0.3 -0.4 -1"
    ET.SubElement(world.find("light"), "cast_shadows").text = "true"
    ET.indent(tree)
    tree.write(path, encoding="unicode", xml_declaration=True)
