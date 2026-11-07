"""Preserve the textured Depot interior and add matching static mesh collisions."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.mission_environment_transfer.assets import verify
from experiments.routing_mission.world_worker import build_world as pilot_world


def build_world(path, *, asset_directory, environment, **kwargs):
    asset_directory = Path(asset_directory).resolve()
    verify(asset_directory)
    pilot_world(path, environment=environment, **kwargs)
    tree = ET.parse(path)
    world = tree.getroot().find("world")
    world.remove(world.find("model[@name='ground']"))
    folder = asset_directory / "depot"
    model = copy.deepcopy(ET.parse(folder / "model.sdf").getroot().find("model"))
    model.set("name", "scenery_depot")
    model.find("static").text = "true"
    for parent in model.iter():
        for child in list(parent):
            if child.tag in {"plugin", "joint"} or (
                child.tag == "script" and parent.find("pbr") is not None
            ):
                parent.remove(child)
    for link in model.findall("link"):
        # Source only supplied a floor-plane collider. Add the exact visual meshes,
        # including roof, furniture and stationary fan blades, with identical poses.
        for index, visual in enumerate(link.findall("visual")):
            collider = ET.SubElement(link, "collision", name=f"visible_mesh_{index}")
            if visual.find("pose") is not None:
                collider.append(copy.deepcopy(visual.find("pose")))
            collider.append(copy.deepcopy(visual.find("geometry")))
        for light in link.findall("light"):
            light.find("intensity").text = str(environment["sunlight"] / 0.9)
    for node in model.iter():
        if node.text and node.tag in {
            "uri",
            "albedo_map",
            "normal_map",
            "metalness_map",
            "roughness_map",
            "emissive_map",
            "light_map",
        }:
            ref = node.text.strip()
            if ref.startswith(("materials/", "meshes/")):
                resolved = (folder / ref).resolve()
                if not resolved.is_relative_to(folder) or not resolved.exists():
                    raise ValueError("Unresolved source resource: " + ref)
                node.text = str(resolved)
    world.append(model)
    for sensor in world.findall(".//sensor"):
        if sensor.get("type") in {"camera", "segmentation"}:
            sensor.find("update_rate").text = "2"
            sensor.find("camera/clip/far").text = "100"
    ET.SubElement(world.find("light"), "cast_shadows").text = "true"
    ET.indent(tree)
    tree.write(path, encoding="unicode", xml_declaration=True)
