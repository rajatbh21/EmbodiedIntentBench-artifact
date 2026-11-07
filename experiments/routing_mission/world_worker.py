"""System-Python compatible SDF builder; parent validates environment."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

from experiments.v4_gazebo_pilot.world import build_world as base_world
from experiments.v4_gazebo_pilot.world import element


def build_world(path, *, environment, **kwargs):
    env = SimpleNamespace(**environment)
    base_world(path, **kwargs)
    root = ET.parse(path)
    world = root.getroot().find("world")
    world.find("scene/ambient").text = f"{env.ambient} {env.ambient} {env.ambient} 1"
    world.find("light/diffuse").text = f"{env.sunlight} {env.sunlight} {env.sunlight} 1"
    if env.fog_density:
        fog = element(world.find("scene"), "fog")
        element(fog, "type", "linear")
        element(fog, "start", 0)
        element(fog, "end", 1 / env.fog_density)
        element(fog, "color", "0.65 0.68 0.7 1")
        element(fog, "density", env.fog_density)
    for sensor in world.findall(".//sensor"):
        if sensor.get("type") not in ("camera", "segmentation"):
            continue
        camera = sensor.find("camera")
        camera.find("image/width").text = str(env.width_px)
        camera.find("image/height").text = str(env.height_px)
        camera.find("clip/far").text = "300"
        if sensor.get("type") == "camera" and env.rgb_noise_stddev:
            noise = element(camera, "noise")
            element(noise, "type", "gaussian")
            element(noise, "mean", 0)
            element(noise, "stddev", env.rgb_noise_stddev)
    root.write(path, encoding="unicode", xml_declaration=True)
