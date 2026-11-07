"""Build a small, offline, local-asset multicopter world for transport checks.

Rotor thrust and rigid-body physics are simulated by Gazebo. Default targets are
labelled boxes; optional verified meshes test synthetic
perception. Neither constitutes real-world human perception evidence.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

WORLD = "missionbench_v4"
STEP_NS = 10_000_000
WIDTH, HEIGHT = 320, 240
HFOV = 1.2
TARGETS = {1: "target_1", 2: "target_2", 3: "target_3"}


def element(parent: ET.Element, tag: str, text: object = "", **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, tag, attrs)
    node.text = str(text)
    return node


def plugin(parent: ET.Element, filename: str, name: str, **params: object) -> ET.Element:
    node = element(parent, "plugin", filename=filename, name=f"gz::sim::systems::{name}")
    for key, value in params.items():
        element(node, key, value)
    return node


def box(
    link: ET.Element,
    name: str,
    size: str,
    color: str,
    pose: str = "0 0 0 0 0 0",
    shape: str = "box",
):
    for kind in ("visual", "collision"):
        node = element(link, kind, name=f"{name}_{kind}")
        element(node, "pose", pose)
        geometry = element(node, "geometry")
        dimensions = list(map(float, size.split()))
        if shape == "box":
            element(element(geometry, "box"), "size", size)
        elif shape == "cylinder":
            cylinder = element(geometry, "cylinder")
            element(cylinder, "radius", dimensions[0] / 2)
            element(cylinder, "length", dimensions[2])
        elif shape == "ellipsoid":
            element(
                element(geometry, "ellipsoid"), "radii", " ".join(str(v / 2) for v in dimensions)
            )
        else:
            raise ValueError("unsupported primitive shape")
        if kind == "visual":
            material = element(node, "material")
            element(material, "ambient", color)
            element(material, "diffuse", color)


def build_world(
    path: Path,
    *,
    objects: list[dict] | None = None,
    initial_position_m: list[float] | None = None,
    camera_pitch_rad: float = math.pi / 2,
    ground_size_m: list[float] | None = None,
) -> None:
    """Write our own geometry and an ideal-state, fixed velocity flight controller."""
    if not math.isfinite(camera_pitch_rad) or not 0 <= camera_pitch_rad <= math.pi / 2:
        raise ValueError("camera pitch must lie in [0, pi/2]")
    root = ET.Element("sdf", version="1.9")
    world = element(root, "world", name=WORLD)
    physics = element(world, "physics", name="fixed_step", type="ignored")
    element(physics, "max_step_size", STEP_NS / 1e9)
    element(physics, "real_time_factor", 0)
    element(world, "gravity", "0 0 -9.81")
    for filename, name in (
        ("physics", "Physics"),
        ("user-commands", "UserCommands"),
        ("scene-broadcaster", "SceneBroadcaster"),
    ):
        plugin(world, f"gz-sim-{filename}-system", name)
    plugin(world, "gz-sim-sensors-system", "Sensors", render_engine="ogre2")
    light = element(world, "light", name="sun", type="directional")
    element(light, "pose", "0 0 10 0 0 0")
    element(light, "diffuse", "0.9 0.9 0.9 1")
    element(light, "direction", "0.2 0.1 -1")
    scene = element(world, "scene")
    element(scene, "ambient", "0.5 0.5 0.5 1")
    element(scene, "background", "0.6 0.7 0.8 1")

    ground = element(world, "model", name="ground")
    element(ground, "static", "true")
    box(
        element(ground, "link", name="ground_link"),
        "ground",
        " ".join(map(str, ground_size_m or [40, 40, 0.1])),
        "0.35 0.4 0.3 1",
        "0 0 -0.05 0 0 0",
    )
    if objects is None:
        objects = [
            dict(
                label=label,
                object_id=TARGETS[label],
                position_m=[x, y, 0.4],
                size_m=[0.6, 0.6, 0.8],
                color_rgba=[0.9, 0.15, 0.1, 1],
            )
            for label, x, y in ((1, 0, 0), (2, 3, 1), (3, 6, -1))
        ]
    for obj in objects:
        target = element(world, "model", name=obj["object_id"])
        element(target, "static", "true")
        element(target, "pose", " ".join(map(str, obj["position_m"])) + " 0 0 0")
        link = element(target, "link", name="target_link")
        if "mesh" in obj:
            from experiments.v4_gazebo_pilot.assets import verify_asset

            mesh_spec = obj["mesh"]
            verify_asset(mesh_spec["asset"])
            # Static visual only: no human-body collision or motion model claimed.
            visual = element(link, "visual", name="target_visual")
            element(visual, "pose", " ".join(map(str, mesh_spec["pose"])))
            mesh = element(element(visual, "geometry"), "mesh")
            element(mesh, "uri", mesh_spec["asset"]["mesh_path"])
            element(mesh, "scale", " ".join(map(str, mesh_spec["scale"])))
        else:
            box(
                link,
                "target",
                " ".join(map(str, obj["size_m"])),
                " ".join(map(str, obj["color_rgba"])),
                shape=obj.get("shape", "box"),
            )
        # Unique semantic labels are deliberately used as stable object IDs here.
        plugin(target, "gz-sim-label-system", "Label", label=obj["label"])

    drone = element(world, "model", name="pilot_quad")
    element(drone, "pose", " ".join(map(str, initial_position_m or [0, 0, 4])) + " 0 0 0")
    body = element(drone, "link", name="body")
    inertia = element(element(body, "inertial"), "inertia")
    for key, value in (("ixx", 0.02), ("iyy", 0.02), ("izz", 0.04)):
        element(inertia, key, value)
    element(body.find("inertial"), "mass", 1.5)
    box(body, "body", "0.3 0.3 0.1", "0.15 0.15 0.2 1")

    for kind, topic in (("camera", "/v4/rgb"), ("segmentation", "/v4/gt")):
        sensor = element(body, "sensor", name=kind, type=kind)
        element(sensor, "pose", f"0 0 -0.1 0 {camera_pitch_rad} 0")
        element(sensor, "topic", topic)
        element(sensor, "update_rate", 10)
        element(sensor, "always_on", "true")
        camera = element(sensor, "camera")
        if kind == "segmentation":
            element(camera, "segmentation_type", "semantic")
        element(camera, "horizontal_fov", HFOV)
        image = element(camera, "image")
        element(image, "width", WIDTH)
        element(image, "height", HEIGHT)
        element(image, "format", "R8G8B8")
        clip = element(camera, "clip")
        element(clip, "near", 0.05)
        element(clip, "far", 100)

    for i, (x, y, direction) in enumerate(
        ((0.2, 0.2, "ccw"), (-0.2, -0.2, "ccw"), (0.2, -0.2, "cw"), (-0.2, 0.2, "cw"))
    ):
        rotor = element(drone, "link", name=f"rotor_{i}")
        element(rotor, "pose", f"{x} {y} 0.06 0 0 0")
        inertial = element(rotor, "inertial")
        element(inertial, "mass", 0.01)
        moment = element(inertial, "inertia")
        for key in ("ixx", "iyy", "izz"):
            element(moment, key, 0.00001)
        joint = element(drone, "joint", name=f"rotor_{i}_joint", type="revolute")
        element(joint, "parent", "body")
        element(joint, "child", f"rotor_{i}")
        element(element(joint, "axis"), "xyz", "0 0 1")
        plugin(
            drone,
            "gz-sim-multicopter-motor-model-system",
            "MulticopterMotorModel",
            jointName=f"rotor_{i}_joint",
            linkName=f"rotor_{i}",
            turningDirection=direction,
            timeConstantUp=0.0125,
            timeConstantDown=0.025,
            maxRotVelocity=1000,
            motorConstant=8.54858e-6,
            momentConstant=0.016,
            commandSubTopic="command/motor_speed",
            actuator_number=i,
            rotorDragCoefficient=8.06428e-5,
            rollingMomentCoefficient=1e-6,
            rotorVelocitySlowdownSim=10,
            motorType="velocity",
        )
    controller = plugin(
        drone,
        "gz-sim-multicopter-control-system",
        "MulticopterVelocityControl",
        robotNamespace="pilot_quad",
        commandSubTopic="cmd_vel",
        comLinkName="body",
        velocityGain="2.7 2.7 2.7",
        attitudeGain="2 3 0.15",
        angularRateGain="0.4 0.52 0.18",
        maximumLinearAcceleration="2 2 2",
    )
    rotors = element(controller, "rotorConfiguration")
    for i in range(4):
        rotor = element(rotors, "rotor")
        element(rotor, "jointName", f"rotor_{i}_joint")
        element(rotor, "forceConstant", 8.54858e-6)
        element(rotor, "momentConstant", 0.016)
        element(rotor, "direction", 1 if i < 2 else -1)
    plugin(
        drone,
        "gz-sim-odometry-publisher-system",
        "OdometryPublisher",
        dimensions=3,
        odom_publish_frequency=100,
        odom_topic="/v4/odometry",
    )
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build_world(parser.parse_args().output)
