"""Bind scenery to recording identity while retaining the frozen base runner."""

from dataclasses import dataclass
from pathlib import Path

from experiments.lake_missions.assets import sha, verify
from experiments.routing_mission.environment import PhysicalScenario
from experiments.routing_mission.environment import load as base_load
from experiments.v4_benchmark.scenario import digest


@dataclass(frozen=True)
class LakeScenario(PhysicalScenario):
    asset_directory: str = ""
    scenery_sha256: str = ""

    @property
    def recording_identity(self):
        return digest(
            dict(
                physical=super().recording_identity,
                scenery=self.scenery_sha256,
                builder_sha256=sha(Path(__file__).with_name("world.py")),
            )
        )


def load(path, environment, assets):
    verify(assets)
    base = base_load(path, environment)
    return LakeScenario(
        **vars(base),
        asset_directory=str(Path(assets).resolve()),
        scenery_sha256=sha(Path(assets) / "manifest.json"),
    )
