"""Executor selection, construction, provenance, and replay end to end.

Covers the plumbing added in fix/v1-executor-plumbing: the CLI --executor option, the
registry-driven construction path, provenance read from the executor object, and the
replay executor reachable through the normal flow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from missionbench.benchmark import BenchmarkData, run_episode, run_suite
from missionbench.executor.base import ExecutionResult, FailureReason, executor_id_of
from missionbench.executor.real_segmentation_executor import (
    NOT_IMPLEMENTED_MESSAGE,
    RealSegmentationExecutor,
)
from missionbench.executor.registry import executor_registry
from missionbench.run_benchmark import build_parser, main
from missionbench.schemas.loading import SchemaValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = str(REPO_ROOT / "data" / "contracts" / "contract_001.json")
EPISODE = str(REPO_ROOT / "data" / "episodes" / "episode_001.json")


@pytest.fixture(scope="module")
def data() -> BenchmarkData:
    return BenchmarkData(REPO_ROOT / "data")


@pytest.fixture
def cli(tmp_path: Path):
    def _run(*args: str, output: str = "result.json") -> tuple[int, Path]:
        target = tmp_path / output
        status = main(
            ["--data-root", str(REPO_ROOT / "data"), *args, "--output", str(target), "--quiet"]
        )
        return status, target

    return _run


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- CLI selection -----------------------------------------------------------------------


def test_omitting_executor_selects_profile() -> None:
    args = build_parser().parse_args(["--suite", "--contract", CONTRACT])
    assert args.executor == "profile"


def test_executor_choices_come_from_the_registry() -> None:
    parser = build_parser()
    for name in executor_registry.names():
        args = parser.parse_args(["--suite", "--contract", CONTRACT, "--executor", name])
        assert args.executor == name


def test_an_unknown_executor_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--suite", "--contract", CONTRACT, "--executor", "nope"])


def test_profile_runs_through_the_cli(cli) -> None:
    status, output = cli("--episode", EPISODE, "--contract", CONTRACT, "--executor", "profile")
    assert status == 0
    assert _payload(output)["executor_id"] == "profile"


def test_replay_runs_through_the_cli(cli) -> None:
    status, output = cli("--episode", EPISODE, "--contract", CONTRACT, "--executor", "replay")
    assert status == 0
    assert _payload(output)["executor_id"] == "replay"


def test_real_segmentation_fails_immediately(cli, capsys) -> None:
    status, output = cli(
        "--episode", EPISODE, "--contract", CONTRACT, "--executor", "real_segmentation"
    )
    assert status == 2
    assert not output.exists(), "no result file should be written for a failed selection"
    assert "not implemented in V1" in capsys.readouterr().err


def test_real_segmentation_fails_before_any_episode_runs() -> None:
    """Construction raises, so the failure cannot arrive after a mission's worth of work."""
    with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED_MESSAGE):
        RealSegmentationExecutor()


# --- provenance --------------------------------------------------------------------------


def test_profile_records_profile_provenance(data) -> None:
    result = run_episode(data=data, episode=_ep(data), contract=_ct(data), executor_name="profile")
    assert result.record.executor_id == "profile"
    assert result.metrics.executor_id == "profile"


def test_replay_records_replay_provenance(data) -> None:
    result = run_episode(data=data, episode=_ep(data), contract=_ct(data), executor_name="replay")
    assert result.record.executor_id == "replay"
    assert result.metrics.executor_id == "replay"


def test_injected_executor_cannot_be_mislabelled(data) -> None:
    """The regression this branch fixes: a ReplayExecutor once recorded as 'profile'.

    Provenance is read from the object, so injecting a replay executor while every other
    default still says 'profile' cannot record 'profile'.
    """
    from missionbench.executor.replay_executor import ReplayExecutor

    replay = ReplayExecutor({}, record_set_id="INJECTED")
    result = run_episode(
        data=data,
        episode=_ep(data),
        contract=_ct(data),
        policy_name="always_local_light",
        executor=replay,
    )
    assert result.record.executor_id == "replay"
    assert result.record.executor_id != "profile"


def test_an_unregistered_executor_is_labelled_honestly(data) -> None:
    """A backend with no executor_id is reported as unregistered:<ClassName>, never guessed."""

    class HomebrewExecutor:
        def execute(self, request) -> ExecutionResult:
            return ExecutionResult(
                success=False,
                latency_s=1.0,
                onboard_energy_j=0.0,
                failure_reason=FailureReason.BACKEND_UNAVAILABLE,
            )

    assert executor_id_of(HomebrewExecutor()) == "unregistered:HomebrewExecutor"
    result = run_episode(
        data=data,
        episode=_ep(data),
        contract=_ct(data),
        policy_name="always_local_light",
        executor=HomebrewExecutor(),
    )
    assert result.record.executor_id == "unregistered:HomebrewExecutor"


def test_serialised_output_preserves_executor_identity(data) -> None:
    # Only EPISODE_001 has a replay set, so the suite is restricted to it. Selecting replay
    # for an episode without a set is a deliberate error, covered separately.
    result = run_suite(data=data, episodes=[_ep(data)], contract=_ct(data), executor_name="replay")
    payload = result.to_dict()
    assert payload["executor_id"] == "replay"
    assert all(e["record"]["executor_id"] == "replay" for e in payload["episodes"])


# --- replay uses replayed values ---------------------------------------------------------


def test_replay_reproduces_the_profile_resource_values(data) -> None:
    """Replaying a set recorded from the profile run reproduces its resources exactly."""
    profile = run_episode(data=data, episode=_ep(data), contract=_ct(data), executor_name="profile")
    replay = run_episode(data=data, episode=_ep(data), contract=_ct(data), executor_name="replay")

    for attr in (
        "mission_completion_time_s",
        "total_communication_mb",
        "total_energy_j",
        "final_battery_fraction",
        "mean_end_to_end_inference_latency_ms",
    ):
        assert getattr(profile.metrics, attr) == pytest.approx(getattr(replay.metrics, attr)), attr


def test_replay_serves_recorded_values_not_profile_generated_ones(data) -> None:
    """A doctored record set changes the result, proving replay reads its records."""
    from missionbench.executor.replay_executor import ReplayExecutor, ReplayRecord

    # One record for every (frame, config) the light-only policy can reach, with a latency
    # that is not any profile value, so a match proves the record was used.
    records = {
        (frame, "CFG_LOCAL_LIGHT"): ReplayRecord(
            frame_id=frame,
            config_id="CFG_LOCAL_LIGHT",
            success=True,
            latency_ms=321.0,
            onboard_energy_j=7.0,
            prediction={"instances": []},
        )
        for frame in range(901)
    }
    result = run_episode(
        data=data,
        episode=_ep(data),
        contract=_ct(data),
        policy_name="always_local_light",
        executor=ReplayExecutor(records),
    )
    assert result.metrics.mean_end_to_end_inference_latency_ms == pytest.approx(321.0)


def test_a_missing_replay_record_reports_context_in_strict_mode(data) -> None:
    from missionbench.executor.replay_executor import ReplayExecutor

    executor = ReplayExecutor({}, strict=True, record_set_id="EMPTY_SET")
    with pytest.raises(SchemaValidationError) as error:
        run_episode(
            data=data,
            episode=_ep(data),
            contract=_ct(data),
            policy_name="always_local_light",
            executor=executor,
        )
    message = str(error.value)
    assert "EMPTY_SET" in message
    assert "EPISODE_001" in message
    assert "frame 0" in message
    assert "CFG_LOCAL_LIGHT" in message


def test_replay_without_a_record_set_names_how_to_make_one(data) -> None:
    """Selecting replay for an episode with no set fails clearly, never falls back to profile."""
    from missionbench.schemas.episode import load_episode

    # episode_002 has no replay set under predictions/.
    episode = load_episode(REPO_ROOT / "data" / "episodes" / "episode_002_stable_network.json")
    with pytest.raises(SchemaValidationError, match="needs a replay record set"):
        run_episode(data=data, episode=episode, contract=_ct(data), executor_name="replay")


# --- the same runner drives two executors ------------------------------------------------


def test_one_runner_drives_two_executors_without_modification(data) -> None:
    """Extensibility: the same EpisodeRunner, episode, and policy, two backends, no changes."""
    episode, contract = _ep(data), _ct(data)
    profile = run_episode(data=data, episode=episode, contract=contract, executor_name="profile")
    replay = run_episode(data=data, episode=episode, contract=contract, executor_name="replay")

    assert profile.record.executor_id == "profile"
    assert replay.record.executor_id == "replay"
    assert profile.record.step_count == replay.record.step_count


def _ep(data: BenchmarkData):
    return next(e for e in data.episodes() if e.episode_id == "EPISODE_001")


def _ct(data: BenchmarkData):
    return next(c for c in data.contracts() if c.contract_id == "CONTRACT_001")
