"""The command-line interface, exercised the way a user runs it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from missionbench import SCHEMA_VERSION
from missionbench.run_benchmark import build_parser, main


@pytest.fixture
def cli(request, tmp_path: Path):
    """Run the CLI against the repository's data root, writing into a temporary directory."""
    data_root = Path(request.config.rootdir) / "data"

    def _run(*args: str, output: str | None = "result.json") -> tuple[int, Path | None]:
        target = tmp_path / output if output else None
        argv = ["--data-root", str(data_root), *args]
        if target is not None:
            argv += ["--output", str(target), "--quiet"]
        return main(argv), target

    return _run


CONTRACT = "data/contracts/contract_001.json"


def _contract(request) -> str:
    return str(Path(request.config.rootdir) / CONTRACT)


def test_a_single_episode_runs_and_writes_json(cli, request, capsys) -> None:
    root = Path(request.config.rootdir)
    status, output = cli(
        "--episode",
        str(root / "data/episodes/episode_001.json"),
        "--contract",
        _contract(request),
        "--policy",
        "rule_based",
    )
    assert status == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["policy"] == "rule_based"
    assert payload["aggregate"]["episode_count"] == 1
    assert payload["episodes"][0]["episode_id"] == "EPISODE_001"
    assert payload["episodes"][0]["mission_success"] is True


def test_the_suite_runs_every_episode(cli, request) -> None:
    status, output = cli("--suite", "--contract", _contract(request), "--policy", "rule_based")
    assert status == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    assert aggregate["episode_count"] == 3
    # Deliberately not pinning the rate: over three episodes it can only be 0, 1/3, 2/3 or 1,
    # and which of those it lands on is a property of the fixtures, not of the CLI. The
    # reference suite pins the numbers; this checks the plumbing.
    assert 0.0 <= aggregate["mission_success_rate"] <= 1.0
    assert aggregate["mission_success_count"] <= aggregate["episode_count"]
    low, high = aggregate["mission_success_ci_95"]
    assert low <= aggregate["mission_success_rate"] <= high


def test_rerunning_produces_a_byte_identical_file(cli, request) -> None:
    """A determinism regression should show up as a diff in review."""
    _, first = cli("--suite", "--contract", _contract(request), output="first.json")
    _, second = cli("--suite", "--contract", _contract(request), output="second.json")
    assert first.read_bytes() == second.read_bytes()


def test_a_result_file_does_not_publish_the_answers(cli, request) -> None:
    _, output = cli("--suite", "--contract", _contract(request))
    text = output.read_text(encoding="utf-8")
    assert "ground_truth_track_id" not in text
    assert "mask_iou" not in text


def test_detail_is_available_when_asked_for(cli, request) -> None:
    _, output = cli("--suite", "--contract", _contract(request), "--include-detail")
    payload = json.loads(output.read_text(encoding="utf-8"))
    record = payload["episodes"][0]["record"]
    assert len(record["steps"]) == record["step_count"]
    assert record["evidence"]["instances"]


def test_hiding_profiles_changes_the_outcome(cli, request) -> None:
    _, disclosed = cli("--suite", "--contract", _contract(request), output="disclosed.json")
    _, hidden = cli(
        "--suite", "--contract", _contract(request), "--hide-profiles", output="hidden.json"
    )

    seen = json.loads(disclosed.read_text(encoding="utf-8"))["aggregate"]["mission_success_rate"]
    blind = json.loads(hidden.read_text(encoding="utf-8"))["aggregate"]["mission_success_rate"]
    assert seen > blind


def test_repeats_pool_seeds_and_narrow_the_interval(cli, request) -> None:
    """Three episodes give an interval ~73 points wide; that is the problem repeats solve."""
    _, single = cli("--suite", "--contract", _contract(request), output="single.json")
    _, pooled = cli(
        "--suite", "--contract", _contract(request), "--repeats", "10", output="pooled.json"
    )

    one = json.loads(single.read_text(encoding="utf-8"))["aggregate"]
    many = json.loads(pooled.read_text(encoding="utf-8"))["aggregate"]

    assert one["episode_count"] == 3
    assert many["episode_count"] == 30
    width = lambda a: a["mission_success_ci_95"][1] - a["mission_success_ci_95"][0]  # noqa: E731
    assert width(many) < width(one) / 2


def test_repeats_are_deterministic(cli, request) -> None:
    _, first = cli("--suite", "--contract", _contract(request), "--repeats", "4", output="a.json")
    _, second = cli("--suite", "--contract", _contract(request), "--repeats", "4", output="b.json")
    assert first.read_bytes() == second.read_bytes()


def test_the_printed_summary_reports_the_headline(cli, request, capsys) -> None:
    main(
        [
            "--data-root",
            str(Path(request.config.rootdir) / "data"),
            "--suite",
            "--contract",
            _contract(request),
            "--policy",
            "always_local_light",
        ]
    )
    printed = capsys.readouterr().out
    assert "Mission Success Rate: 0.0% (0/3)" in printed
    assert "95% CI" in printed
    assert "EPISODE_001" in printed


def test_a_bad_data_root_exits_with_a_message(request, capsys, tmp_path: Path) -> None:
    status = main(
        [
            "--data-root",
            str(tmp_path / "absent"),
            "--suite",
            "--contract",
            _contract(request),
        ]
    )
    assert status == 2
    assert "benchmark data directory not found" in capsys.readouterr().err


def test_a_malformed_contract_exits_with_a_message(request, capsys, tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text('{"schema_version": "9.9"}', encoding="utf-8")
    status = main(
        [
            "--data-root",
            str(Path(request.config.rootdir) / "data"),
            "--suite",
            "--contract",
            str(broken),
        ]
    )
    assert status == 2
    assert "unsupported schema_version" in capsys.readouterr().err


# --- argument surface ---------------------------------------------------------------------


def test_episode_selection_is_required() -> None:
    """Neither --episode nor --suite would silently run nothing."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--contract", CONTRACT])


def test_episode_and_suite_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--contract", CONTRACT, "--suite", "--episode", "data/episodes/episode_001.json"]
        )


def test_an_unknown_policy_is_refused_at_parse_time() -> None:
    """argparse choices come from the registry, so the error lists the real options."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--contract", CONTRACT, "--suite", "--policy", "nope"])


def test_every_registered_policy_is_selectable() -> None:
    from missionbench.policies.registry import policy_registry

    for name in policy_registry.names():
        args = build_parser().parse_args(["--contract", CONTRACT, "--suite", "--policy", name])
        assert args.policy == name
