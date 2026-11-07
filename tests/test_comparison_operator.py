"""Quality-operator comparisons.

The contract's quality constraint and a task's matching rule both hinge on these four
operators, and boundary behaviour decides pass/fail for an entire episode: a run scoring
exactly 0.80 against ``target_f1 >= 0.80`` must succeed. Each operator is pinned at the
boundary for that reason.
"""

from __future__ import annotations

import pytest

from missionbench.schemas.common import ComparisonOperator


@pytest.mark.parametrize(
    ("operator", "value", "threshold", "expected"),
    [
        # >= : boundary passes
        (ComparisonOperator.GREATER_EQUAL, 0.84, 0.80, True),
        (ComparisonOperator.GREATER_EQUAL, 0.80, 0.80, True),
        (ComparisonOperator.GREATER_EQUAL, 0.79, 0.80, False),
        # >  : boundary fails
        (ComparisonOperator.GREATER, 0.81, 0.80, True),
        (ComparisonOperator.GREATER, 0.80, 0.80, False),
        (ComparisonOperator.GREATER, 0.79, 0.80, False),
        # <= : boundary passes
        (ComparisonOperator.LESS_EQUAL, 0.19, 0.20, True),
        (ComparisonOperator.LESS_EQUAL, 0.20, 0.20, True),
        (ComparisonOperator.LESS_EQUAL, 0.21, 0.20, False),
        # <  : boundary fails
        (ComparisonOperator.LESS, 0.19, 0.20, True),
        (ComparisonOperator.LESS, 0.20, 0.20, False),
        (ComparisonOperator.LESS, 0.21, 0.20, False),
    ],
)
def test_compare_reads_as_value_operator_threshold(
    operator: ComparisonOperator, value: float, threshold: float, expected: bool
) -> None:
    assert operator.compare(value, threshold) is expected


def test_every_operator_is_implemented() -> None:
    """A new enum member without a comparison must fail here, not at episode scoring time."""
    for operator in ComparisonOperator:
        assert isinstance(operator.compare(1.0, 0.0), bool)


def test_operators_use_their_specification_spelling() -> None:
    assert sorted(member.value for member in ComparisonOperator) == ["<", "<=", ">", ">="]


def test_operator_parses_from_its_specification_string() -> None:
    assert ComparisonOperator(">=") is ComparisonOperator.GREATER_EQUAL
