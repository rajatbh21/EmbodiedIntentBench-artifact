"""Primitives shared by more than one schema.

Currently the threshold comparison used in two independent places: the contract's quality
constraint (``quality_operator`` / ``quality_threshold``) and a task specification's
matching rule (``matching_rule.operator`` / ``matching_rule.threshold``). Both express
"this measured value must relate to this threshold in this way", so both use one operator
type and one comparison function rather than each re-deriving ``>=`` from a string.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from enum import StrEnum
from typing import Final

__all__ = ["ComparisonOperator"]


class ComparisonOperator(StrEnum):
    """A threshold comparison direction, as written in specification files."""

    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    GREATER = ">"
    LESS = "<"

    def compare(self, value: float, threshold: float) -> bool:
        """Return whether ``value`` satisfies this operator against ``threshold``.

        Reads in the same order as the specification: ``value <op> threshold``. For a
        contract with ``target_f1 >= 0.80``, ``GREATER_EQUAL.compare(0.84, 0.80)`` is
        ``True``.
        """
        return _COMPARISONS[self](value, threshold)


_COMPARISONS: Final[dict[ComparisonOperator, Callable[[float, float], bool]]] = {
    ComparisonOperator.GREATER_EQUAL: operator.ge,
    ComparisonOperator.LESS_EQUAL: operator.le,
    ComparisonOperator.GREATER: operator.gt,
    ComparisonOperator.LESS: operator.lt,
}
