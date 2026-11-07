"""Interface stub for executing a real segmentation model.

**Not implemented, and deliberately so.** V1 must run on CPU with zero runtime dependencies;
importing a model framework here would make every benchmark run depend on a GPU stack to
measure a decision policy that does not need one.

The stub exists to fix the seam. Everything a real backend needs is already in
``ExecutionRequest``, and everything it must return is already in ``ExecutionResult``, so
adding one later is a new module plus a registry entry -- not a change to the runner, the
metrics, or any policy.

How this would be implemented
-----------------------------
1. **Optional dependency.** Declare the model stack under an extra
   (``pip install missionbench[segmentation]``) and import it inside ``execute``, so the
   base package keeps its zero-dependency guarantee.
2. **Model resolution.** Map ``configuration.model_id`` to weights through a model registry
   populated from a configuration file -- never by parsing the ID string.
3. **Strategy application.** ``strategy.precision`` selects the quantised variant;
   ``strategy.input_compression`` is applied before a remote transfer;
   ``strategy.parameters`` carries future dimensions such as input resolution or an
   early-exit index.
4. **Measurement.** Time the inference and read energy from a platform power sensor, then
   report both in the result. A profile then becomes a *recording* of these runs rather
   than a hand-authored fixture -- which is the point of keeping the two separable.
5. **Placement.** A local backend runs in-process. A remote backend posts to a service
   endpoint and must map transport failures onto ``FailureReason`` rather than raising,
   because a failed inference is a benchmark condition, not a crash.
6. **Determinism.** Real inference is not reproducible across hardware. Such runs are for
   *collecting* replay record sets; scored benchmark runs should replay them, so results
   stay comparable across machines.
"""

from __future__ import annotations

from typing import Final

from missionbench.executor.base import ExecutionRequest, ExecutionResult

__all__ = ["NOT_IMPLEMENTED_MESSAGE", "RealSegmentationExecutor"]


#: Raised on construction and on execute, so the two paths cannot drift apart.
NOT_IMPLEMENTED_MESSAGE: Final = (
    "RealSegmentationExecutor is not implemented in V1. Use profile or replay."
)


class RealSegmentationExecutor:
    """Placeholder for a backend that runs an actual segmentation model.

    Conforms to the ``Executor`` protocol so that the seam is type-checked, and fails loudly
    if selected. It does not return a failed ``ExecutionResult``: a missing backend is a
    configuration error to fix, not a mission condition to measure, and quietly reporting
    "inference failed" for every frame would look like a benchmark result.

    **Construction raises**, so selecting this backend fails before step 0 rather than after
    a mission's worth of work. Waiting until the first ``execute`` would spend nine hundred
    steps of accounting to reach a conclusion available immediately.
    """

    #: Provenance, equal to this backend's registry name.
    executor_id: Final = "real_segmentation"

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        del request
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)  # pragma: no cover - unreachable
