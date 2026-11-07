"""Simulated remote inference behind deployment-ready boundaries (V3 P1).

Remote execution is a distinct path with explicit stages —

    capture -> request encoding -> upload -> network transit -> remote compute
            -> response transit -> response decoding -> completion or failure

— never "a local executor with a bigger constant latency". The stages are computed
deterministically here, but they are kept architecturally separate so a later real
deployment can place them on real machines without touching the mission runner,
policies, or the evaluator:

    UAV / Jetson edge runtime          Remote inference service
      policy + local executors          request validation
      request encoding                  remote model execution
      network client (Transport) ─────► response encoding
      fallback handling          ◄─────
              │
              ▼
    Benchmark coordinator (this repo): scenario, runtime state, evidence,
    resource accounting, mission evaluation

Boundaries:

- :class:`InferenceTransport` — everything the *network* does. The simulated
  implementation computes stage times from the capture-time network snapshot
  (**Model A**: conditions are frozen at request submission; a regime change during a
  transfer does not affect it — documented, tested, and replaceable later by an
  interval-integrating transport behind the same protocol).
- :class:`RemoteInferenceBackend` — everything the *server* does. The simulated
  implementation runs an existing local model-strategy in-process.
- :class:`SimulatedRemoteExecutor` — the benchmark-facing executor. It conforms to the
  V2 result shape (`ImageExecutionResult`), charges communication volume and radio
  energy, and applies the fallback semantics.

Protocol models (`InferenceRequest`, transport results) are wire-representable: every
field is a JSON scalar/list, request ids are stable, and the payload travels as an
explicit reference plus metadata (a real transport would serialize the encoded bytes;
the in-process simulation passes the array alongside the request). ``PROTOCOL_VERSION``
stamps every request so a future client/server pair can negotiate.

Accounting rules (each tested):

- **Upload counts even when the mission gains nothing**: a timed-out or lost request
  still transmitted bytes. Partial transfers charge proportionally to transfer time.
- **Radio activation energy** is charged on every attempt, including an unreachable
  probe. Communication energy = uploaded_mb x uplink_j_per_mb + downloaded_mb x
  downlink_j_per_mb + activation_j — configured simulation parameters, labelled so.
- **RTT is charged exactly once** per request (a single end-to-end round-trip
  contribution between upload and remote compute); it is never also folded into the
  upload/download terms.
- **Remote failure is not an empty prediction**: a failed request without a fallback
  yields ``success=False`` plus a status/reason, and the runner records it as a failed
  inference — distinguishable end-to-end from a healthy "nothing detected".

Fallback semantics (executor-level, distinct from the ActionValidator's *action*
fallback): if the remote attempt fails and the config names a ``fallback_config_id``,
the named local executor runs **on the same captured frame**; the failed remote
attempt's elapsed time, transmitted bytes, and radio energy remain charged; the
fallback's compute time and energy are added; the result records the full path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np

from missionbench.schemas.loading import SchemaValidationError
from missionbench.v2.executors import ImageExecutionResult, ImageExecutor
from missionbench.v2.network import NetworkRegime

if TYPE_CHECKING:
    from missionbench.v2.scenario import ExecutorConfigSpec

__all__ = [
    "PROTOCOL_VERSION",
    "ExecutionContext",
    "InferenceRequest",
    "InferenceTransport",
    "RemoteInferenceBackend",
    "RemoteStatus",
    "SimulatedRemoteBackend",
    "SimulatedRemoteExecutor",
    "SimulatedTransport",
    "TransportResult",
]

#: Version of the request/response protocol models; a real client/server pair
#: negotiates on this.
PROTOCOL_VERSION: Final = "1.0"

#: Defaults for the optional simulated-remote parameters (scenario ``parameters`` may
#: override any of them). All are configured simulation values, not measurements.
REMOTE_PARAMETER_DEFAULTS: Final = {
    "download_mb_per_call": 0.05,
    "request_encoding_s": 0.02,
    "response_decoding_s": 0.01,
    "remote_queue_s": 0.0,
    "max_loss_frac": 0.3,
    "unreachable_detect_s": 0.2,
    "uplink_energy_j_per_mb": 2.0,
    "downlink_energy_j_per_mb": 1.0,
    "radio_activation_j": 0.5,
    "onboard_codec_energy_j": 0.5,
}


class RemoteStatus(StrEnum):
    """Outcome of one remote inference attempt."""

    SUCCESS = "success"
    #: The link was down at request time; nothing was transmitted.
    UNREACHABLE = "unreachable"
    #: The upload was transmitted but lost (packet loss above the transport's ceiling).
    UPLOAD_FAILED = "upload_failed"
    #: The configured timeout elapsed before the response was decoded.
    TIMEOUT = "timeout"
    #: The remote backend itself failed after a successful upload.
    REMOTE_ERROR = "remote_error"


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Mission-side context an executor may need beyond the RGB frame.

    Local executors ignore it; the remote path needs the capture time (request
    identity), the deadline (a real client would cancel), and the current network
    sample. Carries nothing ground-truth-derived and nothing about the future.
    """

    scenario_id: str
    observation_id: int
    capture_time_s: float
    deadline_s: float
    network: NetworkRegime


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """One remote inference request; every field is wire-representable.

    The payload itself is referenced (``payload_ref``) rather than embedded: a real
    transport serializes the encoded bytes it points at, while the in-process
    simulation resolves it directly. ``to_wire()`` is the process-independent view.
    """

    request_id: str
    scenario_id: str
    observation_id: int
    capture_time_s: float
    deadline_time_s: float
    requested_config_id: str
    payload_kind: str  # "raw_rgb" is the only kind V3 P1 ships
    payload_shape: tuple[int, ...]
    payload_mb: float
    privacy_level: str
    protocol_version: str = PROTOCOL_VERSION
    payload_ref: str = ""

    def to_wire(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scenario_id": self.scenario_id,
            "observation_id": self.observation_id,
            "capture_time_s": self.capture_time_s,
            "deadline_time_s": self.deadline_time_s,
            "requested_config_id": self.requested_config_id,
            "payload_kind": self.payload_kind,
            "payload_shape": list(self.payload_shape),
            "payload_mb": self.payload_mb,
            "privacy_level": self.privacy_level,
            "protocol_version": self.protocol_version,
            "payload_ref": self.payload_ref,
        }


@dataclass(frozen=True, slots=True)
class TransportResult:
    """What one transport attempt did: status, stage timings, transferred bytes."""

    status: RemoteStatus
    elapsed_s: float
    uploaded_mb: float
    downloaded_mb: float
    #: Stage breakdown; every component separately observable, never one opaque number.
    stages_s: dict[str, float]
    prediction_mask: np.ndarray | None = None
    remote_diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""


class RemoteInferenceBackend(Protocol):
    """The server side: turn a validated request + payload into a prediction.

    A future real deployment implements this on the inference server; the transport
    delivers requests to it. It never sees the mission, the evaluator, or GT.
    """

    model_strategy_id: str

    def infer(self, request: InferenceRequest, rgb: np.ndarray) -> np.ndarray: ...


class InferenceTransport(Protocol):
    """The network side: move a request/response and report what it cost.

    ``rgb`` is the resolved payload for the in-process simulation (a real transport
    would carry serialized bytes referenced by ``request.payload_ref``).
    """

    def execute(
        self,
        request: InferenceRequest,
        rgb: np.ndarray,
        network: NetworkRegime,
        timeout_s: float,
    ) -> TransportResult: ...


class SimulatedRemoteBackend:
    """A remote model simulated by running an existing local model-strategy in-process."""

    def __init__(self, mask_executor: ImageExecutor) -> None:
        self._mask_executor = mask_executor
        self.model_strategy_id = f"REMOTE({mask_executor.model_strategy_id})"

    def infer(self, request: InferenceRequest, rgb: np.ndarray) -> np.ndarray:
        del request  # the simulated server needs only the payload
        return self._mask_executor.run(rgb).prediction_mask


class SimulatedTransport:
    """Deterministic Model-A transport: stage times from the capture-time snapshot.

    Stage order and the latency formula (documented once, here):

        total = request_encoding_s
              + upload_mb x 8 / uplink_mbps          (upload_s)
              + rtt_ms / 1000                         (charged once, end to end)
              + remote_queue_s + remote_compute_s
              + download_mb x 8 / downlink_mbps       (download_s)
              + response_decoding_s

    Failure rules, in precedence order:

    1. Link unreachable (either direction has zero bandwidth): ``UNREACHABLE`` after
       ``unreachable_detect_s`` (capped by the timeout); nothing transmitted.
    2. Packet loss above ``max_loss_frac``: the upload transmits fully but is lost —
       ``UPLOAD_FAILED`` after encode + upload + one RTT (the missing-ack wait),
       capped by the timeout; upload charged, nothing downloaded.
    3. Backend exception: ``REMOTE_ERROR`` at the end of remote compute (error
       responses are treated as size-zero); upload charged.
    4. Total exceeding ``timeout_s``: ``TIMEOUT`` at exactly ``timeout_s``; transfers
       are charged **proportionally to transfer time completed** (partial upload /
       partial download), and any response is discarded.

    Model A means a regime change during the transfer does not affect an in-flight
    request; requests submitted after the change see the new regime. An
    interval-integrating Model B can replace this class behind ``InferenceTransport``.
    """

    def __init__(self, params: dict[str, float], backend: RemoteInferenceBackend) -> None:
        self._p = params
        self._backend = backend

    def execute(
        self,
        request: InferenceRequest,
        rgb: np.ndarray,
        network: NetworkRegime,
        timeout_s: float,
    ) -> TransportResult:
        p = self._p
        if not network.reachable:
            elapsed = min(p["unreachable_detect_s"], timeout_s)
            return TransportResult(
                status=RemoteStatus.UNREACHABLE,
                elapsed_s=elapsed,
                uploaded_mb=0.0,
                downloaded_mb=0.0,
                stages_s={"unreachable_detect_s": elapsed},
                failure_reason=f"link unreachable in regime {network.regime_id!r}",
            )

        encode_s = p["request_encoding_s"]
        upload_s = request.payload_mb * 8.0 / network.uplink_mbps
        rtt_s = network.rtt_ms / 1000.0
        queue_s = p["remote_queue_s"]
        download_mb = p["download_mb_per_call"]
        download_s = download_mb * 8.0 / network.downlink_mbps
        decode_s = p["response_decoding_s"]
        stages = {
            "request_encoding_s": encode_s,
            "upload_s": upload_s,
            "rtt_s": rtt_s,
            "remote_queue_s": queue_s,
            "download_s": download_s,
            "response_decoding_s": decode_s,
        }

        if network.packet_loss_frac > p["max_loss_frac"]:
            elapsed = min(encode_s + upload_s + rtt_s, timeout_s)
            return TransportResult(
                status=RemoteStatus.UPLOAD_FAILED,
                elapsed_s=elapsed,
                uploaded_mb=request.payload_mb,
                downloaded_mb=0.0,
                stages_s=stages | {"remote_compute_s": 0.0},
                failure_reason=(
                    f"packet loss {network.packet_loss_frac} above the transport ceiling "
                    f"{p['max_loss_frac']}"
                ),
            )

        try:
            prediction = self._backend.infer(request, rgb)
            compute_s = p["remote_compute_s"]
            backend_error = None
        except Exception as error:  # a real server returns an error response
            prediction = None
            compute_s = p["remote_compute_s"]
            backend_error = f"{type(error).__name__}: {error}"
        stages["remote_compute_s"] = compute_s

        upload_done = encode_s + upload_s
        server_done = upload_done + rtt_s + queue_s + compute_s
        total = server_done + download_s + decode_s

        if backend_error is not None:
            elapsed = min(server_done, timeout_s)
            return TransportResult(
                status=RemoteStatus.REMOTE_ERROR,
                elapsed_s=elapsed,
                uploaded_mb=request.payload_mb,
                downloaded_mb=0.0,
                stages_s=stages,
                failure_reason=backend_error,
            )

        if total > timeout_s:
            uploaded = request.payload_mb * _fraction_done(timeout_s, encode_s, upload_s)
            downloaded = download_mb * _fraction_done(timeout_s, server_done, download_s)
            return TransportResult(
                status=RemoteStatus.TIMEOUT,
                elapsed_s=timeout_s,
                uploaded_mb=uploaded,
                downloaded_mb=downloaded,
                stages_s=stages,
                failure_reason=(
                    f"end-to-end latency {total:.3f}s exceeds the configured timeout {timeout_s}s"
                ),
            )

        return TransportResult(
            status=RemoteStatus.SUCCESS,
            elapsed_s=total,
            uploaded_mb=request.payload_mb,
            downloaded_mb=download_mb,
            stages_s=stages,
            prediction_mask=prediction,
            remote_diagnostics={"backend_model_strategy_id": self._backend.model_strategy_id},
        )


def _fraction_done(timeout_s: float, stage_start_s: float, stage_duration_s: float) -> float:
    """Fraction of a transfer stage completed when the timeout fires."""
    if stage_duration_s <= 0.0:
        return 1.0 if timeout_s >= stage_start_s else 0.0
    return min(1.0, max(0.0, (timeout_s - stage_start_s) / stage_duration_s))


_REMOTE_PROVENANCE: Final = {
    "mission_latency_s": (
        "derived (capture-time network snapshot x payload size + configured stage "
        "components; simulated, Model A)"
    ),
    "measured_wall_clock_s": "measured (diagnostic only; not used by the simulation)",
    "energy_j": "simulated (configured onboard codec energy per call)",
    "communication_mb": "derived (transmitted bytes under the simulated transport rules)",
    "communication_energy_j": (
        "derived (transferred MB x configured J/MB + configured radio activation; simulated)"
    ),
}


class SimulatedRemoteExecutor:
    """The benchmark-facing remote executor: transport + backend + fallback, one result.

    Conforms to the V2 executor seam via ``run_with_context`` (local executors keep the
    plain ``run(rgb)``); returns the same ``ImageExecutionResult`` shape so the runner,
    metrics, and replay treat local and remote results uniformly.
    """

    def __init__(
        self,
        spec: ExecutorConfigSpec,
        transport: InferenceTransport,
        *,
        fallback: ImageExecutor | None = None,
    ) -> None:
        import time as _time

        self.config_id = spec.config_id
        self.model_strategy_id = spec.model_strategy_id
        self._spec = spec
        self._transport = transport
        self._fallback = fallback
        self._clock = _time.perf_counter
        self._params = {**REMOTE_PARAMETER_DEFAULTS, **dict(spec.parameters)}

    def run_with_context(self, rgb: np.ndarray, context: ExecutionContext) -> ImageExecutionResult:
        started = self._clock()
        p = self._params
        request = InferenceRequest(
            request_id=f"{context.scenario_id}/obs{context.observation_id:06d}",
            scenario_id=context.scenario_id,
            observation_id=context.observation_id,
            capture_time_s=context.capture_time_s,
            deadline_time_s=context.deadline_s,
            requested_config_id=self.config_id,
            payload_kind="raw_rgb",
            payload_shape=tuple(rgb.shape),
            payload_mb=self._spec.communication_mb_per_call,
            privacy_level=str(p.get("privacy_level", "remote_allowed")),
            payload_ref=f"inline:obs{context.observation_id:06d}",
        )
        transport_result = self._transport.execute(
            request, rgb, context.network, float(p["timeout_s"])
        )

        communication_mb = transport_result.uploaded_mb + transport_result.downloaded_mb
        communication_energy = (
            transport_result.uploaded_mb * p["uplink_energy_j_per_mb"]
            + transport_result.downloaded_mb * p["downlink_energy_j_per_mb"]
            + p["radio_activation_j"]  # charged on every attempt, incl. failed probes
        )
        onboard_energy = p["onboard_codec_energy_j"]
        diagnostics: dict[str, Any] = {
            "execution_location": "remote",
            "protocol_version": PROTOCOL_VERSION,
            "request": request.to_wire(),
            "remote_status": transport_result.status.value,
            "failure_reason": transport_result.failure_reason,
            "network": context.network.to_dict(),
            "latency_breakdown_s": dict(transport_result.stages_s),
            "uploaded_mb": transport_result.uploaded_mb,
            "downloaded_mb": transport_result.downloaded_mb,
            "timing_model": "capture_time_snapshot (Model A)",
            "fallback": None,
            **transport_result.remote_diagnostics,
        }

        if transport_result.status is RemoteStatus.SUCCESS:
            assert transport_result.prediction_mask is not None
            return ImageExecutionResult(
                config_id=self.config_id,
                model_strategy_id=self.model_strategy_id,
                success=True,
                prediction_mask=transport_result.prediction_mask,
                mission_latency_s=transport_result.elapsed_s,
                measured_wall_clock_s=self._clock() - started,
                energy_j=onboard_energy,
                communication_mb=communication_mb,
                communication_energy_j=communication_energy,
                measurement_provenance=dict(_REMOTE_PROVENANCE),
                diagnostics=diagnostics,
            )

        if self._fallback is not None:
            local = self._fallback.run(rgb)  # the same captured frame, never a re-render
            diagnostics["fallback"] = {
                "fallback_config_id": self._fallback.config_id,
                "fallback_model_strategy_id": self._fallback.model_strategy_id,
                "fallback_latency_s": local.mission_latency_s,
                "fallback_energy_j": local.energy_j,
            }
            return ImageExecutionResult(
                config_id=self.config_id,
                model_strategy_id=self.model_strategy_id,
                success=True,
                prediction_mask=local.prediction_mask,
                # Both attempts happened in sequence on the mission clock.
                mission_latency_s=transport_result.elapsed_s + local.mission_latency_s,
                measured_wall_clock_s=self._clock() - started,
                energy_j=onboard_energy + local.energy_j,
                communication_mb=communication_mb,
                communication_energy_j=communication_energy,
                measurement_provenance=dict(_REMOTE_PROVENANCE),
                diagnostics=diagnostics,
            )

        # No fallback: a failed remote attempt is a failed inference, not an empty
        # prediction. The mask exists only so downstream shapes hold; success=False and
        # the status keep the distinction end to end.
        return ImageExecutionResult(
            config_id=self.config_id,
            model_strategy_id=self.model_strategy_id,
            success=False,
            prediction_mask=np.zeros(rgb.shape[:2], dtype=bool),
            mission_latency_s=transport_result.elapsed_s,
            measured_wall_clock_s=self._clock() - started,
            energy_j=onboard_energy,
            communication_mb=communication_mb,
            communication_energy_j=communication_energy,
            measurement_provenance=dict(_REMOTE_PROVENANCE),
            diagnostics=diagnostics,
        )


def build_remote_executor(
    spec: ExecutorConfigSpec, local_registry: dict[str, ImageExecutor]
) -> SimulatedRemoteExecutor:
    """Wire a simulated remote executor from its spec and the already-built locals.

    The backend model is an existing local kind (``backend_kind``) run server-side;
    the optional ``fallback_config_id`` must name an already-built local executor.
    """
    from missionbench.v2.executors import build_executors
    from missionbench.v2.scenario import ExecutorConfigSpec as Spec

    params = {**REMOTE_PARAMETER_DEFAULTS, **dict(spec.parameters)}
    backend_kind = str(spec.parameters["backend_kind"])
    backend_spec = Spec(
        config_id=f"{spec.config_id}__backend",
        model_strategy_id=f"{spec.model_strategy_id}__backend",
        kind=backend_kind,
        mission_latency_s=float(params["remote_compute_s"]),
        energy_j_per_call=0.0,  # server-side energy is not charged to the vehicle
        communication_mb_per_call=0.0,
        quality_tier=spec.quality_tier,
        parameters={
            k.removeprefix("backend_"): v
            for k, v in spec.parameters.items()
            if k.startswith("backend_") and k != "backend_kind"
        },
    )
    backend = SimulatedRemoteBackend(build_executors((backend_spec,))[backend_spec.config_id])
    transport = SimulatedTransport(
        {k: float(params[k]) for k in REMOTE_PARAMETER_DEFAULTS}
        | {
            "remote_compute_s": float(params["remote_compute_s"]),
        },
        backend,
    )

    fallback: ImageExecutor | None = None
    fallback_id = spec.parameters.get("fallback_config_id")
    if fallback_id is not None:
        if fallback_id not in local_registry:
            raise SchemaValidationError(
                f"remote config {spec.config_id!r} names fallback_config_id "
                f"{fallback_id!r}, which is not a local executor in this scenario"
            )
        fallback = local_registry[str(fallback_id)]
    return SimulatedRemoteExecutor(spec, transport, fallback=fallback)
