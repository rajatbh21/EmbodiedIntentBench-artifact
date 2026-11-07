"""Lossless causal payload preparation and selected-output virtual placement.

Only the selected delivered action may load a prediction mask. The response size
is a fixed raw bit-mask ceiling derived from RGB dimensions, never predictions.
"""

import hashlib
import time
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from missionbench.v2.executors import ImageExecutionResult
from experiments.mission_network_emulation import account_radio_energy, emulate_job
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


@dataclass(frozen=True)
class PreparedPayload:
    payload: bytes
    rgb_sha256: str
    response_bytes: int

    def summary(self):
        return dict(
            bytes=len(self.payload),
            sha256=hashlib.sha256(self.payload).hexdigest(),
            rgb_sha256=self.rgb_sha256,
            response_bytes=self.response_bytes,
            codec="PNG_RGB_compress_level_6_optimize_false",
            decoded_rgb_exact=True,
        )


def prepare_payload(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB uint8 input required")
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG", compress_level=6, optimize=False)
    payload = stream.getvalue()
    with Image.open(BytesIO(payload)) as image:
        decoded = np.asarray(image.convert("RGB"))
    if not np.array_equal(decoded, rgb):
        raise ValueError("Lossless payload did not preserve the inference RGB")
    return PreparedPayload(payload, rgb_key(rgb), (rgb.shape[0] * rgb.shape[1] + 7) // 8)


class PlacementExecutor:
    def __init__(self, action, predictions, manifest_sha256, trace, costs):
        self.action, self.predictions = action, predictions
        self.manifest_sha256, self.trace, self.costs = manifest_sha256, trace, costs

    def run_selected(self, rgb, prepared, dispatch_network_ns, deadline_network_ns):
        started = time.perf_counter()
        action, costs = self.action, self.costs
        if prepared.rgb_sha256 != rgb_key(rgb):
            raise ValueError("Prepared payload belongs to another RGB")
        transport, radio, delivered = None, 0.0, True
        latency, energy, communication = action.latency_s, action.energy_j, 0.0
        status, decode_s, decode_j = "completed", 0.0, 0.0
        radio_details = None
        if action.placement == "remote":
            job = emulate_job(
                self.trace,
                start_ns=dispatch_network_ns,
                upload_bits=len(prepared.payload) * 8,
                download_bits=prepared.response_bytes * 8,
                service_ns=seconds_to_ns(action.latency_s, allow_zero=True),
                timeout_ns=seconds_to_ns(costs.timeout_s),
                deadline_ns=max(dispatch_network_ns, deadline_network_ns),
            )
            transport, status, delivered = job.to_dict(), job.status, job.prediction_eligible
            radio_details = account_radio_energy(
                job,
                upload_power_w=costs.tx_power_w,
                download_power_w=costs.rx_power_w,
                idle_power_w=costs.wait_power_w,
            )
            radio = radio_details["configured_radio_j"]
            communication = float(job.uploaded_bits + job.downloaded_bits) / 8e6
            decode_s = costs.decode_s if delivered else 0.0
            decode_j = decode_s * costs.codec_power_w
            latency, energy = job.elapsed_ns / 1e9 + decode_s, decode_j
        # No mask access at all for a canceled request. No fallback is scheduled.
        prediction = np.zeros(rgb.shape[:2], dtype=np.bool_)
        if delivered:
            prediction = self.predictions[prepared.rgb_sha256].copy()
            if prediction.dtype != np.bool_ or prediction.shape != rgb.shape[:2]:
                raise ValueError("Wrong selected cached prediction")
            if action.placement == "remote":
                packed = np.packbits(prediction.reshape(-1), bitorder="big")
                restored = np.unpackbits(packed, bitorder="big")[: prediction.size].reshape(
                    prediction.shape
                )
                if len(packed) != prepared.response_bytes or not np.array_equal(
                    restored, prediction
                ):
                    raise ValueError("Fixed bit-packed response parity failed")
        return ImageExecutionResult(
            config_id=action.action_id,
            model_strategy_id=action.source_config_id,
            success=delivered,
            prediction_mask=prediction,
            mission_latency_s=latency,
            measured_wall_clock_s=time.perf_counter() - started,
            energy_j=energy,
            communication_mb=communication,
            communication_energy_j=radio,
            measurement_provenance=dict(
                inference="Cached real masks; local warm p95; virtual remote service",
                network="Authored piecewise virtual link; no remote network or server call",
                energy="Configured onboard assumptions; server electrical energy excluded",
            ),
            diagnostics=dict(
                source_config_id=action.source_config_id,
                source_signature=action.source_signature,
                placement=action.placement,
                cache_sha256=self.manifest_sha256,
                attempt_status=status,
                delivered_prediction=delivered,
                transport=transport,
                radio=radio_details,
                decode_latency_s=decode_s,
                decode_energy_j=decode_j,
                no_prediction_loaded_on_failure=not delivered,
            ),
        )
