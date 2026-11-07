"""Hardware-in-the-loop check: frame-paced, blocking execution on the Orin.

For each recording, captures are released on the wall clock at their recorded
mission times (every 2 s). A single blocking executor runs the selected
configuration with real inference. A capture released while the executor is busy
is skipped (Eq. 5). Non-inference stages (encode+route before, tracking after,
one model load before the first call) are emulated with their declared durations,
exactly as charged by the replay, so the only measured component is inference.

Output: per recording, processed capture IDs, wall completion times, per-call
inference latency and rail-sum energy, for comparison with measured-profile replay.
"""
import sys, os, json, time, argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orin_profile as op  # reuses the shim, sampler and spec construction
import numpy as np

PRE_S, POST_S, LOAD_S = 0.07, 0.10, 0.35  # replay: encode 0.02 + router 0.05; tracking 0.1; load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="Path to the benchmark repository root")
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--config", default="CFG_MASKRCNN_1280")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release).resolve()
    sys.path.insert(0, str(rel)); sys.path.insert(0, str(rel / "tools"))
    cache = rel / ".cache"
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    import torch
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    from missionbench.v2.scenario import ExecutorConfigSpec
    from missionbench.v2.instance_models import InstanceBackend, TorchInstanceSegmentationExecutor
    import experiments.mission_execution_profile as mep
    from dataclasses import replace

    specs = {s["config_id"]: s for s in json.load(open(rel / "data/native_models.json"))["executor_configs"]}
    control = ExecutorConfigSpec(**specs[mep.CONTROL])
    if a.device == "cuda":
        control = replace(control, parameters={**control.parameters, "device": "cuda"})
    factory = lambda spec: TorchInstanceSegmentationExecutor(spec, backend=InstanceBackend(spec.parameters))
    pe = mep.PipelineExecutor(control, control, factory=factory)
    bindings = {b["case_id"]: b for b in json.load(open(rel / "data/replay/native_bindings.json"))["bindings"]}

    sp = op.Sampler(); sp.start()
    out = dict(device=a.device, threads=a.threads, config=a.config, runs={})
    for case, b in bindings.items():
        rec = rel / "data/replay" / Path(b["recording_path"]).parent
        frames = []
        for f in b["frames"]:
            with np.load(rec / f["source_file"], allow_pickle=False) as z:
                frames.append((f["capture_id"], f["mission_time_s"], z["rgb"].copy()))
        for _ in range(3):  # warm the executor before the mission clock starts
            pe.run(frames[0][2])
        time.sleep(2.0)
        t0 = time.perf_counter() + 0.5
        busy_until, events, skipped, first = 0.0, [], [], True
        for cid, mt, rgb in frames:
            if mt < busy_until:  # released while busy -> unavailable (Eq. 5)
                skipped.append(cid)
                continue
            while time.perf_counter() < t0 + mt:
                time.sleep(0.0005)
            start = time.perf_counter() - t0
            time.sleep(PRE_S + (LOAD_S if first else 0.0))
            i0 = time.perf_counter()
            pe.run(rgb)
            i1 = time.perf_counter()
            time.sleep(POST_S)
            done = time.perf_counter() - t0
            p, n = op.mean_power(sp.s, i0, i1)
            events.append(dict(i=cid, t=mt, start=start, done=done, inference_s=i1 - i0,
                               inference_energy_j=p * (i1 - i0), power_w=p, samples=n))
            busy_until, first = done, False
        out["runs"][case] = dict(events=events, skipped=skipped,
                                 processed=[e["i"] for e in events])
        print(case, "processed", [e["i"] for e in events],
              "lat", [round(e["inference_s"], 2) for e in events][:8], flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    sp.stop = True


if __name__ == "__main__":
    main()
