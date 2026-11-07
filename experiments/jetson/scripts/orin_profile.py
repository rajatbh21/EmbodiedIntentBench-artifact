"""Orin execution profile for the EmbodiedIntentBench perception configurations.

Reuses the release's own PipelineExecutor (experiments/mission_execution_profile.py),
so preprocessing, rotation, tiling, union fusion and mask post-processing are exactly
those behind the paper's declared latencies. Only the device changes.

Latency  = wall time of PipelineExecutor.run(rgb) per frame (returns a host numpy
           mask, so GPU work is synchronised inside the timed region).
Energy   = INA3221 module power VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 (VDDQ_VDD2_1V8AO is a subset of VIN_SYS_5V0 and is logged but NOT added)
           integrated over a back-to-back block of calls, divided by the call count.
           'incremental' subtracts the idle rail sum measured just before.
Parity   = IoU / exact match of the produced mask against the stored prediction bank
           (only the three dynamic-study configurations have a bank in the release).
"""
import argparse, json, math, os, sys, threading, time
import enum, typing, datetime
if not hasattr(enum, "StrEnum"):  # Python 3.10 shim (Orin JetPack 6.1); 3.11 semantics
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()
    enum.StrEnum = StrEnum
if not hasattr(typing, "Self"):
    typing.Self = typing.Any
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc
from dataclasses import replace
from pathlib import Path

import numpy as np

import glob as _glob


def _discover_rails(names=("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0", "VDDQ_VDD2_1V8AO")):
    """Resolve INA3221 rails by label; hwmon numbering changes across reboots."""
    found = {}
    for label in _glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"):
        name = open(label).read().strip()
        if name in names:
            idx = label.rsplit("/in", 1)[1].split("_")[0]
            found[name] = label.rsplit("/", 1)[0] + "/{}" + idx + "_input"
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f"INA3221 rails not found: {sorted(missing)}")
    return found


RAILS = _discover_rails()
GPU_FREQ = "/sys/class/devfreq/17000000.gpu/cur_freq"


TOTAL_RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")  # VDDQ_VDD2_1V8AO is INCLUDED in VIN_SYS_5V0 (NVIDIA r36.4.4 docs)


def read_rails():
    return {k: int(open(p.format("in")).read()) * int(open(p.format("curr")).read()) / 1e6 for k, p in RAILS.items()}


def rail_sum(r=None):
    r = read_rails() if r is None else r
    return sum(r[k] for k in TOTAL_RAILS)


class Sampler(threading.Thread):
    def __init__(self, hz=100):
        super().__init__(daemon=True)
        self.dt, self.s, self.stop = 1 / hz, [], False

    def run(self):
        self.raw = []  # timestamped per-rail samples, retained in the output
        while not self.stop:
            r = read_rails(); t = time.perf_counter()
            self.s.append((t, rail_sum(r), int(open(GPU_FREQ).read())))
            self.raw.append((t, r))
            time.sleep(self.dt)


def mean_power(samples, t0, t1):
    w = [p for t, p, _ in samples if t0 <= t <= t1]
    return (float(np.mean(w)), len(w)) if w else (float("nan"), 0)


def stats(xs):
    a = np.asarray(xs, float)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), max=float(a.max()), min=float(a.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="Path to the benchmark repository root")
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--configs", default="all")
    ap.add_argument("--cases", default="sector_revisit/sector_3,shoreline_loop/sector_7")
    ap.add_argument("--frames-per-case", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rel = Path(a.release).resolve()
    sys.path.insert(0, str(rel))
    sys.path.insert(0, str(rel / "tools"))
    cache = rel / ".cache"
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["YOLO_CONFIG_DIR"] = str(cache / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    from missionbench.v2.scenario import ExecutorConfigSpec
    from missionbench.v2.cabinet_models import CabinetBackend, TorchCabinetSegmentationExecutor
    from missionbench.v2.instance_models import InstanceBackend, TorchInstanceSegmentationExecutor
    from experiments.lake_fast_perception import Segmenter
    import experiments.mission_execution_profile as mep

    specs = {s["config_id"]: s for s in json.load(open(rel / "data/native_models.json"))["executor_configs"]}
    for s in specs.values():
        for k in ("checkpoint", "bundle_manifest"):
            if k in s["parameters"]:
                s["parameters"][k] = str(cache / s["parameters"][k])
    control = ExecutorConfigSpec(**specs[mep.CONTROL])
    if a.device == "cuda":
        control = replace(control, parameters={**control.parameters, "device": "cuda"})
    yolo = ExecutorConfigSpec(**specs[mep.YOLO])
    cab = ExecutorConfigSpec(**specs[mep.CABINET])
    nano_ckpt = cache / "models/yolo11n-seg.pt"
    tiled = lambda name, turns, gamma: replace(
        yolo, config_id=name, parameters={**yolo.parameters, "turns": turns, "gamma": gamma})
    all_specs = {
        mep.CONTROL: control,
        mep.ROTATED: replace(control, config_id=mep.ROTATED),
        mep.UNION: replace(control, config_id=mep.UNION),
        mep.YOLO: yolo,
        mep.NANO: replace(yolo, config_id=mep.NANO,
                          parameters={**yolo.parameters, "checkpoint": str(nano_ckpt), "imgsz": 640}),
        mep.TILED: tiled(mep.TILED, 0, 1.0),
        mep.TILED_ROTATED: tiled(mep.TILED_ROTATED, 1, 1.0),
        mep.TILED_GAMMA: tiled(mep.TILED_GAMMA, 0, 0.5),
        mep.CABINET: cab,
    }
    wanted = list(all_specs) if a.configs == "all" else a.configs.split(",")

    def segmenter(ckpt, size):
        seg = Segmenter(ckpt, size)  # constructor warms up on CPU with 4 threads
        torch.set_num_threads(a.threads)
        if a.device == "cuda":
            orig = seg.model.predict
            seg.model.predict = lambda **kw: orig(**{**kw, "device": 0})
            seg.model.predictor = None  # predictor was set up on CPU during warmup; rebuild on GPU
            seg.predict(np.zeros((720, 960, 3), dtype=np.uint8), 0)
            assert next(seg.model.predictor.model.parameters()).is_cuda, "YOLO not on GPU"
        return seg

    def factory(spec):
        if spec.kind == "torch_instance_segmentation":
            return TorchInstanceSegmentationExecutor(spec, backend=InstanceBackend(spec.parameters))
        b = CabinetBackend(spec.parameters)
        if a.device == "cuda":
            b.model = b.model.cuda()
            b.mean, b.std = b.mean.cuda(), b.std.cuda()

            def infer(rgb, b=b):
                image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
                image = (image - b.mean) / b.std
                start = time.perf_counter()
                with torch.inference_mode():
                    logits = b.model(image)[0]
                    prob = logits.softmax(dim=1)[0, b.target].cpu().numpy()
                return prob, time.perf_counter() - start

            b.infer = infer
        return TorchCabinetSegmentationExecutor(spec, backend=b)

    bindings = {b["case_id"]: b for b in json.load(open(rel / "data/replay/native_bindings.json"))["bindings"]}
    frames = []
    for case in a.cases.split(","):
        b = bindings[case]
        rec = rel / "data/replay" / Path(b["recording_path"]).parent
        fr = b["frames"]
        idx = sorted({round(i * (len(fr) - 1) / (a.frames_per_case - 1)) for i in range(a.frames_per_case)})
        for i in idx:
            with np.load(rec / fr[i]["source_file"], allow_pickle=False) as z:
                frames.append((case, fr[i]["capture_id"], z["rgb"].copy()))

    meta = dict(device=a.device, threads=a.threads, torch=torch.__version__,
                gpu=torch.cuda.get_device_name() if a.device == "cuda" else None,
                l4t=open("/etc/nv_tegra_release").readline().strip(),
                nvpmodel=os.popen("nvpmodel -q 2>/dev/null | grep -A1 'NV Power Mode' | tail -1").read().strip(),
                frames=[(c, i) for c, i, _ in frames], warmup=a.warmup)
    sp = Sampler(); sp.start(); time.sleep(1.0)
    t0 = time.perf_counter(); time.sleep(6.0); t1 = time.perf_counter()
    meta["idle_rail_sum_W"], n_idle = mean_power(sp.s, t0, t1)
    if n_idle < 50:
        raise RuntimeError(f"Idle power sampling failed ({n_idle} samples); check INA3221 sysfs permissions")
    out = dict(meta=meta, configs={})
    print(json.dumps(meta)[:400], flush=True)

    for name in wanted:
        spec = all_specs[name]
        pe = mep.PipelineExecutor(spec, control, factory=factory, segmenter=segmenter)
        for i in range(a.warmup):
            pe.run(frames[i % len(frames)][2])
        if a.device == "cuda":
            torch.cuda.synchronize()
        lat, masks = [], []
        b0 = time.perf_counter()
        for case, cid, rgb in frames:
            c0 = time.perf_counter()
            m = pe.run(rgb)
            lat.append(time.perf_counter() - c0)
            masks.append(m)
        b1 = time.perf_counter()
        p, n = mean_power(sp.s, b0, b1)
        if n < 5 or not sp.is_alive():
            raise RuntimeError(f"Power sampler produced {n} samples (thread alive={sp.is_alive()}); refusing to write NaN energy")
        mhz = [f for t, _, f in sp.s if b0 <= t <= b1]
        e_total = p * (b1 - b0) / len(frames)
        e_incr = (p - meta["idle_rail_sum_W"]) * (b1 - b0) / len(frames)
        parity = []
        for (case, cid, _), m in zip(frames, masks):
            f = rel / f"data/replay/predictions/{case}/prediction_{cid:05d}_{name}.npz"
            if f.exists():
                with np.load(f, allow_pickle=False) as z:
                    e = z["prediction_mask"]
                inter, union = np.logical_and(m, e).sum(), np.logical_or(m, e).sum()
                parity.append(dict(exact=bool(np.array_equal(m, e)), iou=float(inter / union) if union else 1.0,
                                   diff_px=int((m != e).sum())))
        s = stats(lat)
        per_rail = {k: float(np.mean([r[k] for t, r in sp.raw if b0 <= t <= b1])) for k in RAILS}
        rec = dict(per_rail_W=per_rail, total_rails=list(TOTAL_RAILS), raw_rail_samples=[(t - b0, r) for t, r in sp.raw if b0 <= t <= b1],
                   latency_samples_s=lat, latency_s=s, declared_style_latency_s=math.ceil(s["p95"] * 100) / 100,
                   block_rail_sum_W=p, power_samples=n, energy_j_per_call_total=e_total,
                   energy_j_per_call_incremental=e_incr, gpu_mhz_median=float(np.median(mhz) / 1e6) if mhz else None,
                   neural_calls=pe.neural_calls)
        if parity:
            rec["parity"] = dict(n=len(parity), exact=sum(x["exact"] for x in parity),
                                 mean_iou=float(np.mean([x["iou"] for x in parity])),
                                 min_iou=float(np.min([x["iou"] for x in parity])))
        out["configs"][name] = rec
        print(f"{name:32s} p50={s['p50']:.3f} p95={s['p95']:.3f} P={p:.1f}W E={e_total:.2f}J "
              f"Einc={e_incr:.2f}J {rec.get('parity', '')}", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        del pe
        if a.device == "cuda":
            torch.cuda.empty_cache()
    sp.stop = True
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
