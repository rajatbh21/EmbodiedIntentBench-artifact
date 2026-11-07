"""Re-run the 648-cell dynamic-urgency study with Orin-measured local execution costs.

Only LOCAL action descriptors change:
  latency_s := measured p95 wall latency of the release PipelineExecutor on the Orin
  energy_j  := measured mean INA3221 rail-sum power during calls x that latency
              (replaces the declared 15 W x latency accounting)
Remote service times, network, load/switch costs, routing, tracking, policies,
contracts and the evaluator are untouched. Policies see the new costs through the
same ActionDescriptor interface they always use.

Usage (from the release root):
  python measured_replay.py --profiles declared,orin_cpu4,orin_cuda \
      --protocols causal,no_skip --orin-dir prof --workers 28 --out results/orin_replay
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
_PROFILES = {}
_CUR = {"name": "declared"}


def load_profiles(orin_dir, names):
    out = {"declared": None}
    for n in names:
        if n == "declared":
            continue
        d = json.loads((Path(orin_dir) / f"{n}.json").read_text())
        out[n] = {
            cid: dict(latency_s=c["latency_s"]["p95"], power_w=c["block_rail_sum_W"])
            for cid, c in d["configs"].items()
        }
    return out


def _init(root, profiles):
    from experiments.validation import execution, runtime

    _PROFILES.update(profiles)
    original = runtime.scale_actions

    def patched(actions, multiplier):
        prof = _PROFILES.get(_CUR["name"])
        if prof is None:
            return original(actions, multiplier)
        assert multiplier == 1
        out = []
        for a in actions:
            if a.placement == "local":
                m = prof[a.source_config_id]
                out.append(replace(a, latency_s=m["latency_s"], energy_j=m["power_w"] * m["latency_s"]))
            else:
                out.append(a)
        return tuple(out)

    runtime.scale_actions = patched
    execution._init(root)


def _run(job):
    from experiments.validation import execution

    _CUR["name"] = job["profile"]
    r = execution.run_job(dict(job, keep_events=True))
    ev = r["events"]
    return dict(
        job_id=job["job_id"], profile=job["profile"], protocol=job.get("protocol", "causal"),
        run_id=job["item"]["run_id"], policy=job["item"]["policy"], split=job["item"]["split"],
        flight=job["item"]["flight"], schedule=job["item"]["intent_schedule"],
        link=job["item"]["link"], energy=job["item"]["energy_condition"],
        success=r["summary"]["success"], failed=r["summary"]["failed"],
        processed=len(ev), skipped=sum(len(e["skipped"]) for e in ev),
        late=sum(bool(e["late"]) for e in ev),
        actions=[e["action"] for e in ev],
        task_energy_j=ev[-1]["energy"] if ev else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="declared,orin_15wv2_cpu4,orin_15wv2_cuda")
    ap.add_argument("--protocols", default="causal")
    ap.add_argument("--orin-dir", required=True)
    ap.add_argument("--root", default=str(REPO / "results/replay-full"), help="Prepared full-study replay directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = str(Path(a.root).resolve())
    grid = json.loads((Path(root) / "plan.json").read_text())["grid"]
    if a.limit:
        grid = grid[: a.limit]
    names = a.profiles.split(",")
    profiles = load_profiles(a.orin_dir, names)
    jobs = [dict(job_id=f"{p}|{pr}|{i['run_id']}", item=i, profile=p, protocol=pr)
            for p in names for pr in a.protocols.split(",") for i in grid]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(root, profiles)) as ex:
        for k, r in enumerate(ex.map(_run, jobs, chunksize=4)):
            rows.append(r)
            if (k + 1) % 200 == 0:
                print(f"{k+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "profiles_used.json").write_text(json.dumps(profiles, indent=1))
    print(f"done {len(rows)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
