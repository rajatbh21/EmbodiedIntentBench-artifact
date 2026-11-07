"""Contract-satisfied gating ("stop when the intent is fulfilled") over the dynamic study.

Gate rule (uses only the router's OWN emitted reports and the PUBLIC contract; no ground truth):
  in the phase containing the current capture, let K = ceil(phase.required_recall * N_FOCAL)
  (N_FOCAL = 3, stated in the public request "at least two of the three people"). Once the
  runtime has issued reports for >= K distinct runtime tracks (component_id) whose evidence lies
  inside this phase and which completed before the current capture, the policy abstains for the
  rest of the phase. Otherwise the wrapped policy decides as usual.

Runs each (profile, contract, gate on/off, policy, condition) through the release's instrumented
runtime with measured local costs (same patch as measured_replay.py).
"""
import argparse, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'jetson' / 'scripts'))
import measured_replay as mr

N_FOCAL = 3
STATE = {"gate": False, "reports": [], "gated_captures": 0}


def _init(root, profiles):
    mr._init(root, profiles)
    from experiments.mission_task_deadlines import pipeline as pl
    import experiments.intent_mission_v2.policy as P

    orig_exec = pl.DeadlinePipeline.execute

    def execute(self, *a, **k):
        out = orig_exec(self, *a, **k)
        STATE["reports"].extend(out.get("emitted_reports") or [])
        return out

    pl.DeadlinePipeline.execute = execute
    orig_make = P.make_policy

    class Gated:
        def __init__(self, inner):
            self.inner = inner

        def select(self, **inp):
            if STATE["gate"]:
                now_s = inp["release_ns"] / 1e9
                phases = inp["intent"]["mission"]["phases"]
                ph = next((p for p in phases if p["start_s"] <= now_s < p["end_s"]), None)
                if ph is not None:
                    k = math.ceil(ph["required_recall"] * N_FOCAL - 1e-9)
                    done = {r["component_id"] for r in STATE["reports"]
                            if r["completion_time_s"] <= now_s + 1e-9
                            and r["evidence"] and all(ph["start_s"] <= e["capture_s"] < ph["end_s"] for e in r["evidence"])}
                    if len(done) >= k:
                        STATE["gated_captures"] += 1
                        return dict(policy="contract_gate", action_id=None,
                                    reason=f"contract satisfied in phase {ph['phase_id']}: {len(done)}>={k} own reports")
            return self.inner.select(**inp)

    P.make_policy = lambda *a, **k: Gated(orig_make(*a, **k))


def _run(job):
    from experiments.validation import execution
    STATE["gate"], STATE["reports"], STATE["gated_captures"] = job["gate"], [], 0
    mr._CUR["name"] = job["profile"]
    spec = {k: job[k] for k in ("job_id", "item")}
    if job["contract"] != "confirm":
        spec["contract"] = job["contract"]
    r = execution.run_job(dict(spec, keep_events=True))
    ev = r["events"]
    return dict(profile=job["profile"], contract=job["contract"], gate=job["gate"],
                policy=job["item"]["policy"], run_id=job["item"]["run_id"], energy=job["item"]["energy_condition"],
                split=job["item"]["split"], flight=job["item"]["flight"],
                success=r["summary"]["success"], failed=r["summary"]["failed"],
                processed=sum(1 for e in ev if e["action"]), gated=STATE["gated_captures"],
                local_calls=sum(1 for e in ev if e["action"] and not e["action"].startswith("REMOTE")),
                remote_calls=sum(1 for e in ev if e["action"] and e["action"].startswith("REMOTE")),
                task_energy_j=ev[-1]["energy"] if ev else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="declared,orin_15wv2_cuda,orin_15wv2_cpu4")
    ap.add_argument("--contracts", default="confirm,confirm_strict,revisit,time_critical")
    ap.add_argument("--policies", default="fixed_local_mask,fixed_remote_mask,risk_current,risk_initial,quality_current,quality_initial")
    ap.add_argument("--orin-dir", required=True, help="Directory containing corrected Jetson profiles")
    ap.add_argument("--replay-root", required=True, help="Prepared replay bundle containing plan.json")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = str(Path(a.replay_root).resolve())
    grid = [i for i in json.loads((Path(root) / "plan.json").read_text())["grid"] if i["policy"] in a.policies.split(",")]
    if a.limit:
        grid = grid[: a.limit]
    profiles = mr.load_profiles(a.orin_dir, a.profiles.split(","))
    jobs = [dict(job_id=f"{p}|{c}|{g}|{i['run_id']}", item=i, profile=p, contract=c, gate=g)
            for p in a.profiles.split(",") for c in a.contracts.split(",") for g in (False, True) for i in grid]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(root, profiles)) as ex:
        for k, r in enumerate(ex.map(_run, jobs, chunksize=4)):
            rows.append(r)
            if (k + 1) % 1000 == 0:
                print(f"{k+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"done {len(rows)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
