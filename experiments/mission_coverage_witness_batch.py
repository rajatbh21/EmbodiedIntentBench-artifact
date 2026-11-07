"""Search and independently validate all post-hoc witness cells without policy tuning."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_repair.protocol import verify


def run(root):
    p = verify(root)
    logs = root / "witness_logs"
    logs.mkdir(exist_ok=False)

    def condition(flight):
        path = logs / f"{flight}.log"
        start = time.monotonic()
        with path.open("x") as stream:
            for module in ("mission_coverage_witness", "mission_coverage_witness_validate"):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments." + module,
                        "--output",
                        str(root),
                        "--flight",
                        flight,
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        return dict(flight=flight, elapsed_s=time.monotonic() - start, log_sha256=sha(path))

    jobs = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(condition, f) for f in p["cases"]]
        for future in as_completed(futures):
            jobs.append(future.result())
            save(logs / "progress.json", dict(expected=len(futures), completed=jobs))
            print(jobs[-1], flush=True)
    rows, hashes, events = [], {}, 0
    for flight in p["cases"]:
        directory = root / "witnesses" / flight
        validation = read(directory / "validation.json")
        if not validation["complete"] or not validation["independent_witness_reconstruction"]:
            raise ValueError("Independent witness audit missing")
        if validation["protocol_sha256"] != sha(root / "protocol.json"):
            raise ValueError("Changed witness protocol")
        for name, expected in validation["source_summaries_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError("Changed witness summary")
        rows.extend(read(directory / "summary.json")["cases"])
        hashes[flight] = sha(directory / "validation.json")
        events += validation["witness_decisions_reproduced"]
    expected = {(f, m) for f in p["cases"] for m in p["missions"]}
    if len(rows) != len(expected) or {(r["flight"], r["mission"]) for r in rows} != expected:
        raise ValueError("Incomplete witness grid")
    save(
        root / "witness_validation.json",
        dict(
            complete=True,
            cases=rows,
            conditions=len(rows),
            full_contract_witnesses=sum(r["full_contract_witness"] for r in rows),
            independently_reconstructed_events=events,
            independent_audits_sha256=hashes,
            source_sha256=sha(Path(__file__)),
            search_source_sha256=sha(Path(__file__).with_name("mission_coverage_witness.py")),
            validator_sha256=sha(Path(__file__).with_name("mission_coverage_witness_validate.py")),
            protocol_sha256=sha(root / "protocol.json"),
            gt_aware=True,
            deployable_policy=False,
            posthoc=True,
            failed_search_is_inconclusive=True,
        ),
    )
    print(
        dict(conditions=len(rows), witnesses=sum(r["full_contract_witness"] for r in rows)),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
