"""Fetch only the pinned evidence files (about 52 MB), verifying SHA-256.

No model execution, installers, account registration or external uploads.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def fetch(output):
    lock = json.loads((Path(__file__).parent / "sources_lock.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    for item in lock["files"]:
        path = output / item["dataset"] / item["name"]
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("existing file differs: " + str(path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            item["url"], headers={"User-Agent": "Mission Routing Benchmark-evidence/0.1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(item["bytes"] + 1)
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("source changed: " + item["url"])
        with path.open("xb") as out:
            out.write(data)
    (output / "download_manifest.json").write_text(json.dumps(lock["files"], indent=2) + "\n")
    (output / "sarenv_manifest.json").write_text(
        json.dumps(lock["sarenv_manifest"], indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    fetch(parser.parse_args().output)
