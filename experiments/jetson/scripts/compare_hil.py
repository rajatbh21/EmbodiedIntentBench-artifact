"""Compare hardware-in-the-loop (real, frame-paced) execution with measured-profile replay."""
import json, sys
import numpy as np

real = json.load(open(sys.argv[1]))      # orin_realtime.py output
replay = json.load(open(sys.argv[2]))    # dump_events.py output (same profile, fixed_local_mask)
ev = replay["events"]
same = 0; tot_r = tot_p = 0; dts = []; lat = []
for case, r in real["runs"].items():
    rp = [e["i"] for e in ev[case]]
    rr = r["processed"]
    same += rp == rr
    tot_r += len(rr); tot_p += len(rp)
    done_p = {e["i"]: e["done"] for e in ev[case]}
    for e in r["events"]:
        if e["i"] in done_p:
            dts.append(e["done"] - done_p[e["i"]])
        lat.append(e["inference_s"])
    print(f"{case:28s} replay={rp}\n{'':28s} real  ={rr}")
m = replay["latency"]["CFG_MASKRCNN_1280"]
print(f"recordings with identical processed sets: {same}/{len(real['runs'])}; processed real={tot_r} replay={tot_p}")
print(f"completion time real-replay: mean={np.mean(dts):+.3f}s  max|.|={np.max(np.abs(dts)):.3f}s  n={len(dts)}")
print(f"inference latency real: mean={np.mean(lat):.3f} p95={np.percentile(lat,95):.3f}  replay charged={m['latency_s']:.3f}")
