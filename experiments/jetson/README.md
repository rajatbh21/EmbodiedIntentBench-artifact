# Jetson AGX Orin cost sensitivity

This directory contains the corrected local-compute profiles, stored-prediction
replay code and reference outcomes for the dynamic Lake study. It also contains
the capture schedules from the frame-paced device check. The profile affects
local actions only: remote service and network behavior remain modeled, and
mission outcomes are evaluated offline from stored predictions. This is not an
end-to-end flight or a device execution of an adaptive router.

## Replay with the recorded profiles

From the repository root on Linux, install the replay dependencies as described
in the top-level README and prepare the complete dynamic study:

```bash
python tools/replay.py --all --workers 4 --output results/replay-full
python experiments/jetson/scripts/measured_replay.py \
  --root results/replay-full \
  --profiles declared,orin_15wv2_cpu4,orin_15wv2_cuda \
  --protocols causal,no_skip \
  --orin-dir experiments/jetson/profiles \
  --workers 4 --out results/jetson-replay
python experiments/jetson/scripts/summarize.py results/jetson-replay
python experiments/jetson/scripts/verify_replay.py results/jetson-replay
```

The replay evaluates 648 executions for each of the three cost profiles and
each of the two protocols. It writes one `rows.jsonl` containing all requested
profiles. Use a fresh output directory. The included references under
`references/replay/` provide the original results separately for each profile.
The verification command requires all three profiles and both protocols and
compares the complete rows exactly, reporting missing, extra or differing jobs.
Summarize the references with:

```bash
python experiments/jetson/scripts/summarize.py \
  experiments/jetson/references/replay/declared \
  experiments/jetson/references/replay/orin_15wv2_cpu4 \
  experiments/jetson/references/replay/orin_15wv2_cuda
```

For the causal protocol the reference totals are 218, 142 and 338 successes
out of 648 for declared, Orin CPU and Orin GPU costs, respectively; for the
no-skip counterfactual they are 255, 192 and 348. Each total combines nine
policies and 72 conditions per policy. The six recordings underlying those
conditions are reused; conditions are not independent flights.

## Profiling protocol and device check

The included profiles were collected on a Jetson AGX Orin Developer Kit in
MODE_15W, using four CPU cores or the GPU. For each model, after five warm-up
calls, latency was measured over 24 captures drawn from two recordings. The
replay uses the 95th-percentile wall time and mean module power during the
profiling block, multiplying them to charge energy per local call. The module
power sum uses VDD_GPU_SOC, VDD_CPU_CV and VIN_SYS_5V0; VDDQ_VDD2_1V8AO is
recorded separately and excluded from the sum because it is included in
VIN_SYS_5V0. The JSON profiles retain individual latency and timestamped rail
samples. They are measurements for this setup, not universal device constants.

For a fresh profiling run, obtain the pinned dynamic-study model weights and
compatible native dependencies using [the model guide](../../docs/MODELS.md).
The device must expose the Jetson INA3221 sensor rails and use compatible
JetPack, PyTorch and TorchVision builds. With the desired power mode already
configured, run on the device:

```bash
python experiments/jetson/scripts/orin_profile.py \
  --release "$PWD" --device cuda \
  --configs CFG_MASKRCNN_1280,CFG_YOLO11S_960_ROT0,CFG_CABINET_UAVID_1280 \
  --out results/my-orin-cuda.json
```

Use `--device cpu --threads 4` to profile the CPU. The script changes neither
the power mode nor device clocks. Hardware profiling requires actual native
inference, model weights and appropriate sensor access, unlike stored-prediction
replay. Differences in hardware, versions and operating conditions can change
the measured profile.

The scheduling check ran fixed local Mask R-CNN with frame-paced inference for
six recordings per processor under one urgency/link/battery setting. Compare
the saved device and replay schedules with:

```bash
python experiments/jetson/scripts/compare_hil.py \
  experiments/jetson/references/scheduling/hil_15w_cpu4.json \
  experiments/jetson/references/scheduling/events_15wv2_cpu4_localmask.json
python experiments/jetson/scripts/compare_hil.py \
  experiments/jetson/references/scheduling/hil_15w_cuda.json \
  experiments/jetson/references/scheduling/events_15wv2_cuda_localmask.json
```

The processed capture sets matched in 4/6 CPU and 6/6 GPU runs. Other stages
were emulated with declared durations. Device masks were not retained and the
mission outcomes of device schedules were scored with stored predictions. The
original device scheduling logs used an outdated power-rail sum, so their power
and energy fields have been removed; use the corrected profiling files for
the sensitivity replay. `orin_realtime.py` permits a new scheduling measurement
on a compatible device using the corrected rail sum.
