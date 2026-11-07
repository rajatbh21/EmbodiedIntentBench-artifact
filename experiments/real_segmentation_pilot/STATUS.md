# Real-segmentation pilot — status

**A real pilot has been run (V3 P4, 2026-07-30): real UAV imagery (UAVid) x real
pretrained models (torchvision LRASPP / DeepLabV3), through the unchanged V1 empirical
bundle pipeline and benchmark CLI.** The earlier blockers table is obsolete: torch,
torchvision, official DEFAULT checkpoints, and an Apple-silicon `mps` device have been
available since V2.1, and UAVid is downloaded locally (never committed).

## What the pilot is — and is not

- **Real**: imagery (UAVid oblique urban UAV keyframes), annotations (semantic Humans
  masks), model inference (tiled 512 px windows), measured per-frame latency (wall
  clock with device sync).
- **Derived**: person *instances* are connected components of the semantic Humans
  class (UAVid has no instance or track annotations); each instance is its own target,
  so track-level recall reads as per-instance recall. Windowed evaluation
  (1280x720 native crops centred on annotated activity) keeps the frozen dense-mask
  wire format writable — full-4K crowded frames produce multi-GB ground truth.
- **Estimated**: energy (assumed 20 W package power x measured time), labelled
  `estimated` end to end.
- **Not a claim**: COCO/VOC checkpoints were not trained on aerial imagery. The
  measured recall (LRASPP 0.030, DeepLabV3 0.122 over 164 derived instances) is a
  **domain-mismatch diagnostic**, consistent with the V2.1/V2.2 synthetic findings —
  it grounds the pipeline in real data; it does not measure attainable aerial-person
  perception. Do not quote these numbers as model quality.

## How to reproduce (local-only)

```bash
# dataset (≈4 GB, anonymous kagglehub mirror of the official layout):
~/.venvs/dstools/bin/python -c "import kagglehub; kagglehub.dataset_download('awsaf49/uavid-semantic-segmentation-dataset')"

# pilot: inference -> sources -> bundle (built AND validated by the standard tools)
.venv/bin/python -m experiments.real_segmentation_pilot.uavid_pilot --output results/uavid_pilot

# the ordinary benchmark CLI over the bundle
.venv/bin/python -m missionbench.run_benchmark \
  --data-root results/uavid_pilot/bundle \
  --episode results/uavid_pilot/bundle/episodes/episode.json \
  --contract results/uavid_pilot/bundle/contracts/contract.json \
  --policy rule_based --executor replay
```

Everything under `results/` is gitignored: UAVid derivatives are CC BY-NC-SA
(non-commercial research) and are never committed or redistributed.

## Module map

- `interfaces.py` / `masks.py` / `measure.py` / `convert.py` / `run_inference.py` /
  `build_pilot.py` — the dataset- and model-agnostic pilot tooling (unchanged; still
  verified by the stub-based CI tests, which never touch torch or the dataset).
- `uavid.py` — the UAVid `DatasetAdapter` (official layout; windowed; derivation and
  curation rules recorded in `provenance()`).
- `torchvision_models.py` — tiled `SegmentationModel` wrappers (lazy torch import,
  person index from weight metadata, physical-size prior on components).
- `uavid_pilot.py` — the end-to-end pilot entry point.

## Honest limitations, kept

No aerial-trained checkpoint (the headline gap); no real track identities; instance
derivation merges touching people; energy is an assumption; the curated subset is 12
keyframes chosen for moderate annotated density. A publication-grade result needs an
aerial-person model (or fine-tuning) and a dataset with true instance/track labels —
`PILOT_ANALYSIS_TEMPLATE.md` remains the form to fill when that exists.
