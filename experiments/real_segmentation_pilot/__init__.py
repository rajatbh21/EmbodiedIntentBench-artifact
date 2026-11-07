"""First real-segmentation empirical pilot: reusable, verifiable tooling.

**Status: the real pilot is blocked, and no real empirical bundle has been produced.** This
environment has no aerial instance-segmentation dataset, no model checkpoints, no PyTorch /
Ultralytics, and no GPU (see ``STATUS.md``). Per the pilot's own scientific rule, nothing here
fabricates a result: there are no hand-authored prediction masks, no invented latency, and no
made-up energy in any committed artifact.

What *is* delivered is the tooling the real pilot needs, built so every part except the actual
model call is exercised by tests using a clearly-labelled stub model over a tiny in-memory
dataset (never a real model, never a committed bundle):

- ``interfaces``  -- the ``DatasetAdapter`` and ``SegmentationModel`` boundaries, plus an
  in-memory adapter and a stub model for tests, and the real-backend stub that fails until
  weights and the optional dependencies are supplied.
- ``masks``       -- deterministic nearest-neighbour resize of a binary mask to GT coordinates.
- ``measure``     -- a latency protocol (warm-up, monotonic clock, per-frame samples, summary
  statistics) and honest energy provenance (``measured`` / ``externally_supplied`` /
  ``estimated``), never a silent zero.
- ``convert``     -- writing model output into the empirical bundle *source* formats.
- ``run_inference`` -- run one configuration over a dataset, timing each frame.
- ``build_pilot`` -- assemble a source manifest and hand it to the existing bundle builder.

When a real dataset, real checkpoints, and a real environment are supplied, these run the
actual pilot with no change to the benchmark core.
"""
