# Pilot analysis — TEMPLATE (all answers PENDING)

**This is a template, not a result.** Every answer is `PENDING — requires a real pilot run`,
because no real model has been executed on a real dataset in this environment (see
`STATUS.md`). Do **not** fill these in from the stub-model test pipeline; the stub measures the
tooling, not any model. Fill them only from a genuine run, and report negative results plainly.

## Subset

- Dataset / subset: PENDING
- Frames (target 50–200, ordered, identical for every config): PENDING
- GT instances / unique tracks / ignored / filtered: PENDING
- Track ids available in the dataset? PENDING (if not, say so — do not infer identity)

## Configurations

- CONFIG_LOCAL_LIGHT (smaller model / lower resolution / INT8): PENDING
- CONFIG_LOCAL_STRONG (larger model / higher resolution / FP16–FP32): PENDING
- Confirm they are actual executable differences, not two labels for one invocation: PENDING

## Per-config results (same frames)

| Config | target_recall | detection_precision | false_positive_detections | mean latency | p95 latency | energy (provenance) |
|---|---|---|---|---|---|---|
| CONFIG_LOCAL_LIGHT | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING (measured/estimated) |
| CONFIG_LOCAL_STRONG | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING (measured/estimated) |

## Questions

1. Does the strong config improve target recall? **PENDING**
2. Does the light config reduce latency? **PENDING**
3. Is either config strictly dominant across quality *and* latency? **PENDING**
4. Do different frames show different relative difficulty? **PENDING**
5. Does the same config dominate every contract? **PENDING**
6. Does the rule-based policy ever switch configs? **PENDING**
7. Does switching improve mission success over static policies? **PENDING**
8. Are results sensitive to contract thresholds? **PENDING**
9. Are energy conclusions measured, externally supplied, or estimated? **PENDING**
   (In this repository, the tooling defaults to `estimated`; treat energy as non-binding until
   real telemetry exists.)
10. What must change before a publication-quality experiment? **PENDING**

## Dominance statement

If, once run, one configuration dominates all others on both quality and latency, state
explicitly:

> "The current pilot does not yet establish the need for configuration-selection policies."

That negative result is scientifically important and must not be hidden.
