"""Transparent authored task selectors; no learned downstream superiority claim."""

from experiments.mission_pipeline_routing.runtime import PublicContext, Selection

ASSOCIATIONS = {
    "image_iou": "ASSOC_IMAGE_IOU",
    "flat_world": "ASSOC_FLAT_WORLD",
    "terrain_world": "ASSOC_TERRAIN_WORLD",
}
VERIFIERS = {"repeat2": "VERIFY_REPEAT2", "repeat3": "VERIFY_REPEAT3"}
FIXED = {
    f"fixed_{a}__{v}": Selection(aid, vid)
    for a, aid in ASSOCIATIONS.items()
    for v, vid in VERIFIERS.items()
}
POLICIES = (*FIXED, "intent_context", "switch_stress")


def select(name, context: PublicContext, history):
    if name in FIXED:
        return FIXED[name]
    if name == "intent_context":
        # A transparent untrained rule, fixed before downstream outcomes.
        association = (
            "ASSOC_IMAGE_IOU"
            if context.battery_frac < 0.3 or context.deadline_ns - context.now_ns < 8_000_000_000
            else "ASSOC_TERRAIN_WORLD"
        )
        verification = "VERIFY_REPEAT3" if context.required_hits >= 3 else "VERIFY_REPEAT2"
        return Selection(association, verification)
    if name == "switch_stress":
        n = history["completed_jobs"]
        return Selection(
            ("ASSOC_IMAGE_IOU", "ASSOC_FLAT_WORLD", "ASSOC_TERRAIN_WORLD")[n % 3],
            ("VERIFY_REPEAT2", "VERIFY_REPEAT3")[n % 2],
        )
    raise ValueError("Unknown predeclared pipeline policy")
