"""English presentation labels for archived lakeside mission identifiers.

Translations are applied to copies. Frozen experiment inputs remain provenance.
This module performs no intent parsing and changes no numeric mission conditions.
"""

import copy

LABELS = {
    "sector_rescue": (
        "Sector search and verification",
        "Search west, prioritize the northern waterfront, search the eastern forest edge, "
        "then revisit the west for fresh evidence.",
        {
            "west_search": "Collect repeated evidence in the west",
            "north_priority": "Prioritize the northern waterfront",
            "east_search": "Search the eastern forest edge",
            "west_revisit": "Revisit west: require fresh evidence",
        },
    ),
    "perimeter_inspection": (
        "Perimeter safety inspection",
        "Inspect all three perimeter sectors. Require repeated observations, suppress "
        "false alarms, and recheck the northern sector.",
        {
            "west_inspect": "Repeatedly inspect the western access route",
            "north_inspect": "Inspect the northern waterfront",
            "east_inspect": "Inspect the eastern perimeter",
            "north_recheck": "Recheck north with fresh evidence",
        },
    ),
    "reserve_revisit": (
        "Battery-constrained revisits",
        "With limited battery, inspect west and north, revisit both for fresh evidence, "
        "and retain at least 20% battery.",
        {
            "west_first": "Collect initial western evidence",
            "north_first": "Collect initial northern evidence",
            "west_fresh": "Verify west with new repeated observations",
            "north_fresh": "Verify north and preserve battery reserve",
        },
    ),
}


def english_missions(missions):
    result = copy.deepcopy(missions)
    for mission in result:
        title, request, phases = LABELS[mission["mission_id"]]
        mission.update(title=title, user_request=request)
        for phase in mission["phases"]:
            phase["instruction"] = phases[phase["phase_id"]]
    return result
