"""
Class name mapping between the part segmentation dataset (Carparts-seg) and
any other naming convention used elsewhere in the project (e.g. the
structural adjacency graph).

Because Carparts-seg, CarDD, and hand-written structural graphs were built
independently, their class names don't line up 1:1 (e.g. "front_door" vs
"Front-door" vs "left_front_door"). This module is the single place where
that gets normalized, so fusion and downstream stages can rely on one
consistent vocabulary.
"""

# Carparts-seg raw class name -> normalized internal name
CARPARTS_SEG_TO_INTERNAL: dict[str, str] = {
    "back_bumper": "rear_bumper",
    "back_door": "rear_door",
    "back_glass": "rear_glass",
    "back_left_door": "rear_left_door",
    "back_left_light": "rear_left_light",
    "back_light": "rear_light",
    "back_right_door": "rear_right_door",
    "back_right_light": "rear_right_light",
    "front_bumper": "front_bumper",
    "front_door": "front_door",
    "front_glass": "front_glass",
    "front_left_door": "front_left_door",
    "front_left_light": "front_left_light",
    "front_light": "front_light",
    "front_right_door": "front_right_door",
    "front_right_light": "front_right_light",
    "hood": "hood",
    "left_mirror": "left_mirror",
    "right_mirror": "right_mirror",
}


def normalize_part_name(raw_name: str) -> str:
    """
    Maps a raw class name from any source dataset to the project's internal
    part vocabulary. Falls back to a lowercased, hyphen-normalized version
    of the input if no explicit mapping exists.
    """
    if raw_name in CARPARTS_SEG_TO_INTERNAL:
        return CARPARTS_SEG_TO_INTERNAL[raw_name]
    return raw_name.strip().lower().replace("-", "_").replace(" ", "_")
