"""
Structural Adjacency Graph — a fixed map of which car parts are
structurally connected to which.

This graph is NOT learned — it is hand-authored (or generated with the help
of an LLM using general automotive engineering knowledge) and serves as the
basis for estimating hidden damage risk.

Each edge means "these two parts are structurally connected" and carries a
base_risk weight representing the rough likelihood that an impact on the
source part transfers damage to the target part. This is a heuristic, not a
measured value — it should be validated/refined against real repair data
whenever such data becomes available.
"""

STRUCTURAL_GRAPH: dict[str, list[dict]] = {
    "front_right_door": [
        {"to": "b_pillar", "base_risk": 0.30, "reason": "Door hinges and latch are bolted to the B-pillar"},
        {"to": "door_wiring_harness", "base_risk": 0.35, "reason": "Electrical wiring runs through the door cavity"},
        {"to": "side_airbag_sensor", "base_risk": 0.10, "reason": "Some vehicles house the side airbag sensor in the B-pillar"},
    ],
    "front_left_door": [
        {"to": "b_pillar", "base_risk": 0.30, "reason": "Door hinges and latch are bolted to the B-pillar"},
        {"to": "door_wiring_harness", "base_risk": 0.35, "reason": "Electrical wiring runs through the door cavity"},
        {"to": "side_airbag_sensor", "base_risk": 0.10, "reason": "Some vehicles house the side airbag sensor in the B-pillar"},
    ],
    "front_bumper": [
        {"to": "radiator", "base_risk": 0.25, "reason": "The radiator sits directly behind the front bumper"},
        {"to": "front_sensors", "base_risk": 0.40, "reason": "Parking/radar sensors are usually mounted inside the bumper"},
        {"to": "headlight_wiring", "base_risk": 0.15, "reason": "Headlight wiring routes past the bumper edge"},
    ],
    "hood": [
        {"to": "engine_bay_components", "base_risk": 0.20, "reason": "The hood sits directly above the engine bay"},
        {"to": "windshield", "base_risk": 0.10, "reason": "A severe hood impact can transfer force to the windshield"},
    ],
    "front_fender": [
        {"to": "wheel_arch_liner", "base_risk": 0.20, "reason": "The fender is attached to the wheel arch liner"},
        {"to": "suspension_strut", "base_risk": 0.08, "reason": "Only likely under very severe impact"},
    ],
    "rear_bumper": [
        {"to": "trunk_latch", "base_risk": 0.15, "reason": "The trunk latch sits close to the rear bumper"},
        {"to": "rear_sensors", "base_risk": 0.35, "reason": "Rear parking sensors are usually mounted inside the bumper"},
    ],
}


def get_adjacent_parts(part_name: str) -> list[dict]:
    """Returns the structurally adjacent parts and their base risk weights."""
    return STRUCTURAL_GRAPH.get(part_name, [])
