from src.hidden_damage_risk.risk_estimator import estimate_hidden_risk
from src.hidden_damage_risk.structural_graph import get_adjacent_parts


def test_get_adjacent_parts_known_part():
    adjacent = get_adjacent_parts("front_right_door")
    assert len(adjacent) > 0
    assert all("to" in edge and "base_risk" in edge for edge in adjacent)


def test_get_adjacent_parts_unknown_part_returns_empty():
    assert get_adjacent_parts("nonexistent_part") == []


def test_estimate_hidden_risk_returns_sorted_results():
    results = estimate_hidden_risk(
        damaged_part="front_right_door",
        severity="severe",
        visible_damage_pct=40,
    )
    assert len(results) > 0
    risks = [r.risk_pct for r in results]
    assert risks == sorted(risks, reverse=True)


def test_estimate_hidden_risk_severity_increases_risk():
    minor_results = estimate_hidden_risk("front_right_door", "minor", 10)
    severe_results = estimate_hidden_risk("front_right_door", "severe", 10)

    minor_total = sum(r.risk_pct for r in minor_results)
    severe_total = sum(r.risk_pct for r in severe_results)

    assert severe_total > minor_total


def test_estimate_hidden_risk_all_have_confidence_and_basis():
    results = estimate_hidden_risk("front_bumper", "medium", 20)
    for r in results:
        assert r.confidence in ("low", "medium", "high")
        assert isinstance(r.basis, str) and len(r.basis) > 0
