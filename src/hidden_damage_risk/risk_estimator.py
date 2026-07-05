"""
Hidden Damage Risk Estimator.

⚠️ This module is NOT a trained model. No dataset of confirmed hidden damage
exists at meaningful scale, so there's nothing to learn from. This is a
rule-based + LLM-reasoning system that combines:
  1. the structural adjacency graph (structural_graph.py)
  2. the severity and location of the *visible* damage (fusion output)
into a probabilistic estimate of risk to non-visible parts.

Every result carries a `confidence` and `basis` field so downstream
consumers (and the end user) can see this is an engineered estimate, not a
measurement.
"""

from dataclasses import dataclass

from src.hidden_damage_risk.structural_graph import get_adjacent_parts

SEVERITY_MULTIPLIER = {
    "minor": 0.4,
    "medium": 0.75,
    "severe": 1.15,
    "unknown": 0.6,
}


@dataclass
class HiddenRiskResult:
    part: str
    risk_pct: float
    confidence: str  # low | medium | high
    basis: str


def estimate_hidden_risk(
    damaged_part: str,
    severity: str,
    visible_damage_pct: float,
) -> list[HiddenRiskResult]:
    """
    Estimates hidden damage risk to adjacent parts based on the structural
    graph and the severity of visible damage.

    This function is a rule-based baseline only. For better accuracy, its
    output plus the raw photo should be passed to Gemini, which can factor
    in visual cues (dent depth, crack shape) that the rules alone can't see
    (see src/report_generation/gemini_report.py).
    """
    adjacent = get_adjacent_parts(damaged_part)
    multiplier = SEVERITY_MULTIPLIER.get(severity, 0.6)
    area_factor = min(visible_damage_pct / 30.0, 1.5)  # bigger visible damage = higher risk

    results: list[HiddenRiskResult] = []
    for edge in adjacent:
        raw_risk = edge["base_risk"] * multiplier * area_factor
        risk_pct = round(min(raw_risk, 0.95) * 100, 1)

        confidence = "low"
        if severity in ("medium", "severe") and visible_damage_pct > 15:
            confidence = "medium"
        if severity == "severe" and visible_damage_pct > 35:
            confidence = "high"

        results.append(HiddenRiskResult(
            part=edge["to"],
            risk_pct=risk_pct,
            confidence=confidence,
            basis=edge["reason"],
        ))

    return sorted(results, key=lambda r: r.risk_pct, reverse=True)
