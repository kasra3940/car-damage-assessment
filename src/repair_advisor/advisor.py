"""
Repair vs. Replace Advisor.

For each damaged part, weighs two paths against each other:
  - Repair: cheaper upfront, but if the part is bodyworked/repainted
    (not original), resale appraisers can often detect it (paint depth
    gauges, panel gap inspection) and buyers discount the price accordingly.
  - Replace: more expensive upfront, but keeps the part "original" on paper,
    which typically preserves resale value better.

⚠️ Like hidden_damage_risk, the `resale_depreciation` figures here are
heuristic estimates, not live market data. No free, reliable dataset of
"resale depreciation by repair type" exists for any market. In production,
this should be validated against real listings data or refreshed via
Gemini + Google Search grounding, and always presented with a confidence
level and reasoning, never as a certified appraisal.
"""

from dataclasses import dataclass

from src.cost_estimation.estimator import CostEstimate

# Rough heuristic: how much of the *replace* cost is typically recovered as
# reduced future depreciation, by severity. This is a placeholder ratio for
# the local baseline — refine with real listings data when available.
DEPRECIATION_RISK_RATIO = {
    "minor":  {"repair": 0.03, "replace": 0.01},
    "medium": {"repair": 0.08, "replace": 0.02},
    "severe": {"repair": 0.15, "replace": 0.03},
}


@dataclass
class RepairOption:
    action: str                 # "repair" | "replace"
    recommendation_pct: float   # how strongly this option is favored (0-100, repair+replace=100)
    upfront_cost: float
    currency: str
    resale_depreciation_estimate: tuple[float, float]  # (low, high) in the same currency
    reasoning: str


def _depreciation_range(replace_cost: float, severity: str, action: str) -> tuple[float, float]:
    ratio = DEPRECIATION_RISK_RATIO.get(severity, DEPRECIATION_RISK_RATIO["medium"])[action]
    low = round(replace_cost * ratio * 0.6, 2)
    high = round(replace_cost * ratio * 1.4, 2)
    return (low, high)


def recommend(
    part: str,
    severity: str,
    repair_cost: CostEstimate,
    replace_cost: CostEstimate,
    is_structural_or_safety_part: bool = False,
) -> list[RepairOption]:
    """
    Compares repair vs. replace for a single part and returns both options
    with a recommendation split (e.g. 60% repair / 40% replace), ranked by
    total expected cost (upfront + expected depreciation midpoint).
    """
    repair_deprec = _depreciation_range(replace_cost.cost, severity, "repair")
    replace_deprec = _depreciation_range(replace_cost.cost, severity, "replace")

    repair_total = repair_cost.cost + sum(repair_deprec) / 2
    replace_total = replace_cost.cost + sum(replace_deprec) / 2

    # Safety/structural parts bias strongly toward replacement regardless of cost
    if is_structural_or_safety_part and severity in ("medium", "severe"):
        repair_pct = 15.0
    else:
        # Inverse-cost weighting: cheaper total option gets a higher share,
        # clamped to a sane range so neither option ever hits 0/100.
        total = repair_total + replace_total
        repair_pct = round(max(10.0, min(90.0, 100 * replace_total / total)), 1)

    replace_pct = round(100 - repair_pct, 1)

    reasoning_repair = (
        f"Damage is {severity}; bodywork/paint is expected to address it at "
        f"lower upfront cost, with a modest resale discount risk if detected "
        f"by an appraiser."
    )
    reasoning_replace = (
        f"Keeps the part factory-original on inspection reports, minimizing "
        f"future resale depreciation, at a higher upfront cost."
    )
    if is_structural_or_safety_part and severity in ("medium", "severe"):
        reasoning_replace += " Recommended due to the part's structural/safety role."

    return [
        RepairOption(
            action="repair",
            recommendation_pct=repair_pct,
            upfront_cost=repair_cost.cost,
            currency=repair_cost.currency,
            resale_depreciation_estimate=repair_deprec,
            reasoning=reasoning_repair,
        ),
        RepairOption(
            action="replace",
            recommendation_pct=replace_pct,
            upfront_cost=replace_cost.cost,
            currency=replace_cost.currency,
            resale_depreciation_estimate=replace_deprec,
            reasoning=reasoning_replace,
        ),
    ]
