"""
Cost Estimation — estimates repair and replacement cost for a damaged part.

Two-layer design:
  1. A local, offline price table (pricing_config.yaml) — always available,
     zero cost, but only as accurate as its manually-researched sample values.
  2. An optional online enrichment layer using Gemini + Google Search
     grounding, which can pull more current, region-specific price signals
     when an API key and internet access are available.

If both are available, the pipeline should surface both numbers and cite
the search sources for transparency. If only the offline table is
available, that's used as a fallback and clearly labeled as an estimate.
"""

from dataclasses import dataclass
import yaml


@dataclass
class CostEstimate:
    part: str
    action: str          # "repair" | "replace"
    cost: float
    currency: str
    source: str          # "local_table" | "gemini_search"


def load_pricing_config(path: str = "configs/pricing_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def estimate_cost_local(
    part: str,
    severity: str,
    action: str,
    region: str,
    config: dict,
) -> CostEstimate | None:
    """
    Offline cost estimate using the local price table. Returns None if the
    part or severity/action combination isn't in the table.
    """
    base_costs = config.get("part_base_replacement_cost", {}).get(part)
    if base_costs is None or region not in base_costs:
        return None

    base_cost = base_costs[region]
    multiplier = (
        config.get("severity_action_multiplier", {})
        .get(severity, {})
        .get(action)
    )
    if multiplier is None:
        return None

    currency = config.get("currency_by_region", {}).get(region, "USD")

    return CostEstimate(
        part=part,
        action=action,
        cost=round(base_cost * multiplier, 2),
        currency=currency,
        source="local_table",
    )


def estimate_cost_with_search(
    part: str,
    severity: str,
    action: str,
    region: str,
    vehicle_model: str,
    gemini_client,
) -> CostEstimate | None:
    """
    Online cost estimate that asks Gemini (with Google Search grounding
    enabled) to search current sources and synthesize a cost range.

    This is an *estimate*, not a certified quote — see the disclaimer in
    the final report. Requires a configured Gemini client with the
    google_search tool enabled (see report_generation/gemini_report.py for
    the client setup pattern).
    """
    # TODO: implement the actual Gemini call with google_search grounding.
    # Example prompt shape:
    #   f"Estimate the average {action} cost for a {severity}-severity "
    #   f"{part.replace('_', ' ')} on a {vehicle_model} in {region}. "
    #   f"Search current sources and return a numeric range with citations."
    raise NotImplementedError(
        "Wire this up to a Gemini client with google_search grounding enabled."
    )
