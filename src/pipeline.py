"""
Full end-to-end pipeline: photo in, report out.

Usage:
    python -m src.pipeline --image path/to/car_photo.jpg --region US
"""

import argparse
import yaml

from src.part_segmentation.predict import PartSegmentationModel
from src.damage_segmentation.predict import DamageSegmentationModel, estimate_severity
from src.fusion.fusion import fuse
from src.hidden_damage_risk.risk_estimator import estimate_hidden_risk
from src.cost_estimation.estimator import load_pricing_config, estimate_cost_local
from src.repair_advisor.advisor import recommend_repair_vs_replace
from src.report_generation.gemini_report import generate_report


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(image_path: str, region: str = "US") -> dict:
    config = load_config()
    pricing_config = load_pricing_config()

    part_model = PartSegmentationModel(config["models"]["part_segmentation_weights"]).load()
    damage_model = DamageSegmentationModel(config["models"]["damage_segmentation_weights"]).load()

    parts = part_model.predict(image_path)
    damages = damage_model.predict(image_path)

    fused = fuse(parts, damages, overlap_threshold=config["fusion"]["overlap_threshold"])

    assessment = []
    for result in fused:
        # Use the highest-severity damage on this part to drive downstream stages
        severities = [d["severity"] for d in result.damages] or ["unknown"]
        severity = max(severities, key=lambda s: ["minor", "medium", "severe", "unknown"].index(s))

        hidden_risks = estimate_hidden_risk(
            damaged_part=result.part_name,
            severity=severity,
            visible_damage_pct=result.total_damage_area_pct,
        )

        repair_cost = estimate_cost_local(result.part_name, severity, "repair", region, pricing_config)
        replace_cost = estimate_cost_local(result.part_name, severity, "replace", region, pricing_config)

        repair_options = []
        if repair_cost and replace_cost:
            repair_options = recommend_repair_vs_replace(
                part=result.part_name,
                severity=severity,
                repair_cost=repair_cost.cost,
                replace_cost=replace_cost.cost,
                currency=repair_cost.currency,
            )

        assessment.append({
            "part": result.part_name,
            "visible_damage_pct": result.total_damage_area_pct,
            "damages": result.damages,
            "hidden_damage_risk": [r.__dict__ for r in hidden_risks],
            "repair_vs_replace": [o.__dict__ for o in repair_options],
        })

    report_text = generate_report({"assessment": assessment})

    return {"assessment": assessment, "report": report_text}


def main():
    parser = argparse.ArgumentParser(description="Run the car damage assessment pipeline on an image")
    parser.add_argument("--image", required=True, help="Path to the input car photo")
    parser.add_argument("--region", default="US", help="Pricing region code (e.g. US, IR)")
    args = parser.parse_args()

    result = run_pipeline(args.image, args.region)
    print(result["report"])


if __name__ == "__main__":
    main()
