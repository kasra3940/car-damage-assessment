"""
Fusion — reconciles part_segmentation and damage_segmentation outputs.

This module has no learning component; it's pure logic. It computes the IoU
overlap between each damage mask and each part mask to determine which
damage belongs to which part, and what percentage of that part's area is
damaged.
"""

from dataclasses import dataclass, field
import numpy as np

from src.part_segmentation.predict import PartDetection
from src.damage_segmentation.predict import DamageDetection
from src.fusion.class_mapping import normalize_part_name


@dataclass
class PartDamageResult:
    part_name: str
    total_damage_area_pct: float
    damages: list[dict] = field(default_factory=list)


def mask_overlap_ratio(damage_mask: np.ndarray, part_mask: np.ndarray) -> float:
    """
    What fraction of the damage mask falls inside the part mask.
    Returns a value between 0 and 1.
    """
    intersection = np.logical_and(damage_mask, part_mask).sum()
    damage_area = damage_mask.sum()
    if damage_area == 0:
        return 0.0
    return float(intersection / damage_area)


def fuse(
    parts: list[PartDetection],
    damages: list[DamageDetection],
    overlap_threshold: float = 0.5,
) -> list[PartDamageResult]:
    """
    For each detected part, finds the damages that overlap it above
    `overlap_threshold` and computes the total damaged area percentage.
    """
    results: list[PartDamageResult] = []

    for part in parts:
        part_name = normalize_part_name(part.part_name)
        part_area = part.mask.sum()
        if part_area == 0:
            continue

        matched_damages = []
        total_damage_pixels = 0

        for dmg in damages:
            overlap = mask_overlap_ratio(dmg.mask, part.mask)
            if overlap >= overlap_threshold:
                matched_damages.append({
                    "type": dmg.damage_type,
                    "severity": dmg.severity,
                    "confidence": dmg.confidence,
                    "area_pct": round(100 * dmg.mask.sum() / part_area, 1),
                })
                total_damage_pixels += dmg.mask.sum()

        if matched_damages:
            results.append(PartDamageResult(
                part_name=part_name,
                total_damage_area_pct=round(100 * total_damage_pixels / part_area, 1),
                damages=matched_damages,
            ))

    return results
