"""
Damage Segmentation — pixel-level detection of visible damage.

Trained completely separately from part_segmentation, since it uses a
different dataset and label set (damage type: dent / scratch / crack /
glass_shatter / lamp_broken / tire_flat — see CarDD).

Note: this model doesn't know which car part each damage belongs to — it
only detects the visual pattern of damage. Linking damage to part happens
in the fusion stage.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DamageDetection:
    damage_type: str        # dent | scratch | crack | glass_shatter | lamp_broken | tire_flat
    mask: np.ndarray
    confidence: float
    severity: str = "unknown"   # minor | medium | severe — filled in during post-processing


class DamageSegmentationModel:
    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.model = None

    def load(self) -> "DamageSegmentationModel":
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)
        return self

    def predict(self, image_path: str) -> list[DamageDetection]:
        if self.model is None:
            raise RuntimeError("Model not loaded — call .load() first")

        results = self.model.predict(image_path, verbose=False)
        detections: list[DamageDetection] = []

        # TODO: map ultralytics Results objects to DamageDetection instances.
        # `severity` can be derived from the mask's relative area, or from a
        # separate minor/medium/severe classifier head.

        return detections


def estimate_severity(mask_area_pct: float) -> str:
    """
    Simple heuristic mapping relative damage area to a severity bucket.
    Meant as a placeholder until a dedicated severity classifier is trained.
    """
    if mask_area_pct < 5:
        return "minor"
    if mask_area_pct < 20:
        return "medium"
    return "severe"
