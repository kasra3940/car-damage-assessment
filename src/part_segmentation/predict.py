"""
Part Segmentation — pixel-level detection of car parts.

Loads a YOLOv8-seg model fine-tuned on the Carparts-seg dataset and returns,
for a given input image, a mask for each detected part (door, bumper, fender,
hood, etc).

Note: this model has no notion of "damage" — it only detects parts. Damage
detection is handled separately by damage_segmentation, and the two are
reconciled in fusion.fuse().
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PartDetection:
    part_name: str
    mask: np.ndarray       # binary mask, same size as input image
    confidence: float


class PartSegmentationModel:
    def __init__(self, weights_path: str):
        """
        weights_path: path to trained model weights (.pt)
        """
        self.weights_path = weights_path
        self.model = None

    def load(self) -> "PartSegmentationModel":
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)
        return self

    def predict(self, image_path: str) -> list[PartDetection]:
        """
        Runs the model on a single image and returns the list of detected
        parts.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded — call .load() first")

        results = self.model.predict(image_path, verbose=False)
        detections: list[PartDetection] = []

        # TODO: map ultralytics Results objects to PartDetection instances
        # for r in results:
        #     for mask, cls, conf in zip(r.masks, r.boxes.cls, r.boxes.conf):
        #         detections.append(PartDetection(
        #             part_name=self.model.names[int(cls)],
        #             mask=mask.data.cpu().numpy(),
        #             confidence=float(conf),
        #         ))

        return detections
