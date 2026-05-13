"""YOLOv8 object detection wrapper for subject completeness scoring."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Fix ultralytics settings permission issue: set YOLO_SETTINGS_DIR before import
_yolo_settings_dir = str(Path.home() / ".config" / "ultralytics")
os.makedirs(_yolo_settings_dir, exist_ok=True)
os.environ.setdefault("YOLO_SETTINGS_DIR", _yolo_settings_dir)

import cv2
import numpy as np

from .utils import BBox, DetectedObject, bbox_area, bbox_center, bbox_intersection

logger = logging.getLogger(__name__)


class SubjectDetector:
    """Detect objects using YOLOv8 and compute subject completeness scores."""

    def __init__(self, config: dict):
        ycfg = config.get("models", {}).get("yolo", {})
        self.model_name: str = ycfg.get("model_name", "yolov8n.pt")
        self.confidence_threshold: float = ycfg.get("confidence_threshold", 0.5)
        self.device: str = ycfg.get("device", "cpu")
        self.important_classes: List[int] = ycfg.get(
            "important_classes",
            [0, 1, 2, 3, 5, 7, 9, 16, 17],  # person, bicycle, car, motorcycle, bus, truck, cat, dog
        )

        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """Lazily load YOLOv8 model."""
        if self._model_loaded:
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_name)
            logger.info(f"YOLOv8 model loaded: {self.model_name}")
        except ImportError:
            logger.warning("ultralytics not available. Subject detection disabled.")
        except Exception as e:
            logger.warning(f"Failed to load YOLOv8: {e}. Subject detection disabled.")
        self._model_loaded = True

    def detect(self, image: np.ndarray) -> List[DetectedObject]:
        """Run object detection on the full image.

        Args:
            image: BGR image (H, W, 3).

        Returns:
            List of DetectedObject instances.
        """
        self._load_model()
        if self._model is None:
            return []

        results = self._model(image, conf=self.confidence_threshold, verbose=False)
        objects = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                cls_name = self._model.names.get(cls_id, str(cls_id))
                objects.append(
                    DetectedObject(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                    )
                )
        return objects

    def score_candidates(
        self,
        bboxes: List[BBox],
        detected_objects: List[DetectedObject],
        image_shape: Tuple[int, int],
    ) -> List[Optional[float]]:
        """Compute subject completeness score for each candidate bbox.

        Score = weighted mean of inclusion ratios for detected objects,
        with boundary penetration penalty.

        Args:
            bboxes: List of candidate bboxes.
            detected_objects: List of DetectedObject from detect().
            image_shape: (H, W) of the original image.

        Returns:
            List of scores (0-1), or None for candidates where no objects matter.
            None means the module should be excluded from fusion for that candidate.
        """
        if len(detected_objects) == 0:
            return [None] * len(bboxes)

        # Filter to important objects only
        important_objects = [
            obj for obj in detected_objects if obj.class_id in self.important_classes
        ]
        if len(important_objects) == 0:
            # Also consider all high-confidence objects as potentially important
            important_objects = [obj for obj in detected_objects if obj.confidence >= 0.7]

        if len(important_objects) == 0:
            return [None] * len(bboxes)

        # Compute weights for each object (confidence * area)
        img_area = image_shape[0] * image_shape[1]
        obj_weights = []
        for obj in important_objects:
            area = bbox_area(obj.bbox)
            # Weight by confidence and relative area
            w = obj.confidence * (area / max(1, img_area)) ** 0.3
            obj_weights.append(w)
        total_weight = sum(obj_weights) + 1e-9

        scores = []
        for bbox in bboxes:
            weighted_inclusion = 0.0
            boundary_penalty = 0.0

            for obj, ow in zip(important_objects, obj_weights):
                # Inclusion ratio
                inter_area = bbox_intersection(bbox, obj.bbox)
                obj_area = max(1, bbox_area(obj.bbox))
                inclusion = inter_area / obj_area

                weighted_inclusion += ow * inclusion

                # Boundary penetration penalty: check if bbox boundary cuts through object
                bx1, by1, bx2, by2 = bbox
                ox1, oy1, ox2, oy2 = obj.bbox
                # Is the object partially inside and partially outside?
                if inter_area > 0 and inter_area < obj_area:
                    # How much of the object is outside the bbox
                    outside_ratio = 1.0 - inclusion
                    # Extra penalty for person class (class_id=0)
                    if obj.class_id == 0:
                        boundary_penalty += ow * outside_ratio * 2.0
                    else:
                        boundary_penalty += ow * outside_ratio * 0.5

            raw_score = weighted_inclusion / total_weight
            penalty = min(1.0, boundary_penalty / total_weight)
            score = max(0.0, raw_score - penalty)
            scores.append(score)

        return scores
