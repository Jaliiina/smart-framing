"""Subjectness and distractor map estimation.

This module decouples visual saliency from aesthetic subjectness. Salient
foreground clutter can be penalized through a distractor map, while coherent
objects/structures are promoted through a subjectness map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .utils import BBox, DetectedObject, bbox_area


@dataclass
class SubjectnessMaps:
    subjectness: np.ndarray
    distractor: np.ndarray
    structure: np.ndarray


class SubjectnessScorer:
    """Build subjectness/distractor maps and per-candidate scores."""

    def __init__(self, config: dict):
        cfg = config.get("subjectness", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.foreground_y_start = float(cfg.get("foreground_y_start", 0.55))
        self.corner_width = float(cfg.get("corner_width", 0.28))
        self.saliency_weight = float(cfg.get("saliency_weight", 0.45))
        self.structure_weight = float(cfg.get("structure_weight", 0.30))
        self.object_weight = float(cfg.get("object_weight", 0.25))
        self.distractor_classes = set(
            cfg.get(
                "distractor_classes",
                [24, 26, 27, 28, 31, 32, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
            )
        )

    def build_maps(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        detected_objects: List[DetectedObject],
    ) -> SubjectnessMaps:
        h, w = image.shape[:2]
        sal = self._norm(saliency_map)
        if not self.enabled:
            return SubjectnessMaps(
                subjectness=sal,
                distractor=np.zeros_like(sal, dtype=np.float32),
                structure=np.zeros_like(sal, dtype=np.float32),
            )
        structure = self._structure_map(image)
        obj_map = np.zeros((h, w), dtype=np.float32)
        distractor = self._foreground_activity_map(image, sal, structure)

        image_area = max(1, h * w)
        for obj in detected_objects:
            x1, y1, x2, y2 = obj.bbox
            area_ratio = bbox_area(obj.bbox) / image_area
            target = distractor if obj.class_id in self.distractor_classes and area_ratio < 0.12 else obj_map
            value = max(0.05, obj.confidence)
            target[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = np.maximum(
                target[max(0, y1):min(h, y2), max(0, x1):min(w, x2)],
                value,
            )

        subjectness = (
            self.saliency_weight * sal
            + self.structure_weight * structure
            + self.object_weight * obj_map
        )
        subjectness = np.clip(subjectness - 0.45 * distractor, 0.0, 1.0)
        return SubjectnessMaps(
            subjectness=self._norm(subjectness),
            distractor=self._norm(distractor),
            structure=structure,
        )

    def score_candidates(
        self,
        bboxes: List[BBox],
        maps: SubjectnessMaps,
    ) -> List[Tuple[float, float]]:
        scores = []
        total_subject = float(maps.subjectness.sum()) + 1e-9
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            subj = maps.subjectness[y1:y2, x1:x2]
            dist = maps.distractor[y1:y2, x1:x2]
            coverage = float(subj.sum()) / total_subject
            density = float(subj.mean()) if subj.size else 0.0
            distractor_inside = float(dist.mean()) if dist.size else 0.0
            scores.append((float(np.clip(0.60 * coverage + 0.40 * density, 0.0, 1.0)), distractor_inside))
        return scores

    def _foreground_activity_map(
        self,
        image: np.ndarray,
        sal: np.ndarray,
        structure: np.ndarray,
    ) -> np.ndarray:
        h, w = sal.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = np.clip(hsv[:, :, 1] / 180.0, 0.0, 1.0)
        value = np.clip(hsv[:, :, 2] / 255.0, 0.0, 1.0)
        activity = 0.45 * sal + 0.30 * structure + 0.25 * sat

        yy, xx = np.mgrid[0:h, 0:w]
        y_prior = np.clip((yy / max(1, h) - self.foreground_y_start) / 0.35, 0.0, 1.0)
        left_corner = xx < self.corner_width * w
        right_corner = xx > (1.0 - self.corner_width) * w
        corner_prior = np.where(left_corner | right_corner, 1.0, 0.35)
        dark_foreground = np.clip(0.65 - value, 0.0, 1.0) * y_prior
        return np.clip(activity * (0.45 + 0.55 * y_prior) * corner_prior + 0.35 * dark_foreground, 0.0, 1.0)

    @staticmethod
    def _structure_map(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0
        lines = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        line_strength = np.clip(np.abs(lines) / 120.0, 0.0, 1.0)
        structure = 0.65 * cv2.GaussianBlur(edges, (0, 0), 1.2) + 0.35 * line_strength
        return SubjectnessScorer._norm(structure)

    @staticmethod
    def _norm(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        min_v = float(values.min())
        max_v = float(values.max())
        if max_v - min_v < 1e-9:
            return np.zeros_like(values, dtype=np.float32)
        return (values - min_v) / (max_v - min_v)
