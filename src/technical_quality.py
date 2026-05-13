"""Technical quality scoring: sharpness, brightness, contrast, saturation."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .utils import BBox


class TechnicalQualityScorer:
    """Score candidates on basic visual quality metrics."""

    def __init__(self, config: dict):
        tcfg = config.get("technical_quality", {})
        w = tcfg.get("weights", {})
        self.weight_sharpness: float = w.get("sharpness", 0.30)
        self.weight_brightness: float = w.get("brightness", 0.25)
        self.weight_contrast: float = w.get("contrast", 0.25)
        self.weight_saturation: float = w.get("saturation", 0.20)

        bright_range = tcfg.get("brightness_ideal_range", [50, 200])
        self.brightness_min: float = bright_range[0]
        self.brightness_max: float = bright_range[1]

        sat_range = tcfg.get("saturation_ideal_range", [40, 180])
        self.saturation_min: float = sat_range[0]
        self.saturation_max: float = sat_range[1]

    def score_candidates(
        self,
        image: np.ndarray,
        bboxes: List[BBox],
    ) -> List[Tuple[float, Dict[str, float]]]:
        """Score each candidate on technical quality.

        Args:
            image: Original BGR image.
            bboxes: Candidate bboxes.

        Returns:
            List of (total_technical_score, sub_score_dict) per candidate.
        """
        scores = []
        for bbox in bboxes:
            sub = self._score_single(image, bbox)
            total = (
                self.weight_sharpness * sub["sharpness"]
                + self.weight_brightness * sub["brightness"]
                + self.weight_contrast * sub["contrast"]
                + self.weight_saturation * sub["saturation"]
            )
            scores.append((total, sub))
        return scores

    def _score_single(self, image: np.ndarray, bbox: BBox) -> Dict[str, float]:
        """Compute technical quality sub-scores for one candidate."""
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return {
                "sharpness": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "saturation": 0.0,
            }

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 1. Sharpness: Laplacian variance
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness_var = float(lap.var())
        # Normalize: typical range 0-2000+, map to 0-1
        sharpness = min(1.0, sharpness_var / 500.0)

        # 2. Brightness: mean luminance
        mean_brightness = float(gray.mean())
        brightness = self._interval_score(mean_brightness, self.brightness_min, self.brightness_max)

        # 3. Contrast: standard deviation of luminance
        contrast_std = float(gray.std())
        # Normalize: typical range 0-128, map to 0-1
        contrast = min(1.0, contrast_std / 64.0)

        # 4. Saturation: mean HSV saturation
        mean_saturation = float(hsv[:, :, 1].mean())
        saturation = self._interval_score(mean_saturation, self.saturation_min, self.saturation_max)

        return {
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
        }

    @staticmethod
    def _interval_score(value: float, low: float, high: float) -> float:
        """Score a value that is ideal in [low, high], penalizing extremes.

        Uses Gaussian-like penalty outside the range.
        """
        if low <= value <= high:
            return 1.0
        elif value < low:
            deviation = low - value
            return max(0.0, math.exp(-deviation ** 2 / (2 * 30 ** 2)))
        else:
            deviation = value - high
            return max(0.0, math.exp(-deviation ** 2 / (2 * 30 ** 2)))
