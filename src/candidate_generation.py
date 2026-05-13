"""GAIC-style grid anchor candidate generation + saliency-guided supplement."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import BBox, bbox_area, bbox_aspect_ratio, clamp_bbox, nms


class CandidateGenerator:
    """Generate candidate cropping boxes using GAIC-style grid anchors
    with optional saliency-guided supplementary candidates."""

    def __init__(self, config: dict):
        cg = config.get("candidate_generation", {})
        self.grid_size: int = cg.get("grid_size", 6)
        self.area_ratios: List[float] = cg.get("area_ratios", [0.35, 0.45, 0.60, 0.75, 0.90])
        self.aspect_ratios: List[float] = cg.get("aspect_ratios", [1.0, 1.333, 0.75, 1.778, 0.5625])
        self.use_original_ratio: bool = cg.get("use_original_ratio", True)
        self.top_k: int = cg.get("top_k", 150)
        self.min_area_ratio: float = cg.get("min_area_ratio", 0.10)
        self.max_area_ratio: float = cg.get("max_area_ratio", 0.95)
        self.min_aspect_ratio: float = cg.get("min_aspect_ratio", 0.5)
        self.max_aspect_ratio: float = cg.get("max_aspect_ratio", 2.0)
        self.nms_iou_threshold: float = cg.get("nms_iou_threshold", 0.7)
        self.saliency_supplement: bool = cg.get("saliency_supplement", True)
        self.saliency_peak_threshold: float = cg.get("saliency_peak_threshold", 0.5)
        self.saliency_smooth_sigma: float = cg.get("saliency_smooth_sigma", 5.0)

    def generate(
        self,
        image: np.ndarray,
        saliency_map: Optional[np.ndarray] = None,
    ) -> List[BBox]:
        """Generate candidate bboxes for the given image.

        Args:
            image: BGR image (H, W, 3).
            saliency_map: Optional saliency map (H, W) in [0, 1].
                If provided and self.saliency_supplement is True,
                additional candidates are generated at saliency peaks.

        Returns:
            List of BBox tuples, filtered and NMS-deduped, up to top_k.
        """
        h, w = image.shape[:2]
        img_area = h * w

        # --- GAIC-style grid anchor generation ---
        candidates = self._grid_anchors(h, w, img_area)

        # --- Saliency-guided supplement ---
        if self.saliency_supplement and saliency_map is not None:
            sal_candidates = self._saliency_guided_candidates(saliency_map, h, w, img_area)
            candidates.extend(sal_candidates)

        # --- Filtering ---
        filtered = self._filter_candidates(candidates, h, w, img_area)

        # --- NMS dedup ---
        if len(filtered) > 0:
            scores = [float(bbox_area(b)) / max(1, img_area) for b in filtered]
            keep = nms(filtered, scores, self.nms_iou_threshold)
            filtered = [filtered[i] for i in keep]

        # --- Keep top-K ---
        if len(filtered) > self.top_k:
            # Subsample evenly
            idx = np.linspace(0, len(filtered) - 1, self.top_k).astype(int)
            filtered = [filtered[i] for i in idx]

        return filtered

    def _grid_anchors(self, h: int, w: int, img_area: int) -> List[BBox]:
        """Generate GAIC-style grid anchor candidates."""
        candidates: List[BBox] = []

        # Grid centers (evenly spaced, including near-edges)
        gx = np.linspace(0, w, self.grid_size + 2, dtype=int)[1:-1]
        gy = np.linspace(0, h, self.grid_size + 2, dtype=int)[1:-1]

        aspect_ratios = list(self.aspect_ratios)
        if self.use_original_ratio:
            orig_ratio = w / max(1, h)
            if orig_ratio not in aspect_ratios:
                aspect_ratios.append(orig_ratio)

        for cx in gx:
            for cy in gy:
                for ar in aspect_ratios:
                    for area_r in self.area_ratios:
                        area = int(img_area * area_r)
                        crop_h = int(math.sqrt(area / max(1e-6, ar)))
                        crop_w = int(crop_h * ar)
                        if crop_h < 8 or crop_w < 8:
                            continue
                        if crop_h > h or crop_w > w:
                            continue
                        x1 = cx - crop_w // 2
                        y1 = cy - crop_h // 2
                        x2 = x1 + crop_w
                        y2 = y1 + crop_h
                        bbox = clamp_bbox((x1, y1, x2, y2), h, w)
                        # Verify clamped box still has reasonable size
                        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        if bw >= 8 and bh >= 8:
                            candidates.append(bbox)
        return candidates

    def _saliency_guided_candidates(
        self,
        saliency_map: np.ndarray,
        h: int,
        w: int,
        img_area: int,
    ) -> List[BBox]:
        """Generate additional candidates centered at saliency peaks."""
        candidates: List[BBox] = []

        # Smooth the saliency map
        smoothed = cv2.GaussianBlur(
            saliency_map.astype(np.float32),
            (0, 0),
            self.saliency_smooth_sigma,
        )

        # Find local peaks by finding contours of high-response regions
        binary = (smoothed > self.saliency_peak_threshold * smoothed.max()).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute centroids of significant contours
        peaks = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                peaks.append((cx, cy))

        # Fallback: use global saliency centroid if no peaks found
        if len(peaks) == 0:
            total = smoothed.sum()
            if total > 1e-6:
                ys, xs = np.mgrid[0:h, 0:w]
                cx = int((xs * smoothed).sum() / total)
                cy = int((ys * smoothed).sum() / total)
                peaks.append((cx, cy))

        # Generate candidates around each peak
        aspect_ratios = list(self.aspect_ratios)
        if self.use_original_ratio:
            orig_ratio = w / max(1, h)
            if orig_ratio not in aspect_ratios:
                aspect_ratios.append(orig_ratio)

        for cx, cy in peaks:
            for ar in aspect_ratios:
                for area_r in self.area_ratios:
                    area = int(img_area * area_r)
                    crop_h = int(math.sqrt(area / max(1e-6, ar)))
                    crop_w = int(crop_h * ar)
                    if crop_h < 8 or crop_w < 8:
                        continue
                    if crop_h > h or crop_w > w:
                        continue
                    x1 = cx - crop_w // 2
                    y1 = cy - crop_h // 2
                    x2 = x1 + crop_w
                    y2 = y1 + crop_h
                    bbox = clamp_bbox((x1, y1, x2, y2), h, w)
                    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if bw >= 8 and bh >= 8:
                        candidates.append(bbox)
        return candidates

    def _filter_candidates(
        self,
        candidates: List[BBox],
        h: int,
        w: int,
        img_area: int,
    ) -> List[BBox]:
        """Filter candidates by area, aspect ratio, and validity."""
        filtered = []
        for bbox in candidates:
            area = bbox_area(bbox)
            area_ratio = area / max(1, img_area)
            ar = bbox_aspect_ratio(bbox)

            # Area filter
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue
            # Aspect ratio filter
            if ar < self.min_aspect_ratio or ar > self.max_aspect_ratio:
                continue
            # Must be valid
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            filtered.append(bbox)
        return filtered
