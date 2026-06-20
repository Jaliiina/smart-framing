"""GAIC-style grid anchor candidate generation + saliency-guided supplement."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import BBox, DetectedObject, bbox_area, bbox_aspect_ratio, clamp_bbox, nms


class CandidateGenerator:
    """Generate candidate cropping boxes using GAIC-style grid anchors
    with optional saliency-guided supplementary candidates."""

    def __init__(self, config: dict):
        cg = config.get("candidate_generation", {})
        self.grid_size: int = cg.get("grid_size", 6)
        self.area_ratios: List[float] = cg.get("area_ratios", [0.35, 0.45, 0.60, 0.75, 0.90])
        self.aspect_ratios: List[float] = cg.get("aspect_ratios", [1.0, 1.333, 0.75, 1.778, 0.5625])
        self.use_original_ratio: bool = cg.get("use_original_ratio", True)
        self.preserve_original_aspect: bool = cg.get("preserve_original_aspect", False)
        self.top_k: int = cg.get("top_k", 150)
        self.min_area_ratio: float = cg.get("min_area_ratio", 0.10)
        self.max_area_ratio: float = cg.get("max_area_ratio", 0.95)
        self.min_aspect_ratio: float = cg.get("min_aspect_ratio", 0.5)
        #self.max_aspect_ratio: float = cg.get("max_aspect_ratio", 2.0)
        self.max_aspect_ratio: float = cg.get("max_aspect_ratio", 2.5)
        self.nms_iou_threshold: float = cg.get("nms_iou_threshold", 0.7)
        self.saliency_supplement: bool = cg.get("saliency_supplement", True)
        self.saliency_peak_threshold: float = cg.get("saliency_peak_threshold", 0.5)
        #self.saliency_smooth_sigma: float = cg.get("saliency_smooth_sigma", 5.0)
        self.saliency_smooth_sigma: float = cg.get("saliency_smooth_sigma", 6.0)

    def generate(
        self,
        image: np.ndarray,
        saliency_map: Optional[np.ndarray] = None,
        detected_objects: Optional[List[DetectedObject]] = None,
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

        # --- Subject-guided supplement ---
        if detected_objects:
            candidates.extend(
                self._object_guided_candidates(detected_objects, h, w, img_area)
            )

        # --- Filtering ---
        filtered = self._filter_candidates(candidates, h, w, img_area)

        # --- NMS dedup ---
        if len(filtered) > 0:
            scores = [
                self._prefilter_score(b, img_area, saliency_map)
                for b in filtered
            ]
            keep = nms(filtered, scores, self.nms_iou_threshold)
            filtered = [filtered[i] for i in keep]

        # --- Keep top-K ---
        if len(filtered) > self.top_k:
            filtered = filtered[:self.top_k]

        return filtered

    def _candidate_at_center(
        self,
        cx: float,
        cy: float,
        area_r: float,
        ar: float,
        h: int,
        w: int,
        img_area: int,
    ) -> Optional[BBox]:
        area = int(img_area * area_r)
        crop_h = int(math.sqrt(area / max(1e-6, ar)))
        crop_w = int(crop_h * ar)
        if crop_h < 8 or crop_w < 8:
            return None
        if crop_h > h or crop_w > w:
            return None
        x1 = int(round(cx - crop_w / 2))
        y1 = int(round(cy - crop_h / 2))
        bbox = clamp_bbox((x1, y1, x1 + crop_w, y1 + crop_h), h, w)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bw >= 8 and bh >= 8:
            return bbox
        return None

    def _active_aspect_ratios(self, h: int, w: int) -> List[float]:
        if self.preserve_original_aspect:
            return [w / max(1, h)]
        aspect_ratios = list(self.aspect_ratios)
        if self.use_original_ratio:
            orig_ratio = w / max(1, h)
            if orig_ratio not in aspect_ratios:
                aspect_ratios.append(orig_ratio)
        return aspect_ratios

    @staticmethod
    def _scale_prior(area_ratio: float) -> float:
        """Prefer useful medium crops during NMS without removing all large boxes."""
        if 0.25 <= area_ratio <= 0.60:
            return 1.0
        if area_ratio < 0.25:
            return max(0.0, 1.0 - (0.25 - area_ratio) / 0.20)
        return max(0.0, 1.0 - (area_ratio - 0.60) / 0.35)

    def _prefilter_score(
        self,
        bbox: BBox,
        img_area: int,
        saliency_map: Optional[np.ndarray],
    ) -> float:
        """Score candidates for NMS so medium, content-dense boxes survive."""
        area_ratio = float(bbox_area(bbox)) / max(1, img_area)
        scale_prior = self._scale_prior(area_ratio)

        if saliency_map is None or saliency_map.size == 0:
            return scale_prior

        x1, y1, x2, y2 = bbox
        region = saliency_map[max(0, y1):y2, max(0, x1):x2]
        if region.size == 0:
            return scale_prior

        total_sal = float(saliency_map.sum()) + 1e-9
        coverage = float(region.sum()) / total_sal
        density = float(region.mean())
        return 0.45 * coverage + 0.35 * density + 0.20 * scale_prior

    def _grid_anchors(self, h: int, w: int, img_area: int) -> List[BBox]:
        """Generate GAIC-style grid anchor candidates with dense center sampling."""
        candidates: List[BBox] = []

        # Grid centers (evenly spaced, including near-edges)
        gx = np.linspace(0.05, 0.95, self.grid_size + 2, dtype=float)[1:-1]
        gy = np.linspace(0.05, 0.95, self.grid_size + 2, dtype=float)[1:-1]

        aspect_ratios = self._active_aspect_ratios(h, w)

        for cx_pct in gx:
            for cy_pct in gy:
                cx = int(cx_pct * w)
                cy = int(cy_pct * h)
                for ar in aspect_ratios:
                    for area_r in self.area_ratios:
                        bbox = self._candidate_at_center(cx, cy, area_r, ar, h, w, img_area)
                        if bbox is not None:
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
        aspect_ratios = self._active_aspect_ratios(h, w)

        for cx, cy in peaks:
            for ar in aspect_ratios:
                for area_r in self.area_ratios:
                    bbox = self._candidate_at_center(cx, cy, area_r, ar, h, w, img_area)
                    if bbox is not None:
                        candidates.append(bbox)
        return candidates

    def _object_guided_candidates(
        self,
        detected_objects: List[DetectedObject],
        h: int,
        w: int,
        img_area: int,
    ) -> List[BBox]:
        """Generate candidates centered on individual objects and object groups."""
        centers: List[Tuple[float, float]] = []
        significant = []
        for obj in detected_objects:
            x1, y1, x2, y2 = obj.bbox
            area_ratio = bbox_area(obj.bbox) / max(1, img_area)
            if area_ratio < 0.003:
                continue
            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            if area_ratio >= 0.01:
                significant.append(obj.bbox)

        if significant:
            gx1 = min(box[0] for box in significant)
            gy1 = min(box[1] for box in significant)
            gx2 = max(box[2] for box in significant)
            gy2 = max(box[3] for box in significant)
            centers.append(((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0))

        candidates: List[BBox] = []
        for cx, cy in centers:
            for ar in self._active_aspect_ratios(h, w):
                for area_r in self.area_ratios:
                    bbox = self._candidate_at_center(cx, cy, area_r, ar, h, w, img_area)
                    if bbox is not None:
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
