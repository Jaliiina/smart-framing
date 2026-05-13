"""Photography composition rule scoring: rule of thirds, center balance,
whitespace, edge simplicity, symmetry."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import BBox, bbox_center


class CompositionScorer:
    """Score candidates based on photography composition rules."""

    def __init__(self, config: dict):
        ccfg = config.get("composition", {})
        w = ccfg.get("weights", {})
        self.weight_thirds: float = w.get("rule_of_thirds", 0.35)
        self.weight_balance: float = w.get("center_balance", 0.25)
        self.weight_whitespace: float = w.get("whitespace", 0.15)
        self.weight_edge: float = w.get("edge_simplicity", 0.15)
        self.weight_symmetry: float = w.get("symmetry", 0.10)
        self.thirds_sigma: float = ccfg.get("thirds_sigma", 0.08)
        self.whitespace_ideal: float = ccfg.get("whitespace_ideal_ratio", 0.3)

    def score_candidates(
        self,
        image: np.ndarray,
        bboxes: List[BBox],
        saliency_map: Optional[np.ndarray] = None,
        detected_objects: Optional[list] = None,
    ) -> List[Tuple[float, Dict[str, float]]]:
        """Score each candidate on composition rules.

        Args:
            image: Original BGR image.
            bboxes: Candidate bboxes.
            saliency_map: Optional saliency map (H, W).
            detected_objects: Optional list of DetectedObject.

        Returns:
            List of (total_composition_score, sub_score_dict) per candidate.
        """
        scores = []
        for bbox in bboxes:
            sub = self._score_single(image, bbox, saliency_map, detected_objects)
            total = (
                self.weight_thirds * sub["thirds"]
                + self.weight_balance * sub["center_balance"]
                + self.weight_whitespace * sub["whitespace"]
                + self.weight_edge * sub["edge_simplicity"]
                + self.weight_symmetry * sub["symmetry"]
            )
            scores.append((total, sub))
        return scores

    def _score_single(
        self,
        image: np.ndarray,
        bbox: BBox,
        saliency_map: Optional[np.ndarray],
        detected_objects: Optional[list],
    ) -> Dict[str, float]:
        """Compute all composition sub-scores for one candidate."""
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return {
                "thirds": 0.0,
                "center_balance": 0.0,
                "whitespace": 0.0,
                "edge_simplicity": 0.0,
                "symmetry": 0.0,
            }

        # Determine the "subject center" — use saliency centroid or object centroid
        subject_center = self._find_subject_center(
            bbox, saliency_map, detected_objects
        )

        thirds = self._rule_of_thirds(subject_center, bbox)
        balance = self._center_balance(saliency_map, bbox) if saliency_map is not None else 0.5
        whitespace = self._whitespace(saliency_map, bbox) if saliency_map is not None else 0.5
        edge = self._edge_simplicity(image, bbox)
        symmetry = self._symmetry(crop)

        return {
            "thirds": thirds,
            "center_balance": balance,
            "whitespace": whitespace,
            "edge_simplicity": edge,
            "symmetry": symmetry,
        }

    def _find_subject_center(
        self,
        bbox: BBox,
        saliency_map: Optional[np.ndarray],
        detected_objects: Optional[list],
    ) -> Tuple[float, float]:
        """Find the subject center within a candidate bbox.

        Priority: detected object center > saliency centroid > bbox center.
        """
        x1, y1, x2, y2 = bbox
        bx_cx, bx_cy = bbox_center(bbox)

        # Try detected objects first
        if detected_objects and len(detected_objects) > 0:
            from .utils import bbox_area, bbox_intersection

            best_obj = None
            best_overlap = 0
            for obj in detected_objects:
                inter = bbox_intersection(bbox, obj.bbox)
                obj_area = max(1, bbox_area(obj.bbox))
                overlap_ratio = inter / obj_area
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_obj = obj
            if best_obj is not None and best_overlap > 0.3:
                ox1, oy1, ox2, oy2 = best_obj.bbox
                cx = (ox1 + ox2) / 2.0
                cy = (oy1 + oy2) / 2.0
                return (cx, cy)

        # Try saliency centroid
        if saliency_map is not None:
            region = saliency_map[max(0, y1):y2, max(0, x1):x2]
            region_sum = region.sum()
            if region_sum > 1e-6:
                local_h, local_w = region.shape[:2]
                ys, xs = np.mgrid[0:local_h, 0:local_w]
                cx = float((xs * region).sum() / region_sum) + x1
                cy = float((ys * region).sum() / region_sum) + y1
                return (cx, cy)

        return (bx_cx, bx_cy)

    def _rule_of_thirds(self, subject_center: Tuple[float, float], bbox: BBox) -> float:
        """Score based on proximity of subject center to rule-of-thirds intersection points."""
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Normalized subject position relative to bbox
        sx = (subject_center[0] - x1) / bw
        sy = (subject_center[1] - y1) / bh
        sx = max(0.0, min(1.0, sx))
        sy = max(0.0, min(1.0, sy))

        # Rule of thirds intersection points (normalized)
        points = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]

        # Distance to nearest third-line intersection
        min_dist = float("inf")
        for px, py in points:
            d = math.sqrt((sx - px) ** 2 + (sy - py) ** 2)
            min_dist = min(min_dist, d)

        # Also check proximity to third lines (not just intersections)
        for val in [1 / 3, 2 / 3]:
            d_line_x = abs(sx - val)
            d_line_y = abs(sy - val)
            min_dist = min(min_dist, d_line_x * 0.5, d_line_y * 0.5)

        # Gaussian decay scoring
        sigma = self.thirds_sigma
        score = math.exp(-min_dist ** 2 / (2 * sigma ** 2))
        return score

    def _center_balance(self, saliency_map: np.ndarray, bbox: BBox) -> float:
        """Score visual weight balance (left-right and top-bottom)."""
        x1, y1, x2, y2 = bbox
        region = saliency_map[max(0, y1):y2, max(0, x1):x2]
        h, w = region.shape[:2]
        if h < 4 or w < 4:
            return 0.5

        # Left-right balance
        mid_x = w // 2
        left_weight = region[:, :mid_x].sum()
        right_weight = region[:, mid_x:].sum()
        total_lr = left_weight + right_weight + 1e-9
        lr_balance = 1.0 - abs(left_weight - right_weight) / total_lr

        # Top-bottom balance
        mid_y = h // 2
        top_weight = region[:mid_y, :].sum()
        bottom_weight = region[mid_y:, :].sum()
        total_tb = top_weight + bottom_weight + 1e-9
        tb_balance = 1.0 - abs(top_weight - bottom_weight) / total_tb

        return 0.5 * lr_balance + 0.5 * tb_balance

    def _whitespace(self, saliency_map: np.ndarray, bbox: BBox) -> float:
        """Score appropriate whitespace around subject.

        Too full (no whitespace) or too empty (all whitespace) both penalized.
        """
        x1, y1, x2, y2 = bbox
        region = saliency_map[max(0, y1):y2, max(0, x1):x2]
        h, w = region.shape[:2]
        if h < 4 or w < 4:
            return 0.5

        # Fraction of the crop that is "non-salient" (whitespace)
        threshold = 0.1
        whitespace_ratio = float((region < threshold).mean())

        # Score: ideal is around self.whitespace_ideal
        deviation = abs(whitespace_ratio - self.whitespace_ideal)
        score = math.exp(-deviation ** 2 / (2 * 0.15 ** 2))
        return score

    def _edge_simplicity(self, image: np.ndarray, bbox: BBox) -> float:
        """Score simplicity of the bbox boundary region.

        Lower edge density near the boundary = simpler = better.
        """
        x1, y1, x2, y2 = bbox
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)

        h, w = image.shape[:2]
        strip = max(3, min(h, w) // 80)

        # Compute edge density in boundary strip of the bbox
        boundary_edge_count = 0
        boundary_pixel_count = 0

        # Top strip
        ry1, ry2 = max(0, y1 - strip), max(0, y1 + strip)
        if ry2 > ry1:
            boundary_edge_count += edges[ry1:ry2, max(0, x1):x2].sum()
            boundary_pixel_count += (ry2 - ry1) * (x2 - x1)
        # Bottom strip
        ry1, ry2 = max(0, y2 - strip), min(h, y2 + strip)
        if ry2 > ry1:
            boundary_edge_count += edges[ry1:ry2, max(0, x1):x2].sum()
            boundary_pixel_count += (ry2 - ry1) * (x2 - x1)
        # Left strip
        rx1, rx2 = max(0, x1 - strip), max(0, x1 + strip)
        if rx2 > rx1:
            boundary_edge_count += edges[y1:y2, rx1:rx2].sum()
            boundary_pixel_count += (y2 - y1) * (rx2 - rx1)
        # Right strip
        rx1, rx2 = max(0, x2 - strip), min(w, x2 + strip)
        if rx2 > rx1:
            boundary_edge_count += edges[y1:y2, rx1:rx2].sum()
            boundary_pixel_count += (y2 - y1) * (rx2 - rx1)

        if boundary_pixel_count == 0:
            return 0.5

        edge_density = boundary_edge_count / (boundary_pixel_count * 255)
        # Lower density = simpler boundary = higher score
        score = 1.0 - min(1.0, edge_density * 10)  # scale factor
        return max(0.0, score)

    def _symmetry(self, crop: np.ndarray) -> float:
        """Score left-right and top-bottom structural similarity."""
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Downsample for speed
        if max(h, w) > 128:
            scale = 128 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        h2, w2 = gray.shape[:2]

        # Left-right symmetry
        mid_x = w2 // 2
        left = gray[:, :mid_x]
        right = gray[:, w2 - mid_x:][:, ::-1]  # flipped
        if left.shape == right.shape:
            lr_diff = np.abs(left - right).mean() / 255.0
            lr_sym = 1.0 - lr_diff
        else:
            lr_sym = 0.0

        # Top-bottom symmetry
        mid_y = h2 // 2
        top = gray[:mid_y, :]
        bottom = gray[h2 - mid_y:, :][::-1, :]  # flipped
        if top.shape == bottom.shape:
            tb_diff = np.abs(top - bottom).mean() / 255.0
            tb_sym = 1.0 - tb_diff
        else:
            tb_sym = 0.0

        # Take the max (some images are symmetric in one direction)
        return max(lr_sym, tb_sym)
