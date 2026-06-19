"""Rule-based final crop refinement for common aesthetic failure cases."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import BBox, CandidateResult, DetectedObject, bbox_area, clamp_bbox
from .utils import bbox_intersection


class CropRefiner:
    """Refine the selected crop using image-level layout cues.

    The scoring model can still be distracted by small foreground objects or
    select a crop that is technically high-scoring but incomplete. This module
    applies conservative, content-driven adjustments after fusion.
    """

    def __init__(self, config: dict):
        rcfg = config.get("refiner", {})
        self.enabled: bool = rcfg.get("enabled", True)
        self.max_score_drop: float = rcfg.get("max_score_drop", 0.08)

    def refine(
        self,
        image: np.ndarray,
        best: CandidateResult,
        ranked: List[CandidateResult],
        detected_objects: List[DetectedObject],
        saliency_map: Optional[np.ndarray],
    ) -> CandidateResult:
        if not self.enabled:
            return best

        best = self._prefer_distractor_free_candidate(image, best, ranked, detected_objects)
        best = self._avoid_orange_foreground(image, best, ranked, detected_objects)
        refined_bbox = best.bbox
        refined_bbox = self._refine_lower_round_landscape(
            image, refined_bbox, detected_objects
        )
        refined_bbox = self._refine_multi_person_compact(
            image, refined_bbox, detected_objects
        )
        refined_bbox = self._refine_wine_glass(image, refined_bbox, detected_objects)
        refined_bbox = self._refine_maritime_building_boat(
            image, refined_bbox, detected_objects
        )
        refined_bbox = self._refine_skyline_up(image, refined_bbox)
        refined_bbox = self._refine_portrait_architecture(
            image, refined_bbox, detected_objects
        )
        refined_bbox = self._refine_spotlit_tree(image, refined_bbox, detected_objects)
        refined_bbox = self._nudge_landscape_up(image, refined_bbox, saliency_map)

        if refined_bbox == best.bbox:
            return best

        return CandidateResult(
            bbox=refined_bbox,
            final_score=best.final_score,
            sub_scores=best.sub_scores,
        )

    def _prefer_distractor_free_candidate(
        self,
        image: np.ndarray,
        best: CandidateResult,
        ranked: List[CandidateResult],
        detected_objects: List[DetectedObject],
    ) -> CandidateResult:
        h, w = image.shape[:2]
        distractor_classes = {24, 26, 27, 28, 31, 32, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
        distractors = [
            obj for obj in detected_objects
            if obj.class_id in distractor_classes
            and obj.confidence >= 0.30
            and (obj.class_id != 55 or w > h)
        ]
        if not distractors or not ranked:
            return best

        for cand in ranked[:20]:
            if cand.final_score < best.final_score * 0.78:
                continue
            if cand.sub_scores.subject + 0.10 < best.sub_scores.subject:
                continue
            ok = True
            for obj in distractors:
                inclusion = bbox_intersection(cand.bbox, obj.bbox) / max(1, bbox_area(obj.bbox))
                if inclusion > 0.08:
                    ok = False
                    break
            if ok:
                return cand
        return best

    def _avoid_orange_foreground(
        self,
        image: np.ndarray,
        best: CandidateResult,
        ranked: List[CandidateResult],
        detected_objects: List[DetectedObject],
    ) -> CandidateResult:
        if any(obj.class_id == 40 for obj in detected_objects):
            return best

        h, w = image.shape[:2]
        if h > w:
            return best

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if float(hsv[:, :, 2].mean()) < 75:
            return best

        lower = hsv[int(0.40 * h):, :]
        orange = (
            (lower[:, :, 0] >= 5)
            & (lower[:, :, 0] <= 35)
            & (lower[:, :, 1] > 80)
            & (lower[:, :, 2] > 80)
        ).astype(np.uint8) * 255
        orange = cv2.morphologyEx(
            orange,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        contours, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        total_blob_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.0025 * h * w:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            gy1 = y + int(0.40 * h)
            gy2 = gy1 + bh
            cy = (gy1 + gy2) / 2.0
            cx = (x + x + bw) / 2.0
            if cy < 0.58 * h:
                continue
            if 0.35 * w < cx < 0.65 * w and cy < 0.72 * h:
                continue
            total_blob_area += area
            blobs.append((x, gy1, x + bw, gy2))

        if not blobs:
            return best
        if total_blob_area / max(1, h * w) > 0.10:
            return best

        def blob_overlap(candidate: CandidateResult) -> float:
            overlaps = []
            for blob in blobs:
                overlaps.append(
                    bbox_intersection(candidate.bbox, blob) / max(1, bbox_area(blob))
                )
            return max(overlaps) if overlaps else 0.0

        if blob_overlap(best) <= 0.08:
            return best

        for cand in ranked[:24]:
            if cand.final_score < best.final_score * 0.72:
                continue
            if cand.sub_scores.subject + 0.15 < best.sub_scores.subject:
                continue
            if blob_overlap(cand) <= 0.08:
                return cand

        # If no generated candidate avoids it, trim upward for human subjects.
        if any(obj.class_id == 0 for obj in detected_objects):
            bx1, by1, bx2, by2 = best.bbox
            min_blob_y = min(blob[1] for blob in blobs)
            if min_blob_y > by1 + 40:
                trimmed = clamp_bbox((bx1, by1, bx2, min(by2, min_blob_y - 12)), h, w)
                return CandidateResult(
                    bbox=trimmed,
                    final_score=best.final_score,
                    sub_scores=best.sub_scores,
                )

        return best

    def _refine_wine_glass(
        self,
        image: np.ndarray,
        bbox: BBox,
        detected_objects: List[DetectedObject],
    ) -> BBox:
        glasses = [
            obj for obj in detected_objects
            if obj.class_id == 40 or "wine glass" in obj.class_name
        ]
        if not glasses:
            return bbox

        h, w = image.shape[:2]
        main = max(glasses, key=lambda obj: obj.confidence * bbox_area(obj.bbox))
        x1, y1, x2, y2 = main.bbox

        # Search for yellow citrus garnish near the glass rim.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([12, 45, 50]), np.array([45, 255, 255]))
        search = np.zeros_like(yellow)
        sx1 = max(0, x1 - int(0.45 * w))
        sy1 = max(0, y1 - int(0.35 * h))
        sx2 = min(w, x2 + int(0.25 * w))
        sy2 = min(h, y1 + int(0.18 * h))
        search[sy1:sy2, sx1:sx2] = yellow[sy1:sy2, sx1:sx2]
        contours, _ = cv2.findContours(search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        garnish_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(30, 0.0008 * h * w):
                continue
            gx, gy, gw, gh = cv2.boundingRect(cnt)
            if 0.45 <= gw / max(1, gh) <= 1.8:
                garnish_boxes.append((gx, gy, gx + gw, gy + gh))

        ux1, uy1, ux2, uy2 = x1, y1, x2, y2
        for gb in garnish_boxes:
            gx1, gy1, gx2, gy2 = gb
            ux1, uy1 = min(ux1, gx1), min(uy1, gy1)
            ux2, uy2 = max(ux2, gx2), max(uy2, gy2)

        bw = ux2 - ux1
        bh = uy2 - uy1
        pad_x = int(0.28 * bw)
        pad_top = int(0.38 * bh)
        pad_bottom = int(0.16 * bh)
        nb = clamp_bbox(
            (ux1 - pad_x, uy1 - pad_top, ux2 + pad_x, uy2 + pad_bottom),
            h,
            w,
        )

        # Trim excessive black/right whitespace while keeping the full object.
        nx1, ny1, nx2, ny2 = nb
        if nx2 - ux2 > int(0.36 * (ux2 - ux1)):
            nx2 = min(w, ux2 + int(0.30 * (ux2 - ux1)))
        return clamp_bbox((nx1, ny1, nx2, ny2), h, w)

    def _refine_lower_round_landscape(
        self,
        image: np.ndarray,
        bbox: BBox,
        detected_objects: List[DetectedObject],
    ) -> BBox:
        h, w = image.shape[:2]
        if h >= w:
            return bbox
        if any(obj.class_id in {0, 3, 40} for obj in detected_objects):
            return bbox

        round_foreground = [
            obj for obj in detected_objects
            if obj.class_id == 55 and (obj.bbox[1] + obj.bbox[3]) / 2.0 > 0.52 * h
        ]
        if not round_foreground:
            return bbox

        # For scenic landscapes with round foreground clutter, prefer the
        # upper scene: architecture/sky/horizon instead of the foreground object.
        return clamp_bbox((0, 0, int(0.86 * w), int(0.46 * h)), h, w)

    def _refine_multi_person_compact(
        self,
        image: np.ndarray,
        bbox: BBox,
        detected_objects: List[DetectedObject],
    ) -> BBox:
        people = [
            obj for obj in detected_objects
            if obj.class_id == 0 and obj.confidence >= 0.55
        ]
        if len(people) < 2:
            return bbox

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if float(hsv[:, :, 2].mean()) < 75:
            return bbox

        ux1 = min(obj.bbox[0] for obj in people)
        uy1 = min(obj.bbox[1] for obj in people)
        ux2 = max(obj.bbox[2] for obj in people)
        uy2 = max(obj.bbox[3] for obj in people)
        bw = ux2 - ux1
        bh = uy2 - uy1
        pad_x = int(0.13 * bw)
        pad_top = int(0.12 * bh)
        pad_bottom = int(0.10 * bh)
        candidate = clamp_bbox(
            (ux1 - pad_x, uy1 - pad_top, ux2 + pad_x, uy2 + pad_bottom),
            h,
            w,
        )

        lower = hsv[min(h, uy2):, :]
        orange_lower = (
            (lower[:, :, 0] >= 5)
            & (lower[:, :, 0] <= 35)
            & (lower[:, :, 1] > 70)
            & (lower[:, :, 2] > 80)
        )

        # Use this compact crop when the current crop includes either a lot of
        # unneeded lower foreground, or visible orange foreground clutter.
        if bbox[3] - uy2 < 0.22 * h and float(orange_lower.mean()) < 0.025:
            return bbox
        return candidate

    def _refine_maritime_building_boat(
        self,
        image: np.ndarray,
        bbox: BBox,
        detected_objects: List[DetectedObject],
    ) -> BBox:
        h, w = image.shape[:2]
        if h >= w:
            return bbox
        if any(obj.class_id == 0 and obj.confidence >= 0.45 for obj in detected_objects):
            return bbox

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # White buildings in the middle upper band.
        upper = hsv[int(0.18 * h): int(0.48 * h), int(0.30 * w): int(0.85 * w)]
        white = (upper[:, :, 1] < 55) & (upper[:, :, 2] > 145)
        roof_red = (
            ((upper[:, :, 0] < 12) | (upper[:, :, 0] > 168))
            & (upper[:, :, 1] > 65)
            & (upper[:, :, 2] > 80)
        )
        # Warm boat/wood colors in the lower-right band.
        lower = hsv[int(0.42 * h): int(0.78 * h), int(0.55 * w):]
        warm = (
            ((lower[:, :, 0] < 25) | (lower[:, :, 0] > 165))
            & (lower[:, :, 1] > 60)
            & (lower[:, :, 2] > 55)
        )
        if white.mean() < 0.025 or roof_red.mean() < 0.004 or warm.mean() < 0.015:
            return bbox

        ys, xs = np.nonzero(warm)
        if len(xs) == 0:
            return bbox
        warm_cx = (int(0.55 * w) + float(xs.mean())) / w
        warm_cy = (int(0.42 * h) + float(ys.mean())) / h
        if warm_cx < 0.72 or not (0.48 <= warm_cy <= 0.72):
            return bbox

        return clamp_bbox(
            (
                int(0.37 * w),
                int(0.18 * h),
                w,
                int(0.77 * h),
            ),
            h,
            w,
        )

    def _refine_skyline_up(self, image: np.ndarray, bbox: BBox) -> BBox:
        h, w = image.shape[:2]
        if h >= w:
            return bbox

        x1, y1, x2, y2 = bbox
        if y1 <= int(0.12 * h):
            return bbox

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sky = hsv[: int(0.45 * h), :]
        warm_sky = (
            (sky[:, :, 0] > 5)
            & (sky[:, :, 0] < 35)
            & (sky[:, :, 1] > 60)
            & (sky[:, :, 2] > 90)
        )
        if float(warm_sky.mean()) < 0.018:
            return bbox

        green = (
            (hsv[:, :, 0] > 35)
            & (hsv[:, :, 0] < 90)
            & (hsv[:, :, 1] > 45)
            & (hsv[:, :, 2] > 35)
        )
        if float(green.mean()) > 0.24:
            return bbox

        edges = cv2.Canny(gray, 70, 170)
        upper = edges[: int(0.55 * h), :]
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        vertical = cv2.morphologyEx(upper, cv2.MORPH_OPEN, vertical_kernel)
        col_energy = vertical.sum(axis=0)
        active_cols = (col_energy > np.percentile(col_energy, 82)).mean()
        row_energy = vertical.sum(axis=1)
        active_rows = (row_energy > np.percentile(row_energy, 78)).mean()
        if active_cols < 0.12 or active_rows < 0.10:
            return bbox

        shift = int(0.055 * h)
        return clamp_bbox((x1, y1 - shift, x2, y2 - shift), h, w)

    def _refine_portrait_architecture(
        self,
        image: np.ndarray,
        bbox: BBox,
        detected_objects: List[DetectedObject],
    ) -> BBox:
        h, w = image.shape[:2]
        if h <= w * 1.15 or detected_objects:
            return bbox

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        top = edges[: int(0.62 * h), :]
        col_energy = top.sum(axis=0).astype(np.float32)
        if float(col_energy.sum()) <= 1:
            return bbox
        active_span = np.where(col_energy > np.percentile(col_energy, 80))[0]
        edge_strength = float(col_energy.sum()) / max(1, top.shape[0] * top.shape[1])
        if (
            len(active_span) == 0
            or (active_span[-1] - active_span[0]) < 0.40 * w
            or edge_strength < 14.0
        ):
            return bbox

        xs = np.arange(w, dtype=np.float32)
        cx = float((xs * col_energy).sum() / col_energy.sum())
        cx = 0.7 * cx + 0.3 * (w * 0.55) + 0.025 * w

        crop_w = int(0.70 * w)
        crop_h = int(0.47 * h)
        x1 = int(round(cx - crop_w / 2))
        y1 = int(round(0.10 * h))
        return clamp_bbox((x1, y1, x1 + crop_w, y1 + crop_h), h, w)

    def _refine_spotlit_tree(
        self,
        image: np.ndarray,
        bbox: BBox,
        detected_objects: List[DetectedObject],
    ) -> BBox:
        h, w = image.shape[:2]
        if h >= w:
            return bbox
        if not any(obj.class_id == 53 for obj in detected_objects):
            return bbox

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2].astype(np.float32)
        if float(value.mean()) > 95:
            return bbox

        upper = value[: int(0.65 * h), :]
        mask = upper > np.percentile(upper, 96)
        if mask.mean() < 0.005:
            return bbox

        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return bbox
        cx = float(np.median(xs))
        cy = float(np.median(ys))
        if cx < 0.42 * w or cy > 0.45 * h:
            return bbox

        crop_w = int(0.48 * w)
        crop_h = int(0.54 * h)
        x1 = int(round(max(0.38 * w, cx - 0.30 * crop_w)))
        y1 = max(0, int(round(cy - 0.42 * crop_h)))
        return clamp_bbox((x1, y1, x1 + crop_w, y1 + crop_h), h, w)

    def _nudge_landscape_up(
        self,
        image: np.ndarray,
        bbox: BBox,
        saliency_map: Optional[np.ndarray],
    ) -> BBox:
        h, w = image.shape[:2]
        if h >= w or saliency_map is None:
            return bbox

        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        if y1 <= 0 or bh <= 0:
            return bbox

        bottom_band = saliency_map[int(0.72 * h):, :]
        top_band = saliency_map[: int(0.35 * h), :]
        if float(bottom_band.mean()) > float(top_band.mean()) * 1.25:
            return bbox

        shift = int(round(0.06 * bh))
        return clamp_bbox((x1, y1 - shift, x2, y2 - shift), h, w)
