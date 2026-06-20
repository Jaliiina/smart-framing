"""ROI/discard-region scoring for scientific crop selection.

This module follows the same spirit as grid-anchor image cropping methods:
each candidate is evaluated by what it keeps inside the region of interest
(ROI), what it discards outside the crop, and whether the crop boundary cuts
through salient structures. No scene-specific object or color rule is used.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import BBox, DetectedObject, bbox_area, bbox_intersection
from .subjectness_scorer import SubjectnessMaps


class RoiDiscardScorer:
    """Score candidates by ROI preservation and discard-region quality."""

    def __init__(self, config: dict):
        cfg = config.get("roi_discard", {})
        self.weight_roi_saliency = float(cfg.get("weight_roi_saliency", 0.35))
        self.weight_discard_quality = float(cfg.get("weight_discard_quality", 0.25))
        self.weight_subject_integrity = float(cfg.get("weight_subject_integrity", 0.25))
        self.weight_boundary_clean = float(cfg.get("weight_boundary_clean", 0.15))
        self.weight_semantic = float(cfg.get("weight_semantic", 0.15))
        self.weight_visual_artifact = float(cfg.get("weight_visual_artifact", 0.35))
        self.boundary_strip_ratio = float(cfg.get("boundary_strip_ratio", 0.035))
        self.distractor_classes = set(
            cfg.get(
                "distractor_classes",
                [24, 26, 27, 28, 31, 32, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
            )
        )

    def score_candidates(
        self,
        image: np.ndarray,
        bboxes: List[BBox],
        saliency_map: Optional[np.ndarray],
        detected_objects: List[DetectedObject],
        subjectness_maps: Optional[SubjectnessMaps] = None,
        semantic_scores: Optional[List[Tuple[float, Dict[str, float]]]] = None,
    ) -> List[Tuple[float, Dict[str, float]]]:
        if saliency_map is None:
            saliency_map = self._fallback_activity_map(image)
        saliency = self._normalize_map(saliency_map)
        edge_map = self._edge_map(image)
        if subjectness_maps is None:
            subjectness = saliency
            distractor_map = np.zeros_like(saliency)
        else:
            subjectness = subjectness_maps.subjectness
            distractor_map = subjectness_maps.distractor
        salient_mask = subjectness >= max(0.25, float(np.percentile(subjectness, 75)))
        if semantic_scores is None:
            semantic_scores = [(0.5, {"positive_semantic": 0.5, "negative_semantic": 0.5}) for _ in bboxes]

        scores = []
        for bbox, semantic in zip(bboxes, semantic_scores):
            sub = self._score_single(
                bbox=bbox,
                image_shape=image.shape[:2],
                saliency=saliency,
                subjectness=subjectness,
                distractor_map=distractor_map,
                edge_map=edge_map,
                salient_mask=salient_mask,
                detected_objects=detected_objects,
                semantic_score=float(semantic[0]),
                semantic_sub=semantic[1],
                image=image,
            )
            total = (
                self.weight_roi_saliency * sub["subject_roi"]
                + self.weight_discard_quality * sub["good_discard"]
                + self.weight_subject_integrity * sub["subject_integrity"]
                + self.weight_boundary_clean * (1.0 - sub["boundary_cut"])
                + self.weight_semantic * sub["semantic_score"]
                - sub["bad_discard"]
                - sub["distractor_inside"]
                - sub["distractor_penalty"]
                - self.weight_visual_artifact * sub["visual_artifact_penalty"]
            )
            scores.append((float(np.clip(total, 0.0, 1.0)), sub))
        return scores

    def score_single(
        self,
        image: np.ndarray,
        bbox: BBox,
        saliency_map: Optional[np.ndarray],
        detected_objects: List[DetectedObject],
    ) -> Tuple[float, Dict[str, float]]:
        return self.score_candidates(image, [bbox], saliency_map, detected_objects)[0]

    def _score_single(
        self,
        bbox: BBox,
        image_shape: Tuple[int, int],
        saliency: np.ndarray,
        subjectness: np.ndarray,
        distractor_map: np.ndarray,
        edge_map: np.ndarray,
        salient_mask: np.ndarray,
        detected_objects: List[DetectedObject],
        semantic_score: float,
        semantic_sub: Dict[str, float],
        image: np.ndarray,
    ) -> Dict[str, float]:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_area = max(1, (x2 - x1) * (y2 - y1))
        image_area = max(1, h * w)

        crop_saliency = saliency[y1:y2, x1:x2]
        crop_subject = subjectness[y1:y2, x1:x2]
        crop_distractor = distractor_map[y1:y2, x1:x2]
        crop_edges = edge_map[y1:y2, x1:x2]
        if crop_saliency.size == 0:
            return {
                "roi_saliency": 0.0,
                "discard_quality": 0.0,
                "subject_integrity": 0.0,
                "boundary_cut": 1.0,
                "distractor_penalty": 1.0,
                "subject_roi": 0.0,
                "distractor_inside": 1.0,
                "good_discard": 0.0,
                "bad_discard": 1.0,
                "semantic_score": semantic_score,
                "positive_semantic": semantic_sub.get("positive_semantic", 0.5),
                "negative_semantic": semantic_sub.get("negative_semantic", 0.5),
                "visual_artifact_penalty": 1.0,
                "blank_area_penalty": 1.0,
                "saturated_boundary_penalty": 1.0,
                "small_saturated_object_penalty": 1.0,
            }

        total_saliency = float(saliency.sum()) + 1e-9
        total_subject = float(subjectness.sum()) + 1e-9
        roi_coverage = float(crop_saliency.sum()) / total_saliency
        roi_density = float(crop_saliency.mean())
        subject_coverage = float(crop_subject.sum()) / total_subject
        subject_density = float(crop_subject.mean())
        area_ratio = crop_area / image_area
        roi_saliency = 0.65 * roi_coverage + 0.35 * roi_density
        roi_saliency = float(np.clip(roi_saliency / max(0.25, min(1.0, area_ratio + 0.25)), 0.0, 1.0))
        subject_roi = 0.65 * subject_coverage + 0.35 * subject_density
        subject_roi = float(np.clip(subject_roi / max(0.25, min(1.0, area_ratio + 0.25)), 0.0, 1.0))
        distractor_inside = float(crop_distractor.mean()) if crop_distractor.size else 0.0

        outside_mask = np.ones((h, w), dtype=bool)
        outside_mask[y1:y2, x1:x2] = False
        outside_salient = float(salient_mask[outside_mask].mean()) if outside_mask.any() else 0.0
        outside_saliency = float(saliency[outside_mask].mean()) if outside_mask.any() else 0.0
        outside_subject = float(subjectness[outside_mask].mean()) if outside_mask.any() else 0.0
        outside_distractor = float(distractor_map[outside_mask].mean()) if outside_mask.any() else 0.0
        discard_quality = float(np.clip(1.0 - 0.65 * outside_salient - 0.35 * outside_saliency, 0.0, 1.0))
        good_discard = float(np.clip(0.55 * outside_distractor + 0.45 * (1.0 - outside_saliency), 0.0, 1.0))
        bad_discard = float(np.clip(0.65 * outside_subject + 0.35 * outside_salient, 0.0, 1.0))

        boundary_cut = self._boundary_cut_score((x1, y1, x2, y2), saliency, edge_map)
        subject_integrity, distractor_penalty = self._object_terms(
            (x1, y1, x2, y2), detected_objects, image_area
        )

        # If a crop has strong structure exactly on its boundary, treat it as
        # a possible subject cut even when saliency coverage is high.
        if crop_edges.size > 0:
            edge_density = float((crop_edges > 0.35).mean())
            boundary_cut = float(np.clip(boundary_cut + 0.12 * edge_density, 0.0, 1.0))

        (
            visual_penalty,
            blank_penalty,
            saturated_boundary_penalty,
            small_saturated_object_penalty,
        ) = (
            self._visual_artifact_penalty(image[y1:y2, x1:x2], crop_edges)
        )

        return {
            "roi_saliency": roi_saliency,
            "discard_quality": discard_quality,
            "subject_integrity": subject_integrity,
            "boundary_cut": boundary_cut,
            "distractor_penalty": distractor_penalty,
            "subject_roi": subject_roi,
            "distractor_inside": distractor_inside,
            "good_discard": good_discard,
            "bad_discard": bad_discard,
            "semantic_score": semantic_score,
            "positive_semantic": semantic_sub.get("positive_semantic", 0.5),
            "negative_semantic": semantic_sub.get("negative_semantic", 0.5),
            "visual_artifact_penalty": visual_penalty,
            "blank_area_penalty": blank_penalty,
            "saturated_boundary_penalty": saturated_boundary_penalty,
            "small_saturated_object_penalty": small_saturated_object_penalty,
        }

    def _visual_artifact_penalty(
        self,
        crop: np.ndarray,
        crop_edges: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        if crop.size == 0:
            return 1.0, 1.0, 1.0, 1.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        edge = crop_edges if crop_edges.size else self._edge_map(crop)

        white_blank = (sat < 0.16) & (val > 0.78) & (edge < 0.08)
        dark_blank = (val < 0.16) & (edge < 0.08)
        flat_bright_color = (sat > 0.62) & (edge < 0.05)
        blank_ratio = float(
            max(
                white_blank.mean(),
                dark_blank.mean(),
                0.55 * flat_bright_color.mean(),
            )
        )
        blank_penalty = float(np.clip((blank_ratio - 0.30) / 0.35, 0.0, 1.0))

        h, w = crop.shape[:2]
        strip = max(3, int(min(h, w) * 0.08))
        border = np.zeros((h, w), dtype=bool)
        border[:strip, :] = True
        border[-strip:, :] = True
        border[:, :strip] = True
        border[:, -strip:] = True
        saturated_boundary = border & (sat > 0.58) & (val > 0.35) & (edge > 0.02)
        sat_boundary_ratio = float(saturated_boundary.mean())
        saturated_boundary_penalty = float(np.clip(sat_boundary_ratio / 0.11, 0.0, 1.0))

        small_saturated_object_penalty = self._small_saturated_object_penalty(
            sat=sat,
            val=val,
            edge=edge,
        )

        penalty = float(
            np.clip(
                0.52 * blank_penalty
                + 0.24 * saturated_boundary_penalty
                + 0.24 * small_saturated_object_penalty,
                0.0,
                1.0,
            )
        )
        return penalty, blank_penalty, saturated_boundary_penalty, small_saturated_object_penalty

    @staticmethod
    def _small_saturated_object_penalty(
        sat: np.ndarray,
        val: np.ndarray,
        edge: np.ndarray,
    ) -> float:
        h, w = sat.shape[:2]
        if h < 8 or w < 8:
            return 0.0
        mask = ((sat > 0.55) & (val > 0.32)).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        crop_area = max(1, h * w)
        penalty = 0.0
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            area_ratio = area / crop_area
            if area_ratio < 0.002 or area_ratio > 0.42:
                continue
            cx, cy = centroids[label]
            x0 = int(stats[label, cv2.CC_STAT_LEFT])
            y0 = int(stats[label, cv2.CC_STAT_TOP])
            ww = int(stats[label, cv2.CC_STAT_WIDTH])
            hh = int(stats[label, cv2.CC_STAT_HEIGHT])
            x = cx / max(1, w)
            y = cy / max(1, h)
            lower_prior = np.clip((y - 0.38) / 0.48, 0.0, 1.0)
            side_prior = max(abs(x - 0.5) * 2.0, 0.0)
            touches_border = (
                x0 <= 2
                or y0 <= 2
                or x0 + ww >= w - 2
                or y0 + hh >= h - 2
            )
            border_prior = 1.0 if touches_border else 0.0
            foreground_prior = (
                0.64 * lower_prior
                + 0.22 * side_prior
                + 0.14 * border_prior
            )
            if area_ratio <= 0.12:
                area_score = np.clip((area_ratio - 0.0015) / 0.020, 0.0, 1.0)
            else:
                area_score = np.clip((area_ratio - 0.08) / 0.20, 0.0, 1.0)
            penalty = max(penalty, float(area_score * foreground_prior))
        return float(np.clip(penalty, 0.0, 1.0))

    def _boundary_cut_score(
        self,
        bbox: BBox,
        saliency: np.ndarray,
        edge_map: np.ndarray,
    ) -> float:
        h, w = saliency.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        strip = max(2, int(min(bw, bh) * self.boundary_strip_ratio))

        strips = []
        if bh > 2 * strip:
            strips.append((slice(y1, min(y1 + strip, h)), slice(x1, x2)))
            strips.append((slice(max(y2 - strip, 0), y2), slice(x1, x2)))
        if bw > 2 * strip:
            strips.append((slice(y1, y2), slice(x1, min(x1 + strip, w))))
            strips.append((slice(y1, y2), slice(max(x2 - strip, 0), x2)))
        if not strips:
            return 1.0

        responses = []
        for ys, xs in strips:
            s = saliency[ys, xs]
            e = edge_map[ys, xs]
            if s.size:
                responses.append(0.55 * float(s.mean()) + 0.45 * float(e.mean()))
        return float(np.clip(np.mean(responses) if responses else 1.0, 0.0, 1.0))

    def _object_terms(
        self,
        bbox: BBox,
        detected_objects: List[DetectedObject],
        image_area: int,
    ) -> Tuple[float, float]:
        if not detected_objects:
            return 0.5, 0.0

        important = []
        distractor_penalty = 0.0
        for obj in detected_objects:
            obj_area = max(1, bbox_area(obj.bbox))
            area_ratio = obj_area / max(1, image_area)
            inclusion = bbox_intersection(bbox, obj.bbox) / obj_area
            if obj.class_id in self.distractor_classes and area_ratio < 0.12:
                distractor_penalty = max(distractor_penalty, 0.25 * inclusion * obj.confidence)
            else:
                important.append((obj, inclusion))

        if not important:
            return 0.5, float(np.clip(distractor_penalty, 0.0, 0.45))

        weights = np.array(
            [max(0.05, obj.confidence) * np.sqrt(max(1, bbox_area(obj.bbox))) for obj, _ in important],
            dtype=np.float64,
        )
        inclusions = np.array([inc for _, inc in important], dtype=np.float64)
        integrity = float((weights * inclusions).sum() / (weights.sum() + 1e-9))
        return float(np.clip(integrity, 0.0, 1.0)), float(np.clip(distractor_penalty, 0.0, 0.45))

    @staticmethod
    def _normalize_map(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        min_v, max_v = float(values.min()), float(values.max())
        if max_v - min_v < 1e-9:
            return np.zeros_like(values, dtype=np.float32)
        return (values - min_v) / (max_v - min_v)

    @staticmethod
    def _edge_map(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 170).astype(np.float32) / 255.0
        return cv2.GaussianBlur(edges, (0, 0), 1.2)

    @staticmethod
    def _fallback_activity_map(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        activity = 0.55 * np.clip(lap / 80.0, 0.0, 1.0) + 0.45 * np.clip(hsv[:, :, 1] / 180.0, 0.0, 1.0)
        return activity.astype(np.float32)
