"""Generic local crop optimization.

The optimizer refines candidate selection with a single mathematical objective
instead of scene-specific post-processing. It searches small center/scale
perturbations around the top candidates and accepts boxes that improve the
combined ROI, discard, composition, subject, and boundary score.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .roi_discard_scorer import RoiDiscardScorer
from .semantic_crop_scorer import SemanticCropScorer
from .subjectness_scorer import SubjectnessMaps
from .utils import BBox, CandidateResult, DetectedObject, SubScores, bbox_area, clamp_bbox


class ScientificCropOptimizer:
    """Local coordinate-search optimizer for final crop selection."""

    def __init__(
        self,
        config: dict,
        roi_scorer: RoiDiscardScorer,
        semantic_scorer: Optional[SemanticCropScorer] = None,
    ):
        cfg = config.get("scientific_optimizer", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.top_n = int(cfg.get("top_n", 20))
        self.max_variants_per_seed = int(cfg.get("max_variants_per_seed", 18))
        self.min_area_ratio = float(cfg.get("min_area_ratio", 0.18))
        self.max_area_ratio = float(cfg.get("max_area_ratio", 0.72))
        self.min_improvement = float(cfg.get("min_improvement", 0.015))
        self.roi_scorer = roi_scorer
        self.semantic_scorer = semantic_scorer

        weights = cfg.get("weights", {})
        self.w_fusion = float(weights.get("fusion", 0.35))
        self.w_roi = float(weights.get("roi_discard", 0.25))
        self.w_subject = float(weights.get("subject", 0.16))
        self.w_composition = float(weights.get("composition", 0.14))
        self.w_boundary = float(weights.get("boundary_clean", 0.10))
        self.w_semantic = float(weights.get("semantic", 0.12))
        self.w_subjectness = float(weights.get("subjectness", 0.12))
        self.w_good_discard = float(weights.get("good_discard", 0.08))
        self.w_bad_discard = float(weights.get("bad_discard", 0.12))
        self.w_distractor = float(weights.get("distractor_avoidance", 0.16))
        self.w_visual_artifact = float(weights.get("visual_artifact", 0.14))

    def optimize(
        self,
        image: np.ndarray,
        ranked: List[CandidateResult],
        detected_objects: List[DetectedObject],
        saliency_map: Optional[np.ndarray],
        subjectness_maps: Optional[SubjectnessMaps] = None,
    ) -> List[CandidateResult]:
        if not self.enabled or not ranked:
            return ranked

        h, w = image.shape[:2]
        seeds = ranked[: max(1, self.top_n)]
        variants = []
        seen = {cand.bbox for cand in ranked}
        for seed in seeds:
            for bbox in self._variants(seed.bbox, h, w):
                area_ratio = bbox_area(bbox) / max(1, h * w)
                if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                    continue
                if bbox in seen:
                    continue
                seen.add(bbox)
                variants.append((bbox, seed))
                if len(variants) >= self.top_n * self.max_variants_per_seed:
                    break

        if not variants:
            return ranked

        variant_boxes = [bbox for bbox, _seed in variants]
        semantic_scores = (
            self.semantic_scorer.score_candidates(image, variant_boxes)
            if self.semantic_scorer is not None
            else None
        )
        roi_scores = self.roi_scorer.score_candidates(
            image=image,
            bboxes=variant_boxes,
            saliency_map=saliency_map,
            detected_objects=detected_objects,
            subjectness_maps=subjectness_maps,
            semantic_scores=semantic_scores,
        )
        subjectness_scores = self._score_subjectness(variant_boxes, subjectness_maps)
        new_candidates = []
        for idx, ((bbox, seed), (roi_total, roi_sub)) in enumerate(zip(variants, roi_scores)):
            semantic_sub = (semantic_scores[idx][1] if semantic_scores else {})
            sub = SubScores(
                aesthetic=seed.sub_scores.aesthetic,
                saliency=seed.sub_scores.saliency,
                composition=self._composition_proxy(bbox, saliency_map, image.shape[:2]),
                subject=self._subject_integrity(bbox, detected_objects, image.shape[:2]),
                technical=seed.sub_scores.technical,
                area_prior=seed.sub_scores.area_prior,
                thirds=seed.sub_scores.thirds,
                center_balance=seed.sub_scores.center_balance,
                whitespace=seed.sub_scores.whitespace,
                edge_simplicity=seed.sub_scores.edge_simplicity,
                symmetry=seed.sub_scores.symmetry,
                sharpness=seed.sub_scores.sharpness,
                brightness=seed.sub_scores.brightness,
                contrast=seed.sub_scores.contrast,
                saturation=seed.sub_scores.saturation,
                person_completeness=seed.sub_scores.person_completeness,
                roi_discard=roi_total,
                roi_saliency=roi_sub.get("roi_saliency", 0.0),
                discard_quality=roi_sub.get("discard_quality", 0.0),
                boundary_cut=roi_sub.get("boundary_cut", 0.0),
                distractor_penalty=roi_sub.get("distractor_penalty", 0.0),
                semantic_score=roi_sub.get("semantic_score", 0.0),
                positive_semantic=semantic_sub.get(
                    "positive_semantic", roi_sub.get("positive_semantic", 0.0)
                ),
                negative_semantic=semantic_sub.get(
                    "negative_semantic", roi_sub.get("negative_semantic", 0.0)
                ),
                subjectness=subjectness_scores[idx][0],
                distractor_map_score=subjectness_scores[idx][1],
                good_discard=roi_sub.get("good_discard", 0.0),
                bad_discard=roi_sub.get("bad_discard", 0.0),
                visual_artifact_penalty=roi_sub.get("visual_artifact_penalty", 0.0),
                blank_area_penalty=roi_sub.get("blank_area_penalty", 0.0),
                saturated_boundary_penalty=roi_sub.get("saturated_boundary_penalty", 0.0),
                small_saturated_object_penalty=roi_sub.get(
                    "small_saturated_object_penalty", 0.0
                ),
            )
            score = self._objective(seed.final_score, sub)
            new_candidates.append(CandidateResult(bbox=bbox, final_score=score, sub_scores=sub))

        rescored_existing = []
        for cand in ranked:
            cand.final_score = self._objective(cand.final_score, cand.sub_scores)
            rescored_existing.append(cand)

        all_candidates = rescored_existing + new_candidates
        all_candidates.sort(key=lambda c: c.final_score, reverse=True)

        if len(all_candidates) > 1:
            original_best = rescored_existing[0]
            if all_candidates[0].bbox != original_best.bbox:
                cleaner_takeover = (
                    all_candidates[0].sub_scores.visual_artifact_penalty
                    + 0.5 * all_candidates[0].sub_scores.distractor_map_score
                    + 0.5 * all_candidates[0].sub_scores.distractor_penalty
                    + 0.02
                    < original_best.sub_scores.visual_artifact_penalty
                    + 0.5 * original_best.sub_scores.distractor_map_score
                    + 0.5 * original_best.sub_scores.distractor_penalty
                )
                if (
                    all_candidates[0].final_score < original_best.final_score + self.min_improvement
                    and not cleaner_takeover
                ):
                    all_candidates.remove(original_best)
                    all_candidates.insert(0, original_best)

        return all_candidates

    def _objective(self, base_score: float, sub: SubScores) -> float:
        return float(
            self.w_fusion * base_score
            + self.w_roi * sub.roi_discard
            + self.w_subject * sub.subject
            + self.w_composition * sub.composition
            + self.w_boundary * (1.0 - sub.boundary_cut)
            + self.w_semantic * sub.semantic_score
            + self.w_subjectness * sub.subjectness
            + self.w_good_discard * sub.good_discard
            - self.w_bad_discard * sub.bad_discard
            - self.w_distractor * max(sub.distractor_map_score, sub.distractor_penalty)
            - self.w_visual_artifact * sub.visual_artifact_penalty
        )

    @staticmethod
    def _score_subjectness(
        bboxes: List[BBox],
        maps: Optional[SubjectnessMaps],
    ) -> List[tuple[float, float]]:
        if maps is None:
            return [(0.0, 0.0) for _ in bboxes]
        scores = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            subj = maps.subjectness[max(0, y1):y2, max(0, x1):x2]
            dist = maps.distractor[max(0, y1):y2, max(0, x1):x2]
            scores.append(
                (
                    float(subj.mean()) if subj.size else 0.0,
                    float(dist.mean()) if dist.size else 0.0,
                )
            )
        return scores

    def _variants(self, bbox: BBox, h: int, w: int) -> List[BBox]:
        x1, y1, x2, y2 = bbox
        bw, bh = max(8, x2 - x1), max(8, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        shifts = [-0.08, 0.0, 0.08]
        scales = [0.92, 1.0, 1.08]
        variants = []
        for sx in shifts:
            for sy in shifts:
                for scale in scales:
                    if sx == 0.0 and sy == 0.0 and scale == 1.0:
                        continue
                    nw = bw * scale
                    nh = bh * scale
                    ncx = cx + sx * bw
                    ncy = cy + sy * bh
                    nb = clamp_bbox(
                        (
                            int(round(ncx - nw / 2)),
                            int(round(ncy - nh / 2)),
                            int(round(ncx + nw / 2)),
                            int(round(ncy + nh / 2)),
                        ),
                        h,
                        w,
                    )
                    if nb[2] - nb[0] >= 8 and nb[3] - nb[1] >= 8:
                        variants.append(nb)
        return variants

    @staticmethod
    def _subject_integrity(
        bbox: BBox,
        detected_objects: List[DetectedObject],
        image_shape,
    ) -> float:
        if not detected_objects:
            return 0.5
        vals = []
        weights = []
        for obj in detected_objects:
            obj_area = max(1, bbox_area(obj.bbox))
            inter = max(0, min(bbox[2], obj.bbox[2]) - max(bbox[0], obj.bbox[0])) * max(
                0, min(bbox[3], obj.bbox[3]) - max(bbox[1], obj.bbox[1])
            )
            vals.append(inter / obj_area)
            weights.append(max(0.05, obj.confidence) * np.sqrt(obj_area))
        vals_arr = np.array(vals, dtype=np.float64)
        weights_arr = np.array(weights, dtype=np.float64)
        return float((vals_arr * weights_arr).sum() / (weights_arr.sum() + 1e-9))

    @staticmethod
    def _composition_proxy(
        bbox: BBox,
        saliency_map: Optional[np.ndarray],
        image_shape,
    ) -> float:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2.0) / max(1, w)
        cy = ((y1 + y2) / 2.0) / max(1, h)
        thirds = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
        thirds_score = max(np.exp(-((cx - tx) ** 2 + (cy - ty) ** 2) / (2 * 0.18 ** 2)) for tx, ty in thirds)
        center_score = np.exp(-((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / (2 * 0.28 ** 2))
        if saliency_map is None:
            return float(0.55 * thirds_score + 0.45 * center_score)
        region = saliency_map[max(0, y1):y2, max(0, x1):x2]
        density = float(region.mean()) if region.size else 0.0
        return float(np.clip(0.40 * thirds_score + 0.35 * center_score + 0.25 * density, 0.0, 1.0))
