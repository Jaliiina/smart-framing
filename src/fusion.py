"""Normalized weighted fusion with fallback strategies."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import BBox, CandidateResult, SubScores, minmax_normalize


class FusionModule:
    """Fuse multi-dimensional scores with normalization and fallback strategies."""

    def __init__(self, config: dict):
        fcfg = config.get("fusion", {})
        w = fcfg.get("weights", {})
        self.weight_aesthetic: float = w.get("aesthetic", 0.25)
        self.weight_saliency: float = w.get("saliency", 0.25)
        self.weight_composition: float = w.get("composition", 0.20)
        self.weight_subject: float = w.get("subject", 0.20)
        self.weight_technical: float = w.get("technical", 0.10)
        self.weight_area_prior: float = w.get("area_prior", 0.0)

        area_cfg = fcfg.get("area_prior", {})
        ideal_range = area_cfg.get("ideal_range", [0.25, 0.60])
        self.area_ideal_min: float = float(ideal_range[0])
        self.area_ideal_max: float = float(ideal_range[1])
        self.large_penalty_start: float = area_cfg.get("large_penalty_start", 0.70)
        self.max_allowed_without_reason: float = area_cfg.get(
            "max_allowed_without_reason", 0.85
        )

        self.saliency_uniform_std: float = fcfg.get("saliency_uniform_std_threshold", 0.05)
        self.saliency_uniform_reduction: float = fcfg.get("saliency_uniform_weight_reduction", 0.10)
        self.low_score_threshold: float = fcfg.get("low_score_threshold", 0.3)
        self.top_k_display: int = fcfg.get("top_k_display", 3)

    def fuse(
        self,
        bboxes: List[BBox],
        aesthetic_scores: List[float],
        saliency_scores: List[float],
        composition_scores: List[Tuple[float, Dict[str, float]]],
        subject_scores: List[Optional[float]],
        technical_scores: List[Tuple[float, Dict[str, float]]],
        saliency_is_uniform: bool = False,
        has_subject: bool = True,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[CandidateResult, List[CandidateResult]]:
        """Fuse all sub-scores and select the best candidate.

        Args:
            bboxes: List of candidate bboxes.
            aesthetic_scores: Raw aesthetic scores per candidate.
            saliency_scores: Saliency preservation scores per candidate.
            composition_scores: (total, sub_dict) per candidate.
            subject_scores: Subject completeness scores per candidate (None = no objects).
            technical_scores: (total, sub_dict) per candidate.
            saliency_is_uniform: If True, saliency map was too uniform.
            has_subject: If True, at least some objects were detected.

        Returns:
            (best_candidate, top_k_candidates): Best result and top-K display list.
        """
        n = len(bboxes)
        if n == 0:
            raise ValueError("No candidates to fuse.")

        # --- Determine active weights ---
        w_aesthetic = self.weight_aesthetic
        w_saliency = self.weight_saliency
        w_composition = self.weight_composition
        w_subject = self.weight_subject
        w_technical = self.weight_technical
        w_area_prior = self.weight_area_prior

        # Fallback: if saliency is uniform, reduce its weight
        if saliency_is_uniform:
            w_saliency -= self.saliency_uniform_reduction
            w_aesthetic += self.saliency_uniform_reduction * 0.5
            w_composition += self.saliency_uniform_reduction * 0.5

        # Fallback: if no subject detected, redistribute subject weight
        if not has_subject:
            redistribute = w_subject / 2.0
            w_aesthetic += redistribute
            w_saliency += redistribute
            w_subject = 0.0

        # Normalize weights to sum to 1
        total_w = (
            w_aesthetic
            + w_saliency
            + w_composition
            + w_subject
            + w_technical
            + w_area_prior
        )
        if total_w > 0:
            w_aesthetic /= total_w
            w_saliency /= total_w
            w_composition /= total_w
            w_subject /= total_w
            w_technical /= total_w
            w_area_prior /= total_w

        # --- Normalize each score dimension per-image ---
        norm_aesthetic = minmax_normalize(np.array(aesthetic_scores, dtype=np.float64))
        norm_saliency = minmax_normalize(np.array(saliency_scores, dtype=np.float64))

        comp_totals = np.array([s[0] for s in composition_scores], dtype=np.float64)
        norm_composition = minmax_normalize(comp_totals)

        tech_totals = np.array([s[0] for s in technical_scores], dtype=np.float64)
        norm_technical = minmax_normalize(tech_totals)
        area_prior = np.array(
            [self._area_prior_score(b, image_shape) for b in bboxes],
            dtype=np.float64,
        )

        # Subject scores: handle None
        subject_arr = np.array(
            [s if s is not None else 0.0 for s in subject_scores],
            dtype=np.float64,
        )
        if has_subject:
            norm_subject = minmax_normalize(subject_arr)
        else:
            norm_subject = np.zeros(n)

        # --- Weighted fusion ---
        final_scores = (
            w_aesthetic * norm_aesthetic
            + w_saliency * norm_saliency
            + w_composition * norm_composition
            + w_subject * norm_subject
            + w_technical * norm_technical
            + w_area_prior * area_prior
        )

        # --- Build results ---
        candidates = []
        for i in range(n):
            sub = SubScores(
                aesthetic=float(norm_aesthetic[i]),
                saliency=float(norm_saliency[i]),
                composition=float(norm_composition[i]),
                subject=float(norm_subject[i]) if has_subject else 0.0,
                technical=float(norm_technical[i]),
                area_prior=float(area_prior[i]),
                # Detailed breakdown
                thirds=composition_scores[i][1].get("thirds", 0.0),
                center_balance=composition_scores[i][1].get("center_balance", 0.0),
                whitespace=composition_scores[i][1].get("whitespace", 0.0),
                edge_simplicity=composition_scores[i][1].get("edge_simplicity", 0.0),
                symmetry=composition_scores[i][1].get("symmetry", 0.0),
                sharpness=technical_scores[i][1].get("sharpness", 0.0),
                brightness=technical_scores[i][1].get("brightness", 0.0),
                contrast=technical_scores[i][1].get("contrast", 0.0),
                saturation=technical_scores[i][1].get("saturation", 0.0),
            )
            candidates.append(
                CandidateResult(
                    bbox=bboxes[i],
                    final_score=float(final_scores[i]),
                    sub_scores=sub,
                )
            )

        # Sort by final score descending
        candidates.sort(key=lambda c: c.final_score, reverse=True)

        best = candidates[0]

        # Fallback: if best score too low, consider a conservative large-area crop
        if best.final_score < self.low_score_threshold:
            # Find the candidate with the largest area
            largest = max(candidates, key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]))
            if largest.final_score > best.final_score * 0.8:
                best = largest

        best = self._apply_conservative_crop_fallback(
            best,
            candidates,
            saliency_is_uniform=saliency_is_uniform,
            has_subject=has_subject,
            image_shape=image_shape,
        )

        # Top-K for display
        top_k = [best]
        for cand in candidates:
            if cand != best:
                top_k.append(cand)
            if len(top_k) >= self.top_k_display:
                break

        return best, top_k

    def _area_ratio(
        self,
        bbox: BBox,
        image_shape: Optional[Tuple[int, int]],
    ) -> float:
        if image_shape is None:
            return self.area_ideal_min
        h, w = image_shape[:2]
        img_area = max(1, h * w)
        return ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / img_area

    def _area_prior_score(
        self,
        bbox: BBox,
        image_shape: Optional[Tuple[int, int]],
    ) -> float:
        area_ratio = self._area_ratio(bbox, image_shape)
        if self.area_ideal_min <= area_ratio <= self.area_ideal_max:
            return 1.0
        if area_ratio < self.area_ideal_min:
            span = max(1e-6, self.area_ideal_min)
            return max(0.0, 1.0 - (self.area_ideal_min - area_ratio) / span)
        if area_ratio <= self.large_penalty_start:
            span = max(1e-6, self.large_penalty_start - self.area_ideal_max)
            return max(0.0, 1.0 - 0.35 * (area_ratio - self.area_ideal_max) / span)
        span = max(1e-6, 1.0 - self.large_penalty_start)
        return max(0.0, 0.65 - 0.65 * (area_ratio - self.large_penalty_start) / span)

    def _apply_conservative_crop_fallback(
        self,
        best: CandidateResult,
        candidates: List[CandidateResult],
        saliency_is_uniform: bool,
        has_subject: bool,
        image_shape: Optional[Tuple[int, int]],
    ) -> CandidateResult:
        if saliency_is_uniform:
            return best

        best_area = self._area_ratio(best.bbox, image_shape)
        if best_area <= 0.80:
            return best

        for cand in candidates[:10]:
            area = self._area_ratio(cand.bbox, image_shape)
            subject_ok = (not has_subject) or cand.sub_scores.subject >= 0.70
            score_close = cand.final_score >= best.final_score * 0.92
            if 0.25 <= area <= 0.65 and subject_ok and score_close:
                return cand
        return best

    def grid_search_weights(
        self,
        bboxes_list: List[List[BBox]],
        gt_bboxes: List[BBox],
        score_fn,
        weight_ranges: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Grid search for optimal fusion weights on a validation set.

        Args:
            bboxes_list: List of candidate bbox lists per image.
            gt_bboxes: Ground truth bboxes.
            score_fn: Function that takes (bboxes, weights) -> (best_bbox, ...).
            weight_ranges: Optional dict of weight ranges to search.

        Returns:
            Best weight configuration.
        """
        from .utils import bbox_iou

        if weight_ranges is None:
            weight_ranges = {
                "aesthetic": [0.15, 0.25, 0.35],
                "saliency": [0.15, 0.25, 0.35],
                "composition": [0.10, 0.20, 0.30],
                "subject": [0.10, 0.20, 0.30],
                "technical": [0.05, 0.10, 0.15],
            }

        best_weights = None
        best_miou = -1.0

        # Generate all combinations
        import itertools

        keys = list(weight_ranges.keys())
        value_lists = [weight_ranges[k] for k in keys]

        for combo in itertools.product(*value_lists):
            weights = dict(zip(keys, combo))
            # Normalize to sum to 1
            total = sum(weights.values())
            if total < 1e-9:
                continue
            weights = {k: v / total for k, v in weights.items()}

            # Evaluate
            ious = []
            for i, (cands, gt) in enumerate(zip(bboxes_list, gt_bboxes)):
                pred_bbox, _ = score_fn(cands, weights)
                ious.append(bbox_iou(pred_bbox, gt))

            miou = float(np.mean(ious))
            if miou > best_miou:
                best_miou = miou
                best_weights = weights

        return best_weights or {}
