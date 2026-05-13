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
        total_w = w_aesthetic + w_saliency + w_composition + w_subject + w_technical
        if total_w > 0:
            w_aesthetic /= total_w
            w_saliency /= total_w
            w_composition /= total_w
            w_subject /= total_w
            w_technical /= total_w

        # --- Normalize each score dimension per-image ---
        norm_aesthetic = minmax_normalize(np.array(aesthetic_scores, dtype=np.float64))
        norm_saliency = minmax_normalize(np.array(saliency_scores, dtype=np.float64))

        comp_totals = np.array([s[0] for s in composition_scores], dtype=np.float64)
        norm_composition = minmax_normalize(comp_totals)

        tech_totals = np.array([s[0] for s in technical_scores], dtype=np.float64)
        norm_technical = minmax_normalize(tech_totals)

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

        # Top-K for display
        top_k = candidates[: self.top_k_display]

        return best, top_k

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
