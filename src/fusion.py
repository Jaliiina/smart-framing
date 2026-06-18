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

        # Dual saliency agreement detection
        self.dual_saliency_agreement_threshold: float = fcfg.get(
            "dual_saliency_agreement_threshold", 0.65
        )
        self.dual_saliency_reduction: float = fcfg.get(
            "dual_saliency_weight_reduction", 0.10
        )

        # Stage 1 边界惩罚配置
        self._filter_config: dict = fcfg.get("stage1_filter", {})
        self._filter_config.setdefault("boundary_threshold", 0.25)
        self._filter_config.setdefault("boundary_penalty_strength", 0.15)

        # High-saliency coverage bonus: boost candidates that contain a lot of salient pixels
        self._saliency_bonus_weight: float = fcfg.get("saliency_coverage_bonus_weight", 0.15)
        self._saliency_bonus_threshold: float = fcfg.get("saliency_coverage_threshold", 0.5)

        # Edge-adjacency penalty: penalize candidates that hug image boundaries
        self._edge_penalty_enabled: bool = fcfg.get("edge_adjacency_penalty_enabled", True)
        self._edge_penalty_threshold: float = fcfg.get("edge_adjacency_penalty_threshold", 0.04)
        self._edge_penalty_strength: float = fcfg.get("edge_adjacency_penalty_strength", 0.20)

        # When saliency is concentrated at bottom (e.g. grass/ground),
        # give a bonus to candidates whose saliency center-of-mass is higher (toward sky).
        # This prevents bottom-texture (grass, ground) from hijacking the crop for sky scenes.
        self._saliency_vertical_bias_enabled: bool = fcfg.get("saliency_vertical_bias_enabled", True)
        self._saliency_vertical_bias_strength: float = fcfg.get("saliency_vertical_bias_strength", 0.12)

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
        return_all: bool = False,
        content_center: Optional[Tuple[float, float]] = None,
        content_region: Optional[BBox] = None,
        saliency_map: Optional[np.ndarray] = None,
        subject_source: str = "yolo",
        dual_saliency_scores: Optional[List[float]] = None,
    ) -> Tuple[CandidateResult, List[CandidateResult], Optional[List[CandidateResult]]]:
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
            saliency_map: (H, W) saliency map for Stage 1 hard filtering.

        Returns:
            (best_candidate, top_k_candidates, all_candidates_or_None)
        """
        n = len(bboxes)
        if n == 0:
            raise ValueError("No candidates to fuse.")

        import logging
        _stage1_logger = logging.getLogger("fusion.stage1")

        # --- Compute edge-adjacency penalty (penalize candidates hugging image boundaries) ---
        edge_penalty = np.zeros(n)
        if saliency_map is not None and self._edge_penalty_enabled and image_shape is not None:
            h, w = image_shape[:2]
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                # Fractional distance from each edge to image boundary (0 = touching, 1 = far)
                left_dist_pct = x1 / max(1, w)
                top_dist_pct = y1 / max(1, h)
                right_dist_pct = (w - x2) / max(1, w)
                bottom_dist_pct = (h - y2) / max(1, h)
                # Any edge within threshold fraction of boundary → penalty
                for dist_pct in [left_dist_pct, top_dist_pct, right_dist_pct, bottom_dist_pct]:
                    if dist_pct < self._edge_penalty_threshold:
                        strength = self._edge_penalty_strength * (1.0 - dist_pct / self._edge_penalty_threshold)
                        edge_penalty[i] = max(edge_penalty[i], strength)

        # --- Compute saliency vertical-COM bias ---
        # When saliency is concentrated at the bottom (e.g. ground/grass in landscape photos),
        # give a bonus to candidates whose internal saliency COM is higher (toward sky).
        vertical_bias = np.zeros(n)
        if saliency_map is not None and self._saliency_vertical_bias_enabled and image_shape is not None:
            h, w = image_shape[:2]
            # Overall saliency COM (Y coordinate as fraction of height)
            total_sal = float(saliency_map.sum()) + 1e-9
            ys_full, xs_full = np.mgrid[0:h, 0:w]
            overall_com_y_pct = float((ys_full * saliency_map).sum()) / total_sal / h
            # If overall saliency is bottom-heavy (COM in lower 60%), apply bias
            if overall_com_y_pct > 0.40:
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    region = saliency_map[max(0, y1):y2, max(0, x1):x2]
                    reg_sum = float(region.sum()) + 1e-9
                    if region.size > 0:
                        local_h, local_w = region.shape[:2]
                        ys_local = np.mgrid[0:local_h, 0:local_w][0]
                        local_com_y_pct = float((ys_local * region).sum()) / reg_sum / max(1, local_h)
                        # Bonus for crops whose internal saliency is in upper portion (sky)
                        vertical_bias[i] = self._saliency_vertical_bias_strength * local_com_y_pct

        # --- Compute boundary penalty (soft penalty for edge contamination) ---
        boundary_penalty = np.zeros(n)
        if saliency_map is not None and saliency_is_uniform is False:
            boundary_th = self._filter_config.get("boundary_threshold", 0.25)
            for i, bbox in enumerate(bboxes):
                boundary_sal = self._compute_boundary_saliency(bbox, saliency_map)
                # 线性惩罚：超过阈值越多，扣分越多
                if boundary_sal > boundary_th:
                    penalty_strength = self._filter_config.get("boundary_penalty_strength", 0.15)
                    exceed_ratio = min(1.0, (boundary_sal - boundary_th) / (boundary_th * 2))
                    boundary_penalty[i] = penalty_strength * exceed_ratio

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

        # If no meaningful subject was detected, exclude that dimension.
        # The remaining active weights are normalized below. Avoid transferring
        # subject weight to saliency because low-level saliency can overvalue
        # textured ground, debris, or other high-contrast distractions.
        if not has_subject:
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

        # Apply vertical bias, edge-adjacency and boundary penalties
        final_scores = final_scores + vertical_bias - edge_penalty - boundary_penalty

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
                # Person & composition enhancement
                person_completeness=composition_scores[i][1].get("person_completeness", 0.5)
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

        if return_all:
            return best, top_k, candidates
        return best, top_k, None

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

    def _compute_boundary_saliency(self, bbox: BBox, saliency_map: np.ndarray) -> float:
        """计算框边缘的saliency均值，越低说明切得越干净。

        Args:
            bbox: (x1, y1, x2, y2)
            saliency_map: (H, W) saliency map

        Returns:
            边缘strip的平均saliency值，越低越好
        """
        x1, y1, x2, y2 = bbox
        h, w = saliency_map.shape[:2]

        strip_w = max(2, min(h, w) // 50)
        boundary_sal = 0.0
        boundary_count = 0

        # Top strip
        if y2 - y1 > 2 * strip_w and x2 - x1 > 0:
            boundary_sal += float(saliency_map[y1:min(y1 + strip_w, h), max(0, x1):min(x2, w)].sum())
            boundary_count += strip_w * (x2 - x1)
        # Bottom strip
        if y2 - y1 > 2 * strip_w and x2 - x1 > 0:
            boundary_sal += float(saliency_map[max(0, y2 - strip_w):y2, max(0, x1):min(x2, w)].sum())
            boundary_count += strip_w * (x2 - x1)
        # Left strip
        if x2 - x1 > 2 * strip_w and y2 - y1 > 0:
            boundary_sal += float(saliency_map[max(0, y1):min(y2, h), x1:min(x1 + strip_w, w)].sum())
            boundary_count += (y2 - y1) * strip_w
        # Right strip
        if x2 - x1 > 2 * strip_w and y2 - y1 > 0:
            boundary_sal += float(saliency_map[max(0, y1):min(y2, h), max(0, x2 - strip_w):x2].sum())
            boundary_count += (y2 - y1) * strip_w

        return boundary_sal / max(1, boundary_count)

    @staticmethod
    def _compute_rank_agreement(
        scores_a: List[float], scores_b: List[float]
    ) -> float:
        """Compute normalized rank correlation (Spearman) without scipy.

        Returns value in [0, 1] where 1 = perfect agreement.
        """
        n = len(scores_a)
        if n < 2:
            return 0.0
        a = np.array(scores_a, dtype=np.float64)
        b = np.array(scores_b, dtype=np.float64)
        # Handle ties by averaging ranks
        rank_a = _rank_data(a)
        rank_b = _rank_data(b)
        # Pearson on ranks ≈ Spearman
        da = rank_a - rank_a.mean()
        db = rank_b - rank_b.mean()
        num = float((da * db).sum())
        den = float(np.sqrt((da ** 2).sum()) * np.sqrt((db ** 2).sum()))
        if den < 1e-9:
            return 0.0
        rho = num / den
        if np.isnan(rho):
            return 0.0
        return float((rho + 1.0) / 2.0)

    def _compute_high_threshold_coverage(
        self,
        bbox: BBox,
        saliency_map: np.ndarray,
        threshold: Optional[float] = None,
    ) -> float:
        """计算高阈值覆盖率（saliency > threshold 的像素占比）。

        Args:
            bbox: (x1, y1, x2, y2)
            saliency_map: (H, W) saliency map
            threshold: 高阈值，默认从配置读取

        Returns:
            高阈值覆盖率（框内高saliency像素占比），越高越好
        """
        if threshold is None:
            threshold = self._filter_config.get("high_saliency_threshold", 0.5)

        x1, y1, x2, y2 = bbox
        region = saliency_map[max(0, y1):y2, max(0, x1):x2]
        if region.size == 0:
            return 0.0

        high_sal_mask = region > threshold
        return float(high_sal_mask.sum()) / region.size

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
