"""Lightweight learned candidate reranker.

The normal fusion module is still responsible for generating interpretable
candidate scores. This module is an optional second-stage calibration layer
trained from testA candidate diagnostics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .utils import BBox, CandidateResult


FEATURE_NAMES = [
    "bias",
    "fusion_score",
    "aesthetic",
    "saliency",
    "composition",
    "subject",
    "technical",
    "area_prior",
    "thirds",
    "center_balance",
    "whitespace",
    "edge_simplicity",
    "symmetry",
    "sharpness",
    "brightness",
    "contrast",
    "saturation",
    "person_completeness",
    "roi_discard",
    "roi_saliency",
    "discard_quality",
    "boundary_cut",
    "distractor_penalty",
    "semantic_score",
    "positive_semantic",
    "negative_semantic",
    "subjectness",
    "distractor_map_score",
    "good_discard",
    "bad_discard",
    "visual_artifact_penalty",
    "blank_area_penalty",
    "saturated_boundary_penalty",
    "small_saturated_object_penalty",
    "cx",
    "cy",
    "width",
    "height",
    "area",
    "aspect_log",
    "center_distance",
    "top_margin",
    "bottom_margin",
    "left_margin",
    "right_margin",
    "fusion_x_area_prior",
    "aesthetic_x_composition",
    "saliency_x_center_balance",
    "subject_x_area",
    "roi_x_discard",
    "boundary_x_saliency",
    "semantic_x_subjectness",
    "distractor_x_boundary",
    "subjectness_x_area",
    "semantic_minus_distractor",
]


def _bbox_features(bbox: BBox, image_shape: Sequence[int]) -> dict[str, float]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    cx = ((x1 + x2) / 2.0) / max(1.0, float(w))
    cy = ((y1 + y2) / 2.0) / max(1.0, float(h))
    width = bw / max(1.0, float(w))
    height = bh / max(1.0, float(h))
    return {
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_log": math.log(max(1e-6, width / max(height, 1e-6))),
        "center_distance": math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2),
        "top_margin": y1 / max(1.0, float(h)),
        "bottom_margin": (h - y2) / max(1.0, float(h)),
        "left_margin": x1 / max(1.0, float(w)),
        "right_margin": (w - x2) / max(1.0, float(w)),
    }


def candidate_feature_vector(
    candidate: CandidateResult,
    image_shape: Sequence[int],
) -> list[float]:
    """Build a stable numeric feature vector for one candidate."""
    sub = candidate.sub_scores
    values = {
        "bias": 1.0,
        "fusion_score": float(candidate.final_score),
        "aesthetic": float(sub.aesthetic),
        "saliency": float(sub.saliency),
        "composition": float(sub.composition),
        "subject": float(sub.subject),
        "technical": float(sub.technical),
        "area_prior": float(sub.area_prior),
        "thirds": float(sub.thirds),
        "center_balance": float(sub.center_balance),
        "whitespace": float(sub.whitespace),
        "edge_simplicity": float(sub.edge_simplicity),
        "symmetry": float(sub.symmetry),
        "sharpness": float(sub.sharpness),
        "brightness": float(sub.brightness),
        "contrast": float(sub.contrast),
        "saturation": float(sub.saturation),
        "person_completeness": float(sub.person_completeness),
        "roi_discard": float(sub.roi_discard),
        "roi_saliency": float(sub.roi_saliency),
        "discard_quality": float(sub.discard_quality),
        "boundary_cut": float(sub.boundary_cut),
        "distractor_penalty": float(sub.distractor_penalty),
        "semantic_score": float(sub.semantic_score),
        "positive_semantic": float(sub.positive_semantic),
        "negative_semantic": float(sub.negative_semantic),
        "subjectness": float(sub.subjectness),
        "distractor_map_score": float(sub.distractor_map_score),
        "good_discard": float(sub.good_discard),
        "bad_discard": float(sub.bad_discard),
        "visual_artifact_penalty": float(sub.visual_artifact_penalty),
        "blank_area_penalty": float(sub.blank_area_penalty),
        "saturated_boundary_penalty": float(sub.saturated_boundary_penalty),
        "small_saturated_object_penalty": float(sub.small_saturated_object_penalty),
    }
    values.update(_bbox_features(candidate.bbox, image_shape))
    values["fusion_x_area_prior"] = values["fusion_score"] * values["area_prior"]
    values["aesthetic_x_composition"] = values["aesthetic"] * values["composition"]
    values["saliency_x_center_balance"] = values["saliency"] * values["center_balance"]
    values["subject_x_area"] = values["subject"] * values["area"]
    values["roi_x_discard"] = values["roi_saliency"] * values["discard_quality"]
    values["boundary_x_saliency"] = values["boundary_cut"] * values["saliency"]
    values["semantic_x_subjectness"] = values["semantic_score"] * values["subjectness"]
    values["distractor_x_boundary"] = values["distractor_map_score"] * values["boundary_cut"]
    values["subjectness_x_area"] = values["subjectness"] * values["area"]
    values["semantic_minus_distractor"] = values["semantic_score"] - values["distractor_map_score"]
    return [values[name] for name in FEATURE_NAMES]


@dataclass
class LearnedReranker:
    """Apply a ridge-regression candidate quality model."""

    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]
    blend_with_fusion: float = 0.0
    takeover_margin: float = 0.04
    protect_high_quality_fusion: bool = True
    protect_fusion_score_threshold: float = 0.80
    large_area_takeover_threshold: float = 0.50

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        blend_with_fusion: float | None = None,
        takeover_margin: float | None = None,
        protect_high_quality_fusion: bool | None = None,
        protect_fusion_score_threshold: float | None = None,
        large_area_takeover_threshold: float | None = None,
    ):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        blend = (
            float(blend_with_fusion)
            if blend_with_fusion is not None
            else float(data.get("blend_with_fusion", 0.0))
        )
        margin = (
            float(takeover_margin)
            if takeover_margin is not None
            else float(data.get("takeover_margin", 0.04))
        )
        protect = (
            bool(protect_high_quality_fusion)
            if protect_high_quality_fusion is not None
            else bool(data.get("protect_high_quality_fusion", True))
        )
        score_threshold = (
            float(protect_fusion_score_threshold)
            if protect_fusion_score_threshold is not None
            else float(data.get("protect_fusion_score_threshold", 0.80))
        )
        area_threshold = (
            float(large_area_takeover_threshold)
            if large_area_takeover_threshold is not None
            else float(data.get("large_area_takeover_threshold", 0.50))
        )
        if data.get("type") == "knn_candidate_reranker":
            raise ValueError("KNN reranker is deprecated; retrain with pairwise ridge.")
        if len(data.get("feature_names", [])) != len(FEATURE_NAMES):
            raise ValueError("Reranker feature schema mismatch; retrain the model.")
        return cls(
            coefficients=np.array(data["coefficients"], dtype=np.float64),
            mean=np.array(data["mean"], dtype=np.float64),
            scale=np.array(data["scale"], dtype=np.float64),
            feature_names=list(data["feature_names"]),
            blend_with_fusion=blend,
            takeover_margin=margin,
            protect_high_quality_fusion=protect,
            protect_fusion_score_threshold=score_threshold,
            large_area_takeover_threshold=area_threshold,
        )

    def _area_ratio(self, candidate: CandidateResult, image_shape: Sequence[int]) -> float:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = candidate.bbox
        return (
            max(1.0, float(x2 - x1))
            * max(1.0, float(y2 - y1))
            / max(1.0, float(h * w))
        )

    def _is_protected_fusion_top(
        self,
        candidate: CandidateResult,
        image_shape: Sequence[int],
    ) -> bool:
        """Keep very strong hand-crafted fusion crops from being overruled."""
        if not self.protect_high_quality_fusion:
            return False
        sub = candidate.sub_scores
        area_ratio = self._area_ratio(candidate, image_shape)
        if (
            candidate.final_score >= self.protect_fusion_score_threshold
            and area_ratio <= self.large_area_takeover_threshold
        ):
            return True
        return (
            sub.aesthetic >= 0.97
            and sub.composition >= 0.88
            and sub.saliency >= 0.62
            and sub.area_prior >= 0.78
            and sub.subject <= 0.10
            and area_ratio <= self.large_area_takeover_threshold
        )

    def score_candidates(
        self,
        candidates: Iterable[CandidateResult],
        image_shape: Sequence[int],
    ) -> List[float]:
        rows = [
            candidate_feature_vector(candidate, image_shape)
            for candidate in candidates
        ]
        if not rows:
            return []
        x = np.array(rows, dtype=np.float64)
        # Bias is intentionally not standardized.
        x_norm = x.copy()
        x_norm[:, 1:] = (x[:, 1:] - self.mean[1:]) / self.scale[1:]
        return [float(v) for v in x_norm @ self.coefficients]

    def rerank(
        self,
        candidates: List[CandidateResult],
        image_shape: Sequence[int],
    ) -> List[CandidateResult]:
        if not candidates:
            return candidates
        fusion_top = candidates[0]
        original_scores = [candidate.final_score for candidate in candidates]
        learned_scores = self.score_candidates(candidates, image_shape)
        blended_scores = []
        for candidate, original, learned in zip(candidates, original_scores, learned_scores):
            if self.blend_with_fusion > 0:
                score = (
                    (1.0 - self.blend_with_fusion) * learned
                    + self.blend_with_fusion * original
                )
            else:
                score = learned

            sub = candidate.sub_scores
            blank_excess = max(0.0, float(sub.blank_area_penalty) - 0.45)
            artifact_excess = max(0.0, float(sub.visual_artifact_penalty) - 0.45)
            saturated_foreground_excess = max(
                0.0, float(sub.small_saturated_object_penalty) - 0.35
            )
            low_roi_blank = (
                0.75
                if sub.blank_area_penalty > 0.75
                and sub.roi_discard < 0.15
                else 0.0
            )
            low_info_blank = (
                0.35
                if sub.blank_area_penalty > 0.85
                and sub.roi_saliency < 0.15
                and sub.saliency < 0.15
                else 0.0
            )
            score -= (
                0.45 * blank_excess
                + 0.45 * artifact_excess
                + 0.50 * saturated_foreground_excess
                + low_roi_blank
                + low_info_blank
            )
            blended_scores.append(float(np.clip(score, 0.0, 1.0)))

        raw_best_idx = int(np.argmax(blended_scores))
        best_idx = self._apply_blank_safety_switch(
            candidates,
            blended_scores,
            raw_best_idx,
        )
        best_idx = self._apply_quality_safety_switch(
            candidates,
            blended_scores,
            best_idx,
        )
        safety_switched = best_idx != raw_best_idx
        if safety_switched:
            blended_scores[best_idx] = min(1.0, max(blended_scores) + 1e-6)
        fusion_idx = 0
        learned_advantage = blended_scores[best_idx] - blended_scores[fusion_idx]
        if (
            best_idx != fusion_idx
            and learned_advantage < self.takeover_margin
            and not safety_switched
        ) or self._is_protected_fusion_top(fusion_top, image_shape):
            for candidate, score in zip(candidates, original_scores):
                candidate.final_score = score
            return candidates

        ranked = []
        for candidate, score in zip(candidates, blended_scores):
            candidate.final_score = float(np.clip(score, 0.0, 1.0))
            ranked.append(candidate)
        ranked.sort(key=lambda c: c.final_score, reverse=True)
        return ranked

    @staticmethod
    def _apply_blank_safety_switch(
        candidates: List[CandidateResult],
        scores: List[float],
        best_idx: int,
    ) -> int:
        best = candidates[best_idx]
        best_sub = best.sub_scores
        if best_sub.blank_area_penalty < 0.75 or best_sub.roi_discard >= 0.45:
            return best_idx

        alternatives = []
        for idx, candidate in enumerate(candidates):
            sub = candidate.sub_scores
            if sub.blank_area_penalty > 0.55:
                continue
            if sub.visual_artifact_penalty > 0.50:
                continue
            if sub.roi_discard < best_sub.roi_discard + 0.15:
                continue
            if scores[idx] < scores[best_idx] * 0.45:
                continue
            alternatives.append((scores[idx], sub.roi_discard, idx))
        if not alternatives:
            return best_idx
        alternatives.sort(reverse=True)
        return alternatives[0][2]

    @staticmethod
    def _apply_quality_safety_switch(
        candidates: List[CandidateResult],
        scores: List[float],
        best_idx: int,
    ) -> int:
        best = candidates[best_idx]
        b = best.sub_scores
        best_score = max(1e-9, scores[best_idx])

        alternatives = []
        for idx, candidate in enumerate(candidates):
            if idx == best_idx:
                continue
            s = candidate.sub_scores
            rel = scores[idx] / best_score

            full_subject_gain = (
                b.subject < 0.55
                and s.subject >= 0.88
                and s.roi_discard >= b.roi_discard + 0.12
                and rel >= 0.70
                and s.blank_area_penalty <= max(0.40, b.blank_area_penalty + 0.10)
            )
            cleaner_roi_gain = (
                s.roi_discard >= b.roi_discard + 0.18
                and s.visual_artifact_penalty <= b.visual_artifact_penalty - 0.10
                and s.blank_area_penalty <= b.blank_area_penalty - 0.12
                and rel >= 0.88
            )
            saturated_tiebreak = (
                rel >= 0.94
                and s.subject + 0.05 >= b.subject
                and s.roi_discard + 0.03 >= b.roi_discard
                and s.small_saturated_object_penalty
                <= b.small_saturated_object_penalty - 0.015
                and s.visual_artifact_penalty <= b.visual_artifact_penalty + 0.02
            )
            no_subject_cleaner = (
                b.subject <= 0.05
                and b.blank_area_penalty + b.visual_artifact_penalty > 0.18
                and s.blank_area_penalty <= max(0.08, b.blank_area_penalty - 0.08)
                and s.visual_artifact_penalty <= max(0.08, b.visual_artifact_penalty - 0.04)
                and s.roi_discard >= b.roi_discard - 0.12
            )

            if full_subject_gain or cleaner_roi_gain or saturated_tiebreak or no_subject_cleaner:
                cleanliness = (
                    s.roi_discard
                    + 0.25 * s.subject
                    - 0.35 * s.blank_area_penalty
                    - 0.35 * s.visual_artifact_penalty
                    - 0.20 * s.small_saturated_object_penalty
                )
                alternatives.append((rel + 0.25 * cleanliness, idx))

        if not alternatives:
            return best_idx
        alternatives.sort(reverse=True)
        return alternatives[0][1]
