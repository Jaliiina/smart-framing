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
    }
    values.update(_bbox_features(candidate.bbox, image_shape))
    values["fusion_x_area_prior"] = values["fusion_score"] * values["area_prior"]
    values["aesthetic_x_composition"] = values["aesthetic"] * values["composition"]
    values["saliency_x_center_balance"] = values["saliency"] * values["center_balance"]
    values["subject_x_area"] = values["subject"] * values["area"]
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
            return KNNReranker(
                train_features=np.array(data["train_features"], dtype=np.float64),
                train_targets=np.array(data["train_targets"], dtype=np.float64),
                mean=np.array(data["mean"], dtype=np.float64),
                scale=np.array(data["scale"], dtype=np.float64),
                feature_names=list(data["feature_names"]),
                k=int(data.get("k", 1)),
                blend_with_fusion=blend,
                takeover_margin=margin,
                protect_high_quality_fusion=protect,
            )
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
        for original, learned in zip(original_scores, learned_scores):
            if self.blend_with_fusion > 0:
                blended_scores.append(
                    (1.0 - self.blend_with_fusion) * learned
                    + self.blend_with_fusion * original
                )
            else:
                blended_scores.append(learned)

        best_idx = int(np.argmax(blended_scores))
        fusion_idx = 0
        learned_advantage = blended_scores[best_idx] - blended_scores[fusion_idx]
        if (
            best_idx != fusion_idx
            and learned_advantage < self.takeover_margin
        ) or self._is_protected_fusion_top(fusion_top, image_shape):
            for candidate, score in zip(candidates, original_scores):
                candidate.final_score = score
            return candidates

        ranked = []
        for candidate, score in zip(candidates, blended_scores):
            candidate.final_score = float(score)
            ranked.append(candidate)
        ranked.sort(key=lambda c: c.final_score, reverse=True)
        return ranked


@dataclass
class KNNReranker:
    """Case-based reranker using candidate examples labeled by IoU."""

    train_features: np.ndarray
    train_targets: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]
    k: int = 1
    blend_with_fusion: float = 0.0
    takeover_margin: float = 0.04
    protect_high_quality_fusion: bool = True

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
        x[:, 1:] = (x[:, 1:] - self.mean[1:]) / self.scale[1:]
        scores = []
        k = max(1, min(self.k, len(self.train_targets)))
        for row in x:
            dist = np.sqrt(((self.train_features - row) ** 2).sum(axis=1))
            nn_idx = np.argpartition(dist, k - 1)[:k]
            if k == 1:
                scores.append(float(self.train_targets[nn_idx[0]]))
            else:
                weights = 1.0 / (dist[nn_idx] + 1e-6)
                scores.append(float((weights * self.train_targets[nn_idx]).sum() / weights.sum()))
        return scores

    def rerank(
        self,
        candidates: List[CandidateResult],
        image_shape: Sequence[int],
    ) -> List[CandidateResult]:
        if not candidates:
            return candidates
        learned_scores = self.score_candidates(candidates, image_shape)
        ranked = []
        for candidate, learned in zip(candidates, learned_scores):
            if self.blend_with_fusion > 0:
                candidate.final_score = (
                    (1.0 - self.blend_with_fusion) * learned
                    + self.blend_with_fusion * candidate.final_score
                )
            else:
                candidate.final_score = learned
            ranked.append(candidate)
        ranked.sort(key=lambda c: c.final_score, reverse=True)
        return ranked
