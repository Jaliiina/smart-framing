"""CLIP-backed learned beauty preference judge for crop candidates.

The regular fusion stack scores interpretable signals. This module is the
subjective second-stage judge: it looks at the crop image, the crop in full
image context, and the existing candidate diagnostics, then applies a learned
pairwise preference model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from PIL import Image

from .reranker import FEATURE_NAMES as STRUCTURED_FEATURE_NAMES
from .reranker import candidate_feature_vector
from .utils import BBox, CandidateResult


VISUAL_PROMPTS = [
    "a beautiful well composed photograph",
    "a clean crop with a clear main subject",
    "a professional photographic crop",
    "a balanced crop with pleasing foreground and background",
    "a complete main subject inside the frame",
    "an elegant architecture or landscape composition",
    "a cluttered crop with distracting objects",
    "a crop containing trash garbage or plastic bag",
    "a crop dominated by empty ground or blank space",
    "a poorly framed partial object",
    "a crop that cuts off the main subject",
    "an accidental snapshot crop with messy edges",
]


PIXEL_FEATURE_NAMES = [
    "crop_edge_density",
    "crop_center_activity",
    "crop_border_activity",
    "crop_border_to_center_activity",
    "crop_blank_ratio",
    "crop_vivid_ratio",
    "crop_dark_ratio",
    "crop_bright_ratio",
    "crop_low_saturation_ratio",
    "crop_activity_balance_x",
    "crop_activity_balance_y",
    "outside_edge_density",
    "outside_vivid_ratio",
    "outside_blank_ratio",
    "outside_dark_ratio",
    "edge_contact",
    "area_ratio",
    "aspect_log_abs",
    "center_distance",
]


def _legacy_feature_vector(candidate: CandidateResult) -> list[float]:
    """Feature schema used by the older non-visual beauty_judge model."""
    sub = candidate.sub_scores
    boundary_clean = 1.0 - float(np.clip(sub.boundary_cut, 0.0, 1.0))
    artifact_mix = float(
        np.clip(
            0.45 * sub.visual_artifact_penalty
            + 0.25 * sub.blank_area_penalty
            + 0.20 * sub.small_saturated_object_penalty
            + 0.10 * max(sub.distractor_map_score, sub.distractor_penalty),
            0.0,
            1.0,
        )
    )
    return [
        1.0,
        float(sub.aesthetic),
        float(sub.saliency),
        float(sub.composition),
        float(sub.subject),
        float(sub.technical),
        float(sub.area_prior),
        float(sub.roi_discard),
        float(sub.roi_saliency),
        float(sub.discard_quality),
        float(sub.boundary_cut),
        float(sub.distractor_penalty),
        float(sub.semantic_score),
        float(sub.positive_semantic),
        float(sub.negative_semantic),
        float(sub.subjectness),
        float(sub.distractor_map_score),
        float(sub.good_discard),
        float(sub.bad_discard),
        float(sub.visual_artifact_penalty),
        float(sub.blank_area_penalty),
        float(sub.saturated_boundary_penalty),
        float(sub.small_saturated_object_penalty),
        float(sub.person_completeness),
        float(sub.aesthetic) * float(sub.composition),
        float(sub.semantic_score) * float(sub.subjectness),
        float(sub.roi_discard) * boundary_clean,
        float(sub.good_discard) - float(sub.bad_discard),
        artifact_mix,
    ]


LEGACY_FEATURE_NAMES = [
    "bias",
    "aesthetic",
    "saliency",
    "composition",
    "subject",
    "technical",
    "area_prior",
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
    "person_completeness",
    "aesthetic_x_composition",
    "semantic_x_subjectness",
    "roi_x_boundary_clean",
    "good_minus_bad_discard",
    "artifact_mix",
]


def _safe_crop(image: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))
    return image[y1:y2, x1:x2]


def _context_view(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """Show the selected crop in full-image context to CLIP."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    view = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dim = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    view = cv2.addWeighted(dim, 0.68, view, 0.32, 0)
    view[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    thickness = max(2, int(round(min(h, w) * 0.006)))
    cv2.rectangle(view, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (255, 255, 255), thickness)
    return view


def _image_to_pil_rgb(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _pixel_features(image: np.ndarray, bbox: BBox) -> list[float]:
    h, w = image.shape[:2]
    crop = _safe_crop(image, bbox)
    ch, cw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    edges = cv2.Canny(gray, 70, 170).astype(np.float32) / 255.0

    strip = max(2, int(round(min(ch, cw) * 0.08)))
    border = np.zeros((ch, cw), dtype=bool)
    border[:strip, :] = True
    border[-strip:, :] = True
    border[:, :strip] = True
    border[:, -strip:] = True
    center = ~border
    center_activity = float(edges[center].mean()) if center.any() else float(edges.mean())
    border_activity = float(edges[border].mean()) if border.any() else 0.0

    activity = edges + 0.35 * sat
    total = float(activity.sum())
    if total > 1e-8:
        ys, xs = np.mgrid[0:ch, 0:cw]
        cx = float((xs * activity).sum() / total) / max(1, cw - 1)
        cy = float((ys * activity).sum() / total) / max(1, ch - 1)
    else:
        cx, cy = 0.5, 0.5

    x1, y1, x2, y2 = bbox
    outside_mask = np.ones((h, w), dtype=bool)
    outside_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = False
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    edges_full = cv2.Canny(gray_full, 70, 170).astype(np.float32) / 255.0
    outside = outside_mask
    if outside.any():
        out_sat = hsv_full[:, :, 1][outside] / 255.0
        out_val = hsv_full[:, :, 2][outside] / 255.0
        outside_edge = float(edges_full[outside].mean())
        outside_vivid = float(((out_sat > 0.55) & (out_val > 0.25)).mean())
        outside_blank = float(((out_sat < 0.18) & (out_val > 0.72)).mean())
        outside_dark = float((out_val < 0.18).mean())
    else:
        outside_edge = outside_vivid = outside_blank = outside_dark = 0.0

    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(h * w))
    aspect = (x2 - x1) / max(1.0, float(y2 - y1))
    center_dist = math.sqrt((((x1 + x2) / 2.0) / max(1, w) - 0.5) ** 2 + (((y1 + y2) / 2.0) / max(1, h) - 0.5) ** 2)
    edge_contact = float(x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1)

    return [
        float(edges.mean()),
        center_activity,
        border_activity,
        float(border_activity / max(1e-6, center_activity)),
        float(((sat < 0.18) & (val > 0.72)).mean()),
        float(((sat > 0.55) & (val > 0.25)).mean()),
        float((val < 0.18).mean()),
        float((val > 0.88).mean()),
        float((sat < 0.16).mean()),
        abs(cx - 0.5) * 2.0,
        abs(cy - 0.5) * 2.0,
        outside_edge,
        outside_vivid,
        outside_blank,
        outside_dark,
        edge_contact,
        float(area_ratio),
        abs(float(math.log(max(aspect, 1e-6)))),
        float(center_dist),
    ]


@dataclass
class BeautyFeatureExtractor:
    """Build visual and structured features for learned beauty ranking."""

    clip_model: str = "ViT-B/32"
    device: str = "cuda"
    projection_dim: int = 32
    projection_seed: int = 2026

    def __post_init__(self) -> None:
        self._clip = None
        self._model = None
        self._preprocess = None
        self._text_features = None
        self._projection = None
        self._embedding_dim = None

    @property
    def feature_names(self) -> list[str]:
        prompt_names = []
        for prefix in ("crop_prompt", "context_prompt"):
            prompt_names.extend(f"{prefix}_{idx:02d}" for idx in range(len(VISUAL_PROMPTS)))
        projection_names = []
        for prefix in ("crop_clip_proj", "context_clip_proj", "delta_clip_proj"):
            projection_names.extend(f"{prefix}_{idx:02d}" for idx in range(self.projection_dim))
        return (
            list(STRUCTURED_FEATURE_NAMES)
            + list(PIXEL_FEATURE_NAMES)
            + prompt_names
            + projection_names
        )

    def _load_clip(self) -> None:
        if self._model is not None:
            return
        import torch
        import clip

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._clip = clip
        self._model, self._preprocess = clip.load(self.clip_model, device=device)
        self._model.eval()
        self.device = device
        with torch.no_grad():
            tokens = clip.tokenize(VISUAL_PROMPTS).to(device)
            text = self._model.encode_text(tokens).float()
            text = text / text.norm(dim=-1, keepdim=True)
        self._text_features = text

    def _projection_matrix(self, dim: int) -> np.ndarray:
        if self._projection is None or self._embedding_dim != dim:
            rng = np.random.default_rng(self.projection_seed)
            self._projection = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(dim, self.projection_dim)).astype(np.float32)
            self._embedding_dim = dim
        return self._projection

    def _encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        self._load_clip()
        import torch

        if not images:
            return np.zeros((0, 1), dtype=np.float32)
        tensors = [self._preprocess(_image_to_pil_rgb(img)) for img in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)

    def build_matrix(
        self,
        image: np.ndarray,
        candidates: Sequence[CandidateResult],
        image_shape: Sequence[int] | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        if not candidates:
            return np.zeros((0, len(self.feature_names)), dtype=np.float64)
        shape = image_shape if image_shape is not None else image.shape[:2]
        structured = np.array(
            [candidate_feature_vector(candidate, shape) for candidate in candidates],
            dtype=np.float64,
        )
        pixel = np.array([_pixel_features(image, candidate.bbox) for candidate in candidates], dtype=np.float64)
        crop_images = [_safe_crop(image, candidate.bbox) for candidate in candidates]
        context_images = [_context_view(image, candidate.bbox) for candidate in candidates]

        crop_embeds = []
        context_embeds = []
        for start in range(0, len(candidates), batch_size):
            crop_embeds.append(self._encode_images(crop_images[start:start + batch_size]))
            context_embeds.append(self._encode_images(context_images[start:start + batch_size]))
        crop_embed = np.vstack(crop_embeds)
        context_embed = np.vstack(context_embeds)

        text = self._text_features.detach().cpu().numpy().astype(np.float32)
        crop_prompts = crop_embed @ text.T
        context_prompts = context_embed @ text.T

        proj = self._projection_matrix(crop_embed.shape[1])
        crop_proj = crop_embed @ proj
        context_proj = context_embed @ proj
        delta_proj = (crop_embed - context_embed) @ proj

        return np.hstack(
            [
                structured,
                pixel,
                crop_prompts.astype(np.float64),
                context_prompts.astype(np.float64),
                crop_proj.astype(np.float64),
                context_proj.astype(np.float64),
                delta_proj.astype(np.float64),
            ]
        )


class BeautyJudge:
    """Learned beauty scoring head over visual and candidate signals."""

    def __init__(
        self,
        coefficients: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        feature_names: list[str],
        blend_with_fusion: float = 0.0,
        takeover_margin: float = 0.02,
        top_n: int = 80,
        extractor: BeautyFeatureExtractor | None = None,
        legacy: bool = False,
    ):
        self.coefficients = coefficients
        self.mean = mean
        self.scale = np.where(scale < 1e-9, 1.0, scale)
        self.feature_names = feature_names
        self.blend_with_fusion = blend_with_fusion
        self.takeover_margin = takeover_margin
        self.top_n = top_n
        self.extractor = extractor
        self.legacy = legacy

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        blend_with_fusion: float | None = None,
        takeover_margin: float | None = None,
        top_n: int | None = None,
    ):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model_type = data.get("type", "")
        legacy = model_type.startswith("beauty_judge") and "clip_model" not in data
        blend = (
            float(blend_with_fusion)
            if blend_with_fusion is not None
            else float(data.get("blend_with_fusion", 0.0))
        )
        margin = (
            float(takeover_margin)
            if takeover_margin is not None
            else float(data.get("takeover_margin", 0.02))
        )
        limit = int(top_n if top_n is not None else data.get("top_n", 80))
        extractor = None
        if not legacy:
            extractor = BeautyFeatureExtractor(
                clip_model=str(data.get("clip_model", "ViT-B/32")),
                device=str(data.get("device", "cuda")),
                projection_dim=int(data.get("projection_dim", 32)),
                projection_seed=int(data.get("projection_seed", 2026)),
            )
        return cls(
            coefficients=np.array(data["coefficients"], dtype=np.float64),
            mean=np.array(data["mean"], dtype=np.float64),
            scale=np.array(data["scale"], dtype=np.float64),
            feature_names=list(data["feature_names"]),
            blend_with_fusion=blend,
            takeover_margin=margin,
            top_n=limit,
            extractor=extractor,
            legacy=legacy,
        )

    def _score_matrix(self, x: np.ndarray) -> np.ndarray:
        x_norm = x.copy()
        if x_norm.shape[1] != len(self.coefficients):
            raise ValueError(
                f"Beauty judge feature mismatch: got {x_norm.shape[1]}, expected {len(self.coefficients)}"
            )
        x_norm[:, 1:] = (x_norm[:, 1:] - self.mean[1:]) / self.scale[1:]
        return x_norm @ self.coefficients

    def score_candidates(
        self,
        candidates: List[CandidateResult],
        image: np.ndarray | None = None,
        image_shape: Sequence[int] | None = None,
    ) -> List[float]:
        if not candidates:
            return []
        if self.legacy:
            rows = [_legacy_feature_vector(candidate) for candidate in candidates]
            return [float(v) for v in self._score_matrix(np.array(rows, dtype=np.float64))]
        if image is None:
            raise ValueError("Visual beauty judge requires the source image.")
        rows = self.extractor.build_matrix(image, candidates, image_shape=image_shape)
        return [float(v) for v in self._score_matrix(rows)]

    def rerank(
        self,
        candidates: List[CandidateResult],
        image: np.ndarray | None = None,
        image_shape: Sequence[int] | None = None,
    ) -> List[CandidateResult]:
        if not candidates:
            return candidates

        head = list(candidates[: min(self.top_n, len(candidates))])
        tail = list(candidates[len(head):])
        original_scores = [candidate.final_score for candidate in head]
        beauty_scores = self.score_candidates(head, image=image, image_shape=image_shape)
        raw = np.array(beauty_scores, dtype=np.float64)
        if raw.size and (raw.max() - raw.min()) > 1e-9:
            normalized = (raw - raw.min()) / (raw.max() - raw.min())
        else:
            normalized = np.full_like(raw, 0.5)

        blended_scores = []
        for original, beauty in zip(original_scores, normalized):
            if self.blend_with_fusion > 0:
                score = (
                    (1.0 - self.blend_with_fusion) * float(beauty)
                    + self.blend_with_fusion * float(original)
                )
            else:
                score = float(beauty)
            blended_scores.append(float(np.clip(score, 0.0, 1.0)))

        best_idx = int(np.argmax(blended_scores))
        if best_idx != 0 and blended_scores[best_idx] < blended_scores[0] + self.takeover_margin:
            best_idx = 0

        ranked = []
        for candidate, score in zip(head, blended_scores):
            candidate.final_score = score
            ranked.append(candidate)
        ranked.sort(key=lambda c: c.final_score, reverse=True)
        if best_idx == 0 and ranked and ranked[0] is not head[0]:
            ranked = [head[0]] + [candidate for candidate in ranked if candidate is not head[0]]
        return ranked + tail


def train_pairwise_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a linear reward model from pairwise feature differences."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_norm = x.copy()
    x_norm[:, 1:] = (x[:, 1:] - mean[1:]) / scale[1:]
    reg = np.eye(x_norm.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    system = x_norm.T @ x_norm + reg
    rhs = x_norm.T @ y
    try:
        coef = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return coef, mean, scale


def score_with_model(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.where(scale < 1e-9, 1.0, scale)
    x_norm = x.copy()
    x_norm[:, 1:] = (x[:, 1:] - mean[1:]) / scale[1:]
    return x_norm @ coef
