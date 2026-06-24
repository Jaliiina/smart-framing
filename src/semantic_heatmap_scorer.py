from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch

from .utils import BBox

logger = logging.getLogger(__name__)


@dataclass
class SemanticHeatmaps:
    positive: np.ndarray
    negative: np.ndarray
    subject: np.ndarray


class SemanticHeatmapScorer:

    def __init__(self, config: dict):
        cfg = config.get("semantic_heatmap", {})
        model_cfg = config.get("models", {}).get("aesthetic", {})
        sem_cfg = config.get("semantic_crop", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.device = model_cfg.get("device", "cpu")
        self.clip_model = cfg.get("clip_model", sem_cfg.get("clip_model", model_cfg.get("clip_model", "ViT-L/14")))
        self.window_ratios = [float(v) for v in cfg.get("window_ratios", [0.30, 0.42])]
        self.stride_ratio = float(cfg.get("stride_ratio", 0.18))
        self.context_top_k = max(1, int(cfg.get("context_top_k", sem_cfg.get("context_top_k", 3))))
        self.positive_prompts = cfg.get("positive_prompts", sem_cfg.get("positive_prompts", []))
        self.negative_prompts = cfg.get("negative_prompts", sem_cfg.get("negative_prompts", []))
        if not self.positive_prompts:
            self.positive_prompts = [
                "beautiful landscape photograph",
                "airplane in the sky",
                "pencil tips and hand drawing",
                "ocean waves scenic view",
                "complete building architecture",
                "illuminated tree canopy",
                "complete glass object",
            ]
        if not self.negative_prompts:
            self.negative_prompts = [
                "trash or garbage",
                "plastic bag",
                "orange bucket or plastic barrel",
                "cluttered random foreground",
                "rusty roof edge",
                "large blank white area",
                "cropped cut off object",
            ]
        self._clip = None
        self._model = None
        self._preprocess = None
        self._pos = None
        self._neg = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded or not self.enabled:
            return
        try:
            import clip  # type: ignore

            self._clip = clip
            self._model, self._preprocess = clip.load(self.clip_model, device=self.device)
            with torch.no_grad():
                pos_tokens = clip.tokenize(self.positive_prompts).to(self.device)
                neg_tokens = clip.tokenize(self.negative_prompts).to(self.device)
                pos = self._model.encode_text(pos_tokens).float()
                neg = self._model.encode_text(neg_tokens).float()
            self._pos = pos / pos.norm(dim=-1, keepdim=True)
            self._neg = neg / neg.norm(dim=-1, keepdim=True)
        except Exception as exc:
            logger.warning(f"Semantic heatmap scorer unavailable: {exc}")
        self._loaded = True

    def build_heatmaps(self, image: np.ndarray) -> SemanticHeatmaps:
        self._load()
        h, w = image.shape[:2]
        if (
            not self.enabled
            or self._model is None
            or self._preprocess is None
            or self._pos is None
            or self._neg is None
        ):
            zeros = np.zeros((h, w), dtype=np.float32)
            return SemanticHeatmaps(positive=zeros, negative=zeros, subject=zeros)

        windows = self._generate_windows(h, w)
        if not windows:
            zeros = np.zeros((h, w), dtype=np.float32)
            return SemanticHeatmaps(positive=zeros, negative=zeros, subject=zeros)

        context_idx = self._context_positive_indices(image)
        pos_map = np.zeros((h, w), dtype=np.float32)
        neg_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        rows = self._score_windows(image, windows, context_idx)
        pos_values = self._normalize_array(np.array([r[1] for r in rows], dtype=np.float32))
        neg_values = self._normalize_array(np.array([r[2] for r in rows], dtype=np.float32))
        for (bbox, _pos, _neg), pos_v, neg_v in zip(rows, pos_values, neg_values):
            x1, y1, x2, y2 = bbox
            weight = self._center_weight(y2 - y1, x2 - x1)
            pos_map[y1:y2, x1:x2] += float(pos_v) * weight
            neg_map[y1:y2, x1:x2] += float(neg_v) * weight
            count_map[y1:y2, x1:x2] += weight

        count_map = np.maximum(count_map, 1e-6)
        pos_map = self._smooth_norm(pos_map / count_map)
        neg_map = self._smooth_norm(neg_map / count_map)
        subject_map = self._smooth_norm(np.clip(pos_map - 0.65 * neg_map, 0.0, 1.0))
        return SemanticHeatmaps(
            positive=pos_map.astype(np.float32),
            negative=neg_map.astype(np.float32),
            subject=subject_map.astype(np.float32),
        )

    def _score_windows(
        self,
        image: np.ndarray,
        windows: List[BBox],
        context_idx: torch.Tensor,
    ) -> List[Tuple[BBox, float, float]]:
        from PIL import Image

        crops = []
        for x1, y1, x2, y2 in windows:
            crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crops.append(self._preprocess(Image.fromarray(crop)))

        rows = []
        chunk_size = 32
        for start in range(0, len(crops), chunk_size):
            chunk = torch.stack(crops[start:start + chunk_size]).to(self.device)
            with torch.no_grad():
                features = self._model.encode_image(chunk).float()
                features = features / features.norm(dim=-1, keepdim=True)
                pos = features @ self._pos.T
                neg = features @ self._neg.T
                pos_value = pos[:, context_idx].mean(dim=1)
                neg_value = neg.max(dim=1).values
            for local, bbox in enumerate(windows[start:start + chunk_size]):
                rows.append(
                    (
                        bbox,
                        float(pos_value[local].cpu()),
                        float(neg_value[local].cpu()),
                    )
                )
        return rows

    def _context_positive_indices(self, image: np.ndarray) -> torch.Tensor:
        from PIL import Image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self._model.encode_image(tensor).float()
            feature = feature / feature.norm(dim=-1, keepdim=True)
            sims = (feature @ self._pos.T).squeeze(0)
        k = min(self.context_top_k, sims.numel())
        return torch.topk(sims, k=k).indices

    def _generate_windows(self, h: int, w: int) -> List[BBox]:
        windows: list[BBox] = []
        short = min(h, w)
        for ratio in self.window_ratios:
            win = max(32, int(round(short * ratio)))
            stride = max(16, int(round(short * self.stride_ratio)))
            for y1 in range(0, max(1, h - win + 1), stride):
                for x1 in range(0, max(1, w - win + 1), stride):
                    windows.append((x1, y1, min(w, x1 + win), min(h, y1 + win)))
            if h > win:
                for x1 in range(0, max(1, w - win + 1), stride):
                    windows.append((x1, h - win, min(w, x1 + win), h))
            if w > win:
                for y1 in range(0, max(1, h - win + 1), stride):
                    windows.append((w - win, y1, w, min(h, y1 + win)))
        return sorted(set(windows))

    @staticmethod
    def _center_weight(h: int, w: int) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        sigma = max(1.0, min(h, w) * 0.32)
        weight = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2)))
        return weight.astype(np.float32)

    @staticmethod
    def _normalize_array(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        min_v = float(values.min())
        max_v = float(values.max())
        if max_v - min_v < 1e-9:
            return np.full_like(values, 0.5, dtype=np.float32)
        return ((values - min_v) / (max_v - min_v)).astype(np.float32)

    @staticmethod
    def _smooth_norm(values: np.ndarray) -> np.ndarray:
        smoothed = cv2.GaussianBlur(values.astype(np.float32), (0, 0), 5.0)
        return SemanticHeatmapScorer._normalize_array(smoothed)
