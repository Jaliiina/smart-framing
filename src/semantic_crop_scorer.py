from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from .utils import BBox

logger = logging.getLogger(__name__)


class SemanticCropScorer:

    def __init__(self, config: dict):
        cfg = config.get("semantic_crop", {})
        model_cfg = config.get("models", {}).get("aesthetic", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.device = model_cfg.get("device", "cpu")
        self.clip_model = cfg.get("clip_model", model_cfg.get("clip_model", "ViT-L/14"))
        self.positive_prompts = cfg.get(
            "positive_prompts",
            [
                "beautiful landscape photograph",
                "beautiful seascape with waves",
                "airplane in the sky",
                "pencil tips and hand drawing",
                "complete building architecture",
                "ocean waves scenic view",
                "illuminated tree canopy",
                "symmetric road with trees",
                "complete glass object",
                "motorcycle scenic landscape",
                "well framed main subject",
            ],
        )
        self.negative_prompts = cfg.get(
            "negative_prompts",
            [
                "trash or garbage",
                "plastic bag",
                "orange bucket or plastic barrel",
                "traffic cone or construction object",
                "cluttered random foreground",
                "dirty water or muddy ground",
                "rusty roof edge",
                "metal roof eaves and beams",
                "dark empty ground",
                "net rope foreground",
                "large blank white area",
                "large flat color area",
                "cropped cut off object",
            ],
        )
        self.context_top_k = max(1, int(cfg.get("context_top_k", 3)))
        self.negative_agg = cfg.get("negative_aggregation", "max")
        self.positive_agg = cfg.get("positive_aggregation", "context_mean")
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
            logger.warning(f"Semantic CLIP scorer unavailable: {exc}")
        self._loaded = True

    def set_positive_prompts(self, prompts: List[str]) -> None:
        self.positive_prompts = list(prompts)
        if not self._loaded:
            return
        if self._clip is None or self._model is None:
            return
        try:
            with torch.no_grad():
                pos_tokens = self._clip.tokenize(self.positive_prompts).to(self.device)
                pos = self._model.encode_text(pos_tokens).float()
            self._pos = pos / pos.norm(dim=-1, keepdim=True)
        except Exception as exc:
            logger.warning(f"Failed to refresh semantic positive prompts: {exc}")

    def score_candidates(
        self,
        image: np.ndarray,
        bboxes: List[BBox],
    ) -> List[Tuple[float, Dict[str, float]]]:
        self._load()
        if (
            not self.enabled
            or self._model is None
            or self._preprocess is None
            or self._pos is None
            or self._neg is None
        ):
            return [(0.5, {"positive_semantic": 0.5, "negative_semantic": 0.5}) for _ in bboxes]

        from PIL import Image

        crops = []
        valid = []
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(self._preprocess(Image.fromarray(crop_rgb)))
            valid.append(idx)

        raw_rows = []
        context_idx = self._context_positive_indices(image)
        chunk_size = 32
        for start in range(0, len(crops), chunk_size):
            chunk = torch.stack(crops[start:start + chunk_size]).to(self.device)
            with torch.no_grad():
                features = self._model.encode_image(chunk).float()
                features = features / features.norm(dim=-1, keepdim=True)
                pos = features @ self._pos.T
                neg = features @ self._neg.T
                pos_focus = pos[:, context_idx]
                if self.positive_agg == "max":
                    pos_value = pos_focus.max(dim=1).values
                else:
                    pos_value = pos_focus.mean(dim=1)
                if self.negative_agg == "mean":
                    neg_value = neg.mean(dim=1)
                else:
                    neg_value = neg.max(dim=1).values
                raw = pos_value - neg_value

            for local, idx in enumerate(valid[start:start + chunk_size]):
                raw_rows.append(
                    (
                        idx,
                        float(raw[local].cpu()),
                        float(pos_value[local].cpu()),
                        float(neg_value[local].cpu()),
                    )
                )
        scores = [(0.5, {"positive_semantic": 0.5, "negative_semantic": 0.5}) for _ in bboxes]
        if not raw_rows:
            return scores
        raw_arr = np.array([[r[1], r[2], r[3]] for r in raw_rows], dtype=np.float32)
        norm_arr = self._normalize_array(raw_arr)
        for row, norm in zip(raw_rows, norm_arr):
            idx = row[0]
            scores[idx] = (
                float(norm[0]),
                {
                    "positive_semantic": float(norm[1]),
                    "negative_semantic": float(norm[2]),
                },
            )
        return scores

    def _context_positive_indices(self, image: np.ndarray) -> torch.Tensor:

        from PIL import Image

        if self._model is None or self._preprocess is None or self._pos is None:
            return torch.arange(min(self.context_top_k, len(self.positive_prompts)))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self._model.encode_image(tensor).float()
            feature = feature / feature.norm(dim=-1, keepdim=True)
            sims = (feature @ self._pos.T).squeeze(0)
        k = min(self.context_top_k, sims.numel())
        return torch.topk(sims, k=k).indices

    @staticmethod
    def _normalize_array(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        out = np.zeros_like(values, dtype=np.float32)
        for col in range(values.shape[1]):
            column = values[:, col]
            min_v = float(column.min())
            max_v = float(column.max())
            if max_v - min_v < 1e-9:
                out[:, col] = 0.5
            else:
                out[:, col] = (column - min_v) / (max_v - min_v)
        return out
