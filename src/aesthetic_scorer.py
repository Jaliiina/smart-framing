"""LAION Aesthetic Predictor wrapper with fallback to hand-crafted features."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from .utils import BBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aesthetic MLP matching aesthetic_predictor.pth key names: layers.0, layers.2...
# ---------------------------------------------------------------------------
class _AestheticMLP(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# AestheticScorer
# ---------------------------------------------------------------------------
class AestheticScorer:
    """Score candidate crops using LAION Aesthetic Predictor or fallback."""

    def __init__(self, config: dict):
        acfg = config.get("models", {}).get("aesthetic", {})
        self.model_path: str = acfg.get("model_path", "models/aesthetic_predictor.pth")
        self.clip_model: str = acfg.get("clip_model", "ViT-B/32")
        self.device: str = acfg.get("device", "cpu")
        self.use_laion_predictor: bool = acfg.get("use_laion_predictor", False)
        self.use_fallback: bool = acfg.get("use_fallback", True)
        self.use_clip_prompt_fallback: bool = acfg.get(
            "use_clip_prompt_fallback", True
        )
        self.positive_prompts: List[str] = acfg.get(
            "positive_prompts",
            [
                "a well-composed professional photograph",
                "a visually pleasing photograph with a clear subject",
                "a balanced scenic photograph",
            ],
        )
        self.negative_prompts: List[str] = acfg.get(
            "negative_prompts",
            [
                "a cluttered photograph with distracting objects",
                "an accidental close-up of ground or debris",
                "a poorly composed photograph",
            ],
        )

        self._clip_model = None
        self._aesthetic_head = None
        self._preprocess = None
        self._positive_text_features = None
        self._negative_text_features = None
        self._model_loaded = False

    def _load_model(self):
        """Lazily load CLIP + optional aesthetic head."""
        if self._model_loaded:
            return
        try:
            import clip  # type: ignore

            need_clip = self.use_laion_predictor or self.use_clip_prompt_fallback
            if need_clip:
                self._clip_model, self._preprocess = clip.load(
                    self.clip_model, device=self.device
                )

            # Branch 1: LAION predictor explicitly enabled
            if self.use_laion_predictor:
                if Path(self.model_path).exists() and self._clip_model is not None:
                    state = torch.load(self.model_path, map_location=self.device)
                    embed_dim = self._clip_model.visual.output_dim
                    self._aesthetic_head = _AestheticMLP(embed_dim).to(self.device)
                    if isinstance(state, dict) and "model" in state:
                        self._aesthetic_head.load_state_dict(state["model"])
                    else:
                        self._aesthetic_head.load_state_dict(state)
                    self._aesthetic_head.eval()
                    logger.info(f"Aesthetic predictor loaded from {self.model_path}")
                else:
                    logger.warning(
                        f"Aesthetic weights not found at {self.model_path}. "
                        "Falling back to CLIP prompt-based scoring."
                    )
                    if self.use_clip_prompt_fallback and self._clip_model is not None:
                        positive_tokens = clip.tokenize(self.positive_prompts).to(self.device)
                        negative_tokens = clip.tokenize(self.negative_prompts).to(self.device)
                        with torch.no_grad():
                            positive = self._clip_model.encode_text(positive_tokens).float()
                            negative = self._clip_model.encode_text(negative_tokens).float()
                        self._positive_text_features = positive / positive.norm(
                            dim=-1, keepdim=True
                        )
                        self._negative_text_features = negative / negative.norm(
                            dim=-1, keepdim=True
                        )
                        logger.info("Using CLIP prompt-based aesthetic fallback.")
            # Branch 2: LAION disabled — use CLIP prompts or hand-crafted fallback
            elif self._clip_model is not None and self.use_clip_prompt_fallback:
                positive_tokens = clip.tokenize(self.positive_prompts).to(self.device)
                negative_tokens = clip.tokenize(self.negative_prompts).to(self.device)
                with torch.no_grad():
                    positive = self._clip_model.encode_text(positive_tokens).float()
                    negative = self._clip_model.encode_text(negative_tokens).float()
                self._positive_text_features = positive / positive.norm(
                    dim=-1, keepdim=True
                )
                self._negative_text_features = negative / negative.norm(
                    dim=-1, keepdim=True
                )
                logger.info("Using CLIP prompt-based aesthetic fallback.")
            elif not Path(self.model_path).exists():
                logger.warning(
                    f"Aesthetic weights not found at {self.model_path}. Using fallback."
                )
        except ImportError:
            logger.warning("CLIP not available. Using fallback aesthetic scoring.")
        except Exception as e:
            logger.warning(f"Failed to load aesthetic predictor: {e}. Using fallback.")
        self._model_loaded = True

    def score_candidates(
        self,
        image: np.ndarray,
        bboxes: List[BBox],
    ) -> List[float]:
        """Score each candidate crop for aesthetic quality."""
        self._load_model()

        if self._clip_model is not None and self._aesthetic_head is not None:
            return self._score_clip(image, bboxes)
        elif (
            self._clip_model is not None
            and self._positive_text_features is not None
            and self._negative_text_features is not None
        ):
            return self._score_clip_prompts(image, bboxes)
        elif self.use_fallback:
            return self._score_fallback(image, bboxes)
        else:
            return [0.5] * len(bboxes)

    def _score_clip(self, image: np.ndarray, bboxes: List[BBox]) -> List[float]:
        """Score using CLIP + aesthetic head, with batch chunking."""
        from PIL import Image

        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            if self._preprocess is not None:
                crops.append(self._preprocess(pil_img))

        if len(crops) == 0:
            return []

        all_scores = []
        chunk_size = 32
        for i in range(0, len(crops), chunk_size):
            chunk = torch.stack(crops[i:i + chunk_size]).to(self.device)
            with torch.no_grad():
                features = self._clip_model.encode_image(chunk).float()
                scores = self._aesthetic_head(features).squeeze(-1).cpu().numpy()
            all_scores.extend([float(s) for s in scores])

        return all_scores

    def _score_clip_prompts(
        self, image: np.ndarray, bboxes: List[BBox]
    ) -> List[float]:
        """Score semantic framing quality using positive and negative prompts."""
        from PIL import Image

        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(self._preprocess(Image.fromarray(crop_rgb)))

        all_scores = []
        chunk_size = 32
        for i in range(0, len(crops), chunk_size):
            chunk = torch.stack(crops[i:i + chunk_size]).to(self.device)
            with torch.no_grad():
                features = self._clip_model.encode_image(chunk).float()
                features = features / features.norm(dim=-1, keepdim=True)
                positive = features @ self._positive_text_features.T
                negative = features @ self._negative_text_features.T
                scores = positive.mean(dim=1) - negative.mean(dim=1)
            all_scores.extend(float(score) for score in scores.cpu().numpy())

        return all_scores

    @staticmethod
    def _score_fallback(image: np.ndarray, bboxes: List[BBox]) -> List[float]:
        """Fallback: use hand-crafted features from original smart_framing.py."""
        scores = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            if h < 8 or w < 8:
                scores.append(0.0)
                continue

            img_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
            l, a, b_ch = cv2.split(img_lab)
            blur = cv2.GaussianBlur(img_lab, (0, 0), 7)
            dl = cv2.absdiff(l, blur[:, :, 0]).astype(np.float32)
            da = cv2.absdiff(a, blur[:, :, 1]).astype(np.float32)
            db = cv2.absdiff(b_ch, blur[:, :, 2]).astype(np.float32)
            s = dl + da + db
            s = cv2.GaussianBlur(s, (0, 0), 5)
            s = s - s.min()
            if s.max() > 1e-6:
                s /= s.max()
            saliency_mean = float(s.mean())

            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            xs_n = xs / max(1, w - 1)
            ys_n = ys / max(1, h - 1)
            points = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
            thirds_mask = np.zeros((h, w), dtype=np.float32)
            sigma = 0.08
            for px, py in points:
                d2 = (xs_n - px) ** 2 + (ys_n - py) ** 2
                thirds_mask = np.maximum(thirds_mask, np.exp(-d2 / (2 * sigma * sigma)))
            thirds_alignment = float((s * thirds_mask).sum() / (s.sum() + 1e-6))

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edge_density = float((edges > 0).mean())

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            color_var = float(hsv[:, :, 1].std() / 255.0)

            hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).reshape(-1)
            p = hist / (hist.sum() + 1e-6)
            entropy = float(-(p * np.log(p + 1e-9)).sum() / math.log(len(p)))

            size_ratio = float((h * w) / (image.shape[0] * image.shape[1]))
            compactness = max(0.0, 1.0 - abs(size_ratio - 0.25) / 0.20)

            w_arr = np.array([0.22, 0.28, 0.16, 0.10, 0.14, 0.10], dtype=np.float32)
            feats = np.array(
                [saliency_mean, thirds_alignment, edge_density, color_var, entropy, compactness],
                dtype=np.float32,
            )
            scores.append(float(feats @ w_arr))

        return scores
