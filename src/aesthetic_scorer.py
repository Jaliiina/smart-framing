"""LAION Aesthetic Predictor wrapper with fallback to hand-crafted features."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import BBox

logger = logging.getLogger(__name__)


class AestheticScorer:
    """Score candidate crops using LAION Aesthetic Predictor or fallback."""

    def __init__(self, config: dict):
        acfg = config.get("models", {}).get("aesthetic", {})
        self.model_path: str = acfg.get("model_path", "models/aesthetic_predictor.pth")
        self.clip_model: str = acfg.get("clip_model", "ViT-B/32")
        self.device: str = acfg.get("device", "cpu")
        self.use_fallback: bool = acfg.get("use_fallback", True)

        self._clip_model = None
        self._aesthetic_head = None
        self._preprocess = None
        self._model_loaded = False

    def _load_model(self):
        """Lazily load CLIP + aesthetic head."""
        if self._model_loaded:
            return
        try:
            import torch
            import clip  # type: ignore

            if Path(self.model_path).exists():
                self._clip_model, self._preprocess = clip.load(
                    self.clip_model, device=self.device
                )
                state = torch.load(self.model_path, map_location=self.device)
                # LAION aesthetic predictor: linear layer on top of CLIP embeddings
                embed_dim = self._clip_model.visual.output_dim
                self._aesthetic_head = torch.nn.Linear(embed_dim, 1).to(self.device)
                if isinstance(state, dict) and "weight" in state:
                    self._aesthetic_head.load_state_dict(state)
                elif isinstance(state, dict) and "model" in state:
                    self._aesthetic_head.load_state_dict(state["model"])
                else:
                    # Try loading directly
                    self._aesthetic_head.load_state_dict(state)
                self._aesthetic_head.eval()
                logger.info(f"Aesthetic predictor loaded from {self.model_path}")
            else:
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
        """Score each candidate crop for aesthetic quality.

        Args:
            image: Original BGR image.
            bboxes: List of candidate bboxes.

        Returns:
            List of raw aesthetic scores (before normalization).
        """
        self._load_model()

        if self._clip_model is not None and self._aesthetic_head is not None:
            return self._score_clip(image, bboxes)
        elif self.use_fallback:
            return self._score_fallback(image, bboxes)
        else:
            return [0.5] * len(bboxes)

    def _score_clip(self, image: np.ndarray, bboxes: List[BBox]) -> List[float]:
        """Score using CLIP + aesthetic head."""
        import torch
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

        # Batch inference
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            features = self._clip_model.encode_image(batch).float()
            scores = self._aesthetic_head(features).squeeze(-1).cpu().numpy()

        return [float(s) for s in scores]

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

            # Compute 6 features same as original SmartFramer
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

            # Rule of thirds
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

            # Empirical weights from original code
            w_arr = np.array([0.32, 0.24, 0.16, 0.08, 0.14, 0.06], dtype=np.float32)
            feats = np.array(
                [saliency_mean, thirds_alignment, edge_density, color_var, entropy, size_ratio],
                dtype=np.float32,
            )
            scores.append(float(feats @ w_arr))

        return scores
