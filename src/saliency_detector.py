"""U2-Net saliency detection wrapper with scoring."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import BBox, bbox_center

logger = logging.getLogger(__name__)


class SaliencyDetector:
    """Wrap U2-Net (or U2-Net-lite) for saliency detection.

    Provides:
    - Full-image saliency map (run once, shared across candidates)
    - Per-candidate saliency preservation score
    """

    def __init__(self, config: dict):
        u2cfg = config.get("models", {}).get("u2net", {})
        self.weights_path: str = u2cfg.get("weights_path", "models/u2net.pth")
        self.lite_weights_path: str = u2cfg.get("lite_weights_path", "models/u2netp.pth")
        self.use_lite: bool = u2cfg.get("use_lite", True)
        self.device: str = u2cfg.get("device", "cpu")
        self.input_size: int = config.get("preprocessing", {}).get("u2net_input_size", 320)

        self.uniform_std_threshold: float = config.get("fusion", {}).get(
            "saliency_uniform_std_threshold", 0.05
        )
        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """Lazily load the U2-Net model."""
        if self._model_loaded:
            return
        try:
            import torch
            from .u2net_model import U2Net, U2NetP  # type: ignore

            weights = self.lite_weights_path if self.use_lite else self.weights_path
            if not Path(weights).exists():
                logger.warning(f"U2-Net weights not found at {weights}, using fallback saliency.")
                self._model_loaded = True
                return

            net_cls = U2NetP if self.use_lite else U2Net
            self._model = net_cls(in_ch=3, out_ch=1)
            self._model.load_state_dict(torch.load(weights, map_location=self.device))
            self._model.to(self.device)
            self._model.eval()
            logger.info(f"U2-Net model loaded from {weights}")
        except ImportError:
            logger.warning(
                "U2-Net model definition not available. Using fallback CV saliency."
            )
        except Exception as e:
            logger.warning(f"Failed to load U2-Net: {e}. Using fallback CV saliency.")
        self._model_loaded = True

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Run saliency detection on the image.

        Args:
            image: BGR image (H, W, 3).

        Returns:
            (saliency_map, is_uniform):
                saliency_map: (H, W) float32 in [0, 1].
                is_uniform: True if the map is too uniform (fallback needed).
        """
        self._load_model()

        if self._model is not None:
            sal = self._detect_u2net(image)
        else:
            sal = self._detect_fallback(image)

        # Check uniformity
        is_uniform = float(sal.std()) < self.uniform_std_threshold
        return sal, is_uniform

    def _detect_u2net(self, image: np.ndarray) -> np.ndarray:
        """Run U2-Net inference."""
        import torch

        h, w = image.shape[:2]
        # Preprocess: resize to input_size x input_size
        inp = cv2.resize(image, (self.input_size, self.input_size))
        inp = inp.astype(np.float32) / 255.0
        inp = (inp - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        inp = inp.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
        tensor = torch.from_numpy(inp).float().to(self.device)

        with torch.no_grad():
            d1, *_ = self._model(tensor)
        sal = d1.squeeze().cpu().numpy()
        sal = cv2.resize(sal, (w, h))
        sal = sal - sal.min()
        if sal.max() > 1e-6:
            sal /= sal.max()
        return sal.astype(np.float32)

    @staticmethod
    def _detect_fallback(image: np.ndarray) -> np.ndarray:
        """Fallback CV-based saliency (from the original smart_framing.py)."""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(img)
        blur = cv2.GaussianBlur(img, (0, 0), 7)
        dl = cv2.absdiff(l, blur[:, :, 0]).astype(np.float32)
        da = cv2.absdiff(a, blur[:, :, 1]).astype(np.float32)
        db = cv2.absdiff(b, blur[:, :, 2]).astype(np.float32)
        s = dl + da + db
        s = cv2.GaussianBlur(s, (0, 0), 5)
        s = s - s.min()
        if s.max() > 1e-6:
            s /= s.max()
        return s

    def score_candidates(
        self,
        saliency_map: np.ndarray,
        bboxes: List[BBox],
        image_shape: Tuple[int, int],
    ) -> List[float]:
        """Compute saliency preservation score for each candidate bbox.

        Score = alpha * coverage - beta * center_offset - gamma * boundary_cut

        Args:
            saliency_map: (H, W) float32 in [0, 1].
            bboxes: List of candidate bboxes.
            image_shape: (H, W) of the original image.

        Returns:
            List of saliency preservation scores.
        """
        h, w = image_shape[:2]
        total_sal = saliency_map.sum() + 1e-9
        scores = []

        # Precompute boundary strip mask width
        strip_w = max(2, min(h, w) // 50)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # 1. Coverage: fraction of total saliency inside bbox
            region = saliency_map[max(0, y1):y2, max(0, x1):x2]
            coverage = float(region.sum()) / total_sal

            # 2. Center offset: distance from saliency centroid to bbox visual center
            region_sum = region.sum() + 1e-9
            local_h, local_w = region.shape[:2]
            if local_h > 0 and local_w > 0:
                ys, xs = np.mgrid[0:local_h, 0:local_w]
                sal_cx = float((xs * region).sum() / region_sum) + x1
                sal_cy = float((ys * region).sum() / region_sum) + y1
            else:
                sal_cx, sal_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            bbox_cx, bbox_cy = bbox_center(bbox)
            diag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 1e-6
            center_offset = math.sqrt((sal_cx - bbox_cx) ** 2 + (sal_cy - bbox_cy) ** 2) / diag

            # 3. Boundary cut: average saliency on boundary strip
            boundary_sal = 0.0
            boundary_count = 0
            # Top strip
            if y2 - y1 > 2 * strip_w and x2 - x1 > 0:
                boundary_sal += float(saliency_map[y1:y1 + strip_w, x1:x2].sum())
                boundary_count += strip_w * (x2 - x1)
            # Bottom strip
            if y2 - y1 > 2 * strip_w and x2 - x1 > 0:
                boundary_sal += float(saliency_map[y2 - strip_w:y2, x1:x2].sum())
                boundary_count += strip_w * (x2 - x1)
            # Left strip
            if x2 - x1 > 2 * strip_w and y2 - y1 > 0:
                boundary_sal += float(saliency_map[y1:y2, x1:x1 + strip_w].sum())
                boundary_count += (y2 - y1) * strip_w
            # Right strip
            if x2 - x1 > 2 * strip_w and y2 - y1 > 0:
                boundary_sal += float(saliency_map[y1:y2, x2 - strip_w:x2].sum())
                boundary_count += (y2 - y1) * strip_w
            boundary_cut = boundary_sal / max(1, boundary_count) if boundary_count > 0 else 0.0

            # Combined score
            alpha, beta, gamma = 1.0, 0.5, 0.3
            score = alpha * coverage - beta * center_offset - gamma * boundary_cut
            scores.append(max(0.0, score))

        return scores
