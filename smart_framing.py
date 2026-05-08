import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class CropPrediction:
    bbox: BBox
    aesthetic_score: float
    feature_scores: Dict[str, float]


class LinearAestheticModel:
    """Simple linear model with closed-form ridge regression for reproducible training."""

    def __init__(self, l2: float = 1e-3):
        self.l2 = l2
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        ones = np.ones((x.shape[0], 1), dtype=np.float32)
        x_aug = np.concatenate([x, ones], axis=1)
        i = np.eye(x_aug.shape[1], dtype=np.float32)
        i[-1, -1] = 0.0  # no regularization on bias
        beta = np.linalg.pinv(x_aug.T @ x_aug + self.l2 * i) @ x_aug.T @ y
        self.w = beta[:-1, 0]
        self.b = float(beta[-1, 0])

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        x = np.asarray(x, dtype=np.float32)
        return x @ self.w + self.b


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


class SmartFramer:
    def __init__(self, model: Optional[LinearAestheticModel] = None):
        self.model = model

    @staticmethod
    def _saliency_map(image: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _rule_of_thirds_mask(h: int, w: int, sigma: float = 0.08) -> np.ndarray:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        xs /= max(1, w - 1)
        ys /= max(1, h - 1)
        points = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
        mask = np.zeros((h, w), dtype=np.float32)
        for px, py in points:
            d2 = (xs - px) ** 2 + (ys - py) ** 2
            mask = np.maximum(mask, np.exp(-d2 / (2 * sigma * sigma)))
        return mask

    def extract_features(self, image: np.ndarray, bbox: BBox) -> Tuple[np.ndarray, Dict[str, float]]:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return np.zeros(6, dtype=np.float32), {
                "saliency_mean": 0.0,
                "thirds_alignment": 0.0,
                "edge_density": 0.0,
                "color_var": 0.0,
                "entropy": 0.0,
                "size_ratio": 0.0,
            }

        saliency = self._saliency_map(crop)
        thirds = self._rule_of_thirds_mask(h, w)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        saliency_mean = float(saliency.mean())
        thirds_alignment = float((saliency * thirds).sum() / (saliency.sum() + 1e-6))
        edge_density = float((edges > 0).mean())
        color_var = float(hsv[:, :, 1].std() / 255.0)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).reshape(-1)
        p = hist / (hist.sum() + 1e-6)
        entropy = float(-(p * np.log(p + 1e-9)).sum() / math.log(len(p)))
        size_ratio = float((h * w) / (image.shape[0] * image.shape[1]))

        vals = np.array([
            saliency_mean,
            thirds_alignment,
            edge_density,
            color_var,
            entropy,
            size_ratio,
        ], dtype=np.float32)
        names = ["saliency_mean", "thirds_alignment", "edge_density", "color_var", "entropy", "size_ratio"]
        return vals, {k: float(v) for k, v in zip(names, vals)}

    @staticmethod
    def generate_candidates(image: np.ndarray, max_candidates: int = 240) -> List[BBox]:
        h, w = image.shape[:2]
        ratios = [1.0, 4 / 3, 3 / 4, 16 / 9, 9 / 16]
        scales = [0.45, 0.55, 0.65, 0.75, 0.85]
        out: List[BBox] = []
        for s in scales:
            area = h * w * s
            for r in ratios:
                ch = int(math.sqrt(area / r))
                cw = int(ch * r)
                if ch >= h or cw >= w:
                    continue
                dy = max(1, ch // 5)
                dx = max(1, cw // 5)
                for y1 in range(0, h - ch + 1, dy):
                    for x1 in range(0, w - cw + 1, dx):
                        out.append((x1, y1, x1 + cw, y1 + ch))
        if len(out) > max_candidates:
            idx = np.linspace(0, len(out) - 1, max_candidates).astype(int)
            out = [out[i] for i in idx]
        return out

    def predict_best_crop(self, image: np.ndarray) -> CropPrediction:
        cands = self.generate_candidates(image)
        feats, detail = [], []
        for b in cands:
            f, d = self.extract_features(image, b)
            feats.append(f)
            detail.append(d)
        x = np.stack(feats, axis=0)
        if self.model is None or self.model.w is None:
            w = np.array([0.32, 0.24, 0.16, 0.08, 0.14, 0.06], dtype=np.float32)
            y = x @ w
        else:
            y = self.model.predict(x)
        k = int(np.argmax(y))
        return CropPrediction(cands[k], float(y[k]), detail[k])


def draw_bbox(image: np.ndarray, bbox: BBox, text: str = "best crop") -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out
