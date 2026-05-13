"""Shared utilities: BBox operations, IoU, NMS, normalization, config loading."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DetectedObject:
    """A single detected object from YOLOv8."""
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float


@dataclass
class SubScores:
    """All per-candidate sub-scores."""
    aesthetic: float = 0.0
    saliency: float = 0.0
    composition: float = 0.0
    subject: float = 0.0
    technical: float = 0.0

    # Detailed composition breakdown
    thirds: float = 0.0
    center_balance: float = 0.0
    whitespace: float = 0.0
    edge_simplicity: float = 0.0
    symmetry: float = 0.0

    # Technical quality breakdown
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0


@dataclass
class CandidateResult:
    """Result for a single candidate box."""
    bbox: BBox
    final_score: float = 0.0
    sub_scores: SubScores = field(default_factory=SubScores)


@dataclass
class CropResult:
    """Full pipeline output for one image."""
    image_path: str
    best_bbox: BBox
    best_crop: np.ndarray
    best_score: float
    best_sub_scores: SubScores
    top_candidates: List[CandidateResult]  # top-K results
    explanation: str = ""
    saliency_map: Optional[np.ndarray] = None
    detected_objects: List[DetectedObject] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BBox operations
# ---------------------------------------------------------------------------
def bbox_iou(a: BBox, b: BBox) -> float:
    """Compute IoU between two bounding boxes."""
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


def bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_intersection(a: BBox, b: BBox) -> int:
    """Area of intersection between two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def clamp_bbox(bbox: BBox, img_h: int, img_w: int) -> BBox:
    """Clamp bbox to image boundaries."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return (x1, y1, x2, y2)


def bbox_aspect_ratio(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return w / h


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------
def nms(bboxes: List[BBox], scores: List[float], iou_threshold: float = 0.7) -> List[int]:
    """Non-maximum suppression. Returns indices of kept boxes."""
    if len(bboxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        suppress = []
        for j_idx, j in enumerate(rest):
            if bbox_iou(bboxes[i], bboxes[j]) > iou_threshold:
                suppress.append(j_idx)
        order = np.delete(rest, suppress)
    return keep


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------
def map_bbox_to_original(bbox: BBox, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> BBox:
    """Map bbox from source image coordinates to destination image coordinates."""
    src_h, src_w = src_size[:2]
    dst_h, dst_w = dst_size[:2]
    scale_x = dst_w / max(1, src_w)
    scale_y = dst_h / max(1, src_h)
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]. Returns zeros if all values are equal."""
    values = np.asarray(values, dtype=np.float64)
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-9:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def resize_short_edge(image: np.ndarray, target: int) -> Tuple[np.ndarray, float]:
    """Resize image so that short edge = target. Returns (resized_image, scale)."""
    h, w = image.shape[:2]
    short = min(h, w)
    scale = target / max(1, short)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def load_image(path: str) -> np.ndarray:
    """Load BGR image from path. Raises FileNotFoundError if image cannot be read.

    Uses numpy.fromfile + cv2.imdecode to handle non-ASCII (Chinese) paths
    on Windows, since cv2.imread fails with such paths.
    """
    try:
        # Try cv2.imread first (faster for ASCII paths)
        img = cv2.imread(path)
        if img is not None:
            return img
    except Exception:
        pass

    # Fallback for non-ASCII paths (e.g. Chinese characters on Windows)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    raise FileNotFoundError(f"Cannot read image: {path}")


def save_image(image: np.ndarray, path: str, params: Optional[list] = None) -> bool:
    """Save BGR image to path. Handles non-ASCII (Chinese) paths on Windows.

    Uses cv2.imencode + numpy.tofile to handle non-ASCII paths.

    Args:
        image: BGR image to save.
        path: Output file path.
        params: Optional cv2.imwrite params (e.g. [cv2.IMWRITE_JPEG_QUALITY, 90]).

    Returns:
        True if successful.
    """
    ext = Path(path).suffix.lower()
    try:
        # First try normal cv2.imwrite
        success = cv2.imwrite(path, image, params) if params else cv2.imwrite(path, image)
        if success:
            return True
    except Exception:
        pass

    # Fallback for non-ASCII paths
    try:
        if params:
            ok, buf = cv2.imencode(ext, image, params)
        else:
            ok, buf = cv2.imencode(ext, image)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_bbox(image: np.ndarray, bbox: BBox, text: str = "", color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw bbox on image copy with optional label text."""
    out = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if text:
        cv2.putText(out, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def draw_multiple_bboxes(image: np.ndarray, bboxes: List[BBox], labels: List[str] = None, colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """Draw multiple bboxes on image copy."""
    out = image.copy()
    if colors is None:
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0)]
    for i, bbox in enumerate(bboxes):
        c = colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else ""
        out = draw_bbox(out, bbox, label, c)
    return out
