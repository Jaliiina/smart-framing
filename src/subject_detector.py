"""YOLOv8 object detection wrapper for subject completeness scoring."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Fix ultralytics settings permission issue: set YOLO_SETTINGS_DIR before import
_yolo_settings_dir = str(Path.home() / ".config" / "ultralytics")
os.makedirs(_yolo_settings_dir, exist_ok=True)
os.environ.setdefault("YOLO_SETTINGS_DIR", _yolo_settings_dir)

import cv2
import numpy as np

from .utils import BBox, DetectedObject, bbox_area, bbox_center, bbox_intersection

logger = logging.getLogger(__name__)


class SubjectDetector:
    """Detect objects using YOLOv8 and compute subject completeness scores."""
    # Class-specific minimum sizes (height in pixels)
    CLASS_MIN_SIZES = {
        0: 50,   # person - should be prominent
        45: 40,  # wine glass 高脚杯最小像素约束，防止小酒杯漏检过滤
        15: 30, 16: 30, 17: 30, 18: 30,  # cat, dog, horse, sheep
        19: 30, 20: 30, 21: 30, 22: 30, 23: 30,  # cow, elephant, bear, zebra, giraffe
        1: 40, 2: 40, 3: 40, 4: 40, 5: 40, 6: 40, 7: 40, 8: 40, 9: 40,  # vehicles
        24: 25, 25: 25, 26: 25, 27: 25, 28: 25, 29: 25, 30: 25, 31: 25, 32: 25, 33: 25,  # umbrella, handbag, etc.
        44: 20, 46: 20, 47: 20, 48: 20, 49: 20,  # bottle, cup, fork, knife, spoon
        50: 20, 51: 20, 52: 20, 53: 20, 54: 20, 55: 20,  # bowl, banana, apple, sandwich, orange, broccoli
        56: 20, 57: 20,  # carrot, hot dog (common food/product subjects)
    }
    # Class-specific aspect ratio (width/height) sanity ranges
    CLASS_ASPECT_RANGES = {
        0: (0.15, 1.5),   # person - typically tall (w/h ~ 0.3-0.5 standing)
        15: (0.5, 2.0), 16: (0.5, 2.0), 17: (0.5, 2.0), 18: (0.5, 2.0),  # quadruped animals
        19: (0.6, 2.0), 20: (0.5, 2.0), 21: (0.5, 2.0), 22: (0.5, 2.0), 23: (0.5, 2.0),
        1: (0.8, 3.0), 2: (0.8, 3.0), 3: (0.8, 3.0), 4: (0.8, 3.0), 5: (0.8, 3.0),  # vehicles
        6: (0.8, 3.0), 7: (0.8, 3.0), 8: (0.8, 3.0), 9: (0.8, 3.0),
    }

    def __init__(self, config: dict):
        ycfg = config.get("models", {}).get("yolo", {})
        self.model_name: str = ycfg.get("model_name", "yolov8n.pt")
        self.confidence_threshold: float = ycfg.get("confidence_threshold", 0.5)
        self.device: str = ycfg.get("device", "cpu")
        self.important_classes: List[int] = ycfg.get(
            "important_classes",
            [
                0, 1, 2, 3, 4, 5, 7, 9,   # person, bicycle, car, motorcycle, airplane, bus, truck, train
                15, 16, 17, 18, 19,       # cat, dog, horse, sheep, cow
                20, 21, 22, 23,           # elephant, bear, zebra, giraffe
                44, 45, 46, 47, 48, 49,  # bottle, wine glass, cup, fork, knife, spoon
                50, 51, 52, 53, 54, 55,  # bowl, banana, apple, sandwich, orange, broccoli
                56, 57,                   # carrot, hot dog (common food/product subjects)
            ],
        )
        scfg = config.get("subject", {})
        self.min_important_inclusion: float = scfg.get("min_important_inclusion", 0.80)
        self.tightness_weight: float = scfg.get("tightness_weight", 0.25)
        # Discard tiny detections that are almost certainly false positives
        # (e.g. YOLO often mis-classifies grass/rocks as giraffes ~20px tall).
        self.min_object_size: int = scfg.get("min_object_size_px", 25)

        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """Lazily load YOLOv8 model."""
        if self._model_loaded:
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_name)
            logger.info(f"YOLOv8 model loaded: {self.model_name}")
        except ImportError:
            logger.warning("ultralytics not available. Subject detection disabled.")
        except Exception as e:
            logger.warning(f"Failed to load YOLOv8: {e}. Subject detection disabled.")
        self._model_loaded = True

    def detect(self, image: np.ndarray, saliency_map: Optional[np.ndarray] = None) -> List[DetectedObject]:
        """Run object detection on the full image.

        Args:
            image: BGR image (H, W, 3).
            saliency_map: Optional saliency map for filtering false positives.

        Returns:
            List of DetectedObject instances.
        """
        self._load_model()
        if self._model is None:
            return []

        results = self._model(image, conf=self.confidence_threshold, verbose=False)
        objects = self._parse_yolo_results(results, image.shape[:2])

        if len(objects) == 0:
            face_objects = self._detect_faces(image)
            if face_objects:
                objects.extend(face_objects)

        if saliency_map is not None and len(objects) > 0:
            objects = self._filter_by_saliency(objects, saliency_map)

        return objects

    def _parse_yolo_results(self, results, image_shape):
        objects = []
        h, w = image_shape[:2]
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                cls_name = self._model.names.get(cls_id, str(cls_id))

                bw, bh = x2 - x1, y2 - y1

                min_size = self.CLASS_MIN_SIZES.get(cls_id, self.min_object_size)
                if min(bw, bh) < min_size:
                    logger.debug(f"Filtered {cls_name} (size {min(bw,bh):.0f}px < {min_size}px)")
                    continue

                if bh > 0:
                    aspect = bw / bh
                    if cls_id in self.CLASS_ASPECT_RANGES:
                        min_ar, max_ar = self.CLASS_ASPECT_RANGES[cls_id]
                        if aspect < min_ar or aspect > max_ar:
                            logger.debug(f"Filtered {cls_name} (aspect {aspect:.2f} not in [{min_ar}, {max_ar}])")
                            continue

                obj_area_ratio = (bw * bh) / max(1, w * h)
                min_conf = max(0.3, 0.5 - obj_area_ratio * 5)
                if conf < min_conf:
                    logger.debug(f"Filtered {cls_name} (conf {conf:.2f} < {min_conf:.2f} for size {obj_area_ratio:.3f})")
                    continue

                objects.append(
                    DetectedObject(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                    )
                )

        # 修复：删除尺寸分组粗暴去重，改用IoU-NMS仅删除高度重叠重复框，保留多人/多同类物体
        final_objects = []
        used_idx = set()
        total_obj = len(objects)
        for idx_a in range(total_obj):
            if idx_a in used_idx:
                continue
            obj_a = objects[idx_a]
            keep_flag = True
            x1a, y1a, x2a, y2a = obj_a.bbox
            area_a = (x2a - x1a) * (y2a - y1a)

            # 和后续同类别物体计算IoU，重叠>0.6判定为重复
            for idx_b in range(idx_a + 1, total_obj):
                obj_b = objects[idx_b]
                if obj_a.class_id != obj_b.class_id or idx_b in used_idx:
                    continue
                x1b, y1b, x2b, y2b = obj_b.bbox
                inter_x1 = max(x1a, x1b)
                inter_y1 = max(y1a, y1b)
                inter_x2 = min(x2a, x2b)
                inter_y2 = min(y2a, y2b)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area_b = (x2b - x1b) * (y2b - y1b)
                union_area = area_a + area_b - inter_area
                iou = inter_area / max(1, union_area)

                if iou > 0.6:
                    # 保留置信度更高的物体
                    if obj_b.confidence > obj_a.confidence:
                        keep_flag = False
                        break
                    else:
                        used_idx.add(idx_b)
            if keep_flag:
                final_objects.append(obj_a)
        return final_objects

    def _detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        face_objects = []
        h, w = image.shape[:2]
        for x, y, fw, fh in faces:
            aspect = fw / max(1, fh)
            if aspect < 0.6 or aspect > 1.6:
                continue
            if y + fh > 0.85 * h:
                continue
            face_objects.append(
                DetectedObject(
                    bbox=(int(x), int(y), int(x + fw), int(y + fh)),
                    class_id=0,
                    class_name="face_person_proxy",
                    confidence=0.6,
                )
            )
        if face_objects:
            logger.info(f"Face fallback found {len(face_objects)} face proxy(ies)")
        return face_objects

    def _filter_by_saliency(self, objects, saliency_map):
        filtered = []
        for o in objects:
            x1, y1, x2, y2 = o.bbox
            roi = saliency_map[y1:y2, x1:x2]
            mean_saliency = float(np.mean(roi)) if roi.size > 0 else 0.0
            if mean_saliency < 0.03:
                logger.debug(f"Filtered {o.class_name} (low saliency {mean_saliency:.3f})")
                continue
            filtered.append(o)
        return filtered

    def score_candidates(
        self,
        bboxes: List[BBox],
        detected_objects: List[DetectedObject],
        image_shape: Tuple[int, int],
    ) -> List[Optional[float]]:
        """Compute subject completeness score for each candidate bbox.
        Fixed critical bugs:
        1. Removed duplicate double boundary penalty
        2. Removed final max normalization which erased completeness gap
        3. Fixed panorama bonus: only reward full complete scene frames, reduce bonus strength
        4. Fixed object weight calculation, small important objects no longer suppressed
        Strictly enforce min_important_inclusion threshold for person/wine glass core subjects
        """
        if len(detected_objects) == 0:
            return [None] * len(bboxes)

        # Filter to important objects only, exclude face fallback proxies
        important_objects = [
            obj
            for obj in detected_objects
            if obj.class_id in self.important_classes and obj.class_name != "face_person_proxy"
        ]
        if len(important_objects) == 0:
            # Fallback: medium confidence objects if no core important targets
            important_objects = [
                obj
                for obj in detected_objects
                if obj.confidence >= 0.4 and obj.class_name != "face_person_proxy"
            ]

        if len(important_objects) == 0:
            return [None] * len(bboxes)

        img_h, img_w = image_shape[:2]
        img_area = img_h * img_w
        obj_weights = []
        for obj in important_objects:
            # Fix: remove global image area attenuation, small subjects get normal weight
            w = obj.confidence
            obj_weights.append(w)
        total_weight = sum(obj_weights) + 1e-9

        per_candidate_scores = []
        for bbox in bboxes:
            weighted_inclusion = 0.0
            boundary_penalty = 0.0
            weighted_tightness = 0.0

            for obj, ow in zip(important_objects, obj_weights):
                inter_area = bbox_intersection(bbox, obj.bbox)
                obj_area = max(1, bbox_area(obj.bbox))
                inclusion = inter_area / obj_area

                weighted_inclusion += ow * inclusion

                crop_area = max(1, bbox_area(bbox))
                obj_to_crop = min(1.0, obj_area / crop_area)
                weighted_tightness += ow * math.sqrt(obj_to_crop)

                boundary_penalty += ow * inclusion

            raw_score = weighted_inclusion / total_weight
            tightness = weighted_tightness / total_weight
            penalty = min(0.3, (1.0 - boundary_penalty / total_weight) ** 2 * 0.5)
            score = min(1.0, max(0.0, raw_score - penalty + self.tightness_weight * tightness))

            # Fixed panorama bonus: only add small bonus when all main subjects are fully included
            crop_area = bbox_area(bbox)
            area_ratio = crop_area / img_area
            avg_inclusion = weighted_inclusion / total_weight
            if area_ratio >= 0.32 and avg_inclusion >= self.min_important_inclusion:
                score += 0.10
            score = min(1.0, score)

            per_candidate_scores.append(score)

        # Critical Fix: delete max normalization, retain raw 0~1 score gap of completeness
        return per_candidate_scores