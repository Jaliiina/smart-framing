from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import BBox, bbox_center


class CompositionScorer:

    def __init__(self, config: dict):
        ccfg = config.get("composition", {})
        w = ccfg.get("weights", {})
        self.weight_thirds: float = w.get("rule_of_thirds", 0.35)
        self.weight_balance: float = w.get("center_balance", 0.25)
        self.weight_whitespace: float = w.get("whitespace", 0.15)
        self.weight_edge: float = w.get("edge_simplicity", 0.15)
        self.weight_symmetry: float = w.get("symmetry", 0.10)
        self.weight_person: float = w.get("person_completeness", 0.10)
        self.thirds_sigma: float = ccfg.get("thirds_sigma", 0.08)
        self.whitespace_ideal: float = ccfg.get("whitespace_ideal_ratio", 0.3)
        self.whitespace_ideal_plant: float = ccfg.get("whitespace_ideal_plant", 0.08)
        self.edge_penalty_plant: float = ccfg.get("edge_penalty_plant", 0.5)
        self.person_class_ids = {0}
        self.head_region_ratio = 0.2

    def score_candidates(
        self,
        image: np.ndarray,
        bboxes: List[BBox],
        saliency_map: Optional[np.ndarray] = None,
        detected_objects: Optional[list] = None,
    ) -> List[Tuple[float, Dict[str, float]]]:
        scores = []
        for bbox in bboxes:
            sub = self._score_single(image, bbox, saliency_map, detected_objects)
            if sub.get("is_landscape", False):
                total = (
                    self.weight_thirds * 0.35 * sub["thirds"]
                    + self.weight_balance * 1.8 * sub["center_balance"]
                    + self.weight_whitespace * sub["whitespace"]
                    + self.weight_edge * sub["edge_simplicity"]
                    + self.weight_symmetry * sub["symmetry"]
                    + self.weight_person * sub["person_completeness"]
                )
            else:
                total = (
                    self.weight_thirds * sub["thirds"]
                    + self.weight_balance * sub["center_balance"]
                    + self.weight_whitespace * sub["whitespace"]
                    + self.weight_edge * sub["edge_simplicity"]
                    + self.weight_symmetry * sub["symmetry"]
                    + self.weight_person * sub["person_completeness"]
                )
            scores.append((total, sub))
        return scores

    def _score_single(
        self,
        image: np.ndarray,
        bbox: BBox,
        saliency_map: Optional[np.ndarray],
        detected_objects: Optional[list],
    ) -> Dict[str, float]:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return {
                "thirds": 0.0,
                "center_balance": 0.0,
                "whitespace": 0.0,
                "edge_simplicity": 0.0,
                "symmetry": 0.0,
                "person_completeness": 0.0,
                "is_landscape": False
            }

        subject_center = self._find_subject_center(bbox, saliency_map, detected_objects)
        thirds = self._rule_of_thirds(subject_center, bbox)
        balance = self._center_balance(saliency_map, bbox) if saliency_map is not None else 0.5
        whitespace = self._whitespace(saliency_map, bbox) if saliency_map is not None else 0.5
        edge = self._edge_simplicity(image, bbox)
        symmetry = self._symmetry(crop)
        person_complete = self._person_completeness(image, bbox, detected_objects)
        gaze_ws = self._gaze_whitespace(image, bbox, detected_objects)
        horizon = self._horizon_level(image, bbox)

        is_landscape_scene = self._is_wide_landscape(crop)
        if is_landscape_scene:
            whitespace = self._whitespace_landscape(crop)

        is_plant = self._is_plant_texture(crop)
        if is_plant:
            whitespace = self._whitespace_plant(crop)
            edge = self._edge_simplicity_plant(crop)

        return {
            "thirds": thirds,
            "center_balance": balance,
            "whitespace": whitespace,
            "edge_simplicity": edge,
            "symmetry": symmetry,
            "person_completeness": person_complete,
            "gaze_whitespace": gaze_ws,
            "horizon_level": horizon,
            "is_landscape": is_landscape_scene
        }

    def _is_wide_landscape(self, crop: np.ndarray) -> bool:

        h, w = crop.shape[:2]
        if h < 60 or w < 60:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        sky_mask = ((S < 60) & (V > 160)).astype(np.float32)
        sky_ratio = sky_mask.mean()
        return sky_ratio > 0.4

    def _whitespace_landscape(self, crop: np.ndarray) -> float:

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_std = float(gray.std() / 255.0)
        if gray_std < 0.03:
            return 0.2
        elif gray_std < 0.12:
            return 0.2 + (gray_std - 0.03) / 0.09 * 0.6
        else:
            return 1.0

    def _find_subject_center(
        self,
        bbox: BBox,
        saliency_map: Optional[np.ndarray],
        detected_objects: Optional[list],
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        bx_cx, bx_cy = bbox_center(bbox)
        if detected_objects and len(detected_objects) > 0:
            from .utils import bbox_area, bbox_intersection
            best_obj = None
            best_overlap = 0
            for obj in detected_objects:
                inter = bbox_intersection(bbox, obj.bbox)
                obj_area = max(1, bbox_area(obj.bbox))
                overlap_ratio = inter / obj_area
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_obj = obj
            if best_obj is not None and best_overlap > 0.3:
                ox1, oy1, ox2, oy2 = best_obj.bbox
                cx = (ox1 + ox2) / 2.0
                cy = (oy1 + oy2) / 2.0
                return (cx, cy)
        if saliency_map is not None:
            region = saliency_map[max(0, y1):y2, max(0, x1):x2]
            region_sum = region.sum()
            if region_sum > 1e-6:
                local_h, local_w = region.shape[:2]
                ys, xs = np.mgrid[0:local_h, 0:local_w]
                cx = float((xs * region).sum() / region_sum) + x1
                cy = float((ys * region).sum() / region_sum) + y1
                return (cx, cy)
        return (bx_cx, bx_cy)

    def _rule_of_thirds(self, subject_center: Tuple[float, float], bbox: BBox) -> float:
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        sx = (subject_center[0] - x1) / bw
        sy = (subject_center[1] - y1) / bh
        sx = max(0.0, min(1.0, sx))
        sy = max(0.0, min(1.0, sy))
        points = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
        min_dist = float("inf")
        for px, py in points:
            d = math.sqrt((sx - px) ** 2 + (sy - py) ** 2)
            min_dist = min(min_dist, d)
        for val in [1 / 3, 2 / 3]:
            d_line_x = abs(sx - val)
            d_line_y = abs(sy - val)
            min_dist = min(min_dist, d_line_x * 0.5, d_line_y * 0.5)
        sigma = self.thirds_sigma
        score = math.exp(-min_dist ** 2 / (2 * sigma ** 2))
        return score

    def _center_balance(self, saliency_map: np.ndarray, bbox: BBox) -> float:
        x1, y1, x2, y2 = bbox
        region = saliency_map[max(0, y1):y2, max(0, x1):x2]
        h, w = region.shape[:2]
        if h < 4 or w < 4:
            return 0.5
        mid_x = w // 2
        left_weight = region[:, :mid_x].sum()
        right_weight = region[:, mid_x:].sum()
        total_lr = left_weight + right_weight + 1e-9
        lr_balance = 1.0 - abs(left_weight - right_weight) / total_lr
        mid_y = h // 2
        top_weight = region[:mid_y, :].sum()
        bottom_weight = region[mid_y:, :].sum()
        total_tb = top_weight + bottom_weight + 1e-9
        tb_balance = 1.0 - abs(top_weight - bottom_weight) / total_tb
        return 0.5 * lr_balance + 0.5 * tb_balance

    def _whitespace(self, saliency_map: np.ndarray, bbox: BBox) -> float:
        x1, y1, x2, y2 = bbox
        region = saliency_map[max(0, y1):y2, max(0, x1):x2]
        h, w = region.shape[:2]
        if h < 4 or w < 4:
            return 0.5
        threshold = 0.1
        whitespace_ratio = float((region < threshold).mean())
        deviation = abs(whitespace_ratio - self.whitespace_ideal)
        score = math.exp(-deviation ** 2 / (2 * 0.15 ** 2))
        return score

    def _edge_simplicity(self, image: np.ndarray, bbox: BBox) -> float:
        x1, y1, x2, y2 = bbox
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        h, w = image.shape[:2]
        strip = max(3, min(h, w) // 80)
        boundary_edge_count = 0
        boundary_pixel_count = 0
        # Top
        ry1, ry2 = max(0, y1 - strip), max(0, y1 + strip)
        if ry2 > ry1:
            boundary_edge_count += edges[ry1:ry2, max(0, x1):x2].sum()
            boundary_pixel_count += (ry2 - ry1) * (x2 - x1)
        # Bottom
        ry1, ry2 = max(0, y2 - strip), min(h, y2 + strip)
        if ry2 > ry1:
            boundary_edge_count += edges[ry1:ry2, max(0, x1):x2].sum()
            boundary_pixel_count += (ry2 - ry1) * (x2 - x1)
        # Left
        rx1, rx2 = max(0, x1 - strip), max(0, x1 + strip)
        if rx2 > rx1:
            boundary_edge_count += edges[y1:y2, rx1:rx2].sum()
            boundary_pixel_count += (y2 - y1) * (rx2 - rx1)
        # Right
        rx1, rx2 = max(0, x2 - strip), min(w, x2 + strip)
        if rx2 > rx1:
            boundary_edge_count += edges[y1:y2, rx1:rx2].sum()
            boundary_pixel_count += (y2 - y1) * (rx2 - rx1)
        if boundary_pixel_count == 0:
            return 0.5
        edge_density = boundary_edge_count / (boundary_pixel_count * 255)
        score = 1.0 - min(1.0, edge_density * 10)
        return max(0.0, score)

    def _symmetry(self, crop: np.ndarray) -> float:
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if max(h, w) > 128:
            scale = 128 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        h2, w2 = gray.shape[:2]
        mid_x = w2 // 2
        left = gray[:, :mid_x]
        right = gray[:, w2 - mid_x:][:, ::-1]
        lr_sym = 0.0
        if left.shape == right.shape:
            lr_diff = np.abs(left - right).mean() / 255.0
            lr_sym = 1.0 - lr_diff
        mid_y = h2 // 2
        top = gray[:mid_y, :]
        bottom = gray[h2 - mid_y:, :][::-1, :]
        tb_sym = 0.0
        if top.shape == bottom.shape:
            tb_diff = np.abs(top - bottom).mean() / 255.0
            tb_sym = 1.0 - tb_diff
        return max(lr_sym, tb_sym)

    def _person_completeness(
        self,
        image: np.ndarray,
        bbox: BBox,
        detected_objects: Optional[list],
    ) -> float:
        if detected_objects is None or len(detected_objects) == 0:
            return 0.5
        persons = [o for o in detected_objects if o.class_id in self.person_class_ids]
        if len(persons) == 0:
            return 0.5
        x1, y1, x2, y2 = bbox
        img_h, img_w = image.shape[:2]
        score_sum = 0.0
        weight_sum = 0.0
        for person in persons:
            px1, py1, px2, py2 = person.bbox
            obj_h = py2 - py1
            head_y2 = int(py1 + obj_h * self.head_region_ratio)
            head_in_crop = (head_y2 <= y2)
            feet_y = py2
            feet_ratio = feet_y / img_h if img_h > 0 else 0
            feet_tolerance = feet_ratio < 0.98
            if not head_in_crop:
                crop_ratio = (py1 - y1) / max(1, y2 - y1)
                score = max(0.0, 0.5 - crop_ratio * 0.5)
            elif not feet_tolerance:
                score = 0.7
            else:
                score = 1.0
            weight_sum += person.confidence
            score_sum += person.confidence * score
        if weight_sum > 0:
            return score_sum / weight_sum
        return 0.5

    def _gaze_whitespace(
        self,
        image: np.ndarray,
        bbox: BBox,
        detected_objects: Optional[list],
    ) -> float:
        if detected_objects is None or len(detected_objects) == 0:
            return 0.5
        persons = [o for o in detected_objects if o.class_id in self.person_class_ids]
        if len(persons) == 0:
            return 0.5
        x1, y1, x2, y2 = bbox
        score_sum = 0.0
        weight_sum = 0.0
        for person in persons:
            px1, py1, px2, py2 = person.bbox
            person_top = py1
            space_above = person_top - y1
            space_ratio = space_above / (y2 - y1) if (y2 - y1) > 0 else 0.0
            if 0.15 <= space_ratio <= 0.35:
                score = 1.0
            elif space_ratio < 0.15:
                score = max(0.3, space_ratio / 0.15)
            else:
                excess = space_ratio - 0.35
                score = max(0.3, 1.0 - excess * 2)
            weight_sum += person.confidence
            score_sum += person.confidence * score
        if weight_sum > 0:
            return score_sum / weight_sum
        return 0.5

    def _is_plant_texture(self, crop: np.ndarray) -> bool:
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 20, 30), (85, 255, 255))
        green_ratio = float(green_mask.mean())
        brown_mask = cv2.inRange(hsv, (10, 15, 50), (30, 120, 200))
        brown_ratio = float(brown_mask.mean())
        golden_mask = cv2.inRange(hsv, (20, 10, 100), (40, 80, 255))
        golden_ratio = float(golden_mask.mean())
        warm_grey_mask = cv2.inRange(hsv, (0, 0, 80), (40, 30, 200))
        warm_grey_ratio = float(warm_grey_mask.mean())
        plant_color_ratio = green_ratio + brown_ratio + golden_ratio + warm_grey_ratio
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float((edges > 0).mean())
        gray_std = float(gray.std() / 255.0)
        has_plant_colors = plant_color_ratio > 5.0
        has_natural_texture = (0.03 < edge_ratio < 0.40) and (gray_std > 0.08)
        return has_plant_colors or has_natural_texture

    def _whitespace_plant(self, crop: np.ndarray) -> float:
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness_std = float(gray.std() / 255.0)
        if brightness_std < 0.02:
            score = 0.2
        elif brightness_std < 0.08:
            score = 0.3 + (brightness_std - 0.02) / 0.06 * 0.3
        else:
            score = min(1.0, 0.6 + (brightness_std - 0.08) / 0.10 * 0.4)
        return score

    def _edge_simplicity_plant(self, crop: np.ndarray) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float((edges > 0).mean())
        if edge_density < 0.05:
            return 0.3
        elif edge_density < 0.15:
            score = 0.5 + (edge_density - 0.05) / 0.10 * 0.5
        elif edge_density <= 0.35:
            score = 1.0
        else:
            score = max(0.4, 1.0 - (edge_density - 0.35) * 0.5)
        return score

    def _horizon_level(self, image: np.ndarray, bbox: BBox) -> float:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 50 or w < 50:
            return 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
        if lines is None or len(lines) == 0:
            return 0.5
        angles = []
        for line in lines[:10]:
            rho, theta = line[0]
            angle_deg = abs(np.degrees(theta) - 90)
            if angle_deg < 15:
                angles.append(angle_deg)
        if len(angles) == 0:
            return 0.5
        avg_angle = sum(angles) / len(angles)
        if avg_angle <= 3:
            score = 1.0
        elif avg_angle <= 10:
            score = 1.0 - (avg_angle - 3) / 7 * 0.5
        else:
            score = max(0.2, 0.5 - (avg_angle - 10) / 20)
        return score