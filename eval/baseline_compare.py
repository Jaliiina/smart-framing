"""Baseline comparison: Center Crop, Rule-based, Saliency-only, Aesthetic-only, YOLO-only, Full."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import AestheticCropper
from src.utils import BBox, bbox_iou, load_image


def center_crop(image: np.ndarray, area_ratio: float = 0.65) -> BBox:
    """Baseline 1: Crop the center region of the image."""
    h, w = image.shape[:2]
    area = h * w * area_ratio
    # Default aspect ratio = original
    ar = w / max(1, h)
    ch = int(math.sqrt(area / max(1e-6, ar)))
    cw = int(ch * ar)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - cw // 2)
    y1 = max(0, cy - ch // 2)
    x2 = min(w, x1 + cw)
    y2 = min(h, y1 + ch)
    return (x1, y1, x2, y2)


def rule_based_crop(image: np.ndarray, saliency_map: np.ndarray = None) -> BBox:
    """Baseline 2: Rule-based crop using composition rules only."""
    h, w = image.shape[:2]
    best_bbox = (0, 0, w, h)
    best_score = -1.0

    # Generate candidates with different scales/positions
    for area_r in [0.45, 0.55, 0.65, 0.75]:
        for ar in [1.0, 4/3, 3/4, 16/9, 9/16]:
            area = int(h * w * area_r)
            ch = int(math.sqrt(area / max(1e-6, ar)))
            cw = int(ch * ar)
            if ch >= h or cw >= w:
                continue
            # Test a few positions
            for cy_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
                for cx_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    cx, cy = int(w * cx_frac), int(h * cy_frac)
                    x1 = max(0, cx - cw // 2)
                    y1 = max(0, cy - ch // 2)
                    x2 = min(w, x1 + cw)
                    y2 = min(h, y1 + ch)
                    bbox = (x1, y1, x2, y2)

                    # Score: rule-of-thirds position
                    cx_norm = (x1 + x2) / 2.0 / w
                    cy_norm = (y1 + y2) / 2.0 / h
                    thirds_points = [(1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)]
                    min_dist = min(
                        math.sqrt((cx_norm - px)**2 + (cy_norm - py)**2)
                        for px, py in thirds_points
                    )
                    score = math.exp(-min_dist**2 / (2 * 0.08**2))

                    if score > best_score:
                        best_score = score
                        best_bbox = bbox

    return best_bbox


def run_baselines(
    cropper: AestheticCropper,
    test_items: List[Dict],
) -> List[Dict]:
    """Run all baseline methods and the full system."""
    results = []

    for item in test_items:
        image = load_image(item["image_path"])
        gt = item["gt_bbox"]
        record = {"image": item["image"]}

        # 1. Center Crop
        cc_bbox = center_crop(image)
        record["center_crop_iou"] = round(bbox_iou(cc_bbox, gt), 4)

        # 2. Rule-based
        rule_bbox = rule_based_crop(image)
        record["rule_based_iou"] = round(bbox_iou(rule_bbox, gt), 4)

        # 3. Saliency-only (weight: saliency=1.0, others=0)
        start = time.time()
        try:
            sal_result = cropper.process_with_custom_weights(
                item["image_path"],
                {"saliency": 1.0, "aesthetic": 0.0, "composition": 0.0, "subject": 0.0, "technical": 0.0},
            )
            record["saliency_only_iou"] = round(bbox_iou(sal_result.best_bbox, gt), 4)
        except Exception:
            record["saliency_only_iou"] = 0.0
        record["saliency_only_time"] = round(time.time() - start, 2)

        # 4. Aesthetic-only
        start = time.time()
        try:
            aes_result = cropper.process_with_custom_weights(
                item["image_path"],
                {"aesthetic": 1.0, "saliency": 0.0, "composition": 0.0, "subject": 0.0, "technical": 0.0},
            )
            record["aesthetic_only_iou"] = round(bbox_iou(aes_result.best_bbox, gt), 4)
        except Exception:
            record["aesthetic_only_iou"] = 0.0
        record["aesthetic_only_time"] = round(time.time() - start, 2)

        # 5. YOLO-only (subject)
        start = time.time()
        try:
            yolo_result = cropper.process_with_custom_weights(
                item["image_path"],
                {"subject": 1.0, "aesthetic": 0.0, "saliency": 0.0, "composition": 0.0, "technical": 0.0},
            )
            record["yolo_only_iou"] = round(bbox_iou(yolo_result.best_bbox, gt), 4)
        except Exception:
            record["yolo_only_iou"] = 0.0
        record["yolo_only_time"] = round(time.time() - start, 2)

        # 6. Full system
        start = time.time()
        try:
            full_result = cropper.process(item["image_path"])
            record["full_iou"] = round(bbox_iou(full_result.best_bbox, gt), 4)
            record["full_score"] = round(full_result.best_score, 4)
        except Exception:
            record["full_iou"] = 0.0
            record["full_score"] = 0.0
        record["full_time"] = round(time.time() - start, 2)

        results.append(record)
        print(f"  {item['image']}: center={record['center_crop_iou']:.4f} "
              f"rule={record['rule_based_iou']:.4f} "
              f"sal={record['saliency_only_iou']:.4f} "
              f"aes={record['aesthetic_only_iou']:.4f} "
              f"yolo={record['yolo_only_iou']:.4f} "
              f"full={record['full_iou']:.4f}")

    # Compute means
    methods = ["center_crop", "rule_based", "saliency_only", "aesthetic_only", "yolo_only", "full"]
    summary = {}
    for m in methods:
        ious = [r.get(f"{m}_iou", 0.0) for r in results]
        summary[f"{m}_mean_iou"] = round(float(np.mean(ious)), 4)

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison experiments")
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="baseline_results.json")
    args = parser.parse_args()

    cropper = AestheticCropper(config_path=args.config)

    data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    test_items = []
    for item in data:
        item["image_path"] = str(Path(args.image_root) / item["image"])
        item["gt_bbox"] = tuple(item["bbox"])
        test_items.append(item)

    print("=== Baseline Comparison ===")
    results, summary = run_baselines(cropper, test_items)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {"summary": summary, "results": results}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
