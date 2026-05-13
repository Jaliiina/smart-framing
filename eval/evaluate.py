"""Evaluation script: compute IoU, scores, and metrics on test set A."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import AestheticCropper
from src.utils import BBox, bbox_iou, load_config


def load_test_set_a(annotations_path: str, image_root: str) -> List[Dict]:
    """Load test set A annotations.

    Expected format:
    [
        {"image": "img001.jpg", "bbox": [x1, y1, x2, y2], "score": 0.88},
        ...
    ]
    """
    data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    for item in data:
        item["image_path"] = str(Path(image_root) / item["image"])
        item["gt_bbox"] = tuple(item["bbox"])
    return data


def evaluate_single(cropper: AestheticCropper, item: Dict) -> Dict:
    """Evaluate a single image and return metrics."""
    start = time.time()
    result = cropper.process(item["image_path"])
    elapsed = time.time() - start

    iou = bbox_iou(result.best_bbox, item["gt_bbox"])

    return {
        "image": item["image"],
        "pred_bbox": list(result.best_bbox),
        "gt_bbox": list(item["gt_bbox"]),
        "iou": round(iou, 4),
        "score": round(result.best_score, 4),
        "time": round(elapsed, 2),
        "explanation": result.explanation,
        "sub_scores": {
            "aesthetic": round(result.best_sub_scores.aesthetic, 4),
            "saliency": round(result.best_sub_scores.saliency, 4),
            "composition": round(result.best_sub_scores.composition, 4),
            "subject": round(result.best_sub_scores.subject, 4),
            "technical": round(result.best_sub_scores.technical, 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on test set A")
    parser.add_argument("--annotations", type=str, required=True, help="Test set A annotations JSON")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory for images")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output results JSON")
    args = parser.parse_args()

    cropper = AestheticCropper(config_path=args.config)
    test_set = load_test_set_a(args.annotations, args.image_root)

    results = []
    for i, item in enumerate(test_set):
        print(f"[{i+1}/{len(test_set)}] Processing {item['image']}...")
        try:
            r = evaluate_single(cropper, item)
            results.append(r)
            print(f"  IoU={r['iou']:.4f} score={r['score']:.4f} time={r['time']:.2f}s")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"image": item["image"], "error": str(e)})

    # Compute summary statistics
    valid = [r for r in results if "iou" in r]
    if valid:
        ious = [r["iou"] for r in valid]
        scores = [r["score"] for r in valid]
        times = [r["time"] for r in valid]

        summary = {
            "num_images": len(valid),
            "mean_iou": round(float(np.mean(ious)), 4),
            "median_iou": round(float(np.median(ious)), 4),
            "min_iou": round(float(np.min(ious)), 4),
            "max_iou": round(float(np.max(ious)), 4),
            "mean_score": round(float(np.mean(scores)), 4),
            "mean_time": round(float(np.mean(times)), 2),
        }

        print("\n=== Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    else:
        summary = {"num_images": 0}

    # Save results
    output = {"summary": summary, "results": results}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
