"""Ablation experiment: evaluate system with modules removed one at a time."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import AestheticCropper
from src.utils import bbox_iou, load_config


ABLATION_CONFIGS = {
    "full": {},  # Use all modules with default weights
    "w/o_aesthetic": {"aesthetic": 0.0},
    "w/o_saliency": {"saliency": 0.0},
    "w/o_yolo": {"subject": 0.0},
    "w/o_composition": {"composition": 0.0},
    "w/o_technical": {"technical": 0.0},
    "grid_only": {"aesthetic": 0.0, "saliency": 0.0, "subject": 0.0, "composition": 0.0, "technical": 0.0},
}


def run_ablation(
    cropper: AestheticCropper,
    test_items: List[Dict],
    name: str,
    weight_overrides: Dict[str, float],
) -> Dict:
    """Run one ablation configuration."""
    # Set custom weights
    if weight_overrides:
        result = cropper.process_with_custom_weights(
            test_items[0]["image_path"],  # dummy call to set weights
            weight_overrides,
        )
        # Actually we need to set weights manually and re-run all items
        for k, v in weight_overrides.items():
            setattr(cropper.fusion, f"weight_{k}", v)
        # Normalize remaining weights
        total = sum(
            getattr(cropper.fusion, f"weight_{k}")
            for k in ["aesthetic", "saliency", "composition", "subject", "technical"]
        )
        if total > 0:
            for k in ["aesthetic", "saliency", "composition", "subject", "technical"]:
                val = getattr(cropper.fusion, f"weight_{k}")
                setattr(cropper.fusion, f"weight_{k}", val / total)

    ious = []
    scores = []
    times = []

    for item in test_items:
        start = time.time()
        try:
            result = cropper.process(item["image_path"])
            iou = bbox_iou(result.best_bbox, item["gt_bbox"])
            ious.append(iou)
            scores.append(result.best_score)
        except Exception:
            ious.append(0.0)
            scores.append(0.0)
        times.append(time.time() - start)

    # Restore default weights
    config = load_config()
    fcfg = config.get("fusion", {}).get("weights", {})
    for k in ["aesthetic", "saliency", "composition", "subject", "technical"]:
        setattr(cropper.fusion, f"weight_{k}", fcfg.get(k, 0.2))

    return {
        "name": name,
        "mean_iou": round(float(np.mean(ious)), 4),
        "mean_score": round(float(np.mean(scores)), 4),
        "mean_time": round(float(np.mean(times)), 2),
        "weight_overrides": weight_overrides,
    }


def run_k_ablation(cropper: AestheticCropper, test_items: List[Dict], k_values: List[int]) -> List[Dict]:
    """Run ablation with different candidate counts K."""
    results = []
    for k in k_values:
        # Temporarily override top_k
        original_k = cropper.candidate_gen.top_k
        cropper.candidate_gen.top_k = k

        ious = []
        times = []
        for item in test_items:
            start = time.time()
            try:
                result = cropper.process(item["image_path"])
                iou = bbox_iou(result.best_bbox, item["gt_bbox"])
                ious.append(iou)
            except Exception:
                ious.append(0.0)
            times.append(time.time() - start)

        results.append({
            "name": f"K={k}",
            "mean_iou": round(float(np.mean(ious)), 4),
            "mean_time": round(float(np.mean(times)), 2),
        })
        cropper.candidate_gen.top_k = original_k

    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation experiments")
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="ablation_results.json")
    args = parser.parse_args()

    cropper = AestheticCropper(config_path=args.config)

    # Load test set
    data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    test_items = []
    for item in data:
        item["image_path"] = str(Path(args.image_root) / item["image"])
        item["gt_bbox"] = tuple(item["bbox"])
        test_items.append(item)

    # Run module ablation
    print("=== Module Ablation ===")
    module_results = []
    for name, overrides in ABLATION_CONFIGS.items():
        print(f"Running {name}...")
        r = run_ablation(cropper, test_items, name, overrides)
        module_results.append(r)
        print(f"  mIoU={r['mean_iou']:.4f} mean_score={r['mean_score']:.4f} time={r['mean_time']:.2f}s")

    # Run K ablation
    print("\n=== K-value Ablation ===")
    k_values = [50, 100, 150, 200, 500]
    k_results = run_k_ablation(cropper, test_items, k_values)
    for r in k_results:
        print(f"  {r['name']}: mIoU={r['mean_iou']:.4f} time={r['mean_time']:.2f}s")

    # Save
    output = {
        "module_ablation": module_results,
        "k_ablation": k_results,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
