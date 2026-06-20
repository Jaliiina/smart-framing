"""Scientific ablation experiments for the aesthetic cropper.

Runs controlled variants on paired TestA data and reports mIoU/Recall metrics.
The variants are designed for presentation: each one adds a named technical
component instead of changing individual test images.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import AestheticCropper
from src.utils import bbox_iou, load_config, load_image


ROW_RE = re.compile(
    r"\|\s*(A\d+\.jpg)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*"
    r"\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)


def load_ground_truth(dataset_dir: Path) -> dict[str, tuple[int, int, int, int]]:
    text = (dataset_dir / "README.md").read_text(encoding="utf-8")
    gt = {}
    for name, cx, cy, bw, bh in ROW_RE.findall(text):
        image = load_image(str(dataset_dir / name))
        h, w = image.shape[:2]
        cx_px, cy_px = float(cx) * w, float(cy) * h
        box_w, box_h = float(bw) * w, float(bh) * h
        gt[name] = (
            max(0, int(round(cx_px - box_w / 2))),
            max(0, int(round(cy_px - box_h / 2))),
            min(w, int(round(cx_px + box_w / 2))),
            min(h, int(round(cy_px + box_h / 2))),
        )
    return gt


def variant_config(base: dict, name: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg.setdefault("reranker", {})["enabled"] = False
    cfg.setdefault("scientific_optimizer", {})["enabled"] = False
    cfg.setdefault("refiner", {})["enabled"] = False
    cfg.setdefault("fusion", {}).setdefault("robust_rank_fusion", {})["enabled"] = False

    weights = cfg.setdefault("fusion", {}).setdefault("weights", {})
    if name == "base_fusion":
        weights["roi_discard"] = 0.0
        weights["semantic"] = 0.0
        weights["subjectness"] = 0.0
        cfg.setdefault("semantic_crop", {})["enabled"] = False
    elif name == "clip_semantic":
        weights["roi_discard"] = 0.0
        weights["semantic"] = 0.12
        weights["subjectness"] = 0.0
        cfg.setdefault("semantic_crop", {})["enabled"] = True
    elif name == "subjectness_distractor":
        weights["roi_discard"] = 0.10
        weights["semantic"] = 0.12
        weights["subjectness"] = 0.12
        cfg.setdefault("semantic_crop", {})["enabled"] = True
    elif name == "robust_rank":
        weights["roi_discard"] = 0.10
        weights["semantic"] = 0.12
        weights["subjectness"] = 0.12
        cfg.setdefault("semantic_crop", {})["enabled"] = True
        cfg["fusion"]["robust_rank_fusion"]["enabled"] = True
    elif name == "pairwise_ranker":
        weights["roi_discard"] = 0.10
        weights["semantic"] = 0.12
        weights["subjectness"] = 0.12
        cfg.setdefault("semantic_crop", {})["enabled"] = True
        cfg["fusion"]["robust_rank_fusion"]["enabled"] = True
        cfg["reranker"]["enabled"] = Path(cfg["reranker"].get("model_path", "")).exists()
    elif name == "optimizer":
        weights["roi_discard"] = 0.10
        weights["semantic"] = 0.12
        weights["subjectness"] = 0.12
        cfg["fusion"]["robust_rank_fusion"]["enabled"] = True
        cfg["scientific_optimizer"]["enabled"] = True
    elif name == "full":
        cfg["scientific_optimizer"]["enabled"] = True
        weights["roi_discard"] = 0.10
        weights["semantic"] = 0.12
        weights["subjectness"] = 0.12
        cfg.setdefault("semantic_crop", {})["enabled"] = True
        cfg["fusion"]["robust_rank_fusion"]["enabled"] = True
        cfg["reranker"]["enabled"] = Path(cfg["reranker"].get("model_path", "")).exists()
    else:
        raise ValueError(f"Unknown variant: {name}")
    return cfg


def evaluate_variant(dataset_dir: Path, gt: dict, cfg: dict) -> dict:
    cropper = AestheticCropper(config=cfg)
    ious = []
    rows = []
    for name, gt_bbox in sorted(gt.items()):
        result = cropper.process(str(dataset_dir / name))
        iou = bbox_iou(result.best_bbox, gt_bbox)
        ious.append(iou)
        rows.append(
            {
                "image": name,
                "iou": float(iou),
                "bbox": [int(v) for v in result.best_bbox],
                "score": float(result.best_score),
            }
        )
    arr = np.array(ious, dtype=np.float64)
    return {
        "mean_iou": float(arr.mean()),
        "median_iou": float(np.median(arr)),
        "recall_iou_0.5": float((arr >= 0.5).mean()),
        "recall_iou_0.7": float((arr >= 0.7).mean()),
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="testA/testA")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs/scientific_ablation")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    gt = load_ground_truth(dataset_dir)
    variants = [
        "base_fusion",
        "clip_semantic",
        "subjectness_distractor",
        "robust_rank",
        "pairwise_ranker",
        "optimizer",
        "full",
    ]
    summary_rows = []
    all_results = {}
    for variant in variants:
        print(f"Running {variant}...")
        result = evaluate_variant(dataset_dir, gt, variant_config(base_cfg, variant))
        all_results[variant] = result
        summary_rows.append(
            {
                "variant": variant,
                "mean_iou": result["mean_iou"],
                "median_iou": result["median_iou"],
                "recall_iou_0.5": result["recall_iou_0.5"],
                "recall_iou_0.7": result["recall_iou_0.7"],
            }
        )

    (output_dir / "ablation_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "mean_iou", "median_iou", "recall_iou_0.5", "recall_iou_0.7"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
