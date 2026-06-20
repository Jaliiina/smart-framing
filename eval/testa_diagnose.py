"""Diagnose AestheticCropper on the paired testA framing dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import AestheticCropper
from src.utils import bbox_iou, draw_bbox, load_image, save_image


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
        gt[name] = tuple(
            int(value)
            for value in (
                max(0, int(round(cx_px - box_w / 2))),
                max(0, int(round(cy_px - box_h / 2))),
                min(w, int(round(cx_px + box_w / 2))),
                min(h, int(round(cy_px + box_h / 2))),
            )
        )
    return gt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="testA/testA")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs/testa_diagnose")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth(dataset_dir)
    cropper = AestheticCropper(config_path=args.config)
    records = []

    for index, (name, gt_bbox) in enumerate(sorted(ground_truth.items()), start=1):
        image_path = dataset_dir / name
        image = load_image(str(image_path))
        start = time.time()

        if hasattr(cropper.saliency_det, "detect_dual"):
            saliency_map, fallback_sal_map, is_uniform, _fallback_uniform = (
                cropper.saliency_det.detect_dual(image)
            )
        else:
            saliency_map, is_uniform = cropper.saliency_det.detect(image)
            fallback_sal_map = saliency_map

        objects = cropper.subject_det.detect(image, saliency_map=saliency_map)
        candidates = cropper.candidate_gen.generate(
            image, saliency_map, detected_objects=objects
        )
        oracle_iou = max((bbox_iou(box, gt_bbox) for box in candidates), default=0.0)

        aesthetic_scores = cropper.aesthetic_scorer.score_candidates(image, candidates)
        saliency_scores = cropper.saliency_det.score_candidates(
            saliency_map, candidates, image.shape
        )
        dual_saliency_scores = (
            cropper.saliency_det.score_candidates(
                fallback_sal_map, candidates, image.shape
            )
            if fallback_sal_map is not saliency_map
            else saliency_scores
        )
        composition_scores = cropper.comp_scorer.score_candidates(
            image, candidates, saliency_map, objects
        )
        subject_scores = cropper.subject_det.score_candidates(
            candidates, objects, image.shape
        )
        technical_scores = cropper.tech_scorer.score_candidates(image, candidates)
        semantic_heatmaps = cropper.semantic_heatmap_scorer.build_heatmaps(image)
        subjectness_maps = cropper.subjectness_scorer.build_maps(
            image=image,
            saliency_map=saliency_map,
            detected_objects=objects,
            semantic_heatmaps=semantic_heatmaps,
        )
        subjectness_scores = cropper.subjectness_scorer.score_candidates(
            candidates, subjectness_maps
        )
        semantic_scores = cropper.semantic_crop_scorer.score_candidates(image, candidates)
        roi_discard_scores = cropper.roi_discard_scorer.score_candidates(
            image=image,
            bboxes=candidates,
            saliency_map=saliency_map,
            detected_objects=objects,
            subjectness_maps=subjectness_maps,
            semantic_scores=semantic_scores,
        )

        original_top_k = cropper.fusion.top_k_display
        cropper.fusion.top_k_display = len(candidates)
        try:
            fused = cropper.fusion.fuse(
                bboxes=candidates,
                aesthetic_scores=aesthetic_scores,
                saliency_scores=saliency_scores,
                composition_scores=composition_scores,
                subject_scores=subject_scores,
                technical_scores=technical_scores,
                roi_discard_scores=roi_discard_scores,
                semantic_scores=semantic_scores,
                subjectness_scores=subjectness_scores,
                saliency_is_uniform=is_uniform,
                has_subject=any(score is not None for score in subject_scores),
                image_shape=image.shape,
                saliency_map=saliency_map,
                return_all=True,
                dual_saliency_scores=dual_saliency_scores,
            )
            best, ranked = fused[0], fused[2] or fused[1]
            if getattr(cropper, "reranker", None) is not None:
                ranked = cropper.reranker.rerank(ranked, image.shape[:2])
                best = ranked[0]
            ranked = cropper.scientific_optimizer.optimize(
                image=image,
                ranked=ranked,
                detected_objects=objects,
                saliency_map=saliency_map,
                subjectness_maps=subjectness_maps,
            )
            best = ranked[0]
        finally:
            cropper.fusion.top_k_display = original_top_k

        elapsed = time.time() - start
        pred_iou = bbox_iou(best.bbox, gt_bbox)
        pred_area = (
            (best.bbox[2] - best.bbox[0])
            * (best.bbox[3] - best.bbox[1])
            / (image.shape[0] * image.shape[1])
        )

        record = {
            "image": name,
            "pred_bbox": [int(x) for x in best.bbox],
            "gt_bbox": list(gt_bbox),
            "iou": pred_iou,
            "oracle_iou": oracle_iou,
            "candidate_count": len(candidates),
            "pred_area_ratio": float(pred_area),
            "saliency_uniform": bool(is_uniform),
            "object_count": len(objects),
            "elapsed": float(elapsed),
            **{
                key: float(value)
                for key, value in best.sub_scores.__dict__.items()
            },
            "candidates": [
                {
                    "bbox": [int(x) for x in candidate.bbox],
                    "iou": float(bbox_iou(candidate.bbox, gt_bbox)),
                    "final_score": float(candidate.final_score),
                    **{
                        key: float(value)
                        for key, value in candidate.sub_scores.__dict__.items()
                    },
                }
                for candidate in ranked
            ],
        }
        records.append(record)

        pred_vis = draw_bbox(image, gt_bbox, "GT", (0, 255, 0), 3)
        pred_vis = draw_bbox(
            pred_vis, best.bbox, f"Pred IoU={pred_iou:.3f}", (0, 0, 255), 3
        )
        save_image(pred_vis, str(output_dir / f"{Path(name).stem}_compare.jpg"))
        print(
            f"[{index:02d}/{len(ground_truth)}] {name} "
            f"IoU={pred_iou:.3f} oracle={oracle_iou:.3f} "
            f"area={pred_area:.3f} candidates={len(candidates)}"
        )

    ious = np.array([r["iou"] for r in records])
    oracle_ious = np.array([r["oracle_iou"] for r in records])
    summary = {
        "count": len(records),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "recall_iou_0.5": float((ious >= 0.5).mean()),
        "recall_iou_0.7": float((ious >= 0.7).mean()),
        "mean_oracle_iou": float(oracle_ious.mean()),
        "oracle_recall_iou_0.7": float((oracle_ious >= 0.7).mean()),
        "mean_elapsed": float(np.mean([r["elapsed"] for r in records])),
    }

    (output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "results": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
