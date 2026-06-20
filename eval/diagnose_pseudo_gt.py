"""Build candidate diagnostics against pseudo ground-truth boxes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import AestheticCropper
from src.utils import bbox_iou, load_image


def load_pseudo_boxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes = {}
    for row in data:
        if "image" in row and "bbox" in row:
            boxes[row["image"]] = tuple(int(v) for v in row["bbox"])
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--pseudo-json", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs/pseudo_diagnose")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pseudo_boxes = load_pseudo_boxes(Path(args.pseudo_json))
    cropper = AestheticCropper(config_path=args.config)
    # Diagnostics should measure the base candidate set, not recursively use
    # the learned reranker being trained.
    cropper.reranker = None
    records = []

    for index, (name, target_bbox) in enumerate(sorted(pseudo_boxes.items()), start=1):
        image_path = image_dir / name
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
        finally:
            cropper.fusion.top_k_display = original_top_k

        elapsed = time.time() - start
        pred_iou = bbox_iou(best.bbox, target_bbox)
        oracle_iou = max((bbox_iou(box, target_bbox) for box in candidates), default=0.0)
        pred_area = (
            (best.bbox[2] - best.bbox[0])
            * (best.bbox[3] - best.bbox[1])
            / (image.shape[0] * image.shape[1])
        )

        record = {
            "image": name,
            "pred_bbox": [int(x) for x in best.bbox],
            "gt_bbox": list(target_bbox),
            "iou": float(pred_iou),
            "oracle_iou": float(oracle_iou),
            "candidate_count": len(candidates),
            "pred_area_ratio": float(pred_area),
            "saliency_uniform": bool(is_uniform),
            "object_count": len(objects),
            "elapsed": float(elapsed),
            **{key: float(value) for key, value in best.sub_scores.__dict__.items()},
            "candidates": [
                {
                    "bbox": [int(x) for x in candidate.bbox],
                    "iou": float(bbox_iou(candidate.bbox, target_bbox)),
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
        print(
            f"[{index:02d}/{len(pseudo_boxes)}] {name} "
            f"IoU={pred_iou:.3f} oracle={oracle_iou:.3f} "
            f"area={pred_area:.3f} candidates={len(candidates)}"
        )

    ious = np.array([r["iou"] for r in records], dtype=np.float64)
    oracle_ious = np.array([r["oracle_iou"] for r in records], dtype=np.float64)
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
