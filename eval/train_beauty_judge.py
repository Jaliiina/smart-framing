"""Train the CLIP-backed beauty preference judge from candidate diagnostics.

The model learns from hard crop pairs. For testA, the human framing box gives a
preference target through IoU; visual cleanliness signals provide a small
auxiliary correction so the model does not learn only geometry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beauty_judge import BeautyFeatureExtractor, score_with_model, train_pairwise_ridge
from src.utils import CandidateResult, SubScores, bbox_iou, load_image


ROW_RE = re.compile(
    r"\|\s*(A\d+\.jpg)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*"
    r"\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)


def load_ground_truth(dataset_dir: Path) -> dict[str, tuple[int, int, int, int]]:
    readme = dataset_dir / "README.md"
    if not readme.exists():
        return {}
    text = readme.read_text(encoding="utf-8")
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


def row_to_candidate(row: dict) -> CandidateResult:
    sub = SubScores(
        aesthetic=float(row.get("aesthetic", 0.0)),
        saliency=float(row.get("saliency", 0.0)),
        composition=float(row.get("composition", 0.0)),
        subject=float(row.get("subject", 0.0)),
        technical=float(row.get("technical", 0.0)),
        area_prior=float(row.get("area_prior", 0.0)),
        thirds=float(row.get("thirds", 0.0)),
        center_balance=float(row.get("center_balance", 0.0)),
        whitespace=float(row.get("whitespace", 0.0)),
        edge_simplicity=float(row.get("edge_simplicity", 0.0)),
        symmetry=float(row.get("symmetry", 0.0)),
        sharpness=float(row.get("sharpness", 0.0)),
        brightness=float(row.get("brightness", 0.0)),
        contrast=float(row.get("contrast", 0.0)),
        saturation=float(row.get("saturation", 0.0)),
        person_completeness=float(row.get("person_completeness", 0.5)),
        roi_discard=float(row.get("roi_discard", 0.0)),
        roi_saliency=float(row.get("roi_saliency", 0.0)),
        discard_quality=float(row.get("discard_quality", 0.0)),
        boundary_cut=float(row.get("boundary_cut", 0.0)),
        distractor_penalty=float(row.get("distractor_penalty", 0.0)),
        semantic_score=float(row.get("semantic_score", 0.0)),
        positive_semantic=float(row.get("positive_semantic", 0.0)),
        negative_semantic=float(row.get("negative_semantic", 0.0)),
        subjectness=float(row.get("subjectness", 0.0)),
        distractor_map_score=float(row.get("distractor_map_score", 0.0)),
        good_discard=float(row.get("good_discard", 0.0)),
        bad_discard=float(row.get("bad_discard", 0.0)),
        visual_artifact_penalty=float(row.get("visual_artifact_penalty", 0.0)),
        blank_area_penalty=float(row.get("blank_area_penalty", 0.0)),
        saturated_boundary_penalty=float(row.get("saturated_boundary_penalty", 0.0)),
        small_saturated_object_penalty=float(row.get("small_saturated_object_penalty", 0.0)),
    )
    return CandidateResult(
        bbox=tuple(int(v) for v in row["bbox"]),
        final_score=float(row.get("final_score", 0.0)),
        sub_scores=sub,
    )


def auxiliary_clean_score(row: dict) -> float:
    artifact = float(row.get("visual_artifact_penalty", 0.0))
    blank = float(row.get("blank_area_penalty", 0.0))
    saturated = float(row.get("saturated_boundary_penalty", 0.0))
    small_sat = float(row.get("small_saturated_object_penalty", 0.0))
    distractor = max(
        float(row.get("distractor_map_score", 0.0)),
        float(row.get("distractor_penalty", 0.0)),
    )
    boundary = float(row.get("boundary_cut", 0.0))
    semantic = float(row.get("semantic_score", 0.0))
    composition = float(row.get("composition", 0.0))
    aesthetic = float(row.get("aesthetic", 0.0))
    clean = 1.0 - np.clip(
        0.36 * artifact
        + 0.22 * blank
        + 0.16 * small_sat
        + 0.12 * saturated
        + 0.10 * distractor
        + 0.08 * boundary,
        0.0,
        1.0,
    )
    return float(np.clip(0.42 * clean + 0.24 * semantic + 0.20 * composition + 0.14 * aesthetic, 0.0, 1.0))


def load_records(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records.extend(data.get("results", []))
    return records


def prepare_examples(
    records: list[dict],
    dataset_dir: Path,
    extractor: BeautyFeatureExtractor,
    top_candidates: int,
    target_iou_weight: float,
) -> list[dict]:
    ground_truth = load_ground_truth(dataset_dir)
    examples = []
    for record in records:
        name = record["image"]
        image_path = dataset_dir / name
        if not image_path.exists():
            continue
        image = load_image(str(image_path))
        gt = tuple(record.get("gt_bbox") or ground_truth.get(name) or ())
        rows = list(record.get("candidates", []))[:top_candidates]
        if not rows:
            continue
        candidates = [row_to_candidate(row) for row in rows]
        x = extractor.build_matrix(image, candidates, image_shape=image.shape[:2])
        labels = []
        for row in rows:
            iou = float(row.get("iou", 0.0))
            if not iou and len(gt) == 4:
                iou = bbox_iou(tuple(int(v) for v in row["bbox"]), gt)
            clean = auxiliary_clean_score(row)
            labels.append(float(np.clip(target_iou_weight * iou + (1.0 - target_iou_weight) * clean, 0.0, 1.0)))
        examples.append(
            {
                "image": name,
                "x": x,
                "labels": np.array(labels, dtype=np.float64),
                "ious": np.array([float(row.get("iou", 0.0)) for row in rows], dtype=np.float64),
                "bboxes": [tuple(int(v) for v in row["bbox"]) for row in rows],
                "fusion_scores": np.array([float(row.get("final_score", 0.0)) for row in rows], dtype=np.float64),
            }
        )
    return examples


def build_pairwise_dataset(
    examples: list[dict],
    min_gap: float,
    max_pairs_per_image: int,
    hard_negative_boost: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    rows = []
    targets = []
    for example in examples:
        labels = example["labels"]
        fusion = example["fusion_scores"]
        pairs = []
        n = len(labels)
        for hi in range(n):
            for lo in range(n):
                gap = labels[hi] - labels[lo]
                if gap < min_gap:
                    continue
                hard = fusion[lo] >= np.percentile(fusion, 70) and labels[lo] <= np.percentile(labels, 45)
                priority = float(gap + (0.20 if hard and hard_negative_boost else 0.0))
                pairs.append((priority, hi, lo, float(gap)))
        pairs.sort(reverse=True)
        if len(pairs) > max_pairs_per_image:
            head = pairs[: max_pairs_per_image // 2]
            tail = pairs[max_pairs_per_image // 2:]
            take = max_pairs_per_image - len(head)
            if take > 0 and tail:
                idx = rng.choice(len(tail), size=min(take, len(tail)), replace=False)
                head.extend(tail[int(i)] for i in idx)
            pairs = head
        for _priority, hi, lo, gap in pairs:
            diff = example["x"][hi] - example["x"][lo]
            diff[0] = 0.0
            rows.append(diff)
            targets.append(min(1.0, gap / 0.45))
    if not rows:
        raise ValueError("No hard pairs produced; lower --min-gap or inspect diagnostics.")
    return np.array(rows, dtype=np.float64), np.array(targets, dtype=np.float64)


def evaluate_examples(examples: list[dict], coef: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> dict:
    picked_ious = []
    baseline_ious = []
    oracle_ious = []
    picked_ranks = []
    for example in examples:
        scores = score_with_model(example["x"], coef, mean, scale)
        pred_idx = int(np.argmax(scores))
        picked_ious.append(float(example["ious"][pred_idx]))
        baseline_ious.append(float(example["ious"][0]))
        oracle_ious.append(float(np.max(example["ious"])))
        order = np.argsort(scores)[::-1]
        best_iou_idx = int(np.argmax(example["ious"]))
        rank = int(np.where(order == best_iou_idx)[0][0]) + 1
        picked_ranks.append(rank)
    return {
        "count": len(examples),
        "mean_iou": float(np.mean(picked_ious)) if picked_ious else 0.0,
        "baseline_mean_iou": float(np.mean(baseline_ious)) if baseline_ious else 0.0,
        "oracle_mean_iou": float(np.mean(oracle_ious)) if oracle_ious else 0.0,
        "recall_iou_0.5": float(np.mean(np.array(picked_ious) >= 0.5)) if picked_ious else 0.0,
        "mean_rank_of_oracle": float(np.mean(picked_ranks)) if picked_ranks else 0.0,
    }


def leave_one_out(
    examples: list[dict],
    alpha: float,
    min_gap: float,
    max_pairs_per_image: int,
) -> dict:
    picked = []
    baseline = []
    oracle = []
    for idx in range(len(examples)):
        train = examples[:idx] + examples[idx + 1:]
        test = [examples[idx]]
        pair_x, pair_y = build_pairwise_dataset(train, min_gap, max_pairs_per_image, True)
        coef, mean, scale = train_pairwise_ridge(pair_x, pair_y, alpha)
        metrics = evaluate_examples(test, coef, mean, scale)
        picked.append(metrics["mean_iou"])
        baseline.append(metrics["baseline_mean_iou"])
        oracle.append(metrics["oracle_mean_iou"])
    return {
        "loo_mean_iou": float(np.mean(picked)) if picked else 0.0,
        "loo_baseline_mean_iou": float(np.mean(baseline)) if baseline else 0.0,
        "loo_oracle_mean_iou": float(np.mean(oracle)) if oracle else 0.0,
        "loo_recall_iou_0.5": float(np.mean(np.array(picked) >= 0.5)) if picked else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIP-backed beauty judge")
    parser.add_argument("--diagnosis-json", required=True, nargs="+")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", default="models/beauty_judge.json")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--top-candidates", type=int, default=120)
    parser.add_argument("--max-pairs-per-image", type=int, default=260)
    parser.add_argument("--min-gap", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=12.0)
    parser.add_argument("--target-iou-weight", type=float, default=0.82)
    parser.add_argument("--blend-with-fusion", type=float, default=0.20)
    parser.add_argument("--takeover-margin", type=float, default=0.015)
    parser.add_argument("--top-n", type=int, default=64)
    parser.add_argument("--skip-loo", action="store_true")
    args = parser.parse_args()

    records = load_records(args.diagnosis_json)
    extractor = BeautyFeatureExtractor(
        clip_model=args.clip_model,
        device=args.device,
        projection_dim=args.projection_dim,
    )
    examples = prepare_examples(
        records=records,
        dataset_dir=Path(args.dataset_dir),
        extractor=extractor,
        top_candidates=args.top_candidates,
        target_iou_weight=args.target_iou_weight,
    )
    if not examples:
        raise ValueError("No examples found. Check --dataset-dir and diagnosis JSON.")

    pair_x, pair_y = build_pairwise_dataset(
        examples,
        min_gap=args.min_gap,
        max_pairs_per_image=args.max_pairs_per_image,
        hard_negative_boost=True,
    )
    coef, mean, scale = train_pairwise_ridge(pair_x, pair_y, args.alpha)
    train_metrics = evaluate_examples(examples, coef, mean, scale)
    loo_metrics = (
        {}
        if args.skip_loo
        else leave_one_out(examples, args.alpha, args.min_gap, args.max_pairs_per_image)
    )
    summary = {
        **train_metrics,
        **loo_metrics,
        "example_count": len(examples),
        "pair_count": int(len(pair_y)),
        "feature_count": int(pair_x.shape[1]),
        "alpha": float(args.alpha),
        "min_gap": float(args.min_gap),
    }

    model = {
        "type": "clip_pairwise_beauty_judge",
        "feature_names": extractor.feature_names,
        "coefficients": [float(v) for v in coef],
        "mean": [float(v) for v in mean],
        "scale": [float(v) for v in scale],
        "clip_model": args.clip_model,
        "device": args.device,
        "projection_dim": int(args.projection_dim),
        "projection_seed": int(extractor.projection_seed),
        "top_n": int(args.top_n),
        "blend_with_fusion": float(args.blend_with_fusion),
        "takeover_margin": float(args.takeover_margin),
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved beauty judge to {output}")


if __name__ == "__main__":
    main()
