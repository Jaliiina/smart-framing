"""Train the scientific crop candidate reranker.

The model predicts candidate crop quality from interpretable features:
fusion score, CLIP/LAION aesthetics, saliency, subject completeness,
ROI/discard separation, boundary-cut penalty, and geometry. The optional
leave-one-out mode reports whether the learned ranker generalizes across
TestA images rather than memorizing the training set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reranker import FEATURE_NAMES, candidate_feature_vector
from src.utils import CandidateResult, SubScores


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


def image_shape(image_roots: list[Path], image_name: str) -> tuple[int, int, int]:
    image = None
    tried = []
    for image_root in image_roots:
        path = image_root / image_name
        tried.append(path)
        image = cv2.imread(str(path))
        if image is not None:
            break
    if image is None:
        raise FileNotFoundError(", ".join(str(path) for path in tried))
    return image.shape


def train_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_norm = x.copy()
    x_norm[:, 1:] = (x[:, 1:] - mean[1:]) / scale[1:]

    reg = np.eye(x_norm.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    # Weighted least squares: X'WX instead of X'X
    w_sqrt = np.sqrt(weights[:, np.newaxis])
    xw = x_norm * w_sqrt
    yw = y * np.sqrt(weights)
    coef = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    pred = x_norm @ coef
    return coef, mean, scale, pred


def build_pairwise_rows(
    x_norm: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    records: list[dict],
    min_gap: float = 0.08,
    max_pairs_per_image: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    targets = []
    weights = []
    offset = 0
    rng = np.random.default_rng(2026)
    for record in records:
        n = len(record["candidates"])
        idx = np.arange(offset, offset + n)
        yy = y[idx]
        pairs = []
        for hi in range(n):
            for lo in range(n):
                gap = yy[hi] - yy[lo]
                if gap >= min_gap:
                    pairs.append((hi, lo, float(gap)))
        if len(pairs) > max_pairs_per_image:
            keep = rng.choice(len(pairs), size=max_pairs_per_image, replace=False)
            pairs = [pairs[int(i)] for i in keep]
        for hi, lo, gap in pairs:
            diff = x_norm[offset + hi] - x_norm[offset + lo]
            diff[0] = 0.0
            rows.append(diff)
            targets.append(min(1.0, gap / 0.5))
            weights.append(float(sample_weights[offset + hi]))
        offset += n
    if not rows:
        return x_norm, y, sample_weights
    return (
        np.array(rows, dtype=np.float64),
        np.array(targets, dtype=np.float64),
        np.array(weights, dtype=np.float64),
    )


def train_pairwise_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    records: list[dict],
    alpha: float,
):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_norm = x.copy()
    x_norm[:, 1:] = (x[:, 1:] - mean[1:]) / scale[1:]

    pair_x, pair_y, pair_w = build_pairwise_rows(x_norm, y, weights, records)
    reg = np.eye(x_norm.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    w_sqrt = np.sqrt(pair_w[:, np.newaxis])
    xw = pair_x * w_sqrt
    yw = pair_y * np.sqrt(pair_w)
    coef = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    pred = x_norm @ coef
    return coef, mean, scale, pred


def evaluate_leave_one_out(
    x: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    records: list[dict],
    alpha: float,
    training_objective: str,
) -> dict:
    offset = 0
    ious = []
    for record in records:
        n = len(record["candidates"])
        test_slice = slice(offset, offset + n)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_slice] = False
        train_records = records[:]
        train_records.pop(len(ious))
        if training_objective == "pairwise":
            coef, mean, scale, _pred = train_pairwise_ridge(
                x[train_mask],
                y[train_mask],
                sample_weights[train_mask],
                train_records,
                alpha,
            )
        else:
            coef, mean, scale, _pred = train_ridge(
                x[train_mask],
                y[train_mask],
                sample_weights[train_mask],
                alpha,
            )
        q = x[test_slice].copy()
        q[:, 1:] = (q[:, 1:] - mean[1:]) / scale[1:]
        scores = q @ coef
        best_idx = int(np.argmax(scores))
        ious.append(float(record["candidates"][best_idx]["iou"]))
        offset += n
    arr = np.array(ious, dtype=np.float64)
    return {
        "mean_iou": float(arr.mean()),
        "median_iou": float(np.median(arr)),
        "recall_iou_0.5": float((arr >= 0.5).mean()),
        "recall_iou_0.7": float((arr >= 0.7).mean()),
        "ious": ious,
    }


def evaluate_by_image(records: list[dict], pred: np.ndarray) -> dict:
    offset = 0
    ious = []
    for record in records:
        n = len(record["candidates"])
        image_pred = pred[offset:offset + n]
        best_idx = int(np.argmax(image_pred))
        ious.append(float(record["candidates"][best_idx]["iou"]))
        offset += n
    arr = np.array(ious, dtype=np.float64)
    return {
        "mean_iou": float(arr.mean()),
        "median_iou": float(np.median(arr)),
        "recall_iou_0.5": float((arr >= 0.5).mean()),
        "recall_iou_0.7": float((arr >= 0.7).mean()),
        "ious": ious,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned crop reranker")
    parser.add_argument("--diagnosis-json", required=True, nargs="+",
                        help="TestA diagnosis JSON(s)")
    parser.add_argument("--image-root", default=["testA/testA"], nargs="+",
                        help="Image root for TestA (order matches --diagnosis-json)")
    parser.add_argument("--flms-json",
                        help="FLMS diagnosis JSON (from prepare_flms.py)")
    parser.add_argument("--flms-image-root", default=["datasets/image"], nargs="+",
                        help="Image root for FLMS")
    parser.add_argument("--flms-weight", type=float, default=1.0,
                        help="Sample weight multiplier for FLMS records (lower = less influence)")
    parser.add_argument("--output", default="models/scientific_ridge_reranker.json")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--blend-with-fusion", type=float, default=0.0)
    parser.add_argument("--training-objective", choices=["pairwise", "direct"], default="pairwise")
    parser.add_argument("--leave-one-out", action="store_true")
    args = parser.parse_args()

    records = []
    image_roots = [Path(root) for root in args.image_root]
    flms_image_roots = [Path(root) for root in args.flms_image_root]

    for diagnosis_json in args.diagnosis_json:
        data = json.loads(Path(diagnosis_json).read_text(encoding="utf-8"))
        records.extend(data["results"])

    # Append FLMS records with FLMS image roots
    flms_records = []
    if args.flms_json:
        flms_data = json.loads(Path(args.flms_json).read_text(encoding="utf-8"))
        flms_records = flms_data["results"]
        print(f"Loaded {len(flms_records)} FLMS records")
        records.extend(flms_records)

    rows = []
    targets = []
    sample_weights = []
    # FLMS records are appended at the end, count them
    n_flms = len(flms_records) if args.flms_json else 0
    n_testa = len(records) - n_flms

    for rec_idx, record in enumerate(records):
        is_flms = rec_idx >= n_testa
        roots = flms_image_roots if is_flms else image_roots
        shape = image_shape(roots, record["image"])
        weight = args.flms_weight if is_flms else 1.0
        for row in record["candidates"]:
            candidate = row_to_candidate(row)
            rows.append(candidate_feature_vector(candidate, shape))
            targets.append(float(row["iou"]))
            sample_weights.append(weight)

    x = np.array(rows, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    sample_weights = np.array(sample_weights, dtype=np.float64)
    if args.training_objective == "pairwise":
        coef, mean, scale, pred = train_pairwise_ridge(
            x, y, sample_weights, records, args.alpha
        )
        metrics = evaluate_by_image(records, pred)
        model = {
            "type": "pairwise_ridge_candidate_reranker",
            "feature_names": FEATURE_NAMES,
            "coefficients": [float(v) for v in coef],
            "mean": [float(v) for v in mean],
            "scale": [float(v) for v in scale],
            "alpha": float(args.alpha),
            "blend_with_fusion": float(args.blend_with_fusion),
            "training_objective": "pairwise",
            "train_summary": metrics,
        }
    else:
        coef, mean, scale, pred = train_ridge(x, y, sample_weights, args.alpha)
        metrics = evaluate_by_image(records, pred)
        model = {
            "type": "ridge_candidate_reranker",
            "feature_names": FEATURE_NAMES,
            "coefficients": [float(v) for v in coef],
            "mean": [float(v) for v in mean],
            "scale": [float(v) for v in scale],
            "alpha": float(args.alpha),
            "blend_with_fusion": float(args.blend_with_fusion),
            "training_objective": "direct_iou",
            "train_summary": metrics,
        }

    if args.leave_one_out:
        model["leave_one_out_summary"] = evaluate_leave_one_out(
            x, y, sample_weights, records, args.alpha, args.training_objective
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    if args.leave_one_out and "leave_one_out_summary" in model:
        print("Leave-one-out:")
        print(json.dumps(model["leave_one_out_summary"], indent=2))
    print(f"Saved reranker to {output}")


if __name__ == "__main__":
    main()
