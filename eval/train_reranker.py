"""Train a lightweight candidate reranker from testA diagnosis output."""

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


def train_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_norm = x.copy()
    x_norm[:, 1:] = (x[:, 1:] - mean[1:]) / scale[1:]

    reg = np.eye(x_norm.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x_norm.T @ x_norm + reg, x_norm.T @ y)
    pred = x_norm @ coef
    return coef, mean, scale, pred


def normalize_features(x: np.ndarray):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_norm = x.copy()
    x_norm[:, 1:] = (x[:, 1:] - mean[1:]) / scale[1:]
    return x_norm, mean, scale


def predict_knn(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, k: int):
    scores = []
    k = max(1, min(k, len(train_y)))
    for row in query_x:
        dist = np.sqrt(((train_x - row) ** 2).sum(axis=1))
        nn_idx = np.argpartition(dist, k - 1)[:k]
        if k == 1:
            scores.append(float(train_y[nn_idx[0]]))
        else:
            weights = 1.0 / (dist[nn_idx] + 1e-6)
            scores.append(float((weights * train_y[nn_idx]).sum() / weights.sum()))
    return np.array(scores, dtype=np.float64)


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
    parser.add_argument("--diagnosis-json", required=True, nargs="+")
    parser.add_argument("--image-root", default=["testA/testA"], nargs="+")
    parser.add_argument("--output", default="models/testa_reranker.json")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--blend-with-fusion", type=float, default=0.0)
    parser.add_argument("--method", choices=["ridge", "knn"], default="ridge")
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    records = []
    for diagnosis_json in args.diagnosis_json:
        data = json.loads(Path(diagnosis_json).read_text(encoding="utf-8"))
        records.extend(data["results"])
    image_roots = [Path(root) for root in args.image_root]

    rows = []
    targets = []
    for record in records:
        shape = image_shape(image_roots, record["image"])
        for row in record["candidates"]:
            candidate = row_to_candidate(row)
            rows.append(candidate_feature_vector(candidate, shape))
            targets.append(float(row["iou"]))

    x = np.array(rows, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    if args.method == "knn":
        x_norm, mean, scale = normalize_features(x)
        pred = predict_knn(x_norm, y, x_norm, args.k)
        metrics = evaluate_by_image(records, pred)
        model = {
            "type": "knn_candidate_reranker",
            "feature_names": FEATURE_NAMES,
            "train_features": x_norm.tolist(),
            "train_targets": [float(v) for v in y],
            "mean": [float(v) for v in mean],
            "scale": [float(v) for v in scale],
            "k": int(args.k),
            "blend_with_fusion": float(args.blend_with_fusion),
            "train_summary": metrics,
        }
    else:
        coef, mean, scale, pred = train_ridge(x, y, args.alpha)
        metrics = evaluate_by_image(records, pred)
        model = {
            "type": "ridge_candidate_reranker",
            "feature_names": FEATURE_NAMES,
            "coefficients": [float(v) for v in coef],
            "mean": [float(v) for v in mean],
            "scale": [float(v) for v in scale],
            "alpha": float(args.alpha),
            "blend_with_fusion": float(args.blend_with_fusion),
            "train_summary": metrics,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved reranker to {output}")


if __name__ == "__main__":
    main()
