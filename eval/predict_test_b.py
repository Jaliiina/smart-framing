"""Predict on test set B and write submission-ready outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import AestheticCropper
from src.utils import BBox, draw_bbox, load_image, save_image


def bbox_to_normalized_center(
    bbox: BBox,
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> dict[str, float]:
    """Convert xyxy pixel bbox to normalized center-format coordinates."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return {
        "cx": ((x1 + x2) / 2.0) / max(1, w),
        "cy": ((y1 + y2) / 2.0) / max(1, h),
        "yw": (x2 - x1) / max(1, w),
        "yh": (y2 - y1) / max(1, h),
    }


def iter_template_names(template_csv: Path | None) -> list[str] | None:
    if template_csv is None or not template_csv.exists():
        return None
    with template_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "result_img_name" not in (reader.fieldnames or []):
            return None
        return [row["result_img_name"] for row in reader if row.get("result_img_name")]


def make_contact_sheet(
    visual_paths: Iterable[Path],
    output_path: Path,
    thumb_width: int = 360,
    columns: int = 4,
) -> None:
    """Create a quick visual overview for manual TestB inspection."""
    images = []
    for path in visual_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = thumb_width / max(1, w)
        thumb = cv2.resize(image, (thumb_width, max(1, int(h * scale))))
        images.append((path.stem, thumb))

    if not images:
        return

    label_h = 28
    max_h = max(img.shape[0] for _, img in images) + label_h
    rows = int(np.ceil(len(images) / columns))
    sheet = np.full((rows * max_h, columns * thumb_width, 3), 255, dtype=np.uint8)

    for idx, (name, thumb) in enumerate(images):
        row, col = divmod(idx, columns)
        y = row * max_h
        x = col * thumb_width
        cv2.putText(
            sheet,
            name.replace("_vis", ""),
            (x + 8, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        sheet[y + label_h:y + label_h + thumb.shape[0], x:x + thumb.shape[1]] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def write_submission_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = ["result_img_name", "cx", "cy", "yw", "yh"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "result_img_name": row["image"],
                    "cx": f"{row['cx']:.6f}",
                    "cy": f"{row['cy']:.6f}",
                    "yw": f"{row['yw']:.6f}",
                    "yh": f"{row['yh']:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on test set B")
    parser.add_argument("--image-dir", type=str, required=True, help="Test set B image directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--output-dir", type=str, default="outputs/test_b", help="Output directory")
    parser.add_argument(
        "--template-csv",
        type=str,
        default="",
        help="Optional submit_teamnumber.csv template to preserve row order",
    )
    args = parser.parse_args()

    cropper = AestheticCropper(config_path=args.config)
    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    vis_dir = out_dir / "visualizations"
    crop_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.image_dir)
    template = Path(args.template_csv) if args.template_csv else img_dir / "submit_teamnumber.csv"
    template_names = iter_template_names(template)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if template_names:
        image_files = [img_dir / name for name in template_names if (img_dir / name).exists()]
    else:
        image_files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in exts)

    print(f"Processing {len(image_files)} images from {img_dir}")

    records = []
    visual_paths = []
    for i, image_file in enumerate(image_files, start=1):
        print(f"[{i:02d}/{len(image_files)}] {image_file.name}...", end=" ", flush=True)
        start = time.time()

        try:
            image = load_image(str(image_file))
            result = cropper.process(str(image_file))
            elapsed = time.time() - start
            norm = bbox_to_normalized_center(result.best_bbox, image.shape)

            vis = draw_bbox(image, result.best_bbox, f"score={result.best_score:.3f}")
            vis_path = vis_dir / f"{image_file.stem}_vis.jpg"
            crop_path = crop_dir / f"{image_file.stem}_crop.jpg"
            save_image(vis, str(vis_path))
            save_image(result.best_crop, str(crop_path))
            visual_paths.append(vis_path)

            record = {
                "image": image_file.name,
                "bbox": [int(v) for v in result.best_bbox],
                **norm,
                "score": round(float(result.best_score), 6),
                "sub_scores": {
                    "aesthetic": round(float(result.best_sub_scores.aesthetic), 6),
                    "saliency": round(float(result.best_sub_scores.saliency), 6),
                    "composition": round(float(result.best_sub_scores.composition), 6),
                    "subject": round(float(result.best_sub_scores.subject), 6),
                    "technical": round(float(result.best_sub_scores.technical), 6),
                    "area_prior": round(float(result.best_sub_scores.area_prior), 6),
                },
                "explanation": result.explanation,
                "time": round(elapsed, 2),
            }
            records.append(record)
            print(
                f"bbox={result.best_bbox} "
                f"norm=({norm['cx']:.3f},{norm['cy']:.3f},{norm['yw']:.3f},{norm['yh']:.3f}) "
                f"score={result.best_score:.4f} ({elapsed:.2f}s)"
            )
        except Exception as exc:
            print(f"Error: {exc}")
            records.append({"image": image_file.name, "error": str(exc)})

    predictions_json = out_dir / "predictions.json"
    predictions_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    valid_records = [row for row in records if "bbox" in row]
    submission_csv = out_dir / "submission.csv"
    write_submission_csv(valid_records, submission_csv)

    detailed_csv = out_dir / "predictions_detail.csv"
    with detailed_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "image",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy",
            "yw",
            "yh",
            "score",
            "explanation",
            "time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in valid_records:
            x1, y1, x2, y2 = row["bbox"]
            writer.writerow(
                {
                    "image": row["image"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": f"{row['cx']:.6f}",
                    "cy": f"{row['cy']:.6f}",
                    "yw": f"{row['yw']:.6f}",
                    "yh": f"{row['yh']:.6f}",
                    "score": row["score"],
                    "explanation": row["explanation"],
                    "time": row["time"],
                }
            )

    make_contact_sheet(visual_paths, out_dir / "contact_sheet.jpg")

    print(f"\nSubmission CSV saved to {submission_csv}")
    print(f"Detailed JSON saved to {predictions_json}")
    print(f"Visualizations saved to {vis_dir}")
    print(f"Crops saved to {crop_dir}")


if __name__ == "__main__":
    main()
