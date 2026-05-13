"""Predict on test set B: output coordinates, cropped images, and visualizations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import AestheticCropper
from src.utils import draw_bbox, load_image, save_image


def main():
    parser = argparse.ArgumentParser(description="Predict on test set B")
    parser.add_argument("--image-dir", type=str, required=True, help="Test set B image directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--output-dir", type=str, default="test_b_output", help="Output directory")
    parser.add_argument("--output-format", type=str, default="both", choices=["json", "csv", "both"],
                        help="Output format for coordinates")
    args = parser.parse_args()

    cropper = AestheticCropper(config_path=args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in exts)

    print(f"Processing {len(image_files)} images from {args.image_dir}")

    results = []
    for i, img_file in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_file.name}...", end=" ", flush=True)
        start = time.time()

        try:
            result = cropper.process(str(img_file))
            elapsed = time.time() - start

            # Save visualization
            image = load_image(str(img_file))
            vis = draw_bbox(image, result.best_bbox, f"score={result.best_score:.3f}")
            save_image(vis, str(out_dir / f"{img_file.stem}_vis.jpg"))

            # Save cropped image
            save_image(result.best_crop, str(out_dir / f"{img_file.stem}_crop.jpg"))

            results.append({
                "image": img_file.name,
                "bbox": list(result.best_bbox),
                "score": round(result.best_score, 4),
                "explanation": result.explanation,
                "time": round(elapsed, 2),
            })
            print(f"IoU-ready bbox={result.best_bbox} score={result.best_score:.4f} ({elapsed:.2f}s)")

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "image": img_file.name,
                "error": str(e),
            })

    # Save coordinates as JSON
    if args.output_format in ("json", "both"):
        json_path = out_dir / "predictions.json"
        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON coordinates saved to {json_path}")

    # Save coordinates as CSV
    if args.output_format in ("csv", "both"):
        csv_path = out_dir / "predictions.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("image,x1,y1,x2,y2,score,explanation\n")
            for r in results:
                if "bbox" in r:
                    f.write(f"{r['image']},{r['bbox'][0]},{r['bbox'][1]},{r['bbox'][2]},{r['bbox'][3]},{r['score']},{r.get('explanation','')}\n")
        print(f"CSV coordinates saved to {csv_path}")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
