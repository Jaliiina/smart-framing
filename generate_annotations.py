"""Generate testA annotations from the README table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - used only outside the CV environment
    cv2 = None


ROW_RE = re.compile(
    r"\|\s*(\S+\.jpg)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*"
    r"\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)


def parse_readme(readme_path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Parse normalized crop boxes from the testA README table."""
    text = readme_path.read_text(encoding="utf-8")
    data: dict[str, tuple[float, float, float, float]] = {}
    for name, cx, cy, w, h in ROW_RE.findall(text):
        data[name] = (float(cx), float(cy), float(w), float(h))
    return data


def normalized_center_to_bbox(
    cx_norm: float,
    cy_norm: float,
    w_norm: float,
    h_norm: float,
    image_w: int,
    image_h: int,
) -> list[int]:
    """Convert normalized center-format boxes to pixel xyxy boxes."""
    box_w = round(w_norm * image_w)
    box_h = round(h_norm * image_h)
    center_x = round(cx_norm * image_w)
    center_y = round(cy_norm * image_h)

    x1 = max(0, int(round(center_x - box_w / 2)))
    y1 = max(0, int(round(center_y - box_h / 2)))
    x2 = min(image_w, int(round(center_x + box_w / 2)))
    y2 = min(image_h, int(round(center_y + box_h / 2)))
    return [x1, y1, x2, y2]


def read_image_size(image_path: Path) -> tuple[int, int] | None:
    """Return (width, height), using OpenCV when available and Pillow otherwise."""
    if cv2 is not None:
        image = cv2.imread(str(image_path))
        if image is not None:
            image_h, image_w = image.shape[:2]
            return image_w, image_h

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None


def generate_annotations(
    image_dir: Path,
    readme_path: Path,
    output_json: Path,
) -> list[dict]:
    """Generate annotation records compatible with eval scripts."""
    readme_data = parse_readme(readme_path)
    annotations = []

    for image_name, (cx_norm, cy_norm, w_norm, h_norm) in sorted(readme_data.items()):
        image_path = image_dir / image_name
        if not image_path.exists():
            print(f"Warning: image not found, skipping: {image_path}")
            continue

        image_size = read_image_size(image_path)
        if image_size is None:
            print(f"Warning: failed to read image, skipping: {image_path}")
            continue

        image_w, image_h = image_size
        annotations.append(
            {
                "image": image_name,
                "bbox": normalized_center_to_bbox(
                    cx_norm, cy_norm, w_norm, h_norm, image_w, image_h
                ),
                "score": 1.0,
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Generated {len(annotations)} annotations: {output_json}")
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate testA annotations JSON")
    parser.add_argument("--image-dir", default="testA/testA")
    parser.add_argument("--readme", default="testA/testA/README.md")
    parser.add_argument("--output", default="testA/testA/annotations.json")
    args = parser.parse_args()

    generate_annotations(
        image_dir=Path(args.image_dir),
        readme_path=Path(args.readme),
        output_json=Path(args.output),
    )


if __name__ == "__main__":
    main()
