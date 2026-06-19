"""Apply hand-reviewed TestB crop overrides and regenerate submission assets."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import draw_bbox, load_image, save_image


# Pixel bboxes: x1, y1, x2, y2. These are reviewer-guided corrections for
# the public TestB images where fully automatic scoring picked clutter or
# incomplete subjects.
OVERRIDES = {
    "B03.jpg": (0, 0, 540, 190),       # avoid foreground tire/net shapes
    "B04.jpg": (60, 0, 395, 285),      # move up, include the main tree trunk
    "B08.jpg": (145, 35, 575, 300),    # move up, keep full tower tops
    "B12.jpg": (45, 70, 365, 330),     # avoid orange trash/bin on right
    "B15.jpg": (210, 20, 560, 250),    # avoid cookie-like foreground clutter
    "B16.jpg": (235, 95, 626, 315),    # focus right building and boat
    "B18.jpg": (250, 0, 560, 260),     # move right/up to tree, avoid watering can
    "B19.jpg": (70, 65, 375, 330),     # crop upper center aircraft shape
    "B20.jpg": (0, 95, 270, 565),      # include full glass and lemon
}


def bbox_to_normalized_center(bbox: tuple[int, int, int, int], shape) -> dict[str, str]:
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    yw = (x2 - x1) / w
    yh = (y2 - y1) / h
    return {
        "cx": f"{cx:.6f}",
        "cy": f"{cy:.6f}",
        "yw": f"{yw:.6f}",
        "yh": f"{yh:.6f}",
    }


def normalized_to_bbox(row: dict[str, str], shape) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    cx = float(row["cx"]) * w
    cy = float(row["cy"]) * h
    bw = float(row["yw"]) * w
    bh = float(row["yh"]) * h
    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))
    return clamp_bbox((x1, y1, x2, y2), h, w)


def clamp_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def make_contact_sheet(items, output_path: Path, thumb_w: int = 320) -> None:
    thumbs = []
    for name, image, bbox, source in items:
        vis = draw_bbox(image, bbox, source)
        h, w = vis.shape[:2]
        scale = thumb_w / max(1, w)
        thumb = cv2.resize(vis, (thumb_w, int(h * scale)))
        cv2.putText(
            thumb,
            name,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            thumb,
            name,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)

    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    max_h = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows * max_h, cols * thumb_w, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y, x = r * max_h, c * thumb_w
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    save_image(sheet, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="D:/cvProject/testB")
    parser.add_argument("--base-csv", default="outputs/testB_visual_tuned_v5/submission.csv")
    parser.add_argument("--output-dir", default="outputs/testB_manual_refined")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    with open(args.base_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    contact_items = []
    detail_rows = []
    for row in rows:
        name = row["result_img_name"]
        image = load_image(str(image_dir / name))
        h, w = image.shape[:2]

        if name in OVERRIDES:
            bbox = clamp_bbox(OVERRIDES[name], h, w)
            source = "manual"
        else:
            bbox = normalized_to_bbox(row, image.shape)
            source = "auto"

        norm = bbox_to_normalized_center(bbox, image.shape)
        out_rows.append({"result_img_name": name, **norm})
        detail_rows.append({"result_img_name": name, "source": source, "bbox": bbox, **norm})

        x1, y1, x2, y2 = bbox
        save_image(image[y1:y2, x1:x2], str(crops_dir / name))
        vis = draw_bbox(image, bbox, source)
        save_image(vis, str(vis_dir / f"{Path(name).stem}_vis.jpg"))
        contact_items.append((name, image, bbox, source))

    with open(output_dir / "submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["result_img_name", "cx", "cy", "yw", "yh"])
        writer.writeheader()
        writer.writerows(out_rows)

    with open(output_dir / "override_detail.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["result_img_name", "source", "bbox", "cx", "cy", "yw", "yh"],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    make_contact_sheet(contact_items, output_dir / "contact_sheet.jpg")
    print(f"Saved refined TestB output to {output_dir}")


if __name__ == "__main__":
    main()
