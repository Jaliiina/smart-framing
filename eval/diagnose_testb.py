"""Print ranked candidates and detected objects for selected TestB images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import AestheticCropper
from src.utils import bbox_area


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="D:/cvProject/testB")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("names", nargs="*", default=["B11.jpg", "B15.jpg"])
    args = parser.parse_args()

    cropper = AestheticCropper(args.config)
    image_dir = Path(args.image_dir)

    for name in args.names:
        result = cropper.process(str(image_dir / name))
        h, w = result.saliency_map.shape[:2]
        print(
            f"\n{name} best={result.best_bbox} "
            f"area={bbox_area(result.best_bbox) / max(1, h * w):.3f} "
            f"score={result.best_score:.3f}"
        )
        objects = [
            (obj.class_name, obj.class_id, round(obj.confidence, 2), obj.bbox)
            for obj in result.detected_objects
        ]
        print("objects:", objects)
        for cand in result.all_candidates[:10]:
            sub = cand.sub_scores
            print(
                f"  bbox={cand.bbox} score={cand.final_score:.3f} "
                f"area={bbox_area(cand.bbox) / max(1, h * w):.3f} "
                f"a={sub.aesthetic:.2f} sal={sub.saliency:.2f} "
                f"comp={sub.composition:.2f} subj={sub.subject:.2f} "
                f"tech={sub.technical:.2f} areaP={sub.area_prior:.2f}"
            )


if __name__ == "__main__":
    main()
