import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from smart_framing import LinearAestheticModel, SmartFramer, bbox_iou


def load_annotations(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(data, list), 'annotations must be a list'
    return data


def build_training_set(samples: List[Dict], img_root: Path) -> tuple:
    framer = SmartFramer()
    xs, ys = [], []
    for s in samples:
        img = cv2.imread(str(img_root / s['image']))
        gt = tuple(s['bbox'])
        cand = framer.generate_candidates(img)
        for b in cand:
            f, _ = framer.extract_features(img, b)
            iou = bbox_iou(b, gt)
            target = 0.65 * iou + 0.35 * float(s.get('score', 0.8))
            xs.append(f)
            ys.append(target)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def evaluate(samples: List[Dict], img_root: Path, model: LinearAestheticModel):
    framer = SmartFramer(model)
    ious, pred_scores = [], []
    for s in samples:
        img = cv2.imread(str(img_root / s['image']))
        gt = tuple(s['bbox'])
        pred = framer.predict_best_crop(img)
        ious.append(bbox_iou(pred.bbox, gt))
        pred_scores.append(pred.aesthetic_score)
    return float(np.mean(ious)), float(np.mean(pred_scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations', type=str, required=True)
    ap.add_argument('--image-root', type=str, required=True)
    ap.add_argument('--model-out', type=str, default='model_weights.npz')
    args = ap.parse_args()

    items = load_annotations(Path(args.annotations))
    split = int(len(items) * 0.8)
    train_items, val_items = items[:split], items[split:]

    x, y = build_training_set(train_items, Path(args.image_root))
    model = LinearAestheticModel(l2=1e-2)
    model.fit(x, y)
    np.savez(args.model_out, w=model.w, b=model.b)

    miou, mscore = evaluate(val_items if val_items else train_items, Path(args.image_root), model)
    print(f'mIoU={miou:.4f}')
    print(f'mean_pred_score={mscore:.4f}')


if __name__ == '__main__':
    main()
