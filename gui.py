import argparse
from pathlib import Path

import cv2
import numpy as np

from smart_framing import LinearAestheticModel, SmartFramer, draw_bbox


def load_model(path: Path) -> LinearAestheticModel:
    d = np.load(path)
    m = LinearAestheticModel()
    m.w = d['w']
    m.b = float(d['b'])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--model', type=str, default='')
    ap.add_argument('--out', type=str, default='framing_result.jpg')
    args = ap.parse_args()

    image = cv2.imread(args.image)
    model = load_model(Path(args.model)) if args.model else None
    framer = SmartFramer(model)
    pred = framer.predict_best_crop(image)

    vis = draw_bbox(image, pred.bbox, f'score={pred.aesthetic_score:.3f}')
    crop = image[pred.bbox[1]:pred.bbox[3], pred.bbox[0]:pred.bbox[2]]

    cv2.imwrite(args.out, vis)
    cv2.imwrite(str(Path(args.out).with_name(Path(args.out).stem + '_crop.jpg')), crop)
    print('bbox:', pred.bbox)
    print('score:', pred.aesthetic_score)
    print('features:', pred.feature_scores)


if __name__ == '__main__':
    main()
