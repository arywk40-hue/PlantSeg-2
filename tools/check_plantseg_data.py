# Copyright (c) OpenMMLab. All rights reserved.
"""Sanity-check the PlantSeg-115 dataset before spending GPU hours on it.

Verifies image/mask pairing, split sizes, label-id range, and per-split class
coverage. Run this first -- the local copy may be an incomplete download.

Usage:
    python tools/check_plantseg_data.py [--data-root data/plantseg115]
"""
import argparse
import os.path as osp
from collections import Counter
from glob import glob

import numpy as np
from PIL import Image

# Paper (arXiv:2409.04038) reports 11,458 annotated images over 115 classes.
PAPER_TOTAL = 11458
NUM_CLASSES = 116  # 1 background + 115 diseases


def parse_args():
    p = argparse.ArgumentParser(description='Check PlantSeg dataset integrity')
    p.add_argument('--data-root', default='data/plantseg115')
    p.add_argument(
        '--sample',
        type=int,
        default=0,
        help='Only scan every Nth mask (0 = scan all). Use for a fast pass.')
    return p.parse_args()


def main():
    args = parse_args()
    root = args.data_root
    total = 0
    all_ids = set()
    problems = []

    for split in ('train', 'val', 'test'):
        img_dir = osp.join(root, 'images', split)
        ann_dir = osp.join(root, 'annotations', split)
        imgs = sorted(glob(osp.join(img_dir, '*.jpg')))
        anns = sorted(glob(osp.join(ann_dir, '*.png')))
        total += len(imgs)

        img_stems = {osp.splitext(osp.basename(f))[0] for f in imgs}
        ann_stems = {osp.splitext(osp.basename(f))[0] for f in anns}
        missing_ann = img_stems - ann_stems
        missing_img = ann_stems - img_stems

        scan = anns[::args.sample] if args.sample else anns
        split_ids = Counter()
        bad_range = []
        for f in scan:
            arr = np.array(Image.open(f))
            if arr.ndim != 2:
                problems.append(f'{f}: mask is not single-channel '
                                f'(shape {arr.shape}) -- must be grayscale P/L')
                continue
            vals, counts = np.unique(arr, return_counts=True)
            for v, c in zip(vals, counts):
                split_ids[int(v)] += int(c)
            over = [int(v) for v in vals if v >= NUM_CLASSES and v != 255]
            if over:
                bad_range.append((osp.basename(f), over))

        all_ids.update(split_ids)
        fg = sum(c for v, c in split_ids.items() if v != 0 and v != 255)
        px = sum(split_ids.values()) or 1

        print(f'[{split}] images={len(imgs)} masks={len(anns)} '
              f'scanned={len(scan)}')
        print(f'    classes present: {len(set(split_ids) - {0, 255})}/115  '
              f'foreground pixels: {100 * fg / px:.2f}%')
        if missing_ann:
            print(f'    !! {len(missing_ann)} images with no mask, '
                  f'e.g. {sorted(missing_ann)[:3]}')
        if missing_img:
            print(f'    !! {len(missing_img)} masks with no image, '
                  f'e.g. {sorted(missing_img)[:3]}')
        if bad_range:
            print(f'    !! {len(bad_range)} masks with label id >= '
                  f'{NUM_CLASSES}, e.g. {bad_range[:3]}')

    print()
    print(f'TOTAL images: {total}   paper reports: {PAPER_TOTAL}')
    if total < PAPER_TOTAL * 0.95:
        pct = 100 * total / PAPER_TOTAL
        print(f'  !! Local copy has {pct:.1f}% of the images described in the '
              f'paper.\n'
              f'  !! Re-download from https://zenodo.org/records/14935094 '
              f'before comparing scores.')

    ids = sorted(i for i in all_ids if i != 255)
    print(f'label id range: {min(ids)}..{max(ids)}  '
          f'distinct: {len(ids)}  (expected 0..115)')
    if max(ids) == 115:
        print('  -> num_classes MUST be 116 in every config.')

    for p in problems[:10]:
        print(f'PROBLEM {p}')


if __name__ == '__main__':
    main()
