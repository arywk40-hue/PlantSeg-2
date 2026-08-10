# Copyright (c) OpenMMLab. All rights reserved.
"""Measure PlantSeg-115 class imbalance and the val-set rare-class noise floor.

Two questions this answers, both of which are needed before drawing conclusions
from a val mIoU delta:

1. How imbalanced is the training set? This is the evidence for or against
   adding a region-based loss (Dice) alongside cross-entropy. CE is dominated
   by whichever class owns most pixels; mIoU weights all 116 classes equally.

2. How many val images back each class? mIoU is a mean over 116 per-class IoUs,
   so a class supported by 2 images contributes as much to the headline number
   as one supported by 400. That sets the floor below which a val improvement
   is indistinguishable from noise, and therefore how large a gain must be
   before it is worth a 3-seed confirmation run.

Pure CPU/IO -- safe to run while a training job owns the GPU.

Usage:
    python tools/class_stats.py [--data-root data/plantseg115] [--split train]
"""
import argparse
import os.path as osp
from collections import Counter, defaultdict
from glob import glob

import numpy as np
from PIL import Image

NUM_CLASSES = 116  # 1 background + 115 diseases
IGNORE = 255


def parse_args():
    p = argparse.ArgumentParser(description='PlantSeg class statistics')
    p.add_argument('--data-root', default='data/plantseg115')
    p.add_argument(
        '--split',
        default='train',
        help='Split to measure pixel imbalance on (default: train).')
    p.add_argument(
        '--val-split',
        default='val',
        help='Split to measure the rare-class noise floor on.')
    return p.parse_args()


def scan(ann_dir):
    """Return (pixels_per_class, images_per_class, n_masks)."""
    pixels = Counter()
    images = defaultdict(int)
    masks = sorted(glob(osp.join(ann_dir, '*.png')))
    for f in masks:
        arr = np.array(Image.open(f))
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals, counts):
            v = int(v)
            if v == IGNORE:
                continue
            pixels[v] += int(c)
            images[v] += 1
    return pixels, images, len(masks)


def main():
    args = parse_args()
    root = args.data_root

    # --- 1. pixel imbalance on the training split ---
    pixels, _, n = scan(osp.join(root, 'annotations', args.split))
    total_px = sum(pixels.values()) or 1
    bg = pixels.get(0, 0)
    fg = {k: v for k, v in pixels.items() if k != 0 and v > 0}

    print(f'=== pixel imbalance [{args.split}, {n} masks] ===')
    print(f'background (id 0): {100 * bg / total_px:.2f}% of all pixels')
    print(f'foreground total : {100 * (total_px - bg) / total_px:.2f}%')

    if fg:
        ranked = sorted(fg.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_px = ranked[0]
        bot_id, bot_px = ranked[-1]
        print(f'commonest disease class: id {top_id} '
              f'({100 * top_px / total_px:.3f}% of pixels)')
        print(f'rarest disease class   : id {bot_id} '
              f'({100 * bot_px / total_px:.5f}% of pixels)')
        print(f'imbalance ratio (commonest / rarest disease): '
              f'{top_px / max(bot_px, 1):.0f}x')
        print(f'background / rarest disease: {bg / max(bot_px, 1):.0f}x')
        missing = [i for i in range(1, NUM_CLASSES) if i not in fg]
        if missing:
            print(f'!! {len(missing)} disease classes have ZERO pixels here: '
                  f'{missing}')

    # --- 2. rare-class noise floor on the val split ---
    _, val_images, n_val = scan(osp.join(root, 'annotations', args.val_split))
    print()
    print(f'=== val support [{args.val_split}, {n_val} masks] ===')

    support = {i: val_images.get(i, 0) for i in range(1, NUM_CLASSES)}
    for thresh in (0, 1, 2, 5):
        rare = [i for i, c in support.items() if c <= thresh and thresh > 0]
        absent = [i for i, c in support.items() if c == 0]
        if thresh == 0:
            print(f'classes with NO val images: {len(absent)} {absent}')
            if absent:
                print('  -> these contribute IoU 0 (or NaN) and drag mIoU down '
                      'regardless of model quality.')
        else:
            # A class present in k val images can flip roughly 1/k of its IoU
            # per image. Worst case that class moves ~100 points, i.e.
            # 100/NUM_CLASSES mIoU points overall.
            swing = 100.0 / NUM_CLASSES
            print(f'classes with <= {thresh} val image(s): {len(rare)}  '
                  f'-> worst-case mIoU swing ~{len(rare) * swing:.2f} points')

    print()
    print('Interpretation: treat a val mIoU gain smaller than the <=2-image '
          'swing above as unconfirmed until it survives 3 seeds.')


if __name__ == '__main__':
    main()
