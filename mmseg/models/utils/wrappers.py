# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_and_argmax(logits, size, align_corners=False, chunk=16):
    """Bilinearly upsample per-class logits to ``size`` and take the argmax.

    Mathematically identical to ``resize(...).argmax(dim=1, keepdim=True)`` but
    never materialises the full ``C x H x W`` float32 tensor. On PlantSeg the
    naive form allocates ~5.3 GiB for a 116-class prediction on a 4032x3024
    image, which is enough to OOM evaluation on its own.

    This is safe to do chunkwise because bilinear interpolation acts on each
    channel independently, and argmax is a running maximum: processing channels
    in groups and keeping the best-so-far value and index gives bit-identical
    results to computing it in one shot.

    Note this is NOT interchangeable with collapsing to labels first and
    upsampling with nearest-neighbour. Interpolating logits lets a class win a
    high-resolution pixel through blending near boundaries; nearest-neighbour
    on an already-argmaxed label map cannot, and loses boundary IoU on small
    objects.

    Args:
        logits (Tensor): Shape (1, C, h, w).
        size (tuple): Target spatial size (H, W).
        align_corners (bool): Passed through to the bilinear resize.
        chunk (int): Channels to upsample at a time. Peak extra memory is
            roughly ``chunk x H x W`` floats.

    Returns:
        Tensor: Predicted labels, shape (1, H, W), dtype long.
    """
    n, c, _, _ = logits.shape
    best_val = None
    best_idx = None

    for start in range(0, c, chunk):
        stop = min(start + chunk, c)
        part = resize(
            logits[:, start:stop],
            size=size,
            mode='bilinear',
            align_corners=align_corners,
            warning=False)
        part_val, part_idx = part.max(dim=1)  # (n, H, W)
        del part
        part_idx = part_idx + start

        if best_val is None:
            best_val, best_idx = part_val, part_idx
        else:
            # Ties resolve to the lower channel index, matching torch.argmax.
            take = part_val > best_val
            best_val = torch.where(take, part_val, best_val)
            best_idx = torch.where(take, part_idx, best_idx)

    return best_idx.reshape(n, *size).long()


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super().__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)
