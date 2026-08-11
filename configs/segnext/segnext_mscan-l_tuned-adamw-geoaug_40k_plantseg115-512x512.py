# EXP-009: geometric augmentation only. ONE variable vs EXP-003.
#
# Evidence this is the right lever, from EXP-003's own logs:
#   * iteration 37,400: decode.acc_seg 91-95% on train, loss 0.1324
#   * best val mIoU 46.47, val plateau 45.87 +/- 0.48 from iter 26k onward
#   * best checkpoint at 26,000 of 40,000, with val drifting DOWN after
# A ~48-point train/val gap with the val curve turning over is memorisation,
# not under-training. EXP-008 (a longer schedule) was cancelled for exactly
# this reason: more iterations over the same views deepens the memorisation.
# Augmentation increases the variety of views instead.
#
# GEOMETRIC ONLY -- and that restriction is deliberate, not conservatism.
# PhotoMetricDistortion is left at its stock defaults. Plant disease classes on
# this dataset are distinguished largely BY colour: chlorosis is defined by
# yellowing, necrosis by browning, and the rusts by hue. Aggressive hue or
# saturation jitter (e.g. hue_delta=25, saturation_range=(0.5, 1.7)) perturbs
# the diagnostic signal itself, not just nuisance variation, and can make two
# distinct classes land on the same appearance. Rotation and flipping cannot do
# that -- a lesion is the same lesion upside down.
#
# Everything else is inherited verbatim from EXP-003: AdamW 6e-5, head
# lr_mult=10, 1500-iteration warmup, bf16, batch 16, CE-only loss, 40k
# schedule, seed 0, val-split validation, save_best='mIoU'.
_base_ = './segnext_mscan-l_tuned-adamw_40k_plantseg115-512x512.py'

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    # NEW. Placed before RandomCrop so the rotation acts on the resized image
    # and the crop can then land anywhere inside it; rotating after the crop
    # would push ignore-padding into every single 512x512 sample.
    #
    # RandomRotate defaults to seg_pad_val=255, so the triangular regions
    # rotated in from outside the image become IGNORE pixels, not class 0.
    # That matters on this dataset: background is already 80.21% of all
    # pixels, and labelling rotation padding as background would both inflate
    # that further and teach the model that black corners are background.
    # It is stated explicitly here rather than relied on as a default.
    #
    # 30 degrees, not 90 or 180: with auto_bound=False the padded area grows
    # with the angle, so large rotations spend real fraction of every crop on
    # ignore pixels that produce no gradient. 30 degrees is most of the
    # invariance for a small fraction of the waste. Larger angles are a
    # separate follow-up, not a free addition.
    dict(type='RandomRotate', prob=0.5, degree=30, pad_val=0, seg_pad_val=255),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # NEW. Leaf close-ups have no canonical up direction, so a vertical flip is
    # label-preserving here in a way it would not be for, say, street scenes.
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # UNCHANGED from the stock pipeline -- see the note above on why colour
    # augmentation is not part of this experiment.
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# batch_size=16, num_workers=8 and pin_memory=True are inherited from EXP-003;
# this dict merges into that one rather than replacing it.
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
