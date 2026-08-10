# EXPERIMENTAL: SegNeXt MSCAN-L at 640x640 (EXP-006).
#
# WARNING -- this config confounds TWO changes (crop size AND loss function)
# against the AdamW baseline. Do not run it as your headline result. Run
# segnext_mscan-l_tuned-adamw_40k_plantseg115-512x512.py first, then use this
# only to test the resolution+loss hypothesis on top of it.
#
# The resolution case here is WEAK and measured, not assumed: on the train
# split the median image is 640x520 and 67.7% have a short side < 640. At
# scale=(2560,640) the median image is upsampled 1.23x and small images up to
# 2.1x -- interpolated pixels, not recovered detail, at 1.56x the compute per
# crop. The counter-argument is that RandomResize ratio_range=(0.5,2.0) already
# samples up to 2x, so a larger crop captures more true context per sample on
# the minority of genuinely high-res images (max observed: 5344x4032, 16 MP).
# Which effect wins is an empirical question. That is why this is an experiment.
#
# The loss case is stronger: measured on the train masks, background is 79.8%
# of pixels and the pixel-count imbalance between the commonest and rarest
# disease class is 519x. Plain CE is dominated by background; Dice is
# region-based and helps the rare classes that mIoU weights equally.
#
# Sized for ~20GB of free VRAM (the GPU is shared): batch 8 x 2 accumulation
# = effective 16, matching the paper's batch size.
_base_ = './segnext_mscan-l_paper-sgd_40k_plantseg115-512x512.py'

crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(drop_path_rate=0.4),
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                avg_non_ignore=True),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=True,
                loss_weight=0.3,
                ignore_index=255),
        ]),
    # Sliding-window inference at test time: the test images are wildly varied
    # in resolution, and whole-image resize loses small lesions.
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(426, 426)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2560, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

# batch 8 x accumulative_counts 2 == effective batch 16 in ~20GB.
train_dataloader = dict(
    batch_size=8, num_workers=8, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=2,
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', power=1.0, begin=1500, end=40000, eta_min=0.0,
         by_epoch=False),
]
