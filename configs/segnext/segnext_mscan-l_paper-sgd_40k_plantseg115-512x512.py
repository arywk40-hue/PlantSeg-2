# Paper-faithful reproduction of the SegNeXt / MSCAN-L row of Table 3.
#
# The paper (arXiv:2409.04038, "Evaluation on PlantSeg") states ALL baselines
# were trained with SGD, lr=0.001, momentum=0.9, weight_decay=0.0005,
# cross-entropy loss, batch size 16. The stock SegNeXt configs in this repo use
# AdamW @ 6e-5, which is the upstream ADE20k recipe -- a different experiment.
# This config reproduces what the paper describes. Use it as the baseline.
#
# NOTE: still validates on `val`, not `test`, so the training-time numbers are
# honest. Report the final number via tools/test.py on the test split.
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
    '../_base_/datasets/plantseg115_valsplit.py',
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_l_20230227-cef260d4.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.3,
        depths=[3, 5, 27, 3],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=1024,
        ham_channels=1024,
        dropout_ratio=0.1,
        # 116 = 1 background + 115 disease classes. Verified against the mask
        # files: label ids run 0..115. Do NOT set this to 115 -- some SAN
        # configs in this repo get it wrong and will silently mistrain.
        num_classes=116,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# --- Paper's stated optimizer: SGD, lr 1e-3, momentum 0.9, wd 5e-4 ---
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=40000,
         by_epoch=False)
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)

# Reproducibility: the repo sets no seed anywhere, so every run draws a random
# one (see the differing numpy_random_seed values across work_dirs/*/*.log).
# Override per run with --cfg-options randomness.seed=<n> for seed studies.
# deterministic=False is deliberate: cudnn deterministic mode costs ~15-20% and
# seed-averaging over 3 runs is the more honest control anyway.
randomness = dict(seed=0, deterministic=False)

# Keep the best-on-val checkpoint so selection never looks at test.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'))
