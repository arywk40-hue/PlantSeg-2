# Tuned SegNeXt MSCAN-L at the PAPER'S native 512x512 crop.
#
# This is the primary "optimized baseline". It changes exactly ONE thing versus
# the paper-faithful baseline: the optimizer (SGD 1e-3 -> AdamW 6e-5 w/ 10x head
# LR + warmup). Everything else -- crop size, loss, schedule, batch -- is held
# constant so the delta is attributable.
#
# Why 512 and not 640: measured on the train split, the median image is
# 640x520 and 67.7% of images have a short side < 640. Training at 640 would
# upsample most of the dataset (median factor 1.23x, small images up to 2.1x),
# spending 1.56x the compute per crop on interpolated detail that is not in the
# source. 640 is worth testing (see EXP-006) but is not the default.
#
# AMP is enabled here because bf16 on Blackwell is numerically safe and the
# speedup is large; it is an engineering change, not a scientific one.
_base_ = './segnext_mscan-l_paper-sgd_40k_plantseg115-512x512.py'

# --- the single scientific change vs baseline: optimizer ---
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
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

# --- engineering only: feed the GPU properly. Does not affect the objective. ---
train_dataloader = dict(batch_size=16, num_workers=8, pin_memory=True)
val_dataloader = dict(num_workers=4, pin_memory=True)
test_dataloader = dict(num_workers=4, pin_memory=True)
