# EXP-008: EXP-003 with the schedule doubled. ONE variable.
#
# Evidence for a longer schedule, not a guess:
#   * EXP-001's best checkpoint was iteration 38,000 of 40,000 -- still
#     improving when the schedule ended.
#   * EXP-003 was at 43.96 val mIoU by ~20k of 40k and still climbing.
# Neither run had plateaued at its schedule end, so the schedule is a live
# constraint rather than a hyperparameter to poke at.
#
# Why 80k and not 160k. At batch 16 over 7,916 train images, 80k iterations is
# ~162 epochs and 160k is ~323. Past a certain point the schedule stops being a
# schedule change and becomes an unmeasured memorization risk -- and we are
# deliberately NOT adding the stronger augmentation that would offset it,
# because that would be a second variable. 80k answers "does longer help?" in
# ~12.5h instead of ~25h. If it wins, 160k becomes justified; if it does not,
# 160k was never going to.
#
# Everything else is inherited verbatim from EXP-003: AdamW 6e-5, head
# lr_mult=10, 1500-iter warmup, bf16, batch 16, CE-only loss, default mmseg
# augmentation, seed 0, and val-split validation with save_best='mIoU'.
_base_ = './segnext_mscan-l_tuned-adamw_40k_plantseg115-512x512.py'

# The base PolyLR is hardcoded to end=40000 (configs/_base_/schedules/
# schedule_40k.py:11 sets it and every config downstream re-states it). Simply
# raising max_iters WITHOUT restating this leaves LR at eta_min=0 from 40k
# onward -- 40k iterations of training followed by 40k of nothing. The scheduler
# has to be redeclared whenever max_iters moves.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', power=1.0, begin=1500, end=80000, eta_min=0.0,
         by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)

# save_best is inherited, but interval and max_keep_ckpts are sized for 40k.
# Widen both so the run keeps a usable window without filling the disk: an
# MSCAN-L checkpoint is ~200 MB.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=8000,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'))

# NOT set: load_from. This trains from the ImageNet MSCAN-L backbone exactly as
# EXP-003 did. Warm-starting from a converged 40k checkpoint and then restarting
# a full poly schedule would make this a two-stage ~120k run whose gain could
# not be attributed to the schedule, and it would not be comparable to EXP-003.
