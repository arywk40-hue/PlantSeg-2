# PlantSeg-115 Reproduction & Improvement Log

## Target (LOCKED)

The benchmark to beat is the SegNeXt / MSCAN-L row of **Table 3** in
arXiv:2409.04038v1:

| Method  | Backbone | mIoU      | mAcc      |
|---------|----------|-----------|-----------|
| SegNeXt | MSCAN-L  | **44.52** | **59.95** |

### On the 53.89 discrepancy

The body text on p.8 states SegNeXt achieves "the MIoU of 53.89% and the mAcc
of 65.91%" while describing Table 3, which says 44.52 / 59.95. arXiv lists
**only v1** (6 Sep 2024, never revised), so this is not a cross-version
mismatch -- it is an uncorrected inconsistency within the single published
version. Table 3 and the official repo README agree on 44.52, so 44.52/59.95
is the benchmark of record. If publishing, footnote the discrepancy.

## Dataset (VERIFIED)

`tools/check_plantseg_data.py` on the GPU box, 2026-08-10:

| Split | Images | Classes present |
|-------|--------|-----------------|
| train | 7,916  | 115/115 |
| val   | 1,247  | 115/115 |
| test  | 2,295  | 115/115 |
| **total** | **11,458** | matches the paper exactly |

Label ids run 0..115, confirming `num_classes=116`. An earlier partial download
(7,774 images, 67.8%) has been replaced; any note referencing 846-image val or
1,561-image test predates that fix.

## Ground rules

1. Model selection and hyperparameter tuning happen on the **val** split
   (1,247 images) only. The **test** split (2,295 images) is touched exactly
   once, at the end, via `tools/test.py`.
2. One variable per experiment. EXP-002 exists solely to prove AMP is
   score-neutral so EXP-003's delta is attributable to the optimizer.
3. Report mean +/- std over 3 seeds for any headline claim. Single-seed
   segmentation runs vary by roughly +/-0.5 mIoU.
4. Rare-class noise floor: mIoU averages 116 classes equally, so a class with a
   handful of val images can swing the mean on its own. **Recompute on the full
   dataset** with `tools/class_stats.py` before quoting a threshold -- the
   previous "19 classes with <=2 val images" figure was measured on the partial
   download and no longer applies.

## Results

Fill in as runs complete. mIoU/mAcc on VAL unless the row says TEST.

| Exp     | Config                                              | Split | mIoU | mAcc | Notes |
|---------|-----------------------------------------------------|-------|------|------|-------|
| EXP-000 | smoke (200 iter)                                    | val   |      |      | pipeline check only |
| EXP-001 | segnext_mscan-l_paper-sgd_40k_...512x512            | val   |      |      | paper recipe: SGD 1e-3, FP32 |
| EXP-002 | EXP-001 + `--amp`                                   | val   |      |      | must match EXP-001 +/-0.3 |
| EXP-003 | segnext_mscan-l_tuned-adamw_40k_...512x512          | val   |      |      | one change vs EXP-002: AdamW |
| EXP-004 | EXP-003, lr in {3e-5, 6e-5, 1.2e-4, 2.4e-4}         | val   |      |      | sweep |
| EXP-005 | EXP-003 + CE+Dice                                   | val   |      |      | justify with `tools/class_stats.py` on the FULL data first |
| EXP-006 | segnext_mscan-l_tuned_40k_...640x640                | val   |      |      | CONFOUNDED: crop + loss |
| EXP-007 | winner, seeds 0/1/2                                 | val   |      |      | report mean +/- std |
| FINAL   | winner, best-on-val ckpt                            | TEST  |      |      | run ONCE |
| FINAL+  | winner + TTA                                        | TEST  |      |      | disclose separately |

## Reproduction gate

EXP-001 is the gate. If val mIoU does not land within ~2 points of 44.52, do
not proceed to EXP-003 -- diagnose first. Remaining likely causes, in order:

1. Training on `train` only (7,916 images) vs what the paper most likely used,
   train+val (9,163). Holding out val costs roughly -1 mIoU. This is
   deliberate and is the honest choice; fold val back in only after
   hyperparameters are frozen, and say so when reporting.
2. The paper does not state its iteration count. 40k is this repo's schedule,
   not a quoted number. If EXP-001 is still improving at 40k, the gap may be
   schedule length rather than a reproduction error.
3. Unstated augmentation. The paper gives optimizer, LR, momentum, weight
   decay, loss, and batch size -- nothing about the crop/resize pipeline. Ours
   comes from the repo's mmseg defaults.

Dataset completeness and `num_classes` are both verified above and are no
longer candidate explanations.

## Run log

**EXP-001, attempt 1 (2026-08-10)** -- died at iteration 1 with
`torch.OutOfMemoryError` in `mscan.py:160`. Cause was environmental, not
configuration: a foreign process (PID 50784) held 26.5 GiB of the 47 GiB card,
leaving 20.7 GiB. No config change was made.

**EXP-001, attempt 2 (2026-08-10, 19:03)** -- running. Foreign process gone;
39,345 MiB in use at batch 16 / 512x512 / FP32, i.e. the paper's batch size with
no gradient accumulation and no AMP. 0.535 s/iter, `data_time` 0.009 s
(GPU-bound, dataloader is not a bottleneck). Training ETA ~6h20m.
