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

## Ground rules

1. Model selection and hyperparameter tuning happen on the **val** split
   (846 images) only. The **test** split (1561 images) is touched exactly once,
   at the end, via `tools/test.py`.
2. One variable per experiment. EXP-002 exists solely to prove AMP is
   score-neutral so EXP-003's delta is attributable to the optimizer.
3. Report mean +/- std over 3 seeds for any headline claim. Single-seed
   segmentation runs vary by roughly +/-0.5 mIoU.
4. Rare-class noise floor: 19 of 115 classes have <=2 images in val. A single
   image flipping moves that class's IoU by ~50%, worth ~0.4 mIoU overall.
   Treat val gains under ~1.5 mIoU as noise until confirmed by seeds.

## Blocking issue

`tools/check_plantseg_data.py` reports **7,774 images locally vs 11,458 in the
paper (67.8%)**. Until the full dataset is present, no number produced here is
comparable to Table 3. Re-download: https://zenodo.org/records/14935094

## Results

Fill in as runs complete. mIoU/mAcc on VAL unless the row says TEST.

| Exp     | Config                                              | Split | mIoU | mAcc | Notes |
|---------|-----------------------------------------------------|-------|------|------|-------|
| EXP-000 | smoke (200 iter)                                    | val   |      |      | pipeline check only |
| EXP-001 | segnext_mscan-l_paper-sgd_40k_...512x512            | val   |      |      | paper recipe: SGD 1e-3, FP32 |
| EXP-002 | EXP-001 + `--amp`                                   | val   |      |      | must match EXP-001 +/-0.3 |
| EXP-003 | segnext_mscan-l_tuned-adamw_40k_...512x512          | val   |      |      | one change vs EXP-002: AdamW |
| EXP-004 | EXP-003, lr in {3e-5, 6e-5, 1.2e-4, 2.4e-4}         | val   |      |      | sweep |
| EXP-005 | EXP-003 + CE+Dice                                   | val   |      |      | bg=79.8%, 519x imbalance |
| EXP-006 | segnext_mscan-l_tuned_40k_...640x640                | val   |      |      | CONFOUNDED: crop + loss |
| EXP-007 | winner, seeds 0/1/2                                 | val   |      |      | report mean +/- std |
| FINAL   | winner, best-on-val ckpt                            | TEST  |      |      | run ONCE |
| FINAL+  | winner + TTA                                        | TEST  |      |      | disclose separately |

## Reproduction gate

EXP-001 is the gate. If val mIoU does not land within ~2 points of 44.52, do
not proceed to EXP-003 -- diagnose first. Most likely cause, in order:
1. Incomplete dataset (see Blocking issue above)
2. `num_classes` != 116 (some SAN configs in this repo wrongly use 115)
3. Training on `train` only (5,367) vs the paper's train+val (6,213), worth
   roughly -1 mIoU. This is deliberate; fold val back in only after
   hyperparameters are frozen.
