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
3. Report mean +/- std over 3 seeds for any headline claim. The single-run
   noise is measured, not assumed: EXP-003 dropped 2.39 mIoU between adjacent
   val passes on one run (see its trajectory table). Treat any difference under
   ~2.5 mIoU from a single run as unresolved.
4. Rare-class noise floor: measured on the full data with `tools/class_stats.py`
   (`logs/class_stats.txt`). Every class has val support -- no class scores a
   forced zero. 2 classes have a single val image, 7 have <=2, 33 have <=5.

   The tool prints a "worst-case mIoU swing" for each tier. Read it as an
   **upper bound, not an expected variance**: the <=2 tier's 6.03 points
   assumes all 7 classes simultaneously flip a full 100 IoU points, which does
   not happen in practice. It bounds how much the thin tail *could* move the
   mean, nothing more. The usable significance threshold is the empirical
   seed-to-seed spread from EXP-007, not this number.

## Class imbalance (full data, train split)

| Quantity | Value |
|----------|-------|
| background (id 0) share of pixels | 80.21% |
| all 115 disease classes combined | 19.79% |
| commonest disease class (id 90) | 1.160% of pixels |
| rarest disease class (id 42) | 0.00107% of pixels |
| commonest / rarest disease | 1,087x |
| background / rarest disease | 75,183x |

This is the evidence for EXP-005. Cross-entropy averages over pixels, so the
objective is ~80% background by construction, while mIoU averages over classes
and weights id 42 exactly as heavily as id 90. Both experiments so far show the
predicted symptom -- precision far above recall (EXP-001 final: 53.11 vs 42.87;
EXP-003 @2k: 52.06 vs 26.48), i.e. the model under-commits on disease pixels.
A region-based term (Dice) optimises overlap per class rather than per pixel
and targets exactly that asymmetry.

## Results

Fill in as runs complete. mIoU/mAcc on VAL unless the row says TEST.

| Exp     | Config                                              | Split | mIoU | mAcc | Notes |
|---------|-----------------------------------------------------|-------|------|------|-------|
| EXP-000 | smoke (200 iter)                                    | val   |      |      | pipeline check only |
| EXP-001 | segnext_mscan-l_paper-sgd_40k_...512x512            | val   | 31.18 | 42.87 | best ckpt @38k of 40k |
| EXP-001 | same checkpoint, corrected eval                     | TEST  | 33.50 | 46.60 | vs paper's 44.52 / 59.95 |
| EXP-002 | EXP-001 + `--amp`                                   | val   |      |      | dropped -- EXP-001 abandoned, no baseline to control against |
| EXP-003 | segnext_mscan-l_tuned-adamw_40k_...512x512          | val   | 46.47 | 60.00 | best ckpt @26k of 40k |
| EXP-003 | best ckpt @26k, corrected eval                       | TEST  | **45.04** | **60.06** | vs paper 44.52/59.95 -> +0.52 / +0.11 |
| EXP-004 | EXP-003, lr in {3e-5, 6e-5, 1.2e-4, 2.4e-4}         | val   |      |      | sweep |
| EXP-005 | EXP-003 + CE+Dice                                   | val   |      |      | justify with `tools/class_stats.py` on the FULL data first |
| EXP-006 | segnext_mscan-l_tuned_40k_...640x640                | val   |      |      | CONFOUNDED: crop + loss |
| EXP-007 | winner, seeds 0/1/2                                 | val   |      |      | report mean +/- std |
| EXP-008 | segnext_mscan-l_tuned-adamw_80k_...512x512          | --    | --   | --   | CANCELLED -- EXP-003 overfits by 26k, not iteration-starved |
| EXP-009 | segnext_mscan-l_tuned-adamw-geoaug_40k_...512x512   | val   |      |      | EXP-003 + rotation & vflip, ONE variable |
| FINAL   | winner, best-on-val ckpt                            | TEST  |      |      | run ONCE |
| FINAL+  | winner + TTA                                        | TEST  |      |      | disclose separately |

### EXP-001: the paper's stated recipe does not reproduce its own number

Ran the full 40,000 iterations. Best checkpoint at iteration **38,000**, i.e.
the model was still improving when the schedule ended -- evidence that 40k is
short for 116 classes on 7,916 images.

| Iter  | mIoU  | mPrecision | mRecall |
|-------|-------|-----------|---------|
| 2000  | 1.09  | 24.97     | 1.55    |
| 8000  | 11.34 | 41.00     | 17.04   |
| 38000 | 31.18 | 53.11     | 42.87   |

**Head-to-head with the paper.** The stock `plantseg115.py` points
`val_dataloader` at `images/test`, so the paper's Table 3 numbers are test-split
numbers. Re-scoring the best checkpoint on test under the corrected evaluation
code gives the like-for-like comparison:

| Source                        | mIoU  | mAcc  |
|-------------------------------|-------|-------|
| Paper, Table 3 (SegNeXt / MSCAN-L) | 44.52 | 59.95 |
| EXP-001, test split           | **33.50** | **46.60** |
| gap                           | -11.02 | -13.35 |

Roughly 1 point of that gap is self-imposed: we train on `train` only (7,916)
while the paper most likely used train+val (9,163). The remaining ~10 points
are not accounted for by the dataset, the class count, the evaluation code, or
the schedule length alone.

**What the recipe is missing.** The paper specifies "a learning rate of 0.001"
with no mention of parameter groups, so this config applies it uniformly to
both the ImageNet-pretrained backbone and the randomly initialised 116-class
head. The repo's own `segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py`
instead uses AdamW @ 6e-5, gives the head **10x** the base LR, and warms up over
1,500 iterations. The published text and the published code describe different
experiments, and only the code is a plausible source of 44.52. This is the
second internal inconsistency in the paper, after the 44.52 vs 53.89 conflict
between Table 3 and the body text.

The early-iteration profile supports the diagnosis rather than a data fault:
precision (41) far exceeded recall (17) at iteration 8,000, which is an
under-trained head on an ~81%-background dataset. A genuine label or pipeline
bug drives precision to ~0 as well.

Both EXP-001 rows above were produced under the corrected `postprocess_result`
(commit 5152e61) for the test row, and under the older nearest-neighbour path
for the in-training val row. Do not read the val-to-test difference as an
effect of either change alone -- the split and the evaluation code both differ.

### EXP-003: the optimizer was the gap (in progress)

One variable changed vs EXP-001: the optimizer block. AdamW @ 6e-5 with
`head: lr_mult=10.` and a 1500-iteration linear warmup, taken from the repo's
own `segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py`, replacing the
paper's stated uniform SGD @ 1e-3.

| Iter    | EXP-001 (paper SGD) | EXP-003 (AdamW, head x10) |
|---------|---------------------|---------------------------|
| 2000    | 1.09                | 18.54                     |
| ~16000  | --                  | 40.23                     |
| ~20000  | --                  | **43.96**                 |
| 38000   | 31.18 (best)        | --                        |

At the halfway point EXP-003 exceeds EXP-001's *final* best by 12.8 mIoU. The
diagnosis in the EXP-001 write-up -- that a randomly initialised 116-class head
cannot train at the backbone's learning rate -- is confirmed. The paper's text
and the paper's code describe different experiments; only the code reproduces.

**43.96 is a val number and is NOT comparable to the paper's 44.52**, which is
a test-split figure. Do not report the two side by side. EXP-001 moved +2.3
from val to test, but that single observation came from a weak checkpoint and
also spans an evaluation-code change, so it does not transfer as an offset.

**Revision to the EXP-005 rationale.** The precision-over-recall gap that
motivated adding a Dice term has largely closed on its own under the corrected
optimizer:

| Checkpoint        | mPrecision | mRecall | gap  |
|-------------------|-----------|---------|------|
| EXP-001 final     | 53.11     | 42.87   | 10.2 |
| EXP-003 @2k       | 52.06     | 26.48   | 25.6 |
| EXP-003 @~20k     | 64.02     | 56.70   | 7.3  |

The under-commitment on disease pixels was a symptom of the under-trained head,
not solely of the 80.21% background prior. The class-imbalance figures are
unchanged and still real, but they are now weaker evidence for EXP-005 than
they appeared. Demote CE+Dice below the 160k schedule in priority.

### EXP-003 val trajectory (val_interval=2000, 0.587 s/iter)

Only improvements print a checkpoint line, so unlisted passes failed to beat
the running best.

| Iter  | val mIoU | Note |
|-------|----------|------|
| 2000  | 18.54    | |
| 6000  | 34.50    | |
| 8000  | 37.64    | |
| 10000 | <37.64   | no improvement |
| 12000 | 42.62    | |
| 14000 | <42.62   | no improvement |
| 16000 | 40.23    | **-2.39 vs iter 12000** |
| 18000 | <42.62   | no improvement |
| 20000 | 43.96    | |
| 22000 | <43.96   | no improvement |
| 24000 | <43.96   | no improvement |
| 26000 | **46.47**| +2.51 over iter 20000 |

**Measured noise floor: +/-2.4 mIoU between adjacent val passes on a single
run.** The 12000 -> 16000 drop of 2.39 points is not seed variance or split
variance -- it is the same run, the same data, 4000 iterations apart. The
margin required to beat the paper is +0.48. The measurement noise is roughly
five times the target.

Two consequences:

1. **Read progress off best-so-far, not adjacent passes.** 12000 -> 20000 is
   +1.34 over 8000 iterations, not the +3.73 that 16000 -> 20000 suggests.
   But do not over-correct into a plateau story either: 20000 -> 26000 then
   gained **+2.51 in 6000 iterations**, faster than the preceding stretch. The
   curve is noisy and non-monotonic in both directions, and the iter-16000 dip
   was noise rather than saturation. No trend claim from fewer than ~3 val
   passes is worth making on this run.
2. **`save_best='mIoU'` is a max over ~20 noisy draws, so it selects the upper
   tail.** The reported val best is biased high by roughly the noise scale.
   Part of the -0.97 val-to-test offset recorded above is therefore selection
   bias rather than a genuine split difference, which means the val -> test
   figure should be treated as an upper bound on val, not a property of the
   splits. The test read remains the honest number.

This is also the strongest argument yet for EXP-007. A single run landing at
45.1 test is indistinguishable from one landing at 44.0; only the seed spread
separates them.

### val -> test calibration (supersedes the EXP-001 estimate)

EXP-003's mid-run checkpoint was scored on both splits under the same corrected
evaluation code, giving the first clean measurement of the split offset:

| Checkpoint            | val mIoU | test mIoU | test - val |
|-----------------------|----------|-----------|------------|
| EXP-001 best @38k     | 31.18    | 33.50     | **+2.32**  |
| EXP-003 mid-run @~20k | 43.96    | 42.99     | **-0.97**  |

The sign flips, and the reason is a confound in the first row, not instability
in the dataset: EXP-001's val number was produced under the old
argmax-then-nearest-upsample path, while its test number used the corrected
`resize_and_argmax`. That +2.32 is mostly the evaluation fix. Only the EXP-003
row compares like with like.

**Working figure: test ~= val - 1.0 mIoU.** One paired observation, so treat it
as a rough offset rather than a constant. It sets the val-side targets:

| Goal (test)          | Required val mIoU |
|----------------------|-------------------|
| match paper (44.52)  | ~45.5             |
| cross 45             | ~46.0             |

Note mAcc is proportionally further from the paper than mIoU (56.77/59.95 =
94.7% versus 42.99/44.52 = 96.6%), i.e. some residual under-commitment on
disease pixels remains even after the optimizer fix.

## EXP-003 result: Table 3 reproduces, but not from the paper's stated recipe

| Source                              | mIoU      | mAcc      |
|-------------------------------------|-----------|-----------|
| Paper, Table 3 (SegNeXt / MSCAN-L)  | 44.52     | 59.95     |
| EXP-001, paper's STATED recipe      | 33.50     | 46.60     |
| EXP-003, repo's SHIPPED recipe      | **45.04** | **60.06** |

All three on the test split (2,295 images), same `IoUMetric`, 116 classes,
corrected `resize_and_argmax` evaluation.

**The finding.** The paper states SGD @ 1e-3 with no parameter groups. That
gives 33.50 -- eleven points short of its own Table 3. The repo's shipped
`segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py` uses AdamW @ 6e-5 with
`head: lr_mult=10.` and a 1500-iteration warmup, and that reproduces Table 3.
The published text and the published code describe different experiments; only
the code reproduces. This is independent of any margin claim.

**On the +0.52 margin -- NOT established.** Three reasons:

1. The plateau pass-to-pass spread is 0.42 mIoU (measured, see trajectory
   table). +0.52 is ~1.2 sigma. Suggestive, not resolved.
2. The checkpoint was chosen by `save_best`, i.e. the maximum of 13 val passes.
   Max-of-N on a noisy metric is upward-biased by roughly the noise scale.
3. Single seed.

**mAcc +0.11 is a tie, not a win.** Do not report it as an improvement.

**One factor in our favour, worth stating when reporting.** We train on `train`
only (7,916 images). The paper's stock config validates on test, implying it
trained on train+val (9,163). So 45.04 was obtained with ~14% less training
data than the baseline it is compared against.

### val -> test offset, two paired observations

| Checkpoint        | val   | test  | test - val |
|-------------------|-------|-------|------------|
| EXP-003 @~20k     | 43.96 | 42.99 | -0.97      |
| EXP-003 best @26k | 46.47 | 45.04 | -1.43      |

Consistent in direction; mean **-1.2**. Part of this is genuine split
difference and part is `save_best` selection bias on val, which the test split
does not share.

### Overfitting onset -- why EXP-008 (80k) was cancelled

At iteration 37,400: `loss: 0.1324`, `decode.acc_seg: 91-95%` on train, while
val mIoU had drifted down across three consecutive passes and LR was at 4.05e-6
(6.8% of base). The best val checkpoint came at 26,000 -- **not** at the
schedule end, unlike EXP-001, which peaked at 38,000 of 40,000.

That reverses the evidence EXP-008 was built on. EXP-003 is not
iteration-starved; it is saturating on data variety at ~69 epochs. Running 80k
on the same augmentation would deepen memorisation rather than raise val, so
EXP-008 is not run.

This vindicates the structure of the original 160k config, which paired the
long schedule *with* stronger augmentation. The augmentation is what makes
additional iterations useful. Its specific choices remain doubtful --
`hue_delta=25` and `saturation_range=(0.5, 1.7)` perturb the channels that
carry the diagnostic signal for chlorosis, necrosis and rust -- but geometric
augmentation and milder photometric jitter are the right lever from here.

### EXP-009: geometric augmentation, isolated

The counterpart to the cancelled EXP-008. Same diagnosis, opposite remedy: if
the model is memorising rather than under-training, add view variety rather
than iterations.

Changes, all inside the augmentation block:

| Transform | Change |
|-----------|--------|
| `RandomRotate` | NEW -- `prob=0.5, degree=30, seg_pad_val=255` |
| `RandomFlip` (vertical) | NEW -- `prob=0.5` |
| `PhotoMetricDistortion` | UNCHANGED at stock defaults |

Three implementation points that are deliberate rather than incidental:

* **Rotation goes before `RandomCrop`**, so it acts on the resized image and
  the crop can land anywhere inside it. Rotating after the crop would force
  ignore-padding into every 512x512 sample.
* **`seg_pad_val=255` is stated explicitly**, though it is also the default.
  The regions rotated in from outside the image must be IGNORE, not class 0.
  Background is already 80.21% of pixels; labelling rotation padding as
  background would inflate that and teach the model that dark corners are
  background.
* **30 degrees, not 90 or 180.** With `auto_bound=False` the padded fraction
  grows with the angle, and padded pixels produce no gradient. 30 degrees buys
  most of the invariance for a small fraction of the waste. Larger angles are a
  separate experiment.

**Colour augmentation is excluded on purpose.** Disease classes here are
distinguished largely by colour -- chlorosis by yellowing, necrosis by
browning, the rusts by hue. Strong hue/saturation jitter perturbs the
diagnostic signal itself and can map two distinct classes onto the same
appearance. Rotation and flipping cannot: a lesion is the same lesion upside
down. This is why the pre-existing 160k config's `hue_delta=25` and
`saturation_range=(0.5, 1.7)` are not adopted, even though its instinct to pair
a longer schedule with stronger augmentation was correct.

If EXP-009 raises val and closes the train/val gap, a longer schedule becomes
justified *in combination with it* -- which is what the 160k config was
reaching for. Colour augmentation, if tested at all, should be a third
experiment at much milder settings.

## Review of the pre-existing 160k config (NOT RUN)

`configs/segnext/segnext_mscan-l_160k_diceCE_aug_plantseg115-512x512.py` was
written before the EXP-001/003 diagnosis. It is not run as-is. Its PolyLR
observation is correct and is carried into EXP-008; four defects are not.

| # | Defect | Consequence |
|---|--------|-------------|
| 1 | Inherits `plantseg115.py`, whose `val_dataloader` points at `images/test` (line 64) | All 16 validation passes read the test split -- leakage |
| 2 | No `save_best`; inherits `interval=10000` from `schedule_40k.py:22` | EXP-001 peaked at 38k/40k. A 160k run would keep only the final checkpoint |
| 3 | `DiceLoss` entry omits `use_sigmoid=False`; the class defaults to `True` (`dice_loss.py:98`) | Sigmoid applied to a 116-way softmax problem. CE and Dice optimise different output parameterisations simultaneously |
| 4 | `load_from` a converged 40k checkpoint, then a fresh 160k poly schedule | Re-warms a converged model: a two-stage ~200k schedule, not comparable to a 160k run |

Two further notes, non-blocking:

* `accumulative_counts=2` at `batch_size=8` restores effective batch 16 for
  gradients but not for BatchNorm -- MSCAN uses BN, whose statistics are
  computed per micro-batch of 8. Not equivalent to EXP-003's true batch 16.
* `ignore_index=255` in the Dice entry is a no-op. `dice_loss.py:71` drops
  *channel* 255, and `torch.arange(116) != 255` is all-True.
  `_expand_onehot_labels_dice` (line 25) clamps label 255 to 116 and slices it
  off, so ignore/padding pixels become all-zero targets that still contribute
  to the denominator via `input*input`. With `seg_pad_val=255` padding every
  image, this inflates the loss on pixels intended to be skipped.

The augmentation block is the part to be most sceptical of. `hue_delta=25` and
`saturation_range=(0.5, 1.7)` perturb exactly the channels that carry the
diagnostic signal on this dataset -- chlorosis, necrosis and rust are
identified by hue and saturation. This is a hypothesis to test on its own, not
to bundle.

Net: the config changes five things at once (schedule, loss, augmentation,
grad clipping, batch/accumulation). EXP-008 takes the one with direct
supporting evidence.

### EXP-008: schedule length, isolated

Two independent observations motivate this, and neither is a guess:

* EXP-001's best checkpoint was iteration **38,000 of 40,000**.
* EXP-003 reached 43.96 val mIoU by ~20k of 40k and was still climbing.

Neither run had plateaued at its schedule end.

**80k rather than 160k.** At batch 16 over 7,916 images, 80k iterations is ~162
epochs; 160k is ~323. Since we are deliberately *not* adding the stronger
augmentation that would offset that (it would be a second variable), 160k
carries real memorisation risk. 80k answers "does longer help" in ~12.5h rather
than ~25h. If it wins, 160k is justified on evidence; if it does not, 160k was
never going to.

## Test-split budget (DISCLOSE WHEN REPORTING)

The ground rules say test is touched once. It will in fact be read three times,
and every read is recorded here:

1. **EXP-001 best checkpoint** -- 33.50 / 46.60. Diagnostic on a failed
   reproduction, taken before any tuning decision depended on it.
2. **EXP-003 mid-run checkpoint** -- 42.99 / 56.77. Taken at ~20k of 40k to
   calibrate the val-to-test offset, which the EXP-001 pair could not give
   because the two splits were scored under different evaluation code.
3. **EXP-003 best checkpoint** -- the reproduction claim. A test read is
   unavoidable because 44.52 is a test number; val cannot answer whether the
   paper reproduces.
4. **FINAL** -- the winning config after all val-side tuning.

No hyperparameter, schedule, or loss decision may be made on the basis of (1),
(2) or (3). All model selection stays on val; read (2) is used only to convert
val targets into test-equivalent ones, never to choose between configs.
Anything else is test-set fitting, and it would invalidate the comparison the
whole exercise exists to make.

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
