#!/bin/bash
# PlantSeg-115 experiment runner for a single CUDA GPU.
#
# Replaces the CPU/MPS run.sh. Run steps in order; each is independently
# resumable. Set PYTHON to your CUDA env's interpreter.
set -euo pipefail

PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Blackwell: prefer TF32 matmuls and reduce allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1

CFG_DIR=configs/segnext
BASE=$CFG_DIR/segnext_mscan-l_paper-sgd_40k_plantseg115-512x512.py
ADAMW=$CFG_DIR/segnext_mscan-l_tuned-adamw_40k_plantseg115-512x512.py
RES640=$CFG_DIR/segnext_mscan-l_tuned_40k_plantseg115-640x640.py

step=${1:-help}

case "$step" in

# 0. Verify the data before spending GPU hours. Full scan.
check)
  $PYTHON tools/check_plantseg_data.py
  ;;

# EXP-000. Smoke test: 200 iters, confirms the pipeline runs and VRAM fits.
smoke)
  $PYTHON tools/train.py "$ADAMW" \
    --cfg-options train_cfg.max_iters=200 train_cfg.val_interval=100 \
                  default_hooks.logger.interval=10 \
    --work-dir work_dirs/smoke
  ;;

# EXP-001. Paper-faithful baseline (SGD @ 1e-3, 512x512, CE, FP32).
#          This is the number that should land near the paper's Table 3 row.
baseline)
  $PYTHON tools/train.py "$BASE" \
    --work-dir work_dirs/exp001_baseline_sgd
  ;;

# EXP-002. Same objective, bf16 AMP. Pure engineering control: verifies AMP
#          does not move the score before it is used everywhere else.
baseline-amp)
  $PYTHON tools/train.py "$BASE" --amp \
    --work-dir work_dirs/exp002_baseline_sgd_amp
  ;;

# EXP-003. Optimized baseline: AdamW instead of SGD. ONE variable vs EXP-002.
adamw)
  $PYTHON tools/train.py "$ADAMW" \
    --work-dir work_dirs/exp003_adamw
  ;;

# EXP-004. LR sweep around the AdamW default. Run sequentially on one GPU.
lr-sweep)
  for LR in 0.00003 0.00006 0.00012 0.00024; do
    $PYTHON tools/train.py "$ADAMW" \
      --cfg-options optim_wrapper.optimizer.lr=$LR \
      --work-dir "work_dirs/exp004_lr${LR}"
  done
  ;;

# EXP-006. Resolution + loss hypothesis. CONFOUNDED (two variables) -- read the
#          header of the config before drawing conclusions from it.
res640)
  $PYTHON tools/train.py "$RES640" \
    --work-dir work_dirs/exp006_640_cedice
  ;;

# Seed replication for the final config. Report mean +/- std over 3 seeds.
seeds)
  CFG=${2:?usage: $0 seeds <config.py>}
  for S in 0 1 2; do
    $PYTHON tools/train.py "$CFG" \
      --cfg-options randomness.seed=$S \
      --work-dir "work_dirs/seed${S}_$(basename "$CFG" .py)"
  done
  ;;

# Resume an interrupted run from its work_dir.
resume)
  CFG=${2:?usage: $0 resume <config.py> <work_dir>}
  WD=${3:?usage: $0 resume <config.py> <work_dir>}
  $PYTHON tools/train.py "$CFG" --resume --work-dir "$WD"
  ;;

# FINAL. Held-out test split, best-on-val checkpoint. Run ONCE, at the end.
test)
  CFG=${2:?usage: $0 test <config.py> <ckpt>}
  CKPT=${3:?usage: $0 test <config.py> <ckpt>}
  $PYTHON tools/test.py "$CFG" "$CKPT" --work-dir work_dirs/final_test
  ;;

# FINAL + multi-scale/flip TTA. Report separately from the single-scale number.
test-tta)
  CFG=${2:?usage: $0 test-tta <config.py> <ckpt>}
  CKPT=${3:?usage: $0 test-tta <config.py> <ckpt>}
  $PYTHON tools/test.py "$CFG" "$CKPT" --tta --work-dir work_dirs/final_test_tta
  ;;

# Throughput benchmark: 100 iters, prints s/iter and data_time.
bench)
  CFG=${2:-$ADAMW}
  $PYTHON tools/train.py "$CFG" \
    --cfg-options train_cfg.max_iters=100 train_cfg.val_interval=100000 \
                  default_hooks.logger.interval=10 \
    --work-dir work_dirs/bench
  ;;

*)
  echo "usage: $0 {check|smoke|baseline|baseline-amp|adamw|lr-sweep|res640|"
  echo "          seeds <cfg>|resume <cfg> <wd>|test <cfg> <ckpt>|"
  echo "          test-tta <cfg> <ckpt>|bench [cfg]}"
  exit 1
  ;;
esac
