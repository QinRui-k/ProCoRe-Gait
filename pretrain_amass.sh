#!/usr/bin/env bash
set -euo pipefail

python pretrain_motionbert.py \
  --config configs/pretrain/PM_adapt_patient_dynamic_anchor_depth5.yaml \
  --checkpoint checkpoint/procore_gait/adapt \
  --pretrained checkpoint/procore_gait/pretrain \
  --selection latest_epoch.bin
