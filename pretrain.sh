#!/usr/bin/env bash
set -euo pipefail

python pretrain_motionbert.py \
  --config configs/pretrain/PM_pretrain_amass_gait_cmu_h36m_depth5.yaml \
  --checkpoint checkpoint/procore_gait/pretrain
