#!/usr/bin/env bash
set -euo pipefail

python tools/convert_amass_to_dppd.py \
  --extract \
  --archives-root data/AMASS/downloads/data/AMASS \
  --raw-root data/AMASS/amass_raw \
  --body-model-root data/AMASS/body_models \
  --joint-regressor data/AMASS/J_regressor_h36m_correct.npy \
  --output-root Dataset/motion3d/MB3D_f81s9/AMASS/0 \
  --target-fps 60 \
  --clip-len 81 \
  --stride 9 \
  --overwrite
