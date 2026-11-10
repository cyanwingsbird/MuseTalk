#!/bin/bash
# Make the chosen stage-1 UNet the stage-2 starting point (models/canto_stage1_final/).
#   sh prepare_canto_stage2.sh <stage1-step>      e.g. 500000
set -euo pipefail
cd "$HOME/musetalk15"
SRC="exp_out/stage1/canto_stage1/unet-$1.pth"
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
mkdir -p models/canto_stage1_final
cp models/musetalkV15/musetalk.json models/canto_stage1_final/musetalk.json
cp "$SRC" models/canto_stage1_final/pytorch_model.bin
echo "stage 2 will start from $SRC"
