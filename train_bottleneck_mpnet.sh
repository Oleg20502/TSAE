#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="4,5"
NP=4

accelerate launch --num_processes $NP --config_file ./accel_configs/fp32.yaml \
    scripts/train_bottleneck.py \
    --config configs/train/ae_mpnet_fineweb.yaml \
    # --resume_from_checkpoint outputs/noise/nl_4_rstd_0.1_sl_10/checkpoint-2000

echo "done"
