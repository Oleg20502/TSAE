#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="0,1"
NP=2

accelerate launch --num_processes $NP \
    --config_file ./accel_configs/bf16.yaml \
    scripts/train_bottleneck.py \
    --config configs/train/ae_mpnet_fineweb.yaml \
    # --resume_from_checkpoint outputs/comp/nl_4_std_0.2_sl_10_test/checkpoint-56250

echo "done"
