#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"
NP=1

accelerate launch --num_processes $NP --config_file ./accel_configs/fp32_single_gpu.yaml \
    scripts/train_bottleneck.py \
    --config configs/train/ae_mpnet_fineweb_compression.yaml \
    # --resume_from_checkpoint outputs/comp/nl_4_std_0.2_sl_10_test/checkpoint-56250

echo "done"
