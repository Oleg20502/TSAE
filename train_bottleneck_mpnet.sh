#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
NP=4

accelerate launch --num_processes $NP --config_file ./accel_configs/fp32.yaml \
    scripts/train_bottleneck.py \
    --config configs/train/ae_mpnet_fineweb_noise.yaml

echo "done"
