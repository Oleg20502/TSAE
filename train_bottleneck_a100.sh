#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"
NP=1

accelerate launch --num_processes $NP --config_file ./accel_configs/fp32.yaml \
    scripts/train_bottleneck.py --config configs/train/ae_mpnet_fineweb_compression.yaml

echo "done"
