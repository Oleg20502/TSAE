#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="0,1"
NP=2

accelerate launch --num_processes $NP --config_file ./accel_configs/bf16.yaml \
    scripts/train_bottleneck_no_repr.py \
    --config configs/train/ae_no_repr_fineweb.yaml

echo "done"
