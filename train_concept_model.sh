#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="2,3"
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
NP=2

accelerate launch --num_processes $NP --config_file ./accel_configs/fp32.yaml \
    scripts/train_concept_model.py \
    --config configs/train/cm_fineweb.yaml

echo "done"
