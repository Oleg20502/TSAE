#!/bin/bash
set -e

# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
NP=4

accelerate launch --num_processes $NP --config_file ./accel_configs/fp32_ds_s2.yaml \
    scripts/train_bottleneck.py --configs \
        configs/model/bottleneck_mpnet.yaml \
        configs/datasets/fineweb_10bt.yaml \
        configs/train/bottleneck_ae.yaml

echo "done"
