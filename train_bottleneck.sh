#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
NP=4

accelerate launch --num_processes $NP --config_file ./accel_configs/fp32_ds_s2.yaml \
    scripts/train_bottleneck.py --configs \
        configs/model/bottleneck_simcse.yaml \
        configs/data/wiki_sentences.yaml \
        configs/train/bottleneck_autoencoder.yaml

echo "done"
