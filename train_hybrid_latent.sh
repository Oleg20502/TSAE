#!/bin/bash
set -e

export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES="1"
NP=1

accelerate launch --num_processes $NP --config_file ./accel_configs/bf16_single_gpu.yaml \
    scripts/train_hybrid_latent_model.py \
    --config configs/train/hybrid_gsm8k.yaml

echo "done"
