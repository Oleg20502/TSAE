#!/bin/bash
set -e

export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES="0,1"
NP=2

accelerate launch --num_processes $NP --config_file ./accel_configs/bf16.yaml \
    scripts/train_hybrid_latent_model.py \
    --config configs/train/hybrid_wikipedia_k8.yaml

echo "done"
