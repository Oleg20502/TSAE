#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"

python scripts/eval_mteb.py \
    --model-type bottleneck \
    --config configs/model/bottleneck_no_repr.yaml \
    --checkpoint outputs/bottleneck_no_repr/checkpoint-200000/model.safetensors \
    --output-dir mteb_results/vavae_no_repr/ \
    --batch-size 32 \
    --no-repr \

echo "done"
