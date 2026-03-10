#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"

python scripts/eval_mteb.py \
    --model-type bottleneck \
    --config configs/model/bottleneck_mpnet.yaml \
    --checkpoint outputs/vavae_mpnet_nl_2_std_0.2_sl_10/checkpoint-425000/model.safetensors \
    --output-dir mteb_results/vavae_mpnet_nl_2_std_0.2_sl_10-425000/ \
    --batch-size 32 \

echo "done"
