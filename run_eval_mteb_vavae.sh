#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"

python scripts/eval_mteb.py \
    --task-type Classification \
    --model-type bottleneck \
    --config configs/train/ae_mpnet_fineweb_nl_2_std_0.2_sl_10.yaml \
    --checkpoint outputs/vavae_mpnet_nl_2_std_0.2_sl_10/checkpoint-739000/model.safetensors \
    --output-dir mteb_results/vavae_mpnet_nl_2_std_0.2_sl_10-739000/ \
    --batch-size 32 \

echo "done"
