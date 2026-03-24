#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"

exp_dir="outputs/vavae_mpnet_nl_2_std_0.2"

python scripts/eval_mteb.py \
    --task-type Classification \
    --model-type bottleneck \
    --config ${exp_dir}/config.yaml \
    --checkpoint ${exp_dir}/checkpoint-2075000/model.safetensors \
    --output-dir mteb_results/${exp_dir}-2075000/ \
    --batch-size 64

echo "done"
