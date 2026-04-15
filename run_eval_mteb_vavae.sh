#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"

exp_dir="outputs/ae/nl_1_rstd_0.05"

python scripts/eval_mteb.py \
    --task-type STS \
    --model-type bottleneck \
    --config ${exp_dir}/config.yaml \
    --checkpoint ${exp_dir}/checkpoint-116000/model.safetensors \
    --output-dir mteb_results/${exp_dir}-116000/ \
    --batch-size 64

echo "done"
