#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="0"

python scripts/eval_mteb.py \
    --model-type bottleneck \
    --config configs/model/bottleneck_bert.yaml \
    --checkpoint outputs/vavae_bert_nl_2/checkpoint-606000/model.safetensors \
    --output-dir mteb_results/vavae/ \
    --batch-size 32

echo "done"
