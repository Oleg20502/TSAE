#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"

python scripts/eval_mteb.py \
    --model-type st \
    --st-model sentence-transformers/all-mpnet-base-v2 \
    --output-dir mteb_results/all-mpnet-base-v2_ml-16/ \
    --max-length 16 \
    --batch-size 32

echo "done"
