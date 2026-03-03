#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"

python scripts/eval_mteb.py \
    --model-type st \
    --st-model sentence-transformers/all-MiniLM-L12-v2 \
    --output-dir mteb_results/all-MiniLM-L12-v2/ \
    --batch-size 32

echo "done"
