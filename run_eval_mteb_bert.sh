#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="2"

python scripts/eval_mteb.py \
    --task-type Classification \
    --model-type st \
    --st-model sentence-transformers/all-MiniLM-L12-v2 \
    --output-dir mteb_results/all-MiniLM-L12-v2_ml-16/ \
    --max-length 16 \
    --batch-size 32

echo "done"
