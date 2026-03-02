#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="0"

python scripts/eval_mteb.py \
    --model-type st \
    --st-model sentence-transformers/nli-bert-base \
    --output-dir mteb_results/nli-bert/ \
    --batch-size 32

echo "done"
