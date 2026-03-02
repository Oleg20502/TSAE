#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1"

python scripts/eval_mteb_nli_bert_cls.py \
    --config configs/model/bottleneck_bert.yaml \
    --output-dir mteb_results/nli-bert-cls/ \
    --batch-size 32

echo "done"
