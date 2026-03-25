#!/bin/bash
set -euo pipefail

python scripts/prepare_hybrid_lm_dataset.py --configs configs/preprocess/hybrid_wiki_k4.yaml
