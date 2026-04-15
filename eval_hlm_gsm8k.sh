#!/bin/bash

python scripts/eval_hybrid_latent_generate.py \
  --config /home/okashurin/okashurin/TSAE/outputs/hybrid_latent_5/nl_1_sl_1.0/config.yaml \
  --checkpoint /home/okashurin/okashurin/TSAE/outputs/hybrid_latent_5/nl_1_sl_1.0/checkpoint-282000/model.safetensors \
  --device cuda:0