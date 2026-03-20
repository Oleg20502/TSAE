# Text Semantic Auto Encoder (TSAE)

## Installation

Use different setups for different GPU types.

### A100 (Ampere / newer)

```bash
conda env create -n tsae python=3.10
pip install -e .
```

### 1080 Ti (Pascal)

DeepSpeed needs `CUDA_HOME` (CUDA toolkit). Use a conda env that provides PyTorch and the CUDA toolkit:

1. Create and activate the env:
   ```bash
   conda env create -f envs/conda-pascal.yaml
   conda activate tsae-1080
   ```
2. Install TSAE in editable mode:
   ```bash
   cp pyproject.pascal.toml pyproject.toml
   pip install -e .
   ```
Don't use bf16 presicion.

If you get `MissingCUDAException: CUDA_HOME does not exist`, set it before training:
   ```bash
   export CUDA_HOME=$CONDA_PREFIX
   ```

Optional: to remove the DeepSpeed `async_io` / `libaio` warning on Debian/Ubuntu, install `libaio-dev` (e.g. `sudo apt-get install libaio-dev`). Training runs without it; only async I/O is affected.

## Dataset preparation (recommended for Wikipedia)

To avoid long preprocessing at training time (paragraph split + shuffle on 6M+ rows), prepare the dataset once and load from disk when training:

Run once; downloads HF dataset, splits paragraphs, shuffles, writes train/val:
   ```bash
   python scripts/prepare_dataset.py --configs configs/preprocess/fineweb_10bt_ml_16.yaml
   ```

## Launching training

   ```bash
   accelerate launch --num_processes 4 --config_file ./accel_configs/fp32.yaml \
        scripts/train_bottleneck.py \
        --config configs/train/ae_mpnet_fineweb.yaml \
   ```
