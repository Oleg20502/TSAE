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