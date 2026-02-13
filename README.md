# Text Semantic Auto Encoder (TSAE)

## Installation

### Environment per GPU type

Use different dependency sets for different servers.

##### A100 (and Ampere / newer)

Use the default `pyproject.toml` at the repo root:

```bash
pip install -e .
```

Safe to use `bf16: true` in train configs.

#### GTX 1080Ti (Pascal)

On the Pascal server, use the Pascal-specific project file:

```bash
# Replace project file then install (one-time per clone)
cp pyproject.pascal.toml pyproject.toml
pip install -e .
```

- In train configs set **fp16: true**, **bf16: false** (Pascal has no bf16 tensor cores).
- Prefer installing a PyTorch CUDA 11.8 wheel for best compatibility.
- To check DeepSpeed on that machine: `ds_report` or `python -m deepspeed.env_report`.

