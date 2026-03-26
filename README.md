# TSAE

TSAE is a research codebase for learning latent text representations and using them in downstream generative models. The repository currently contains three main experiment families:

- `bottleneck` models: text autoencoders with a latent bottleneck over sentence-transformer representations
- `concept` models: sequence models trained in the AE latent space
- `hybrid latent` models: GPT-2 based reasoning / language modeling experiments conditioned on AE latents

## Repository Layout

```text
.
├── accel_configs/          # Accelerate launcher configs for fp32/fp16/bf16 and DeepSpeed
├── configs/
│   ├── eval/              # Evaluation configs
│   ├── preprocess/        # Dataset preprocessing configs
│   └── train/             # Training configs for AE / concept / hybrid models
├── envs/                  # Conda environment definitions
├── scripts/               # Entry points for preprocess, train, eval, hub upload
├── src/
│   ├── backbones/         # Representation encoder helpers
│   ├── data/              # Dataset loaders, collators, preprocess helpers
│   ├── eval/              # Reconstruction / semantic / hybrid metrics
│   ├── losses/            # Reconstruction and semantic losses
│   ├── models/            # AE, concept model, hybrid latent model
│   ├── trainers/          # Training loops
│   └── utils/             # Config loading, checkpoint helpers, training math
├── tests/                 # Unit tests for shapes, bottleneck logic, checkpoint selection
├── train_*.sh             # Machine-specific launch wrappers
└── run_eval_*.sh          # Example evaluation wrappers
```

## Environment

The repo targets Python 3.10 and uses Hugging Face `transformers`, `datasets`, `accelerate`, `sentence-transformers`, and `mteb`.

Recommended setup for Pascal GPUs:

```bash
conda env create -f envs/conda-pascal.yaml
conda activate tsae-1080
pip install -e ".[dev]"
```

If you want to follow the Pascal-specific packaging notes in the repo comments, you can also copy `pyproject.pascal.toml` over `pyproject.toml` before installation.

## Important Config Notes

Most YAML configs in `configs/` contain paths to datasets, checkpoints, and output directories from the original training environment. Before running anything, update at least:

- `data.preprocessed_dir`
- `train.output_dir`
- `model.ae_config_path`
- `model.ae_checkpoint_path`
- any dataset `cache_dir`

The top-level `.sh` launchers also hardcode `CUDA_VISIBLE_DEVICES`; treat them as examples rather than portable scripts.

## Basic Workflows

### 1. Train a bottleneck autoencoder on preprocessed text

Preprocess a text dataset into GPT-2-sized chunks:

This step loads the source dataset via Hugging Face `datasets`, downloading it if needed into the configured `cache_dir`, and then writes the processed dataset to `data.preprocessed_dir`.

```bash
python scripts/prepare_dataset.py --configs configs/preprocess/fineweb_10bt_ml_16.yaml
```

Then train an MPNet-based bottleneck AE:

```bash
accelerate launch --num_processes 1 --config_file accel_configs/fp32_single_gpu.yaml   scripts/train_bottleneck.py   --config configs/train/ae_mpnet_fineweb.yaml
```

Useful variants:

- `configs/train/ae_mpnet_fineweb_compression.yaml`
- `configs/train/ae_mpnet_fineweb_noise.yaml`
- `configs/train/ae_mpnet_fineweb_sem_loss.yaml`
- `scripts/train_bottleneck_no_repr.py` with `configs/train/ae_no_repr_fineweb.yaml`

### 2. Train a concept model over AE latents

First build chunk-grouped sequences:

Like the AE preprocessing step, this will fetch the raw dataset through Hugging Face if it is not already cached, then save the prepared sequences to `data.preprocessed_dir`.

```bash
python scripts/prepare_cm_dataset.py   --config configs/preprocess/cm_fineweb_preprocess.yaml
```

Then train:

```bash
accelerate launch --num_processes 4 --config_file accel_configs/fp16.yaml   scripts/train_concept_model.py   --config configs/train/cm_fineweb.yaml
```

### 3. Train a hybrid latent model

There are two main paths in the repo:

- `configs/train/hybrid_gsm8k.yaml`: reasoning experiments on GSM8K-style data
- `configs/train/hybrid_wikipedia_k8.yaml`: LM-style training on a preprocessed Wikipedia dataset

For the Wikipedia workflow, preprocess first:

This command also loads the raw dataset from Hugging Face, reusing `cache_dir` when available, and writes the hybrid-LM samples to `data.preprocessed_dir`.

```bash
python scripts/prepare_hybrid_lm_dataset.py   --configs configs/preprocess/hybrid_wiki_k8.yaml
```

Then launch training:

```bash
accelerate launch --num_processes 2 --config_file accel_configs/bf16.yaml   scripts/train_hybrid_latent_model.py   --config configs/train/hybrid_wikipedia_k8.yaml
```

## Evaluation

Run MTEB for a trained bottleneck model:

```bash
python scripts/eval_mteb.py   --task-type Classification   --model-type bottleneck   --config outputs/<experiment>/config.yaml   --checkpoint outputs/<experiment>/checkpoint-<step>/model.safetensors   --output-dir mteb_results/<run_name>   --batch-size 64
```

You can also evaluate sentence-transformer baselines:

```bash
python scripts/eval_mteb.py   --model-type st   --st-model sentence-transformers/all-MiniLM-L12-v2   --output-dir mteb_results/all-MiniLM-L12-v2
```

Other helpers:

- `scripts/evaluate_rae_style.py` for reconstruction-style evaluation
- `scripts/eval_mteb_nli_bert_cls.py` for an NLI BERT CLS baseline
- `scripts/push_to_hub.py` for publishing saved datasets

## Development

Run tests with:

```bash
pytest
```

Outputs are typically written to `outputs/`, logs to `logs/`, and MTEB results to `mteb_results/`.