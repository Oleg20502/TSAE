from pathlib import Path

import numpy as np
import datasets
import mteb
from transformers import AutoTokenizer
from datasets import Value

MAX_LEN = 16
TOKENIZER_NAME = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

def get_hf_spec(task):
    meta = task.metadata
    ds_info = meta.dataset
    hf_name = ds_info.get("path")
    hf_subset = ds_info.get("name")
    revision = ds_info.get("revision")
    eval_splits = meta.eval_splits or ["test"]
    return hf_name, hf_subset, revision, eval_splits

def string_columns(ds_split):
    cols = []
    for k, feat in ds_split.features.items():
        if isinstance(feat, Value) and feat.dtype == "string":
            cols.append(k)
    return cols

def token_len_stats(ds_split, sample=20000, seed=0):
    n = len(ds_split)
    if sample is not None and n > sample:
        ds_split = ds_split.shuffle(seed=seed).select(range(sample))

    cols = string_columns(ds_split)
    # Drop obvious non-text ids if present
    cols = [c for c in cols if c.lower() not in {"id", "idx", "guid"}]
    if not cols:
        return None, None

    lengths = []
    bs = 512
    for i in range(0, len(ds_split), bs):
        batch = ds_split[i : i + bs]
        # take per-example max over all string columns (handles STS pairs like sentence1/sentence2)
        per_col_lens = []
        for c in cols:
            texts = batch[c]
            # Some datasets may have None; normalize
            texts = [t if isinstance(t, str) else "" for t in texts]
            enc = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
            per_col_lens.append([len(x) for x in enc["input_ids"]])
        per_ex_max = np.max(np.array(per_col_lens), axis=0)
        lengths.extend(per_ex_max.tolist())

    lengths = np.array(lengths)
    stats = {
        "n": int(len(lengths)),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": float(lengths.max()),
        f"frac_gt": float((lengths > MAX_LEN).mean()),
    }
    return stats, cols

def analyze_tasks(tasks, sample=20000):
    out = []
    for task in tasks:
        hf_name, hf_subset, revision, eval_splits = get_hf_spec(task)
        if hf_name is None:
            continue

        try:
            ds = datasets.load_dataset(hf_name, hf_subset, revision=revision)
        except (ValueError, Exception) as e:
            print(f"  [skip] {task.metadata.name}: {e}")
            continue
        for split in eval_splits:
            if split not in ds:
                continue
            stats, cols = token_len_stats(ds[split], sample=sample)
            if stats:
                out.append((task.metadata.name, split, cols, stats))
    return out

# 1) English STS tasks
sts_tasks = mteb.get_tasks(task_types=["STS"], languages=["eng"])
sts_report = analyze_tasks(sts_tasks, sample=None)  # STS sets are small → full pass

# 2) English linear probing ≈ English Classification tasks
clf_tasks = mteb.get_tasks(task_types=["Classification"], languages=["eng"])
clf_report = analyze_tasks(clf_tasks, sample=20000)  # sample large sets

def print_report(report, title):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    for name, split, cols, stats in report:
        print(f"{name} | {split} | cols={cols}")
        print(f"  n={stats['n']}, p50={stats['p50']:.1f}, p95={stats['p95']:.1f}, p99={stats['p99']:.1f}, max={stats['max']:.0f},  frac>{MAX_LEN}={stats['frac_gt']*100:.2f}%")

def save_report(report, path):
    import json
    data = {name: {**stats, "split": split, "cols": cols} for name, split, cols, stats in report}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {path}")

print_report(sts_report, "STS (English) token lengths")
print_report(clf_report, "Classification (English) token lengths (linear probing)")

save_report(sts_report, "mteb_results/sts_token_lengths.json")
save_report(clf_report, "mteb_results/clf_token_lengths.json")