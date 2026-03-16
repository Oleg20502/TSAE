"""Push preprocessed dataset to HF."""

from datasets import load_from_disk, DatasetDict

train_ds = load_from_disk("../datasets/fineweb_10bt_ml_64/train")
val_ds = load_from_disk("../datasets/fineweb_10bt_ml_64/validation")

ds = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
})

print(len(ds))
print(ds)
print(ds["train"][0])

ds.push_to_hub("okashurin/fineweb_10bt_ml_64")
