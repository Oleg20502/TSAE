"""Evaluate function for running evaluation on autoencoder."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.rae_text import RAEText
from src.eval.reconstruction_metrics import (
    token_accuracy,
    perplexity_from_loss,
)
from src.eval.semantic_metrics import cosine_sim_batch


@torch.no_grad()
def evaluate(
    model: RAEText,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_samples: int = 1000,
    num_qualitative: int = 5,
) -> dict:
    """Run full evaluation on the given dataloader."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_sem = 0.0
    n_batches = 0
    all_tok_acc = []
    all_cos_sim = []
    qualitative = []

    bos_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 101
    eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 102
    n_seen = 0

    for batch in dataloader:
        if n_seen >= max_samples:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass for loss
        out = model(**batch)
        total_loss += out["loss"].item()
        total_recon += out["l_recon"].item()
        total_sem += out["l_sem"].item()
        n_batches += 1

        # Greedy decode for metrics
        gen_ids = model.generate_greedy(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            max_length=batch["input_ids"].size(1),
        )

        # Token accuracy (compare against labels where not -100)
        labels = batch["labels"]
        min_len = min(gen_ids.size(1), labels.size(1))
        acc = token_accuracy(gen_ids[:, :min_len], labels[:, :min_len])
        all_tok_acc.append(acc)

        # Decode to text
        orig_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # Semantic similarity
        sent_emb, _ = model.repr_encoder.encode(batch["input_ids"], batch["attention_mask"])
        z_emb = model.encode(batch["input_ids"], batch["attention_mask"])

        
        sims = cosine_sim_batch(sent_emb, z_emb)
        all_cos_sim.extend(sims.cpu().tolist())

        # Collect qualitative examples
        if len(qualitative) < num_qualitative:
            for orig, gen in zip(orig_texts, gen_texts):
                if len(qualitative) >= num_qualitative:
                    break
                qualitative.append({"original": orig, "reconstruction": gen})

        n_seen += batch["input_ids"].size(0)

    # Aggregate
    cos_tensor = torch.tensor(all_cos_sim)
    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "recon_loss": total_recon / max(n_batches, 1),
        "sem_loss": total_sem / max(n_batches, 1),
        "perplexity": perplexity_from_loss(total_recon / max(n_batches, 1)),
        "token_accuracy": sum(all_tok_acc) / max(len(all_tok_acc), 1),
        "sem_cos_mean": cos_tensor.mean().item() if len(cos_tensor) > 0 else 0.0,
        "sem_cos_median": cos_tensor.median().item() if len(cos_tensor) > 0 else 0.0,
        "num_samples": n_seen,
    }
    return {"metrics": metrics, "qualitative": qualitative}