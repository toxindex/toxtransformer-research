"""Evaluate each target with a declared number of observed context properties."""

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import collate


@torch.no_grad()
def evaluate(model, rows, property_names, device, batch_size, context_count=0):
    if context_count < 0 or batch_size < 1 or not rows:
        raise ValueError("Evaluation requires rows, a positive batch size, and nonnegative context")
    model.eval()
    predictions = []
    pending = []

    def flush():
        batch = collate(pending, device)
        logits = model(*batch)
        target_positions = batch[3].sum(dim=1) - 1
        target_logits = logits[torch.arange(len(pending), device=device), target_positions]
        probabilities = torch.softmax(target_logits, dim=-1)[:, 1].cpu().tolist()
        for row, probability in zip(pending, probabilities):
            prop, label = row["pairs"][-1]
            predictions.append(
                {
                    "smiles": row["smiles"],
                    "property_id": property_names[prop],
                    "value": label,
                    "probability": probability,
                    "n_context": len(row["pairs"]) - 1,
                }
            )
        pending.clear()

    for row in rows:
        for target, label in row["pairs"]:
            # Context consists only of other observed labels for this held-out molecule.
            context = [(p, v) for p, v in row["pairs"] if p != target][:context_count]
            pending.append({**row, "pairs": context + [(target, label)]})
            if len(pending) == batch_size:
                flush()
    if pending:
        flush()
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["property_id"]].append(prediction)

    def metrics(items):
        truth = np.array([item["value"] for item in items])
        probability = np.array([item["probability"] for item in items])
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        return {
            "n": len(items),
            "positive": int(truth.sum()),
            "log_loss": float(
                -(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)).mean()
            ),
            "accuracy": float(((probability >= 0.5) == truth).mean()),
            "brier_score": float(((probability - truth) ** 2).mean()),
            "roc_auc": float(roc_auc_score(truth, probability)) if len(set(truth)) == 2 else None,
        }

    per_property = {prop: metrics(items) for prop, items in sorted(grouped.items())}
    aucs = [result["roc_auc"] for result in per_property.values() if result["roc_auc"] is not None]
    return {
        "overall": metrics(predictions),
        "per_property": per_property,
        "macro_roc_auc": float(np.mean(aucs)) if aucs else None,
        "properties_with_auc": len(aucs),
        "properties_without_auc": len(per_property) - len(aucs),
        "requested_context": context_count,
        "mean_actual_context": float(np.mean([p["n_context"] for p in predictions])),
    }, predictions
