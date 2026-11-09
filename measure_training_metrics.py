from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader

from src.data import ExternalFractureDataset, FracAtlasDataset
from src.model import load_checkpoint


def main():
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frac = FracAtlasDataset(root, "train", training=False)
    bone = ExternalFractureDataset(root, root / "outputs" / "bonefract_train.csv", training=False)
    pooled = ConcatDataset([frac, bone])
    loader = DataLoader(pooled, batch_size=64, shuffle=False, num_workers=6,
                        pin_memory=device.type == "cuda", persistent_workers=True)
    model, checkpoint = load_checkpoint(str(root / "artifacts" / "best_model.pt"), device)
    truth_parts, score_parts = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True)).squeeze(1)
            truth_parts.append(labels.numpy().astype(np.int64))
            score_parts.append(torch.sigmoid(logits).cpu().numpy())
    truth = np.concatenate(truth_parts)
    scores = np.concatenate(score_parts)
    threshold = float(checkpoint.get("threshold", 0.5))
    predicted = scores >= threshold
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    result = {
        "split": "pooled_training_posthoc",
        "sample_count": int(truth.size),
        "threshold": threshold,
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "fracture_recall_sensitivity": float(recall_score(truth, predicted, zero_division=0)),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "note": "Measured after training on the training pool; an in-sample diagnostic, not a generalization estimate.",
    }
    (root / "outputs" / "training_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
