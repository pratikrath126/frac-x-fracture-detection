from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from src.data import FracAtlasDataset
from src.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Choose a fracture-recall target using validation data only")
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/resnet50/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/best_model.pt"))
    args = parser.parse_args()
    if not 0 < args.target_recall <= 1:
        raise SystemExit("--target-recall must be greater than 0 and at most 1")

    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(str(args.checkpoint), device)
    loader = DataLoader(FracAtlasDataset(root, "validation"), batch_size=32, shuffle=False, num_workers=0)
    truth, scores = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            truth.extend(labels.int().tolist())
    truth, scores = np.asarray(truth), np.asarray(scores)
    positive_scores = np.sort(scores[truth == 1])[::-1]
    required_true_positives = math.ceil(args.target_recall * len(positive_scores))
    threshold = float(positive_scores[required_true_positives - 1])
    predicted = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(truth, predicted, labels=[0, 1]).ravel())
    metrics = {
        "split": "validation",
        "target_recall": args.target_recall,
        "threshold": threshold,
        "sample_count": int(len(truth)),
        "fracture_recall_sensitivity": float(recall_score(truth, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "pr_auc": float(average_precision_score(truth, scores)),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
    }
    checkpoint["threshold"] = threshold
    checkpoint["threshold_selection"] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    output_metrics = args.output.parent / "validation_recall_target_metrics.json"
    output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved validation-selected threshold to {args.output}")


if __name__ == "__main__":
    main()
