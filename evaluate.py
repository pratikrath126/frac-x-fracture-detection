from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from torch.utils.data import ConcatDataset, DataLoader

from src.data import ExternalFractureDataset, FracAtlasDataset
from src.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved FRAC-X checkpoint on the held-out test split")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/test_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--include-bonefract", action="store_true",
                        help="Evaluate on a pooled test set from FracAtlas and BoneFract")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(str(args.checkpoint), device)
    fracatlas_test = FracAtlasDataset(project_root, "test")
    datasets = [("FracAtlas", fracatlas_test)]
    dataset = fracatlas_test
    if args.include_bonefract:
        bonefract_test = ExternalFractureDataset(
            project_root, project_root / "outputs" / "bonefract_test.csv", training=False
        )
        datasets.append(("BoneFract", bonefract_test))
        dataset = ConcatDataset([fracatlas_test, bonefract_test])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    truth, scores = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            truth.extend(labels.int().tolist())

    truth = np.asarray(truth)
    scores = np.asarray(scores)
    threshold = float(checkpoint.get("threshold", 0.5))
    predicted = scores >= threshold
    fpr, tpr, _ = roc_curve(truth, scores, drop_intermediate=True)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    metrics = {
        "split": "test",
        "sample_count": int(len(truth)),
        "threshold_from_validation": threshold,
        "accuracy": float((truth == predicted).mean()),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "fracture_recall_sensitivity": float(recall_score(truth, predicted, zero_division=0)),
        "specificity": float(matrix[0, 0] / max(1, matrix[0].sum())),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "pr_auc": float(average_precision_score(truth, scores)),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }
    per_dataset = {}
    offset = 0
    for name, part in datasets:
        size = len(part)
        part_truth = truth[offset:offset + size]
        part_predictions = predicted[offset:offset + size]
        part_scores = scores[offset:offset + size]
        part_matrix = confusion_matrix(part_truth, part_predictions, labels=[0, 1])
        per_dataset[name] = {
            "sample_count": int(size),
            "fracture_count": int(part_truth.sum()),
            "accuracy": float((part_truth == part_predictions).mean()),
            "fracture_recall_sensitivity": float(recall_score(part_truth, part_predictions, zero_division=0)),
            "specificity": float(part_matrix[0, 0] / max(1, part_matrix[0].sum())),
            "precision": float(precision_score(part_truth, part_predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(part_truth, part_scores)),
            "pr_auc": float(average_precision_score(part_truth, part_scores)),
            "confusion_matrix_labels_0_1": part_matrix.tolist(),
        }
        offset += size
    metrics["per_dataset"] = per_dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved test metrics to {args.output}")


if __name__ == "__main__":
    main()
