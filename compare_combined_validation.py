from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader

from src.data import ExternalFractureDataset, FracAtlasDataset
from src.model import load_checkpoint
from train import best_accuracy_threshold, get_probabilities, evaluate_scores


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints on the pooled validation split")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frac_data = FracAtlasDataset(root, "validation")
    bone_data = ExternalFractureDataset(root, root / "outputs" / "bonefract_validation.csv", training=False)
    combined = ConcatDataset([frac_data, bone_data])
    loader = DataLoader(combined, batch_size=args.batch_size, shuffle=False, num_workers=6,
                        pin_memory=device.type == "cuda", persistent_workers=True)
    labels = np.concatenate((frac_data.rows["label"].to_numpy(dtype=np.int64),
                             bone_data.rows["label"].to_numpy(dtype=np.int64)))
    print(f"Pooled validation: {len(labels)} images ({len(frac_data)} FracAtlas, {len(bone_data)} BoneFract)")

    candidates = []
    for checkpoint_path in args.checkpoints:
        model, checkpoint = load_checkpoint(str(checkpoint_path), device)
        truth, scores = get_probabilities(model, loader, device)
        if not np.array_equal(truth, labels):
            raise RuntimeError("Validation order/labels changed unexpectedly.")
        threshold = best_accuracy_threshold(truth, scores)
        predictions = scores >= threshold
        matrix = confusion_matrix(truth, predictions, labels=[0, 1])
        n_frac = len(frac_data)
        metrics = {
            "checkpoint": str(checkpoint_path),
            "model": checkpoint["model_name"],
            "epoch": checkpoint.get("epoch"),
            "threshold": threshold,
            "combined_accuracy": float(accuracy_score(truth, predictions)),
            "combined_fracture_recall": float(recall_score(truth, predictions, zero_division=0)),
            "combined_precision": float(precision_score(truth, predictions, zero_division=0)),
            "combined_confusion_matrix": matrix.tolist(),
            "combined_roc_auc": float(evaluate_scores(truth, scores, threshold)["roc_auc"]),
            "combined_pr_auc": float(evaluate_scores(truth, scores, threshold)["pr_auc"]),
            "FracAtlas_validation": evaluate_scores(truth[:n_frac], scores[:n_frac], threshold),
            "BoneFract_validation": evaluate_scores(truth[n_frac:], scores[n_frac:], threshold),
        }
        candidates.append(metrics)

    candidates.sort(key=lambda item: (item["combined_accuracy"], item["combined_fracture_recall"]), reverse=True)
    for item in candidates:
        print(item)
    print("Validation selection: highest accuracy on the pooled FracAtlas + BoneFract validation set.")


if __name__ == "__main__":
    main()
