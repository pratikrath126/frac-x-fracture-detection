from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from src.data import ExternalFractureDataset, FracAtlasDataset
from src.model import IMAGE_SIZE, MODEL_NAMES, create_model, freeze_backbone, unfreeze_top_stage


def parse_args():
    parser = argparse.ArgumentParser(description="Train a stronger pretrained FRAC-X classifier")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dynamic-oversampling", action="store_true",
                        help="Use original training images only and oversample fractures with fresh online augmentation")
    parser.add_argument("--include-bonefract", action="store_true",
                        help="Pool BoneFract and FracAtlas train/validation partitions for combined evaluation")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Initialize from a compatible trained checkpoint before fine-tuning")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_probabilities(model, loader, device):
    model.eval()
    truth, scores = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True)).squeeze(1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            truth.extend(labels.int().tolist())
    return np.asarray(truth), np.asarray(scores)


def best_accuracy_threshold(truth, scores):
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    results = [(accuracy_score(truth, scores >= threshold), threshold) for threshold in thresholds]
    best_accuracy = max(item[0] for item in results)
    # Among equally accurate cutoffs, use the one with higher fracture recall.
    best = [threshold for accuracy, threshold in results if accuracy == best_accuracy]
    return float(min(best))


def evaluate_scores(truth, scores, threshold):
    predictions = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(truth, predictions, labels=[0, 1]).ravel())
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "fracture_recall_sensitivity": float(recall_score(truth, predictions, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "pr_auc": float(average_precision_score(truth, scores)),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
    }


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_dir = args.output_dir or (project_root / "artifacts" / args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; model: {args.model}; seed: {args.seed}")

    fracatlas_train = FracAtlasDataset(project_root, "train", training=True,
                                       include_augmented=not args.dynamic_oversampling)
    train_data = fracatlas_train
    sampler = None
    if args.include_bonefract:
        if args.dynamic_oversampling:
            raise SystemExit("Use either --dynamic-oversampling or --include-bonefract, not both.")
        manifest = project_root / "outputs" / "bonefract_train.csv"
        if not manifest.exists():
            raise SystemExit(f"Missing {manifest}; run prepare_bonefract.py first.")
        bonefract_train = ExternalFractureDataset(project_root, manifest, training=True)
        train_data = ConcatDataset([fracatlas_train, bonefract_train])
        source_groups = np.concatenate((np.zeros(len(fracatlas_train), dtype=np.int64),
                                        np.ones(len(bonefract_train), dtype=np.int64)))
        all_labels = np.concatenate((fracatlas_train.rows["label"].to_numpy(dtype=np.int64),
                                     bonefract_train.rows["label"].to_numpy(dtype=np.int64)))
        sample_weights = np.empty(len(all_labels), dtype=np.float64)
        print(f"BoneFract supplemental training images: {len(bonefract_train)} "
              f"({int((all_labels[source_groups == 1] == 1).sum())} fracture, "
              f"{int((all_labels[source_groups == 1] == 0).sum())} non-fracture)")
        for source_id in (0, 1):
            for label in (0, 1):
                group = (source_groups == source_id) & (all_labels == label)
                sample_weights[group] = 1.0 / int(group.sum())
                print(f"  {'FracAtlas' if source_id == 0 else 'BoneFract'} "
                      f"{'fracture' if label else 'non-fracture'}: {int(group.sum())}")
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                        num_samples=24000, replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
        print("Balanced source/class sampling: 6,000 draws per group per epoch (with replacement).")
    validation_data = FracAtlasDataset(project_root, "validation")
    if args.include_bonefract:
        bonefract_validation = ExternalFractureDataset(
            project_root, project_root / "outputs" / "bonefract_validation.csv", training=False
        )
        validation_data = ConcatDataset([validation_data, bonefract_validation])
        print(f"Combined validation images: {len(validation_data)} "
              f"({len(validation_data.datasets[0])} FracAtlas, {len(validation_data.datasets[1])} BoneFract)")
    if args.dynamic_oversampling:
        labels = train_data.rows["label"].to_numpy(dtype=np.int64)
        counts = np.bincount(labels, minlength=2)
        sample_weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=2 * int(counts.max()), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
        print(f"Dynamic oversampling from originals only: {counts[1]} fracture and {counts[0]} non-fracture source images")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers, pin_memory=device.type == "cuda",
                              persistent_workers=args.workers > 0)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=device.type == "cuda",
                                   persistent_workers=args.workers > 0)

    model = create_model(args.model, pretrained=args.init_checkpoint is None).to(device)
    if args.init_checkpoint is not None:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if initial_checkpoint.get("model_name") != args.model:
            raise SystemExit("--init-checkpoint model architecture must match --model.")
        model.load_state_dict(initial_checkpoint["model_state_dict"])
        print(f"Initialized from checkpoint: {args.init_checkpoint}")
    freeze_backbone(model, args.model)
    head = model.fc if args.model == "resnet50" else model.classifier
    criterion = nn.BCEWithLogitsLoss()
    head_lr = 1e-4 if args.init_checkpoint is not None else 8e-4
    stage_lr = 1e-5 if args.init_checkpoint is not None else 8e-5
    optimizer = torch.optim.AdamW(head.parameters(), lr=head_lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    best_loss = float("inf")
    patience = 4
    stale_epochs = 0
    history = []
    best_path = output_dir / "best_model.pt"

    for epoch in range(args.epochs):
        if epoch == 2:
            unfreeze_top_stage(model, args.model)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                          lr=stage_lr, weight_decay=2e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"{args.model} epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / (len(train_loader.sampler) if train_loader.sampler is not None else len(train_data))
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                val_loss += criterion(model(images).squeeze(1), labels).item() * images.size(0)
        val_loss /= len(validation_data)
        truth, scores = get_probabilities(model, validation_loader, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_roc_auc": float(roc_auc_score(truth, scores)),
            "validation_pr_auc": float(average_precision_score(truth, scores)),
        })
        print(f"loss train={train_loss:.4f} val={val_loss:.4f}; "
              f"val ROC-AUC={history[-1]['validation_roc_auc']:.4f}; "
              f"val PR-AUC={history[-1]['validation_pr_auc']:.4f}")

        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "model_name": args.model,
                        "image_size": IMAGE_SIZE, "seed": args.seed, "epoch": epoch + 1}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    truth, scores = get_probabilities(model, validation_loader, device)
    threshold = best_accuracy_threshold(truth, scores)
    checkpoint["threshold"] = threshold
    checkpoint["validation_metrics"] = evaluate_scores(truth, scores, threshold)
    torch.save(checkpoint, best_path)
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: epoch {checkpoint['epoch']}; validation threshold={threshold:.4f}")
    print(json.dumps(checkpoint["validation_metrics"], indent=2))
    print("Training used only the supplied training and validation splits; test data was not loaded.")


if __name__ == "__main__":
    main()
