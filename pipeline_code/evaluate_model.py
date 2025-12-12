"""
Evaluate trained PV segmentation model and generate comprehensive metrics.
Computes accuracy, precision, recall, F1-score, Dice coefficient, and IoU.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .train_model import PVDataset, get_validation_augmentation


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Model output probabilities (0-1)
        targets: Ground truth binary masks (0-1)
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with metrics
    """
    # Binarize predictions
    pred_binary = (predictions > threshold).astype(np.int32)
    targets_binary = targets.astype(np.int32)
    
    # Flatten for pixel-level metrics
    pred_flat = pred_binary.flatten()
    target_flat = targets_binary.flatten()
    
    # Pixel-level metrics
    accuracy = accuracy_score(target_flat, pred_flat)
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    
    # Dice coefficient
    intersection = np.sum(pred_binary * targets_binary)
    dice = 2.0 * intersection / (np.sum(pred_binary) + np.sum(targets_binary) + 1e-6)
    
    # IoU (Intersection over Union)
    union = np.sum(pred_binary) + np.sum(targets_binary) - intersection
    iou = intersection / (union + 1e-6)
    
    # Per-sample metrics (image-level)
    n_samples = predictions.shape[0]
    sample_dices = []
    for i in range(n_samples):
        pred_sample = pred_binary[i]
        target_sample = targets_binary[i]
        intersection_s = np.sum(pred_sample * target_sample)
        dice_s = 2.0 * intersection_s / (np.sum(pred_sample) + np.sum(target_sample) + 1e-6)
        sample_dices.append(dice_s)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "dice_coefficient": float(dice),
        "iou": float(iou),
        "mean_sample_dice": float(np.mean(sample_dices)) if sample_dices else 0.0,
        "std_sample_dice": float(np.std(sample_dices)) if sample_dices else 0.0,
    }


def evaluate_model(
    model_path: Path,
    dataset_dir: Path,
    device: torch.device = None,
) -> dict:
    """
    Evaluate model on validation dataset and return metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with weights_only=False for compatibility
    model = torch.load(str(model_path), map_location=device, weights_only=False)
    model.eval()
    
    # Load validation dataset
    val_dataset = PVDataset(
        images_dir=dataset_dir / "images" / "val",
        masks_dir=dataset_dir / "masks" / "val",
        transform=get_validation_augmentation(),
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    print("Evaluating model on validation set...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets = masks.cpu().numpy()
            
            all_preds.append(probs)
            all_targets.append(targets)
    
    # Concatenate all predictions and targets
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Remove channel dimension if present
    if preds.shape[1] == 1:
        preds = preds.squeeze(1)
    if targets.shape[1] == 1:
        targets = targets.squeeze(1)
    
    # Compute metrics
    metrics = compute_metrics(preds, targets)
    
    return metrics


def save_model_metrics(
    model_path: Path,
    metrics: dict,
    dataset_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
):
    """
    Save model metrics to a JSON file for tracking across runs.
    """
    metrics_file = Path("training_logs/model_metrics.json")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics if available
    if metrics_file.exists():
        with open(metrics_file) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    
    # Create entry for this training run
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
        },
        "metrics": metrics,
    }
    
    all_metrics.append(entry)
    
    # Save updated metrics
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Model metrics saved to {metrics_file}")
    
    return metrics_file


def print_metrics(metrics: dict):
    """Pretty print metrics."""
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1-Score:           {metrics['f1_score']:.4f}")
    print(f"Dice Coefficient:   {metrics['dice_coefficient']:.4f}")
    print(f"IoU (Jaccard):      {metrics['iou']:.4f}")
    print(f"Mean Sample Dice:   {metrics['mean_sample_dice']:.4f} ± {metrics['std_sample_dice']:.4f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PV segmentation model")
    parser.add_argument("--model_path", default="trained_model/pv_segmentation.pt", help="Path to trained model")
    parser.add_argument("--dataset_dir", default="data/training_dataset", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size used in training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate used in training")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Evaluate model
    metrics = evaluate_model(
        model_path=Path(args.model_path),
        dataset_dir=Path(args.dataset_dir),
        device=device,
    )
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics
    save_model_metrics(
        model_path=Path(args.model_path),
        metrics=metrics,
        dataset_dir=Path(args.dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
