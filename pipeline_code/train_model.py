"""
Train PV segmentation model using prepared dataset.
Uses a simple U-Net architecture with ResNet18 encoder.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .architecture import SimpleUNet


class PVDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(list(images_dir.glob("*.png")))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Fallback transform if none provided
            image = image.astype("float32") / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask).unsqueeze(0)

        # Ensure mask dim
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask


def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.2,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        bce_loss = nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        d_loss = dice_loss(outputs, masks)
        loss = bce_loss + d_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(outputs, masks)
            d_loss = dice_loss(outputs, masks)
            loss = bce_loss + d_loss
            
            total_loss += loss.item()
    
    return total_loss / len(loader)


def train_model(
    dataset_dir: Path,
    output_model_path: Path,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 0.001,
    log_dir: Path | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = PVDataset(
        images_dir=dataset_dir / "images" / "train",
        masks_dir=dataset_dir / "masks" / "train",
        transform=get_training_augmentation(),
    )
    val_dataset = PVDataset(
        images_dir=dataset_dir / "images" / "val",
        masks_dir=dataset_dir / "masks" / "val",
        transform=get_validation_augmentation(),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    
    best_val_loss = float("inf")
    metrics_log_dir = log_dir or Path("training_logs")
    metrics_log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_log_dir / "metrics.jsonl"
    summary_path = metrics_log_dir / "training_summary.json"
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with metrics_path.open("a", encoding="utf-8") as log_f:
            json.dump(epoch_record, log_f)
            log_f.write("\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model, str(output_model_path))
            print(f"Saved best model to {output_model_path}")
    
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    summary = {
        "best_val_loss": best_val_loss,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dataset": str(dataset_dir),
        "model_path": str(output_model_path),
        "metrics_log": str(metrics_path),
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    with summary_path.open("w", encoding="utf-8") as summary_f:
        json.dump(summary, summary_f, indent=2)
    print(f"Wrote metrics to {metrics_path} and summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PV segmentation model")
    parser.add_argument("--dataset_dir", default="data/training_dataset", help="Dataset directory")
    parser.add_argument("--output_model", default="trained_model/pv_segmentation.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log_dir", default="training_logs", help="Directory to write metrics and logs")
    
    args = parser.parse_args()
    
    train_model(
        dataset_dir=Path(args.dataset_dir),
        output_model_path=Path(args.output_model),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_dir=Path(args.log_dir),
    )


if __name__ == "__main__":
    main()
