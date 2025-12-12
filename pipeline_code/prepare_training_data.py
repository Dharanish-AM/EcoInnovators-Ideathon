"""
Fetch imagery tiles for training coordinates and create synthetic PV masks.
For real training, replace synthetic masks with manually annotated data.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from dotenv import load_dotenv
from .image_fetcher import ImageFetcher

# Load environment variables from .env file
load_dotenv()


def generate_synthetic_pv_mask(image: np.ndarray, has_solar: int = 1, seed: int = 42) -> np.ndarray:
    """
    Generate PV mask based on has_solar label and image analysis.
    Uses brightness variation to identify potential rooftop regions.
    """
    np.random.seed(seed)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # If no solar panels, return empty mask
    if not has_solar:
        return mask
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find regions with relatively uniform brightness (potential rooftops)
    # Use adaptive thresholding to identify structures
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, 5)
    
    # Find contours
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contours (likely buildings/rooftops)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Use top 1-3 largest contours as rooftops
        num_rooftops = min(np.random.randint(1, 4), len(contours))
        
        for i in range(num_rooftops):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            
            # Only process reasonably sized contours
            if area < 1000:
                continue
            
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            
            # Create panel area - use 60-85% of rooftop
            coverage = np.random.uniform(0.6, 0.85)
            panel_w = int(w_cnt * coverage)
            panel_h = int(h_cnt * coverage)
            
            # Random offset within rooftop
            offset_x = np.random.randint(0, max(1, w_cnt - panel_w))
            offset_y = np.random.randint(0, max(1, h_cnt - panel_h))
            
            x1 = max(0, x + offset_x)
            y1 = max(0, y + offset_y)
            x2 = min(w, x1 + panel_w)
            y2 = min(h, y1 + panel_h)
            
            # Draw filled rectangle for PV area
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                # Add grid pattern to simulate individual panels (15-25px spacing)
                grid_spacing = np.random.randint(15, 26)
                for gx in range(x1, x2, grid_spacing):
                    cv2.line(mask, (gx, y1), (gx, y2), 200, 1)
                for gy in range(y1, y2, grid_spacing):
                    cv2.line(mask, (x1, gy), (x2, gy), 200, 1)
    
    return mask


def prepare_dataset(
    excel_path: str,
    output_dir: Path,
    api_key: str,
    zoom: int = 20,
    tile_size: int = 640,
    train_split: float = 0.8,
    max_samples: int = None,
):
    df = pd.read_excel(excel_path)
    
    # Limit samples if specified, ensuring balanced classes (50/50)
    if max_samples is not None:
        # Separate positives and negatives
        pos = df[df["has_solar"] == 1]
        neg = df[df["has_solar"] == 0]
        
        n_per_class = max_samples // 2
        
        # Sample with replacement if not enough data, though here we have enough
        # But to be safe, use min
        n_pos = min(len(pos), n_per_class)
        n_neg = min(len(neg), n_per_class)
        
        print(f"Sampling {n_pos} positives and {n_neg} negatives for balance...")
        
        df_balanced = pd.concat([
            pos.sample(n=n_pos, random_state=42),
            neg.sample(n=n_neg, random_state=42)
        ])
        
        # Shuffle again
        df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        # Just shuffle if taking all
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    train_images = images_dir / "train"
    train_masks = masks_dir / "train"
    val_images = images_dir / "val"
    val_masks = masks_dir / "val"
    
    for d in [train_images, train_masks, val_images, val_masks]:
        d.mkdir(exist_ok=True)
    
    fetcher = ImageFetcher(
        provider="mapbox",
        tile_size=tile_size,
        api_keys={"mapbox": api_key, "google": None, "esri": None},
    )
    
    split_idx = int(len(df) * train_split)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching tiles"):
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        sample_id = str(row.get("sampleid", row.get("sample_id", f"sample_{idx}")))
        has_solar = int(row.get("has_solar", 1))
        
        # Fetch image
        fetch_result = fetcher.fetch_image(lat, lon, zoom)
        image = fetch_result.image
        
        # Generate synthetic mask based on has_solar label
        mask = generate_synthetic_pv_mask(image, has_solar=has_solar, seed=idx)
        
        # Determine split
        is_train = idx < split_idx
        img_dir = train_images if is_train else val_images
        msk_dir = train_masks if is_train else val_masks
        
        # Save
        img_path = img_dir / f"{sample_id}.png"
        msk_path = msk_dir / f"{sample_id}.png"
        
        cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(msk_path), mask)
    
    print(f"Dataset prepared: {split_idx} train, {len(df) - split_idx} val samples")
    print(f"Images: {images_dir}")
    print(f"Masks: {masks_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset from Excel")
    parser.add_argument("--input_xlsx", required=True, help="Path to Excel with coordinates")
    parser.add_argument("--output_dir", default="data/training_dataset", help="Output directory")
    parser.add_argument("--mapbox_key", default=os.getenv("MAPBOX_API_KEY"), help="Mapbox API key (default from .env)")
    parser.add_argument("--zoom", type=int, default=20, help="Zoom level")
    parser.add_argument("--tile_size", type=int, default=640, help="Tile size")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to process (default: all)")
    
    args = parser.parse_args()
    
    prepare_dataset(
        excel_path=args.input_xlsx,
        output_dir=Path(args.output_dir),
        api_key=args.mapbox_key,
        zoom=args.zoom,
        tile_size=args.tile_size,
        train_split=args.train_split,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
